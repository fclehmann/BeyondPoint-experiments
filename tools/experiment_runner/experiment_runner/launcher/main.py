from functools import partial
from itertools import chain
from experiment_runner.launcher.utils import batched
import retry
import json
import os
from pathlib import Path
from random import sample
import shutil
import subprocess
from multiprocessing import Process
import sys
from typing import Any, Callable, Literal, Optional
import uuid

import optuna
import optuna.storages.journal
import optuna.trial
from submitit import AutoExecutor
import submitit
import tomlkit
import pandas as pd
import tqdm
import yaml
import logging
from concurrent.futures import ProcessPoolExecutor

_logger = logging.getLogger(__name__)

from experiment_runner.launcher.compat.omniopt import (
    create_sampler,
    CLIObjective,
    parse_search_space,
)

from experiment_runner.run.experiment_config import (
    ExperimentConfig,
    read_experiment_config,
)
from dataclasses import asdict


def wait_job_with_rescheduling(
    job: submitit.Job,
    executor: submitit.Executor,
    f: Callable,
    params: list[dict],
    n: int = 5,
) -> tuple[Optional[Any], submitit.Job]:
    i = 0
    while True:
        try:
            result = job.result()
            return (result, job)
        except Exception as e:

            if i + 1 == n:
                _logger.warning(
                    f"Job failed (after {n} retries) for params: {params}", exc_info=e
                )
                return (None, job)

            _logger.info(f"Retrying ({i+1}/{n}) job for params: {params}")
            job = executor.submit(f, params)

        i += 1


def requeue_stale_trials(study: optuna.Study, path_runs: Path):
    stale_trials = study.get_trials(
        states=[optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.FAIL]
    )
    failed_trial_cb = optuna.storages.RetryFailedTrialCallback(max_retry=None)
    for trial in stale_trials:
        run_uid = trial.user_attrs.get("run_uid", None)

        if run_uid is not None:
            _logger.info(f"Removing run path of running trial={trial.number} ...")
            shutil.rmtree(path_runs / run_uid)

        if trial.state == optuna.trial.TrialState.RUNNING:
            _logger.info(f"Seting state of running trial={trial.number} to failed.")
            study.tell(trial=trial.number, state=optuna.trial.TrialState.FAIL)

        _logger.info(
            f"Requeuing running trial={trial.number} with params={trial.params} ..."
        )
        failed_trial_cb(study=study, trial=trial)


def launch_sweepers(path_experiment_config: Path):
    exp_conf = asdict(read_experiment_config(path_experiment_config))
    seeds = exp_conf["DATA"]["seed"]
    exp_work_dir = Path(exp_conf["ENV_PARAMS"]["base_params"]["work_dir"])

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(exp_work_dir / "launcher_logs.log"),
            logging.StreamHandler(stream=sys.stderr),
        ],
    )

    if isinstance(seeds, int):
        seeds = [seeds]

    n_parallel_studies = exp_conf["LAUNCHER"].get("n_parallel_studies", 25)

    with ProcessPoolExecutor(max_workers=n_parallel_studies) as pool:
        for s in seeds:
            pool.submit(run_hopt, exp_conf=exp_conf, hopt_seed=s)


def run_hopt(exp_conf: dict, hopt_seed: Optional[int] = None):
    _logger.info(f"Starting study with hopt_seed={hopt_seed}...")
    try:
        # Exp config
        exp_work_dir = Path(exp_conf["ENV_PARAMS"]["base_params"]["work_dir"])

        # Parameters
        db_port = exp_conf["LAUNCHER"]["db_port"]
        db_host = exp_conf["LAUNCHER"]["db_host"]
        db_pass = os.environ.get("DB_PASS")
        db_uri = f"postgresql://postgres:{db_pass}@{db_host}:{db_port}"
        mode = exp_conf["LAUNCHER"]["mode"]
        run_retries = exp_conf["LAUNCHER"]["run_retries"]
        simd_enabled = exp_conf["LAUNCHER"]["simd"]
        run_time = exp_conf["LAUNCHER"]["run_time"]

        n_runs_per_job = exp_conf["LAUNCHER"][
            "n_runs_per_job"
        ]  # Number of runs executed within same job in parallel
        batch_size = exp_conf["LAUNCHER"][
            "batch_size"
        ]  # Number of parameters asked from study without telling result

        # Paths
        path_hopt = exp_work_dir / "hopt" / f"seed={hopt_seed}"
        path_hopt_jobs = path_hopt / "outputs"
        path_runs = exp_work_dir / "results"
        path_runs.mkdir(parents=True, exist_ok=True)
        path_hopt.mkdir(exist_ok=True, parents=True)

        # Study
        study_name = f'{exp_conf["ENV_PARAMS"]["base_params"]["experiment_name"]}_SEED_{hopt_seed}'
        _logger.info(f"Study name: {study_name}")
        storage = optuna.storages.RDBStorage(url=db_uri)
        sampler = create_sampler(exp_conf, seed=hopt_seed)
        study = optuna.create_study(
            storage=storage, study_name=study_name, load_if_exists=True, sampler=sampler
        )

        # Optimization
        search_space = parse_search_space(dimensions=exp_conf["DIMENSIONS"])
        n_trials_configured = exp_conf["DATA"]["max_evals"]

        if mode == "continue":
            n_success_trials = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            n_iter = max(n_trials_configured - n_success_trials, 0)
        elif mode == "append":
            n_iter = n_trials_configured
        else:
            raise Exception(f"Mode {mode} is not supported.")

        # Reque the trials previously running or failed, instead of letting study to create new ones
        requeue_stale_trials(study=study, path_runs=path_runs)

        _logger.info(f"Will run N={n_iter} iterations.")

        objective = CLIObjective(
            objective_program_path=exp_work_dir
            / exp_conf["ENV_PARAMS"]["base_params"]["objective_program_name"],
            runs_path=path_runs,
            data_option_name=exp_conf["LAUNCHER"]["data_root_option"],
            data_path=Path(
                exp_conf["FIXED_EXPERIMENT_PARAMS"][
                    exp_conf["LAUNCHER"]["data_root_option"]
                ]
            ),
            data_archive_path=Path(exp_conf["LAUNCHER"]["data_root_archive"]),
        )

        executor = submitit.AutoExecutor(
            folder=str(path_hopt_jobs),
        )
        executor.update_parameters(
            job_name=f"{study_name}",
            ntasks_per_node=1,
            nodes=1,
            gpus_per_task=exp_conf["DATA"]["num_gpus_per_worker"],
            cpus_per_task=exp_conf["EXTRA_HARDWARE_PARAMS"]["num_cpus_per_worker"],
            mem_per_cpu=exp_conf["DATA"]["mem_per_cpu"],
            gpus_per_node=1,
            time=run_time,  # mins
        )

        if simd_enabled:
            os.environ["SLURM_HINT"] = "multithread"

        for batch_i, batch in tqdm.tqdm(
            enumerate(batched(range(n_iter), n=batch_size))
        ):
            batch_trial_numbers = []
            batch_params = []
            batch_jobs = []
            batch_run_uids = []
            batch_trials: list[optuna.trial.Trial] = []

            # Prepare parameter counstellations
            for batch_internal_run_i in batch:
                trial = retry.retry(tries=5, exceptions=Exception, delay=120)(
                    study.ask
                )(search_space)

                # Possibly trial is requed => use existing run_uid. It will diverge with actual
                # number of the trial in the study, but will still match the run_uid in user attrs and
                # failed_trial (=number of failed/original trial) in system attrs of the trial.
                # NOTE: in study history dataframe the trial number matches the number in database
                # and "actual" number is stored under system attributes as "failed_trial"
                run_uid = trial.user_attrs.get("run_uid", None)

                if run_uid is None:
                    run_uid = (
                        f"S_{hopt_seed}-N_{trial.number}-ID_{str(uuid.uuid4())[:6]}"
                    )

                trial.set_user_attr(key="run_uid", value=run_uid)
                trial.set_user_attr(key="hopt_seed", value=hopt_seed)

                params = {"ExperimentConfig.run_uid": run_uid, **trial.params}
                _logger.info(
                    f"Asking trial {trial.number} with params: {trial.params} and run_uid={run_uid}"
                )

                batch_params.append(params)
                batch_trials.append(trial)
                batch_trial_numbers.append(trial.number)
                batch_run_uids.append(batch_run_uids)

            per_job_params = list(batched(batch_params, n=n_runs_per_job))
            per_job_trials = list(batched(batch_trials, n=n_runs_per_job))
            per_job_trial_nums = list(batched(batch_trial_numbers, n=n_runs_per_job))

            # Submit jobs
            with executor.batch():
                for parallel_runs_params_batch, parallel_trial_numbers_batch in zip(
                    per_job_params,
                    per_job_trial_nums,
                ):
                    _logger.info(
                        f"Submitting trials as single task: {parallel_trial_numbers_batch}"
                    )
                    batch_jobs.append(
                        executor.submit(objective, parallel_runs_params_batch)
                    )

            # Await for results and handle outputs
            job: submitit.Job
            for i, job in enumerate(batch_jobs):
                job_return, job = wait_job_with_rescheduling(
                    job=job,
                    executor=executor,
                    f=objective,
                    params=per_job_params[i],
                    n=run_retries,
                )
                for j, trial in enumerate(per_job_trials[i]):
                    v = None if job_return is None else job_return[j]
                    _logger.info(f"Trial {trial.number} result: {v}")
                    tell = retry.retry(
                        tries=5, exceptions=Exception, delay=120, backoff=1.5
                    )(study.tell)
                    try:
                        if job_return is None:
                            _logger.info(f"Trial {trial.number} result: None")
                            tell(
                                trial.number,
                                values=None,
                                state=optuna.trial.TrialState.FAIL,
                            )
                        else:
                            tell(trial.number, values=job_return[j])
                    except RuntimeError:
                        pass

        df_hist: pd.DataFrame = study.trials_dataframe()
        df_hist.to_csv(path_hopt / "trials.csv")
        study_result = {"params": study.best_params, "value": study.best_value}
        _logger.info(f"Study result: {study_result}")
        with open(path_hopt / "study_result.json", "w") as fp:
            json.dump(study_result, fp)
    except Exception as e:
        _logger.warning(
            f"An exception occured during the study (hopt_seed = {hopt_seed}), skipping study.",
            exc_info=e,
        )

import concurrent.futures
from functools import partial
import logging
import os
import pathlib
import typing
from typing import Annotated, List, Optional, Type, Tuple
import multiprocessing

import concurrent
from experiment_runner.backup.backup import BackupHandler
from experiment_runner.post_process.post_processor import PostProcessor
from experiment_runner.run.experiment_config import BaseExperimentEnvParams, ExperimentConfig, read_experiment_config
from matplotlib.pyplot import stem
import pandas as pd
import typer
import glob

import experiment_runner.run.run_experiment
import experiment_runner.export_omniopt_results
import experiment_runner.status.hopt
import experiment_runner.status.opt

import submitit

_logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)

@app.command(name="run")
def run_experiment(
    experiment_config_path: pathlib.Path,
    clean_workdir: bool = typer.Option(
        False, "--clean_workdir", help="Remove * from workdir on taurus"
    ),
    submit_jobs: bool = typer.Option(False, "--submit_jobs", help="Submit jobs"),
    log_level: int = typer.Option(logging.INFO, "--log_level", help="Logging level"),
):
    """
    Runs experiment defined by given experiment config.
    """
    experiment_runner.run.run_experiment.run_experiment(
        experiment_config_path=experiment_config_path,
        clean_workdir=clean_workdir,
        submit_jobs=submit_jobs,
        log_level=log_level,
    )


@app.command(name="continue")
def continue_experiment(experiment_path: pathlib.Path):
    """
    Runs experiment defined by given experiment config.
    """
    experiment_runner.run.run_experiment.continue_experiment(
        experiment_path=experiment_path
    )

@app.command(name="status")
def experiment_status(experiments_glob: str):
    for path_str in sorted(glob.glob(experiments_glob)):

        p = pathlib.Path(path_str)
        if "hopt" in p.stem:
            experiment_runner.status.hopt.get_experiment_state(p)
        else:
            experiment_runner.status.opt.get_experiment_status_handler(p.stem)(
                p
            ).show_status()

        print("\n\n")

@app.command("get-backup-status")
def get_backup_status(
    experiments: List[pathlib.Path]
):
    logging.basicConfig()
    _logger.setLevel(logging.DEBUG)

    for path_e in experiments:
        _logger.debug(f"Veryfying {path_e} ...")
        status = BackupHandler.last_backup_job_status(path_e)
        _logger.info(f"Backup job status of \n{path_e.stem}: {status}.")
        if status != "COMPLETED":
            _logger.warning(f"Backup is not completed for {path_e.stem}!")
        

@app.command("backup")
def backup(
    output_path: Annotated[
        pathlib.Path,
        typer.Option("--output_path", "-o", file_okay=False, dir_okay=True),
    ],
    experiments: Annotated[
        List[pathlib.Path],
        typer.Option("--experiment", "-e", file_okay=True, dir_okay=False),
    ],
    run_patterns: Annotated[List[str], typer.Option("--run_patterns", "-rp")],
    hopt_patterns: Annotated[List[str], typer.Option("--hopt_patterns", "-hp")],
    compression: Annotated[str, typer.Option("--compression", "-c")] = "xz",
    lazy: Annotated[bool, typer.Option("--lazy")] = False,
    resolve_path_from_ec: Annotated[
        bool, typer.Option("--resolve_path_from_ec")
    ] = False,
    n_workers: Annotated[int, typer.Option("--n_workers", "-w")] = 1,
    as_job: Annotated[bool, typer.Option("--as_job")] = False,
    mem: Annotated[
        str,
        typer.Option("--mem"),
    ] = "12GB",
    time: Annotated[
        str,
        typer.Option("--time"),
    ] = "64:00:00",
    cpus: Annotated[
        int,
        typer.Option("--cpus"),
    ] = 6,

):
    logging.basicConfig()
    _logger.setLevel(logging.DEBUG)
    
    backup_handler = BackupHandler(
        compression=compression, # type: ignore
        run_glob_patterns=run_patterns,
        hopt_glob_patterns=hopt_patterns,
        lazy=lazy,
        resolve_workdir_from_ec=resolve_path_from_ec,
    )

    paths_ec = [
        pathlib.Path(p)
        for path_ec_glob in experiments
        for p in glob.glob(str(path_ec_glob))
    ]
    paths_archive = [
        output_path / f"{path_ec.stem}.tar.{compression}" for path_ec in paths_ec
    ]

    for path_ec, path_archive in zip(paths_ec, paths_archive):
        ec: ExperimentConfig = read_experiment_config(path_ec)
        base_params: BaseExperimentEnvParams = ec.ENV_PARAMS.base_params
        path_experiment = pathlib.Path(base_params.work_dir)
        
        if resolve_path_from_ec:
            path_experiment = path_ec.parent
        
        path_job = backup_handler.prepare_job_folder(path_experiment)

        if as_job:
            # Have to create executor each time to set different logs folder...
            executor = submitit.SlurmExecutor(folder=path_job)

            executor.update_parameters(
                job_name=f"isml-archive-{path_ec.stem}",
                ntasks_per_node=1,
                cpus_per_task=cpus,
                mem=mem,
                time=time,
                account="p_scads"
            )
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)

        _logger.info(f"Starting backup job for: {path_experiment}...")
        executor.submit(backup_handler, path_experiment, path_archive)
        _logger.info(f"Archive job path: {path_job}.")

        if not as_job:
            executor.shutdown(wait=True)

@app.command("process")
def process(
    path_experiment: pathlib.Path,
    metrics: Annotated[
        List[str],
        typer.Option("--metric", "-m"),
    ] = list(),
    workers: Annotated[
        int,
        typer.Option("--workers"),
    ] = 32,
    as_job: Annotated[
        bool,
        typer.Option("--as_job"),
    ] = False,
    mem: Annotated[
        str,
        typer.Option("--mem"),
    ] = "32GB",
    time: Annotated[
        str,
        typer.Option("--time"),
    ] = "12:00:00",
):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    _logger.info(f"workers={workers}, mem={mem}, time={time}")

    kwargs = dict(path_experiment=path_experiment, metrics=metrics, n_workers=workers)

    if as_job:
        path_job = path_experiment / "post-process-job"
        path_job.mkdir(exist_ok=True)
        executor = submitit.SlurmExecutor(folder=path_job)
        executor.update_parameters(
            nodes=1,
            job_name=f"isml-pp-{path_experiment.stem}",
            ntasks_per_node=1,
            cpus_per_task=workers,
            mem=mem,
            time=time,
        )
        executor.submit(PostProcessor.process, **kwargs)
    else:
        PostProcessor.process(**kwargs) # type: ignore


if __name__ == "__main__":
    app()

import logging
from typing import Any, Awaitable, Callable, Optional
import uuid
from experiment_runner.launcher.utils import batched
import optuna
import retry
import submitit
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

_logger = logging.getLogger(__name__)


class MultiTrialObjective:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class ObjectiveExecutorWithRetriesWrapper:
    def __init__(
        self,
        executor: submitit.Executor,
        objective: MultiTrialObjective,
        n: int = 5,
    ) -> None:
        self._executor = executor
        self._objective = objective
        self._n = n

    def execute(self, params: list[dict]) -> Any:
        i = 0
        while True:
            try:
                result = job.result()  # uses time.sleep() internally => non-blocking!
                return (result, job)
            except Exception as e:

                if i + 1 == self._n:
                    _logger.warning(
                        f"Job failed (after {self._n} retries) for params: {params}",
                        exc_info=e,
                    )
                    return (None, job)

                _logger.info(f"Retrying ({i+1}/{self._n}) job for params: {params}")
                job = self._executor.submit(self._objective, params)

            i += 1


class StudyLauncher:
    def __init__(
        self,
        batch_size: int,
        n_precompute_trials: int,
        n_trials_per_job: int,
        seed: int,
    ) -> None:
        self._batch_size = batch_size
        self._n_precompute_trials = n_precompute_trials
        self._n_trials_per_job = n_trials_per_job
        self._seed = seed

    def create_paths(self): ...

    def obtain_study(self) -> optuna.Study: ...

    def obtain_rest_iter_num(self) -> int: ...

    def obtain_completed_iter_num(self) -> int: ...

    def handle_dangling_trials(self, study: optuna.Study): ...

    def obtain_search_space(
        self,
    ) -> dict[str, optuna.distributions.BaseDistribution]: ...

    def obtain_sampler(self) -> optuna.samplers.BaseSampler: ...

    def obtain_executor(self) -> submitit.Executor: ...

    def ask_trial(self) -> optuna.Trial: ...

    def store_trial(self, trial: optuna.Trial): ...

    def save_history(self): ...

    @staticmethod
    def create_run_uid(hopt_seed, trial_number, uuid4: Optional[str] = None):
        if uuid4 is None:
            uuid4 = str(uuid.uuid4())
        return f"S_{hopt_seed}-N_{trial_number}-ID_{str(uuid.uuid4())[:6]}"

    def handle_per_job_batch(
        self,
        n_iters: int,
        study: optuna.Study,
        objective_executor: ObjectiveExecutorWithRetriesWrapper,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ):

        trials_batch = []
        params_batch = []
        uids_batch = []

        for _ in range(n_iters):
            trial = retry.retry(tries=5, exceptions=Exception, delay=120, backoff=1.5)(study.ask)(
                search_space
            )

            # Possibly trial is requed => use existing run_uid. It will diverge with actual
            # number of the trial in the study, but will still match the run_uid in user attrs and
            # failed_trial (=number of failed/original trial) in system attrs of the trial.
            # NOTE: in study history dataframe the trial number matches the number in database
            # and "actual" number is stored under system attributes as "failed_trial"
            run_uid = trial.user_attrs.get("run_uid", None)

            if run_uid is None:
                run_uid = self.create_run_uid(
                    hopt_seed=self._seed, trial_number=trial.number
                )

            trial.set_user_attr(key="run_uid", value=run_uid)
            trial.set_user_attr(key="hopt_seed", value=self._seed)

            params = {"ExperimentConfig.run_uid": run_uid, **trial.params}
            _logger.info(
                f"Asking trial {trial.number} with params: {trial.params} and run_uid={run_uid}"
            )

            params_batch.append(params)
            trials_batch.append(trial)

        job_return, job = objective_executor.execute(params_batch)

        for j, trial in enumerate(trials_batch):
            v = None if job_return is None else job_return[j]
            _logger.info(f"Trial {trial.number} result: {v}")
            tell = retry.retry(tries=5, exceptions=Exception, delay=120, backoff=1.5)(study.tell)
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
            except Exception as e:
                _logger.warning(f"Trial {trial.number} store failed.", exc_info=e)

    def launch(self):
        self.create_paths()
        study = self.obtain_study()
        n_iter = self.obtain_rest_iter_num()
        self.handle_dangling_trials(study)

        objective = MultiTrialObjective()

        executor = self.obtain_executor()

        n_iters_completed = self.obtain_completed_iter_num()

        n_iters_pre = min(0, self._n_precompute_trials - n_iters_completed)
        n_iters_post = min(0, n_iter - n_iters_pre)

        with ThreadPoolExecutor(max_workers=self._batch_size) as pool:

            futures_pre = []
            for per_job_iters_batch in batched(
                range(n_iters_pre), self._n_trials_per_job
            ):
                future = pool.submit(
                    self.handle_per_job_batch,
                    n_iters=len(per_job_iters_batch),
                    study=study,
                )
                futures_pre.append(future)

            wait(futures_pre)

            futures_post = []
            for per_job_iters_batch in batched(
                range(n_iters_post), self._n_trials_per_job
            ):
                future = pool.submit(
                    self.handle_per_job_batch,
                    n_iters=len(per_job_iters_batch),
                    study=study,
                )
                futures_post.append(future)

            wait(futures_post)

import logging
import os
from pathlib import Path
import pickle
from typing import Any, Iterator, Literal, Sequence
from experiment_runner.run.experiment_config import (
    read_experiment_config,
    ExperimentConfig,
    BaseExperimentEnvParams,
)
import tarfile
from experiment_runner.utils import get_sacct_job_status
import submitit
import submitit.core.utils
import tqdm
from itertools import chain

class BackupHandler:
    def __init__(
        self,
        compression: Literal["gz", "xz"],
        run_glob_patterns: list[str],
        hopt_glob_patterns: Sequence[str],
        lazy: bool = False,
        resolve_workdir_from_ec: bool = False,
    ) -> None:
        self.compression = compression
        self.run_patterns = run_glob_patterns
        self.hopt_patterns = hopt_glob_patterns
        self.lazy = lazy
        self.resolve_workdir_from_ec = resolve_workdir_from_ec

    def get_results_files(self, path_experiment: Path) -> Iterator[Path]:
        return chain.from_iterable(
            (path_experiment / "results").glob(f"*/{g}") for g in self.run_patterns
        )

    def get_hopt_files(self, path_experiment: Path) -> Iterator[Path]:
        return chain.from_iterable(path_experiment.glob(g) for g in self.hopt_patterns)

    def get_data_files(self, path_experiment: Path) -> Iterator[Path]:
        return iter([path_experiment / "data"])

    def get_ec_files(self, path_experiment: Path) -> Iterator[Path]:
        return path_experiment.glob("*.toml")

    def prepare_job_folder(self, path_experiment: Path) -> Path:
        path_job = path_experiment / "job-archive"
        path_job.mkdir(exist_ok=True)
        return path_job

    def setup_logging(self, path_experiment: Path):
        path_job = self.prepare_job_folder(path_experiment)
        path_logs = path_job / "archive-logs.log"

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        file_handler = logging.FileHandler(path_logs)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger

    @staticmethod
    def last_backup_job_status(path_experiment: Path):
        logger = logging.getLogger(__name__)
        try:
            path_job = path_experiment / "job-archive"
            path_submisions = path_job.glob("./*_submission.sh")

            job_ids = []
            for p_s in path_submisions:
                name = p_s.stem
                job_id = name[:-len("_submission")]
                job_ids.append(int(job_id))
            
            job_ids = sorted(job_ids, reverse=True)

            status = get_sacct_job_status(job_id=job_ids[0])

            return status
        except Exception as e:
            logger.warning(f"Failed to get backup job status", exc_info=True)


    def __call__(self, path_experiment: Path, path_output_archive: Path):
        
        logger = self.setup_logging(path_experiment)

        logger.info(f"Backing up {path_experiment} to {path_output_archive}")

        files = chain(
            self.get_results_files(path_experiment),
            self.get_hopt_files(path_experiment),
            self.get_data_files(path_experiment),
            self.get_ec_files(path_experiment)
        )
        if not self.lazy:
            files = list(files)

        try:
            with tarfile.open(path_output_archive, f"w:{self.compression}") as tar: # type: ignore

                for path_file in tqdm.tqdm(
                    files, desc=f"Compressing {path_experiment.stem}"
                ):
                    path_file_rel = path_file.relative_to(path_experiment.parent)
                    tar.add(name=path_file, arcname=path_file_rel)
        except Exception as e:
            logger.exception(e)
            logger.warning(f"Will be removed: {path_output_archive}")
            os.remove(path_output_archive)

    
if __name__ == "__main__":
    BackupHandler.verify_backup_job(Path("/data/horse/ws/s4610340-isml/cifar10_clf/experiments_v2/008_cifar10_clf_hopt_tpe_SGDwlrs_bs128_s"))
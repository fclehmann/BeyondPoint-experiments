from collections import defaultdict
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Optional, Union
from uuid import uuid4

import pandas as pd
import torch

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only
from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
)


_logger = logging.getLogger(__name__)


class Run:
    def __init__(self, out_path: Path, run_uid: Optional[str] = None) -> None:
        if run_uid is None:
            self.uid = str(uuid4())
        else:
            self.uid = run_uid
        self.path = out_path / self.uid
        self.params = {}
        self.metrics = defaultdict(list)
        self.status = "ACTIVE"

    def create(self):
        _logger.info(f"Creating run with uid={self.uid}, full path: {self.path}")
        if self.path.exists():
            _logger.warning(f"Run {self.path} already exists and will be replaced!")
        self.path.mkdir(exist_ok=True, parents=True)
        return self

    def save(self):
        with open(self.path / "metrics.json", "w") as fp:
            json.dump(self.metrics, fp)
        with open(self.path / "params.json", "w") as fp:
            json.dump(self.params, fp)

    def finish(self):
        self.status = "FINISHED"

    def end(self):
        self.save()
        self.finish()

    def __enter__(self):
        self.create()
        return self

    def __exit__(self):
        self.end()


class FileSystemClient:

    def __init__(self, out_path: Union[str, Path]) -> None:
        self._out_path = Path(out_path)

    def start_run(self, run_id: Optional[str] = None, *args, **kwargs) -> Run:
        self._run = Run(out_path=self._out_path, run_uid=run_id)
        self._run.create()
        return self._run

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        synchronous: Optional[bool] = None,
        timestamp: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        self._run.metrics[key].append({"step": step, "value": value})

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[int] = None,
    ):
        for key, val in metrics.items():
            self.log_metric(key, val, step)

    def log_param(self, key: str, value: Any, synchronous: Optional[bool] = None):
        self._run.params[key] = value

    def log_params(
        self,
        params: dict[str, Any],
        synchronous: Optional[bool] = None,
        run_id: Optional[str] = None,
    ):
        self._run.params.update(params)

    def log_table(
        self,
        data: pd.DataFrame,
        artifact_file: str,
        run_id: Optional[str] = None,
    ):
        path_artifact_file = self._run.path / artifact_file
        path_artifact_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_json(path_artifact_file)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str,
        run_id: Optional[str] = None,
    ):
        path_artifact_file = self._run.path / artifact_path
        path_artifact_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, path_artifact_file)

    def log_torch_model(self, model: torch.nn.Module, artifact_file: str):
        torch.save(model, self._run.path / artifact_file)


class LightningFileSystemLogger(Logger):
    def __init__(self, out_path: Path, run_uid: str) -> None:
        super().__init__()
        self._initialized = False
        self._out_path = out_path
        self._run_uid = run_uid

    @property
    def name(self):
        return "FileSystemLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @property
    @rank_zero_experiment
    def experiment(self) -> "FileSystemClient":
        if self._initialized:
            return self._fs_client

        self._fs_client = FileSystemClient(out_path=self._out_path)
        self._run = self._fs_client.start_run(run_id=self._run_uid)
        self._initialized = True

        return self._fs_client

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        params = _convert_params(params)
        params = _flatten_dict(params)

        for k, v in params.items():
            params[k] = str(v)[:500]

        self.experiment.log_params(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        self.experiment.log_metrics(metrics=metrics, step=step)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        self._run.save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        self._run.end()

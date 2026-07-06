from pathlib import Path
from tempfile import tempdir
import tempfile
from typing import Any, Protocol
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from isml_train.tracking import FileSystemClient
from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_params,
    _flatten_dict,
)
import pandas as pd
import torch


class TrackingProtocol(Protocol):
    def log_metrics(self, metrics: dict[str, Any], step: int, **kwargs): ...

    def log_table(self, data: pd.DataFrame, artifact_file: str): ...

    def log_hyperparams(self, params: dict[str, Any]): ...

    def log_model(
        self,
        pytorch_model,
        artifact_path,
        signature,
    ): ...


class MlFlowLogger(MLFlowLogger):
    @rank_zero_only
    def log_model(
        self,
        pytorch_model,
        artifact_path,
        signature,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_model_tmp = Path(tmpdir) / artifact_path
            path_model_tmp.parent.mkdir(parents=True)
            torch.save(pytorch_model, path_model_tmp)
            self.experiment.log_artifact(self.run_id, path_model_tmp, artifact_path)

    @rank_zero_only
    def log_table(self, data: pd.DataFrame, artifact_file: str):
        self.experiment.log_table(self.run_id, data=data, artifact_file=artifact_file)


class FileSystemLogger(Logger):
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
    def log_table(self, data: pd.DataFrame, artifact_file: str):
        self.experiment.log_table(data=data, artifact_file=artifact_file)

    @rank_zero_only
    def log_model(
        self,
        pytorch_model,
        artifact_path,
        signature,
    ):
        self.experiment.log_pytorch_model(pytorch_model, artifact_path)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        self._run.end()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

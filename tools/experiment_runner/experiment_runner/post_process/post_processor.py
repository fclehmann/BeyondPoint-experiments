from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Type, Union

from experiment_runner.post_process.utils import read_ec
import pandas as pd

_logger = logging.getLogger(__name__)


class PostProcessor(ABC):
    def __init__(self, path_experiment: Path, n_workers: int) -> None:
        self.path_experiment = path_experiment
        self.n_workers = n_workers
        self.ec = read_ec(path_experiment)

    @property
    @abstractmethod
    def decision_metric(self) -> str: ...

    @property
    @abstractmethod
    def best_metric_decision_type(self) -> str: ...

    @property
    def path_runs(self) -> Path:
        return self.path_experiment / "results"

    @staticmethod
    @abstractmethod
    def supports(path_experiment: Path) -> bool: ...

    @abstractmethod
    def create_plots(
        self, df: pd.DataFrame, metrics: Optional[Sequence[str]] = None
    ) -> None: ...

    def save(
        self,
        data_varied: pd.DataFrame,
        example_params: dict[str, Any],
        path_output_file: Optional[Path] = None,
    ) -> None:
        if path_output_file is None:
            path_output_file = self.path_experiment / "data" / "processed" / "data.json"

        path_output_file.parent.mkdir(parents=True, exist_ok=True)

        if path_output_file.exists():
            _logger.warning(f"Will overwrite {path_output_file}!")

        result = {
            "varied": data_varied.reset_index().to_dict(orient="list"),
            "params": example_params,
            "decision_metric": self.decision_metric,
            "best_metric_decision_type": self.best_metric_decision_type,
        }

        with path_output_file.open("w+") as fp:
            json.dump(result, fp)

    @abstractmethod
    def process_varied(self) -> pd.DataFrame: ...

    @abstractmethod
    def process_fixed(self) -> dict[str, Any]: ...

    @staticmethod
    def process(
        path_experiment: Path, n_workers: int, metrics: Sequence[str] = tuple()
    ) -> None:
        from experiment_runner.post_process.hopt import post_processors as hopt_pcs
        from experiment_runner.post_process.opt import post_processors as opt_pcs

        processors: list[Type[PostProcessor]] = [
            *opt_pcs,
            *hopt_pcs
        ]

        for p_class in processors:
            if p_class.supports(path_experiment):
                _logger.info(f"Picked processor class: {p_class}")
                break
        else:
            raise Exception(
                f"Post processor not found for experiment: {path_experiment.stem}"
            )

        p = p_class(path_experiment=path_experiment, n_workers=n_workers)
        df = p.process_varied()
        ps = p.process_fixed()
        p.save(data_varied=df, example_params=ps)
        p.create_plots(df, metrics)

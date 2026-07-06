from abc import ABC, abstractmethod
from itertools import islice
import json
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import tomlkit

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import tqdm
import yaml

from operator import is_not

from experiment_runner.post_process.utils import macro_f1_score, read_ec
from experiment_runner.post_process.post_processor import PostProcessor

_logger = logging.getLogger(__name__)


class OptPostProcessor(PostProcessor):

    @property
    @abstractmethod
    def seed_attr(self) -> Sequence[str]: ...

    @abstractmethod
    def read_run_data(self, path_run: Path) -> dict[str, Any]: ...

    def _read_run_data_handling(self, path_run: Path) -> tuple[str, Optional[dict[str, Any]]]:
        uid = path_run.stem
        try:
            return (uid, self.read_run_data(path_run))
        except Exception as e:
            _logger.debug(f"Error when reading or parsing {uid}.", exc_info=True)
        return (uid, None)

    def read_runs(self) -> dict[str, dict[str, Any]]:
        path_results = self.path_runs
        pbar = islice(tqdm.tqdm(list(path_results.glob("*"))), None)
        with multiprocessing.Pool(self.n_workers) as p:
            d = p.imap(self._read_run_data_handling, pbar)
            d = filter(lambda x: x[1] is not None, d)
            d = dict(d)
        return d

    @abstractmethod
    def get_run_metrics(self, run_data: dict) -> dict[str, float]: ...

    @abstractmethod
    def get_run_seeds(self, run_data: dict) -> dict[str, int]: ...

    def get_metric_values(
        self,
        runs: dict[str, dict],
    ) -> pd.DataFrame:

        rows = []
        for experiment_uid, run_data in runs.items():
            metrics = self.get_run_metrics(run_data)
            seeds = self.get_run_seeds(run_data)
            rows.append({**seeds, **metrics})

        s = pd.DataFrame(data=rows).set_index(self.seed_attr)
        return s

    @abstractmethod
    def plot_metric(self, metric: pd.Series, ax: matplotlib.axes.Axes): ...

    def create_plots(self, df_metrics: pd.DataFrame, metrics: Sequence[str] = tuple()):
        path_plots = self.path_experiment / "graphics"
        path_plots.mkdir(exist_ok=True, parents=True)

        sns.set_style("darkgrid")

        for metric in metrics:
            fig: matplotlib.figure.Figure = plt.figure(figsize=(8, 6))
            ax: matplotlib.axes.Axes = fig.add_subplot(1, 1, 1)
            self.plot_metric(df_metrics[metric], ax=ax)
            fig.tight_layout()
            fig.savefig(path_plots / f"{metric}.png")
            fig.clear()


    def process_varied(self) -> pd.DataFrame:
        _logger.info("Reading runs ...")
        opt_results = self.read_runs()
        _logger.info(f"Obtained N={len(list(opt_results.keys()))} runs.")

        return self.get_metric_values(opt_results)

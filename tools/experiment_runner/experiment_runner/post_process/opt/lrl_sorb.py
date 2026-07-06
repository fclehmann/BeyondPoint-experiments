from experiment_runner.post_process.opt.post_processor import OptPostProcessor

from abc import ABC, abstractmethod
import json
import logging
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

from experiment_runner.post_process.utils import (
    macro_f1_score,
    macro_precision,
    macro_recall,
    string_to_numpy_array,
)

_logger = logging.getLogger(__name__)


class LRLSorbOptPostProcessor(OptPostProcessor):
    @property
    def seed_attr(self):
        """
        Returns sequence of seed attrs
        """
        if "_hopt_" in self.path_experiment.stem:
            return ["omniopt_seed"]
        else:
            return [d["name"] for d in self.ec["DIMENSIONS"]]

    @staticmethod
    def supports(path_experiment: Path) -> bool:
        """
        Returns if supports given experiment
        """
        name = path_experiment.stem
        return "_lrl_sorb_" in name and "_opt_" in name

    def read_run_data(self, path_run: Path) -> dict[str, Any]:
        """
        Reads run data from run path
        """
        path_metrics = next(path_run.glob("./all_results.json"))
        path_params = next(path_run.glob("./pipeline_config.json"))

        with open(path_metrics, "r") as fp:
            metrics = json.load(fp)
        with open(path_params) as fp:
            params = json.load(fp)

        return {
            "metrics": metrics,
            "params": params,
        }

    def _get_metric(self, opt_result, metric, stage):
        stage_to_hf = {"val": "eval", "train": "train", "test": "test"}
        return opt_result["metrics"][f"{stage_to_hf[stage]}_{metric}"]

    def _get_stage_metrics(
        self,
        opt_result: dict,
        stage: str,
    ) -> dict[str, float]:
        """
        Get fold averaged metrics of stage
        """

        conf_matrix = string_to_numpy_array(
            self._get_metric(opt_result, "confusion matrix", stage)
        )

        return {
            f"{stage}_macro_F1": macro_f1_score(conf_matrix),
            f"{stage}_macro_recall": macro_recall(conf_matrix),
            f"{stage}_macro_precision": macro_precision(conf_matrix),
            f"{stage}_accuracy": self._get_metric(opt_result, "accuracy", stage),
        }  # type: ignore

    def get_run_metrics(
        self, run_data: dict, conf_matrix: bool = True
    ) -> dict[str, float]:
        """
        Gets fold averaged run metrics for all stages
        """
        metrics = {}
        for stage in ["train", "val", "test"]:
            metrics.update(self._get_stage_metrics(run_data, stage))
        return metrics

    def get_run_seeds(self, run_data: dict) -> dict[str, int]:
        """
        Obtains seeds from run data
        """
        r = {}
        for seed_attr in self.seed_attr:
            if seed_attr == "train_val_split_algorithm_seed":
                r[seed_attr] = run_data["params"]["train_test_split_args"][seed_attr]
            else:
                raise Exception("Seed not supported")
        return r

    def process_fixed(self) -> dict[str, Any]:
        example_run_params_path = next(self.path_runs.glob("*/*.yaml"))
        with example_run_params_path.open("r") as fp:
            params = yaml.safe_load(fp)
        example_run_metrics_path = next(self.path_runs.glob("*/*.json"))
        with example_run_metrics_path.open("r") as fp:
            metrics = json.load(fp)
        return {
            "SLURM_tracking": metrics["SLURM_tracking"],
            "uid": metrics["uid"],
            **params,
        }

    def create_plots(
        self,
        df_metrics: pd.DataFrame,
        metrics: Sequence[str] = ("val_macro_F1", "val_accuracy", "val_loss"),
    ):
        super().create_plots(df_metrics, metrics)

    def plot_metric(self, metric: pd.Series, ax: plt.Axes):
        ax: matplotlib.axes.Axes = sns.histplot(metric, stat="count", ax=ax)
        ax.set(
            title=f"Empirical distribution of {metric.name}.\n"
            f"Case: simpsons; \n"
            f"Varying {self.seed_attr};\n "
            f"Observations count: {len(metric)};\n"
            f"Experiment: {self.path_experiment.stem}"
        )

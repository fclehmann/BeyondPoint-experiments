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

_logger = logging.getLogger(__name__)

class Cifar10OptPostProcessor(OptPostProcessor):
    @property
    def decision_metric(self):
        return "val_MulticlassAccuracy"
    
    @property
    def best_metric_decision_type(self):
        return "max"
    
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
        return "_cifar10_clf_" in name and "_opt_" in name

    def read_run_data(self, path_run: Path) -> dict[str, Any]:
        """
        Reads run data from run path
        """
        path_metrics = next(path_run.glob("./metrics.json"))
        path_params = next(path_run.glob("./params.json"))

        with open(path_metrics, "r") as fp:
            metrics = json.load(fp)
        with open(path_params) as fp:
            params = yaml.safe_load(fp)

        assert len(metrics["train_loss"]) == int(params["num_epochs"])

        return {
            "metrics": metrics,
            "params": params,
        }

    def _get_max_metric_step(self, run_data, metric) -> int:
        max_idx = np.array([
            m_obj["value"] for m_obj in run_data["metrics"][metric]
        ]).argmax()
        max_step = run_data["metrics"][metric][max_idx]["step"]
        return max_step

    def get_run_metrics(self, run_data: dict, conf_matrix: bool = True) -> dict[str, float]:
        """
        Gets fold averaged run metrics for all stages
        """
        metrics = {}
        metric_names = ("MulticlassAccuracy", "MulticlassF1Score")
        max_step = self._get_max_metric_step(run_data, self.decision_metric)
        
        for stage in ["train", "val"]:
            for metric_name in metric_names:
                metric = f"{stage}_{metric_name}"
                value = next(m_obj["value"] for m_obj in run_data["metrics"][metric] if m_obj["step"]==max_step)
                metrics.update({
                    metric: value
                })
        
        for stage in ["test"]:
            for metric_name in metric_names:
                metric = f"{stage}_{metric_name}"
                value = run_data["metrics"][metric][-1]["value"]
                metrics.update({
                    metric: value
                })

        return metrics

    def get_run_seeds(self, run_data: dict) -> dict[str, int]:
        """
        Obtains seeds from run data
        """
        r = {}
        for seed_attr in self.seed_attr:
            seed_name = seed_attr.split(".")[1]
            r[seed_attr] = run_data["params"][seed_name]
        return r

    def process_fixed(self) -> dict[str, Any]:
        example_run_params_path = next(self.path_runs.glob("*/params.json"))
        with example_run_params_path.open("r") as fp:
            params = json.load(fp)
        return dict(params)

    def plot_metric(self, metric: pd.Series, ax: plt.Axes):
        ax: matplotlib.axes.Axes = sns.histplot(metric, stat="count", ax=ax)
        ax.set(
            title=f"Empirical distribution of {metric.name}.\n"
            f"Case: cifar10_clf; \n"
            f"Varying {self.seed_attr};\n "
            f"Observations count: {len(metric)};\n"
            f"Experiment: {self.path_experiment.stem}"
        )
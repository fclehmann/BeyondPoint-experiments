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
)

_logger = logging.getLogger(__name__)


class SimpsonsOptPostProcessor(OptPostProcessor):
    @property
    def decision_metric(self):
        return "val_accuracy"

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
        return "_simpsons_" in name and "_opt_" in name

    def read_run_data(self, path_run: Path) -> dict[str, Any]:
        """
        Reads run data from run path
        """
        path_metrics = next(path_run.glob("./result-*.json"))
        path_params = next(path_run.glob("./*.yaml"))

        with open(path_metrics, "r") as fp:
            metrics = json.load(fp)
        with open(path_params) as fp:
            params = yaml.safe_load(fp)

        # Drop train-val idx as e don't need them here
        for i in range(len(metrics["result_data"]["train_results"])):
            if "train_indices" in metrics["result_data"]["train_results"][i]:
                del metrics["result_data"]["train_results"][i]["train_indices"]
            if "test_indices" in metrics["result_data"]["train_results"][i]:
                del metrics["result_data"]["train_results"][i]["test_indices"]

        # Drop experiment setup as we have params
        del metrics["experiment_setup"]

        return {
            "metrics": metrics,
            "params": params,
        }
    
    def _get_min_metric(
        self,
        opt_result: dict,
        stage: Literal["train", "val"],
        metric: str,
        fold_i: int = 0,
    ) -> Union[float, list]:
        """
        Gets the last epoch metric (val or train)
        """
        decision_metric_name = "acc"
        decision_metric_stage = "val"
        values = []
        for m_obj in opt_result["metrics"]["result_data"]["train_results"][fold_i]["metrics"]:
            values.append(m_obj[decision_metric_stage][decision_metric_name])
        idxmax = np.array(values).argmax()
        return opt_result["metrics"]["result_data"]["train_results"][fold_i]["metrics"][idxmax][stage][metric]

    def _get_test_metric(
        self,
        opt_result: dict,
        metric: str,
    ):
        """
        Gets the test metric
        """
        return opt_result["metrics"]["result_data"]["test_metrics"][metric]

    def _get_num_folds(self, opt_result) -> int:
        return len(opt_result["metrics"]["result_data"]["train_results"])

    def _get_fold_averaged_metric(
        self,
        opt_result: dict,
        stage: str,
        metric: str,
        f_pp: Callable[[Any], float] = lambda x: float(x),
    ) -> float:
        """
        Obtains the metric and averages it over the folds
        """
        return np.mean(
            [
                f_pp(self._get_min_metric(opt_result, stage, metric, fold_i=i))
                for i in range(self._get_num_folds(opt_result))
            ],
            axis=0,
        )  # type: ignore

    def _get_stage_metrics(
        self,
        opt_result: dict,
        stage: str,
    ) -> dict[str, float]:
        """
        Get fold averaged metrics of stage
        """
        return {
            f"{stage}_accuracy": self._get_fold_averaged_metric(
                opt_result, stage, "acc"
            ),
            f"{stage}_loss": self._get_fold_averaged_metric(opt_result, stage, "loss"),
            f"{stage}_macro_F1": self._get_fold_averaged_metric(
                opt_result,
                stage,
                "confusion_matrix",
                f_pp=lambda x: macro_f1_score(np.array(x)),  # type: ignore
            ),
            f"{stage}_macro_precision": self._get_fold_averaged_metric(
                opt_result,
                stage,
                "confusion_matrix",
                f_pp=lambda x: macro_precision(np.array(x)),  # type: ignore
            ),
            f"{stage}_macro_recall": self._get_fold_averaged_metric(
                opt_result,
                stage,
                "confusion_matrix",
                f_pp=lambda x: macro_recall(np.array(x)),  # type: ignore
            ),
        }

    def get_run_metrics(
        self, run_data: dict, conf_matrix: bool = True
    ) -> dict[str, float]:
        """
        Gets fold averaged run metrics for all stages
        """
        metrics = {}
        for stage in ["train", "val"]:
            metrics.update(self._get_stage_metrics(run_data, stage))
        if "test_metrics" in run_data["metrics"]["result_data"]:
            metrics.update(
                {
                    "test_accuracy": self._get_test_metric(run_data, "acc"),
                    "test_loss": self._get_test_metric(run_data, "loss"),
                    "test_macro_F1": macro_f1_score(
                        np.array(self._get_test_metric(run_data, "confusion_matrix"))
                    ),
                    "test_macro_precision": macro_precision(
                        np.array(self._get_test_metric(run_data, "confusion_matrix"))
                    ),
                    "test_macro_recall": macro_recall(
                        np.array(self._get_test_metric(run_data, "confusion_matrix"))
                    ),
                }
            )
        return metrics

    def get_run_seeds(self, run_data: dict) -> dict[str, int]:
        """
        Obtains seeds from run data
        """
        r = {}
        for seed_attr in self.seed_attr:
            if seed_attr == "omniopt_seed":
                r[seed_attr] = int(
                    run_data["result_data"]["environment"]["omniopt_seed"]
                )
            else:
                seed_name = seed_attr.split(".")[1]
                r[seed_attr] = run_data["params"]["ExperimentConfig"][seed_name]
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

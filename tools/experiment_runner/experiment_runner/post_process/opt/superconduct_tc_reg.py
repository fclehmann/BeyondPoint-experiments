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

from experiment_runner.post_process.utils import macro_f1_score

_logger = logging.getLogger(__name__)

class SuperconductOptPostProcessor(OptPostProcessor):
    @property
    def decision_metric(self):
        if "gbdt" in self.path_experiment.stem:
            return "val_RMSE"
        elif "dnn" in self.path_experiment.stem:
            return "val_MSE"
        else:
            raise Exception("Decision metric not defined")
    
    @property
    def best_metric_decision_type(self):
        return "min"
    
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
        return "_superconduct_tc_reg_" in name and "_opt_" in name

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

        if params["pipeline/_target_"] == "superconduct_tc_reg.pipeline.dnn.DNNPipeline":
            if params["split/_target_"] == "superconduct_tc_reg.folds_gen.CVFoldsGeneratorAlgorithm":
                for i in range(int(params["split/n_folds"])):
                    assert f"train_loss-{i}" in metrics
                    assert metrics[f"train_loss-{i}"][-1]["step"] >= 1199
            else:
                assert f"train_loss" in metrics
                # NOTE: Hard coded, have to change this if changing max_epochs
                assert metrics["train_loss"][-1]["step"] >= 1199
        else:
            if params["split/_target_"] == "superconduct_tc_reg.folds_gen.CVFoldsGeneratorAlgorithm":
                for i in range(int(params["split/n_folds"])):
                    assert f"train_RMSE-{i}" in metrics
            else:
                assert f"train_RMSE" in metrics

        return {
            "metrics": metrics,
            "params": params,
        }

    def _metric_name(self, stage, metric, fold):
        k = f"{stage}_{metric}"
        if fold is not None:
            k += f"-{fold}"
        return k
    
    def _get_min_metric_step(self, run_data, metric) -> int:
        min_idx = np.array([
            m_obj["value"] for m_obj in run_data["metrics"][metric]
        ]).argmin()
        return run_data["metrics"][metric][min_idx]["step"]

    def _min_metric(self, run_data, key, fold):
        decision_metric = self.decision_metric
        if fold is not None:
            decision_metric += f"-{fold}"
        step = self._get_min_metric_step(run_data, metric=decision_metric)
        for m_obj in run_data["metrics"][key]:
            if m_obj["step"]==step:
                return m_obj["value"]
        raise Exception(f"Step not found for {key} in fold={fold}")

    def _get_MSE(self, run_data, stage, fold) -> float:
        if run_data["params"]["pipeline/_target_"] == "superconduct_tc_reg.pipeline.dnn.DNNPipeline":
            if stage == "train":
                mse = float(self._min_metric(run_data, self._metric_name("train", "loss", fold), fold))
                return mse
            else:
                return float(self._min_metric(run_data, self._metric_name(stage, "MSE", fold), fold))
        else:
            rmse = float(self._min_metric(run_data, self._metric_name(stage, "RMSE", fold), fold))
            return rmse**2
    
    def _get_RMSE(self, run_data, stage, fold) -> float:
        return np.sqrt(self._get_MSE(run_data, stage, fold))

    def get_run_metrics(self, run_data: dict, conf_matrix: bool = True) -> dict[str, float]:
        """
        Gets fold averaged run metrics for all stages
        """
        n_folds = int(run_data["params"].get("split/n_folds", 1))
        metrics = {}
        for stage in ["train", "val"]:
            values = []
            for i in range(n_folds):
                fold_i = i if n_folds > 1 else None
                values.append(self._get_RMSE(run_data, stage, fold_i))
            metrics[f"{stage}_RMSE"] = np.mean(values)
        
        try:
            metrics[f"test_RMSE"] = self._get_RMSE(run_data, "test", None)
        except Exception:
            pass

        return metrics

    def get_run_seeds(self, run_data: dict) -> dict[str, int]:
        """
        Obtains seeds from run data
        """
        r = {}
        for seed_attr in self.seed_attr:
            seed_name = seed_attr.replace(".", "/")
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
            f"Case: superconduct_tc_reg; \n"
            f"Varying {self.seed_attr};\n "
            f"Observations count: {len(metric)};\n"
            f"Experiment: {self.path_experiment.stem}"
        )
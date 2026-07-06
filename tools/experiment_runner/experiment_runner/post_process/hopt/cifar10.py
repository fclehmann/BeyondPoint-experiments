from functools import partial
from itertools import groupby, islice
import json
import logging
import multiprocessing
from pathlib import Path
import re
from typing import Any, Hashable, Optional, Sequence

import matplotlib.axes
import numpy as np
import pandas as pd

from experiment_runner.post_process.hopt.post_processor import HOptPostProcessor
from experiment_runner.post_process.opt.cifar10_clf import Cifar10OptPostProcessor

import tqdm
import yaml
import joblib

_logger = logging.getLogger(__name__)


class Cifar10HOptPostProcessor(Cifar10OptPostProcessor, HOptPostProcessor):
    @staticmethod
    def supports(path_experiment: Path) -> bool:
        return (
            "_hopt_" in path_experiment.name and "_cifar10_clf_" in path_experiment.name
        )

    def read_run_data(self, path_run: Path) -> dict[str, Any]:
        run_data = super().read_run_data(path_run)

        processed_metrics = self.get_run_metrics(run_data)
        processed_params = {
            d["name"]: run_data["params"][d["name"].split(".")[1]]
            for d in self.ec["DIMENSIONS"]
        }
        processed_params.update(
            {
                "SLURM_JOBID": run_data["params"]["SLURM_tracking/SLURM_JOBID"],
                "omniopt_seed": int(float(run_data["params"]["omniopt_seed"])),
                "run_uid": run_data["params"]["run_uid"],
            }
        )

        return {"metrics": processed_metrics, "params": processed_params}

    def process_fixed(self) -> dict[str, Any]:
        example_run_params_path = next(self.path_runs.glob("*/params.json"))
        with example_run_params_path.open("r") as fp:
            params = json.load(fp)
        return dict(params)

    def _read_hopt_dfs(self):
        dfs = []
        for p_trials in (self.path_experiment / "hopt").glob("*/trials.csv"):
            df = pd.read_csv(p_trials, index_col=0)
            if "system_attrs_failed_trial" in df.columns:
                repeated_trials = df.loc[~df["system_attrs_failed_trial"].isna()]
                df.loc[repeated_trials.index, "number"] = repeated_trials[
                    "system_attrs_failed_trial"
                ]
            if len(df) < 200:
                print(len(df))
                assert False
            df = df.drop(df.loc[df["state"] == "FAIL"].index)
            dfs.append(df)
        return pd.concat(dfs).set_index("user_attrs_run_uid")

    def process_varied(self, iters_thr: float = 0.85) -> pd.DataFrame:
        # Read oo results in single dataframe
        df_hopt = self._read_hopt_dfs()

        _logger.info(df_hopt)

        # Read all runs
        _logger.info("Reading runs ...")
        runs = self.read_runs()

        # Match runs with oo results to obtain hopt_iter
        _logger.info("Matching runs ...")
        df_hopt_matched = pd.DataFrame(
            (
                {
                    "hopt_iter": df_hopt.loc[run_uid]["number"],
                    **run["params"],
                    **run["metrics"],
                }
                for run_uid, run in runs.items()
            )
        )

        _logger.info(df_hopt_matched.columns)
        df_hopt_matched.set_index(["omniopt_seed", "hopt_iter"], inplace=True)

        return df_hopt_matched

    def plot_metric(self, metric: pd.Series, ax: matplotlib.axes.Axes):
        if (
            "val_macro_F1" in metric.name
            or "val_macro_precision" in metric.name
            or "val_macro_recall" in metric.name
            or "val_accuracy" in metric.name
        ):
            cum_func = np.maximum.accumulate
        else:
            cum_func = np.minimum.accumulate
        for _, series_seed in metric.groupby(level="omniopt_seed"):
            series_seed = series_seed.sort_index(level="hopt_iter")
            ax.plot(
                series_seed.index.get_level_values("hopt_iter"),
                cum_func(series_seed),
                color="blue",
            )

        n_obs = len(metric.index.get_level_values("omniopt_seed").unique())
        ax.set(
            title=f"HOpt paths.\n"
            f"Case: superconduct_tc_reg; \n"
            f"Varying {self.seed_attr};\n "
            f"Observations count: {n_obs};\n"
            f"Experiment: {self.path_experiment.stem}",
            yscale="log",
            xlabel="hopt iter",
            ylabel=metric.name,
        )

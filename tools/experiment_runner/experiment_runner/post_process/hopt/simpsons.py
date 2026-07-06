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
from experiment_runner.post_process.opt.simpsons import SimpsonsOptPostProcessor

import tqdm
import yaml
import joblib

_logger = logging.getLogger(__name__)


class SimpsonsHOptPostProcessor(SimpsonsOptPostProcessor, HOptPostProcessor):
    @staticmethod
    def supports(path_experiment: Path) -> bool:
        return (
            "_hopt_" in path_experiment.name
            and "_simpsons_" in path_experiment.name
        )

    def _simplify_oo_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename({"RESULT": "val_loss"}, axis=1)
        df = df[
            [d["name"] for d in self.ec["DIMENSIONS"]] + ["val_loss"] + ["omniopt_seed"]
        ]
        df.index.set_names(["hopt_iter"], inplace=True)
        df.reset_index(inplace=True)
        return df

    def _get_oo_results(self) -> pd.DataFrame:
        _logger.info(f"Exporting results from {self.path_experiment}")

        if not (self.path_experiment / "omniopt").exists():
            df = pd.read_csv(
                self.path_experiment
                / "data"
                / "processed"
                / "hopt"
                / "omniopt_results.csv",
                index_col=0,
            )
            return self._simplify_oo_df(df)

        omniopt_project_regex = rf"{self.path_experiment.name}_SEED_(\d*\.\d+|\d+)"
        path_to_omni_projects = self.path_experiment / "omniopt" / "projects"

        _logger.debug(f"Path to omniopt projects: {path_to_omni_projects}")

        omni_results: list[pd.DataFrame] = []

        not_exist_count: int = 0

        progress_bar = tqdm.tqdm(
            list(path_to_omni_projects.glob(f"{self.path_experiment.name}_SEED_*"))
        )

        for project in islice(progress_bar, None):

            _logger.debug(
                f"--------------------------------- {project.name} ---------------------------------"
            )

            _logger.debug(f"Processing {project} ...")

            match = re.search(omniopt_project_regex, project.name)
            seed_str: str = match.group(1)

            if seed_str.isdigit():
                seed = int(seed_str)
            else:
                seed = float(seed_str)

            progress_bar.set_description(f"s={seed}, nec={not_exist_count}")

            _logger.debug(f"Extracted seed: {seed}")

            csv_path = project / "csv"

            if not csv_path.exists():
                not_exist_count += 1
                _logger.warning(f"Csv files not found for project: {project}")
                continue

            csv_files = list(csv_path.iterdir())

            if len(csv_files) > 1:
                _logger.warning("Multiple csv files => sorting in descending order ...")
                csv_files = sorted(csv_files, reverse=True)
                _logger.warning(f"Will export: {csv_files[0]}")

            try:
                omni_result_df: pd.DataFrame = pd.read_csv(csv_files[0], sep=";")
                omni_result_df["omniopt_seed"] = seed
                omni_results.append(omni_result_df)
            except Exception as e:
                _logger.exception(e, exc_info=True)

        _logger.debug(f"Seeds count successfully extracted: {len(omni_results)}")
        _logger.debug(f"Seeds count export failed: {not_exist_count}")

        df = pd.concat(omni_results)
        df = self._simplify_oo_df(df)

        return df

    def read_run_data(self, path_run: Path) -> dict[str, Any]:
        run_data = super().read_run_data(path_run)

        processed_metrics = self.get_run_metrics(run_data)
        processed_params = {
            d["name"]: run_data["params"]["ExperimentConfig"][d["name"].split(".")[1]]
            for d in self.ec["DIMENSIONS"]
        }
        processed_params.update(
            {
                "SLURM_JOBID": run_data["metrics"]["SLURM_tracking"]["SLURM_JOBID"],
                "omniopt_seed": int(run_data["metrics"]["environment"]["omniopt_seed"]),
                "uid": run_data["metrics"]["uid"],
            }
        )

        return {"metrics": processed_metrics, "params": processed_params}

    def _match_run(
        self,
        run_row: tuple[Hashable, pd.Series],
        runs_grouped: dict[int, list[dict]],
        match_params: Sequence[str],
        match_metric: str = "val_loss",
        decimals: int = 4,
    ) -> tuple[Hashable, Optional[str]]:
        """
        Returns a run uid of the run matching run_row
        """
        results = []

        for run_data in runs_grouped[int(run_row[1]["omniopt_seed"])]:

            match_condition = all(
                np.round(run_row[1][p], decimals)
                == np.round(run_data["params"][p], decimals)
                for p in match_params
            )

            if match_condition:
                results.append((run_row[0], run_data["params"]["uid"]))
                # NOTE: We don't break here, because we want to be sure of 1-to-1 correspondence

        if len(results) == 0:
            _logger.warning(f"No run data found for run_row={run_row[0]}")
            return (run_row[0], None)
        if len(results) > 1:
            _logger.warning(f"Multiple run data found for run_row={run_row[0]}")
            return (run_row[0], None)

        return results[0]

    def _remove_seeds_with_unsufficient_runs(self, df: pd.DataFrame, iters_thr: float):
        max_iters = int(self.ec["DATA"]["max_evals"])  # type: ignore
        n_iters_thr = int(iters_thr * max_iters)
        list_of_valid_hopt_dfs = []
        for seed, df_g in df.groupby("omniopt_seed"):
            n_iters = df_g["hopt_iter"].max()
            if n_iters >= n_iters_thr:
                list_of_valid_hopt_dfs.append(df_g)
            else:
                _logger.info(
                    f"Seed={seed} removed because hopt iterations count = {n_iters} < threshold = {n_iters_thr}"
                )

        return pd.concat(list_of_valid_hopt_dfs)

    def _group_runs_by_seed(self, runs: dict[str, dict]) -> dict[int, list[dict]]:
        runs_sorted = sorted(runs.values(), key=lambda r: r["params"]["omniopt_seed"])
        runs_grouped: dict[int, list[dict]] = {}
        for seed, runs_of_seed in groupby(
            runs_sorted, key=lambda r: r["params"]["omniopt_seed"]
        ):
            runs_grouped[seed] = list(runs_of_seed)
        return runs_grouped

    def process_fixed(self) -> dict[str, Any]:
        example_run_params_path = next(self.path_runs.glob("*/*.yaml"))
        with example_run_params_path.open("r") as fp:
            params = yaml.safe_load(fp)
        return dict(params)

    def process_varied(self, iters_thr: float = 0.85) -> pd.DataFrame:
        # Read oo results in single dataframe
        df_hopt = self._get_oo_results()

        _logger.info(df_hopt)

        # Read all runs
        _logger.info("Reading runs ...")
        runs = self.read_runs()

        # Group runs by oo seed to do less iterations when looking for runs
        _logger.info("Grouping runs ...")
        runs_grouped = self._group_runs_by_seed(runs)

        # Match runs with oo results to obtain hopt_iter
        _logger.info("Matching runs ...")
        match_run = partial(
            self._match_run,
            runs_grouped=runs_grouped,
            match_params=[d["name"] for d in self.ec["DIMENSIONS"]],
            decimals=6,
        )

        runs_uid_matched = (match_run(t) for t in df_hopt.iterrows())
        runs_uid_matched = (
            (idx, uid) for idx, uid in runs_uid_matched if uid is not None
        )

        # - Add hopt iter to the params of each run and form a dataframe
        df_hopt_matched = pd.DataFrame(
            (
                {
                    "hopt_iter": df_hopt.loc[row_id]["hopt_iter"],
                    **runs[run_uid]["params"],
                    **runs[run_uid]["metrics"],
                }
                for row_id, run_uid in tqdm.tqdm(runs_uid_matched, total=len(df_hopt))
            ),
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
            f"Case: simpsons; \n"
            f"Varying {self.seed_attr};\n "
            f"Observations count: {n_obs};\n"
            f"Experiment: {self.path_experiment.stem}",
            yscale="log",
            xlabel="hopt iter",
            ylabel=metric.name,
        )
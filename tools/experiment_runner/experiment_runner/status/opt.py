from abc import ABC, abstractmethod
from collections import defaultdict
from email.policy import default
from itertools import islice
import json
from pathlib import Path
from typing import Type

import pandas as pd
import numpy as np
import rich.console 
import rich.style
import rich.table
import rich.progress

import experiment_runner.run.experiment_config as experiment_config

def progress_with_percentage(n_1: int, n_2: int) -> str:
    return f"{n_1} / {n_2} ({(n_1 / n_2) * 100:.2f}%)"

class OptExperimentStatus(ABC):
    def __init__(self, path: Path, show_n_eid: int = 3) -> None:
        self.show_n_eid = show_n_eid
        self.path = path

    @abstractmethod
    def run_result_exists(self, run_id: str) -> bool:
        pass
    
    @abstractmethod
    def run_config_exists(self, run_id: str) -> bool:
        pass

    def show_status(self):
        path = self.path
        show_n_eid = self.show_n_eid

        path_experiment_config = next(path.glob("*.toml"))
        config = experiment_config.read_experiment_config(path_experiment_config)
        
        assert isinstance(config.DIMENSIONS[0], experiment_config.ChoiceDimension)

        n_seeds_configured = len(config.DIMENSIONS[0].options)
        n_iterations_configured = config.DATA.max_evals

        if n_seeds_configured > n_iterations_configured:
            print("Warning: N seeds > Max iterations!")

        path_results = path / "results"
        
        n_seeds_failed_or_running = 0
        n_seeds_failed_without_cfg = 0
        n_seeds_success = 0

        eid_failed_or_running = []
        eid_failed_without_cfg = []

        for path_result in rich.progress.track(list(path_results.iterdir())):
            eid = path_result.stem

            if not self.run_config_exists(eid):
                n_seeds_failed_without_cfg += 1
                n_seeds_failed_or_running += 1
                eid_failed_without_cfg.append(eid)
                continue

            if not self.run_result_exists(eid):
                n_seeds_failed_or_running += 1
                eid_failed_or_running.append(path_result.stem)
                continue
            
            n_seeds_success += 1
            
        table = rich.table.Table(
            title=f"State of {config.ENV_PARAMS.base_params.experiment_name}",
            show_lines=True,
            expand=True,
            title_style=rich.style.Style(color="red", bold=True),
            border_style=rich.style.Style(color="bright_black")
        )

        table.add_column("Statistic")
        table.add_column("Value")
        table.add_column("IDs")

        table.add_row(
            "N seeds configured", 
            str(n_iterations_configured), 
            ""
        )
        table.add_row(
            "N successfull runs", 
            progress_with_percentage(n_seeds_success, n_iterations_configured), 
            ""
        )
        table.add_row(
            "N runs failed or active", 
            progress_with_percentage(n_seeds_failed_or_running, n_iterations_configured), 
            f"{list(islice(eid_failed_or_running, show_n_eid))} ..."
        )
        table.add_row(
            "N runs without input cfg", 
            progress_with_percentage(n_seeds_failed_without_cfg, n_iterations_configured), 
            f"{list(islice(eid_failed_without_cfg, show_n_eid))} ..."
        )

        console = rich.console.Console()
        console.print(table)


class SimpsonsOptExperimentStatus(OptExperimentStatus):
    def run_result_exists(self, run_id: str) -> bool:
        path_run = self.path / "results" / run_id
        path_result = next(path_run.glob("*.json"), None)
        return path_result is not None

    def run_config_exists(self, run_id: str) -> bool:
        path_run = self.path / "results" / run_id
        path_conf = next(path_run.glob("*.yaml"), None)
        return path_conf is not None
    

class LrlSorbOptExperimentStatus(OptExperimentStatus):
    def run_result_exists(self, run_id: str) -> bool:
        path_run = self.path / "results" / run_id
        path_result = next(path_run.glob("eval_results.json"), None)
        return path_result is not None

    def run_config_exists(self, run_id: str) -> bool:
        path_run = self.path / "results" / run_id
        path_conf = next(path_run.glob("pipeline_config.json"), None)
        return path_conf is not None

class SuperConductRegExperimentStatus(OptExperimentStatus):
    def run_result_exists(self, run_id: str) -> bool:
        path_run = self.path / "results" / run_id
        
        with open(path_run / "params.json") as fp:
            params = json.load(fp)

        with open(path_run / "metrics.json") as fp:
            metrics = json.load(fp)
        
        if "pipeline/_target_" not in params:
            return False
        
        # dnn last epoch == max_epochs
        if params["pipeline/_target_"] == "superconduct_tc_reg.pipeline.dnn.DNNPipeline":
            if "epoch" not in metrics:
                return False
            return metrics["epoch"][-1]["value"] == int(params["pipeline/trainer/max_epochs"]) - 1
        
        # xgb gbdt last metric step == n_estimators
        elif params["pipeline/_target_"] == "superconduct_tc_reg.pipeline.gbdt.XGBPipeline":
            # Early stopping case
            if "pipeline/early_stopping/_target_" in params:
                return next(path_run.glob("metrics.json"), None) is not None
            return int(metrics["val_RMSE"][-1]["step"]) == int(params["pipeline/model/n_estimators"]) -1
        
        # metrics.json exist?
        else:
            return next(path_run.glob("metrics.json"), None) is not None

    def run_config_exists(self, run_id: str) -> bool:
        path_run = self.path / "results" / run_id
        path_conf = next(path_run.glob("params.json"), None)
        return path_conf is not None


def get_experiment_status_handler(experiment_name: str) -> Type[OptExperimentStatus]:
    if "lrl_sorb" in experiment_name:
        return LrlSorbOptExperimentStatus
    elif "simpsons" in experiment_name:
        return SimpsonsOptExperimentStatus
    elif "superconduct_tc_reg" in experiment_name:
        return SuperConductRegExperimentStatus
    else:
        raise Exception(f"Unsupported experiment {experiment_name}")
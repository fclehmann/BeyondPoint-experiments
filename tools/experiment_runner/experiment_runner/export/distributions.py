from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any
import pandas as pd
import tomlkit
import yaml
import traceback


def read_ec(exp_path: Path) -> dict[str, Any]:
    return tomlkit.parse(next(exp_path.glob("*.toml")).read_text())


class ExportHandler(ABC):

    @abstractmethod
    def __call__(self, exp_path: Path) -> dict[str, Any]: ...


class SimpsonsHandler(ExportHandler):
    _seed_param_to_uncertainty_source = {
        "ExperimentConfig.train_test_split_algorithm_seed": "train-test split",
        "ExperimentConfig.augmentation_seed": "data augmentation",
        "ExperimentConfig.dropout_seed": "dropout",
        "ExperimentConfig.weights_init_seed": "weights init",
        "ExperimentConfig.data_shuffle_seed": "train set order",
        "omniopt_seed": "HOpt algorithm",
    }

    def _legend(self, name, seed_param, model, transfer_learning, optimizer):
        number = name[:2]
        if isinstance(seed_param, list):
            uncertainty_source = " and ".join(
                [self._seed_param_to_uncertainty_source[p] for p in seed_param]
            )
        else:
            uncertainty_source = self._seed_param_to_uncertainty_source[seed_param]
        if transfer_learning:
            model = f"{model} - tl"
        return f"Simpsons({number}), {model}, {optimizer}, {uncertainty_source}"

    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = {}
        ec = read_ec(exp_path)
        self._ec = ec

        with open(exp_path / "data" / "processed" / "data.json", "r") as fp:
            data_processed = json.load(fp)
            self._data_processed = data_processed

        ed["experiment"] = exp_path.stem
        ed["experiments_path"] = str(exp_path)
        ed["model"] = ec["FIXED_EXPERIMENT_PARAMS"].get(
            "ExperimentConfig.model", "SimpsonsNetConv"
        )
        ed["dataset"] = "simpsons"
        ed["metric_name"] = ec["FIXED_EXPERIMENT_PARAMS"][
            "ExperimentConfig.target_metric"
        ]
        ed["train_test_split_algorithm"] = ec["FIXED_EXPERIMENT_PARAMS"][
            "ExperimentConfig.train_test_split_algorithm"
        ]
        ed["optimizer"] = ec["FIXED_EXPERIMENT_PARAMS"]["ExperimentConfig.optimizer"]
        ed["transfer_learning"] = "_tl_" in exp_path.stem

        ed["decision_metric"] = data_processed["decision_metric"]
        ed["best_metric_decision_type"] = data_processed["best_metric_decision_type"]

        return ed


class SimpsonsOptSingleSourceHandler(SimpsonsHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = self._ec

        ed["seed_param"] = ec["DIMENSIONS"][0]["name"]
        assert ed["seed_param"] != ""

        ed["seeds"] = self._data_processed["varied"][ed["seed_param"]]
        ed["metric_values"] = {
            col: self._data_processed["varied"][col]
            for col in self._data_processed["varied"].keys()
            if col != ed["seed_param"]
        }

        ed["process"] = "opt"
        ed["n_sources"] = 1

        ed["legend"] = self._legend(
            name=ed["experiment"],
            seed_param=ed["seed_param"],
            model=ed["model"],
            transfer_learning=ed["transfer_learning"],
            optimizer=ed["optimizer"],
        )

        return ed


class SimpsonsOptMultiSourceHangler(SimpsonsHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = self._ec

        ed["seed_params"] = [
            ec["DIMENSIONS"][0]["name"],
            ec["DIMENSIONS"][1]["name"],
        ]

        ed["seeds"] = {
            name: self._data_processed["varied"][name] for name in ed["seed_params"]
        }
        ed["metric_values"] = {
            col: self._data_processed["varied"][col]
            for col in self._data_processed["varied"].keys()
            if col not in ed["seed_params"]
        }

        ed["process"] = "opt"
        ed["n_sources"] = 2

        ed["legend"] = self._legend(
            name=ed["experiment"],
            seed_param=ed["seed_params"],
            model=ed["model"],
            transfer_learning=ed["transfer_learning"],
            optimizer=ed["optimizer"],
        )

        return ed


class SimpsonsHOptHangler(SimpsonsHandler):

    def _get_optima_if_sufficient_runs(self, df_varied: pd.DataFrame):
        dfs = []
        thr_iters = 200
        for key, df_g in df_varied.groupby("omniopt_seed"):
            if len(df_g) > thr_iters:
                dfs.append(df_g)
            else:
                print(f"Seed {key} removed because of unsuficcient runs: {len(df_g)}")
        df = pd.concat(dfs)
        return df.loc[df.groupby("omniopt_seed")["val_accuracy"].idxmax()]

    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)
        varied_params = [d["name"] for d in ec["DIMENSIONS"]]

        ed["seed_param"] = "omniopt_seed"

        df_varied = pd.DataFrame(self._data_processed["varied"])
        df_hopt_min = self._get_optima_if_sufficient_runs(df_varied)

        ed["metric_values"] = {
            col: df_hopt_min[col].to_list()
            for col in df_hopt_min.columns
            if col != ed["seed_param"] and col not in varied_params
        }
        ed["parameter_values"] = {
            col: df_hopt_min[col].to_list() for col in varied_params
        }
        ed["seeds"] = df_hopt_min[ed["seed_param"]].to_list()

        ed["seed_param"] = "omniopt_seed"
        ed["hopt_algorithm"] = ec["DATA"]["algo_name"]
        ed["process"] = "hopt"
        ed["n_sources"] = 1

        ed["hopt"] = {
            "algorithm": ec["DATA"]["algo_name"],
            "max_iter": ec["DATA"]["max_evals"],
        }

        ed["legend"] = self._legend(
            name=ed["experiment"],
            seed_param=ed["seed_param"],
            model=ed["model"],
            transfer_learning=ed["transfer_learning"],
            optimizer=ed["optimizer"],
        )

        return ed


class LrlSorbHandler(ExportHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = {}
        ec = read_ec(exp_path)
        name = exp_path.stem
        ed["metric_name"] = "macro_f1"

        model_path = Path(ec["FIXED_EXPERIMENT_PARAMS"]["model_name_or_path"])
        ed["model"] = model_path.stem
        if "train_test_split_algorithm" in ec["FIXED_EXPERIMENT_PARAMS"]:
            ed["train_test_split_algorithm"] = ec["FIXED_EXPERIMENT_PARAMS"][
                "train_test_split_algorithm"
            ]
        else:
            ed["train_test_split_algorithm"] = "split"
        ed["optimizer"] = ec["FIXED_EXPERIMENT_PARAMS"]["optim"]
        ed["dataset"] = (
            f'{ec["FIXED_EXPERIMENT_PARAMS"]["dataset_name"]}:{ec["FIXED_EXPERIMENT_PARAMS"]["wikiann_language_tag"]}'
        )
        ed["experiment"] = name
        ed["experiments_path"] = str(exp_path)
        ed["transfer_learning"] = "_tl_" in name
        ed["n_sources"] = 1

        return ed


class LrlSorbHOptHandler(LrlSorbHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)

        ed["seed_param"] = "omniopt_seed"
        ed["hopt_algorithm"] = ec["DATA"]["algo_name"]
        ed["process"] = "hopt"
        df_distribution = pd.read_csv(
            exp_path / "data" / "processed" / "hopt" / "metric_values.csv"
        )
        ed["metric_name"] = "macro_f1"
        ed["seeds"] = df_distribution[ed["seed_param"]].to_list()
        ed["metric_values"] = df_distribution[ed["metric_name"]].to_list()

        return ed


class LrlSorbOptHandler(LrlSorbHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)

        ed["seed_param"] = ec["DIMENSIONS"][0]["name"]
        ed["process"] = "opt"
        df_distribution = pd.read_csv(
            exp_path / "data" / "processed" / "opt" / "metric_values.csv"
        )
        ed["metric_name"] = "macro_f1"
        ed["seeds"] = df_distribution[ed["seed_param"]].to_list()
        ed["metric_values"] = df_distribution[ed["metric_name"]].to_list()

        return ed


class Cifar10Handler(ExportHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = {}
        ec = read_ec(exp_path)
        path_params_example = next((exp_path / "results").glob("*/params.json"))
        with open(path_params_example, "r") as fp:
            params_example = json.load(fp)
        name = exp_path.stem

        ed["experiment"] = name
        ed["image"] = ec["ENV_PARAMS"]["singularity_image_file"]
        ed["model"] = "VGG11BN"
        ed["dataset"] = "cifar10"
        ed["experiments_path"] = str(exp_path)
        ed["transfer_learning"] = "_tl_" in name
        ed["params_example"] = params_example

        return ed


class Cifar10OptSingleSourceHandler(Cifar10Handler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)
        df_distribution = pd.read_csv(
            exp_path / "data" / "processed" / "metric_values.csv"
        )

        ed["seed_param"] = ec["DIMENSIONS"][0]["name"][len("ExperimentConfig.") :]
        assert ed["seed_param"] != ""

        ed["seeds"] = df_distribution[ed["seed_param"]].to_list()
        ed["metric_values"] = {
            col: df_distribution[col].to_list()
            for col in df_distribution.columns
            if col != ed["seed_param"]
        }

        ed["process"] = "opt"
        ed["n_sources"] = 1

        return ed


class SoilAnalysisHandler(ExportHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = {}
        ec = read_ec(exp_path)
        path_params_example = next((exp_path / "results").glob("*/params.json"))
        with open(path_params_example, "r") as fp:
            params_example = json.load(fp)
        name = exp_path.stem
        # ed["metric_name"] = "MSE"
        ed["experiment"] = name
        ed["image"] = ec["ENV_PARAMS"]["singularity_image_file"]
        ed["model"] = "DNN"
        ed["train_val_split_algorithm"] = params_example["split/_target_"]
        ed["optimizer"] = params_example["optimizer/_target_"]
        ed["dataset"] = params_example["dataset/name"]
        ed["experiments_path"] = str(exp_path)
        ed["transfer_learning"] = "_tl_" in name

        return ed


class SoilAnalysisOptSingleSourceHandler(SoilAnalysisHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)
        df_distribution = pd.read_csv(
            exp_path / "data" / "processed" / "metric_values.csv"
        )

        ed["seed_param"] = ec["DIMENSIONS"][0]["name"]
        assert ed["seed_param"] != ""

        ed["seeds"] = df_distribution[ed["seed_param"]].to_list()
        ed["metric_values"] = {
            col: df_distribution[col].to_list()
            for col in df_distribution.columns
            if col != ed["seed_param"]
        }

        ed["process"] = "opt"
        ed["n_sources"] = 1

        return ed


class SoilAnalysisHOptHandler(SoilAnalysisHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)

        df_distribution = pd.read_csv(
            exp_path / "data" / "processed" / "hopt" / "metric_values.csv"
        )

        ed["seed_param"] = "omniopt_seed"
        ed["seeds"] = df_distribution[ed["seed_param"]].to_list()
        ed["metric_values"] = {
            col: df_distribution[col].to_list()
            for col in df_distribution.columns
            if col != ed["seed_param"]
        }
        ed["hopt_algorithm"] = ec["DATA"]["algo_name"]
        ed["process"] = "hopt"
        ed["n_sources"] = 1

        return ed


class SuperconductHandler(ExportHandler):

    _seed_param_to_uncertainty_source = {
        "split.split_algorithm_seed": "train-test split",
        "pipeline.model.random_state": "GBDT model",
        "pipeline.weights_init.seed": "weights init",
        "pipeline.model.dropout_seed": "dropout",
        "pipeline.data_loader.seed": "train set order",
        "omniopt_seed": "HOpt algorithm",
        "pipeline.train_shuffle_seed": "train set order",
    }

    def _legend(self, name, seed_param):
        number = name[:3]
        model = "GBDT" if "gbdt" in name else "DNN"
        uncertainty_source = self._seed_param_to_uncertainty_source[seed_param]

        if model == "GBDT":
            if "pipeline/model/subsample" in self._data_processed["params"]:
                model = "GBDT(row subsample, col subsample)"
            else:
                model = "GBDT(without subsample)"

        return f"SCTCR({number}), {model}, {uncertainty_source}"

    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = {}
        ec = read_ec(exp_path)
        with open(exp_path / "data" / "processed" / "data.json", "r") as fp:
            data_processed = json.load(fp)
            self._data_processed = data_processed
        name = exp_path.stem
        ed["experiment"] = name
        ed["image"] = ec["ENV_PARAMS"]["singularity_image_file"]
        ed["model"] = data_processed["params"]["pipeline/model/_target_"]
        ed["dataset"] = "UCIML_superconductivity_data"
        ed["experiments_path"] = str(exp_path)
        ed["transfer_learning"] = "_tl_" in name
        ed["params_example"] = data_processed["params"]
        ed["decision_metric"] = data_processed["decision_metric"]
        ed["best_metric_decision_type"] = data_processed["best_metric_decision_type"]

        return ed


class SuperconductOptSingleSourceHandler(SuperconductHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)
        data_processed = self._data_processed

        ed["seed_param"] = ec["DIMENSIONS"][0]["name"]
        assert ed["seed_param"] != ""

        ed["seeds"] = data_processed["varied"][ed["seed_param"]]
        ed["metric_values"] = {
            col: data_processed["varied"][col]
            for col in data_processed["varied"].keys()
            if col != ed["seed_param"]
        }

        ed["process"] = "opt"
        ed["n_sources"] = 1
        ed["legend"] = self._legend(ed["experiment"], ed["seed_param"])

        return ed


class SuperconductHOptHandler(SuperconductHandler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)

        data_processed = self._data_processed

        varied_params = [d["name"] for d in ec["DIMENSIONS"]]

        ed["seed_param"] = "omniopt_seed"

        df_varied = pd.DataFrame(data_processed["varied"])
        df_hopt_min = df_varied.loc[
            df_varied.groupby("omniopt_seed")["val_RMSE"].idxmin()
        ]

        ed["metric_values"] = {
            col: df_hopt_min[col].to_list()
            for col in df_hopt_min.columns
            if col != ed["seed_param"] and col not in varied_params
        }
        ed["parameter_values"] = {
            col: df_hopt_min[col].to_list() for col in varied_params
        }
        ed["seeds"] = df_hopt_min[ed["seed_param"]].to_list()

        ed["hopt_algorithm"] = ec["DATA"]["algo_name"]
        ed["process"] = "hopt"
        ed["n_sources"] = 1

        ed["legend"] = self._legend(ed["experiment"], ed["seed_param"])

        ed["hopt"] = {
            "algorithm": ec["DATA"]["algo_name"],
            "max_iter": ec["DATA"]["max_evals"],
        }

        return ed


class CifarV2Handler(ExportHandler):

    _seed_param_to_uncertainty_source = {
        "ExperimentConfig.train_val_split_algorithm_seed": "train-test split",
        "ExperimentConfig.augmentation_seed": "data augmentation",
        "ExperimentConfig.dropout_seed": "dropout",
        "ExperimentConfig.weights_init_seed": "weights init",
        "ExperimentConfig.data_shuffle_seed": "train set order",
        "omniopt_seed": "HOpt algorithm",
    }

    def _legend(self, name, seed_param):
        number = name[:3]
        uncertainty_source = self._seed_param_to_uncertainty_source[seed_param]
        return f"Cifar10({number}), {uncertainty_source}"

    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = {}
        ec = read_ec(exp_path)
        with open(exp_path / "data" / "processed" / "data.json", "r") as fp:
            data_processed = json.load(fp)
            self._data_processed = data_processed
        name = exp_path.stem
        ed["experiment"] = name
        ed["image"] = ec["ENV_PARAMS"]["singularity_image_file"]
        ed["model"] = "VGG11BN"
        ed["dataset"] = "cifar10"
        ed["experiments_path"] = str(exp_path)
        ed["transfer_learning"] = "_tl_" in name
        ed["params_example"] = data_processed["params"]

        ed["decision_metric"] = data_processed["decision_metric"]
        ed["best_metric_decision_type"] = data_processed["best_metric_decision_type"]

        return ed


class CifarV2OptSingleSourceHandler(CifarV2Handler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)
        data_processed = self._data_processed

        ed["seed_param"] = ec["DIMENSIONS"][0]["name"]
        assert ed["seed_param"] != ""

        ed["seeds"] = data_processed["varied"][ed["seed_param"]]
        ed["metric_values"] = {
            col: data_processed["varied"][col]
            for col in data_processed["varied"].keys()
            if col != ed["seed_param"]
        }

        ed["process"] = "opt"
        ed["n_sources"] = 1

        ed["legend"] = self._legend(ed["experiment"], ed["seed_param"])

        return ed


class CifarV2HOptHandler(CifarV2Handler):
    def __call__(self, exp_path: Path) -> dict[str, Any]:
        ed = super().__call__(exp_path)
        ec = read_ec(exp_path)

        data_processed = self._data_processed

        varied_params = [d["name"] for d in ec["DIMENSIONS"]]

        ed["seed_param"] = "omniopt_seed"

        df_varied = pd.DataFrame(data_processed["varied"])
        df_hopt_min = df_varied.loc[
            df_varied.groupby("omniopt_seed")["val_MulticlassAccuracy"].idxmax()
        ]

        ed["metric_values"] = {
            col: df_hopt_min[col].to_list()
            for col in df_hopt_min.columns
            if col != ed["seed_param"] and col not in varied_params
        }
        ed["parameter_values"] = {
            col: df_hopt_min[col].to_list() for col in varied_params
        }
        ed["seeds"] = df_hopt_min[ed["seed_param"]].to_list()

        ed["hopt_algorithm"] = ec["DATA"]["algo_name"]
        ed["process"] = "hopt"
        ed["n_sources"] = 1

        ed["legend"] = self._legend(ed["experiment"], ed["seed_param"])

        ed["hopt"] = {
            "algorithm": ec["DATA"]["algo_name"],
            "max_iter": ec["DATA"]["max_evals"],
        }

        return ed


def pick_handler(exp_path: Path):
    name = exp_path.stem
    ec = read_ec(exp_path)
    n_dims = len(ec["DIMENSIONS"])
    if "simpsons" in name and "_opt_" in name and n_dims == 1:
        return SimpsonsOptSingleSourceHandler()
    elif "simpsons" in name and "_opt_" in name and n_dims > 1:
        return SimpsonsOptMultiSourceHangler()
    elif "simpsons" in name and "_hopt_" in name:
        return SimpsonsHOptHangler()
    elif "lrl_sorb" in name and "_opt_" in name and n_dims == 1:
        return LrlSorbOptHandler()
    elif "lrl_sorb" in name and "_hopt_" in name:
        return LrlSorbHOptHandler()
    elif "soil_analysis" in name and "_opt_" in name and n_dims == 1:
        return SoilAnalysisOptSingleSourceHandler()
    elif "soil_analysis" in name and "_hopt_" in name and n_dims > 1:
        return SoilAnalysisHOptHandler()
    elif (
        "cifar10" in name
        and "_opt_" in name
        and n_dims == 1
        and "experiments_v2" not in str(exp_path)
    ):
        return Cifar10OptSingleSourceHandler()
    elif (
        "cifar10" in name
        and "_opt_" in name
        and n_dims == 1
        and "experiments_v2" in str(exp_path)
    ):
        return CifarV2OptSingleSourceHandler()
    elif "cifar10" in name and "_hopt_" in name and "experiments_v2" in str(exp_path):
        return CifarV2HOptHandler()
    elif "superconduct_tc_reg" in name and "_opt_" in name and n_dims == 1:
        return SuperconductOptSingleSourceHandler()
    elif "superconduct_tc_reg" in name and "_hopt_" in name and "_test" not in name:
        return SuperconductHOptHandler()
    else:
        raise ValueError(f"Experiment not supported: {name}")


def export_data(exp_paths: list[Path]) -> list[dict[str, Any]]:
    data = []
    for p in exp_paths:
        try:
            exp_data = pick_handler(p)(p)
            data.append(exp_data)
        except Exception as e:
            print(f"Skiping {p.stem}")
            traceback.print_exc()
    return data

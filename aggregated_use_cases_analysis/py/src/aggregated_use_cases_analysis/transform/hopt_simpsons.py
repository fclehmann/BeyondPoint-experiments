from functools import partial
import logging
from typing import Optional
import pandas as pd
import pathlib
import json
from collections import defaultdict
import multiprocessing
import tqdm
import yaml
import aggregated_use_cases_analysis.data as data
from tomlkit import parse

_logger = logging.getLogger(__name__)

def read_result_from_run(
        path: pathlib.Path
) -> tuple[dict, dict]:
    """Reads run config and run results file 

    :param path: _description_
    :return: config, results tuple
    """
    path_input = next(path.glob("*.yaml"), None)
    if path_input is None:
        return None, None
    input_config = yaml.load(path_input.read_text(), Loader=yaml.Loader)

    path_result = next(path.glob("*.json"), None)

    if path_result is None:
        return input_config, None

    with (path_result).open("r") as file:
        result = json.load(file)

    result = {
        "environment": result["environment"],
        "result_data": {
            "test_metrics": {
                "acc": result["result_data"]["test_metrics"]["acc"]
            }
        }
    }

    return input_config, result


def read_results_from_experiment(path_experiment: pathlib.Path, n_processes: int = 24, chunksize: int = 10):
    paths = list((path_experiment / "results").iterdir())

    with multiprocessing.Pool(processes=n_processes) as p:

        results = [
            (c, r) for c, r in
            tqdm.tqdm(
                p.imap(
                    read_result_from_run,
                    paths,
                    chunksize=chunksize
                ),
                total=len(paths)
            ) if c is not None and r is not None
        ]

    return results

def convert_idx_val(raw_idx_val):
    if type(raw_idx_val) is int:
        idx_val = f"{raw_idx_val}"
    elif type(raw_idx_val) is float:
        idx_val = f"{raw_idx_val:.10f}"
    else:
        idx_val = str(raw_idx_val)
    return idx_val


def get_index_from_run_config(config, index: list[str]):
    index_values = []

    for idx_key in index:
        idx_key_1, idx_key_2 = idx_key.split(".")
        raw_idx_val = config[idx_key_1][idx_key_2]
        index_values.append(convert_idx_val(raw_idx_val))

    return tuple(index_values)


def get_oo_seed_from_result(result) -> int:
    return int(result["environment"]["omniopt_seed"])


def index_results(results, index: list[str]):
    indexed_results = defaultdict(dict)
    for config, result in results:
        indexed_results[get_oo_seed_from_result(
            result)][get_index_from_run_config(config, index)] = result
    return indexed_results


def extract_latest_minima_from_hopt_df(df: pd.DataFrame, result_col: str = "RESULT"):
    def _latest_min_row(df_seed: pd.DataFrame) -> pd.Series:
        min_metric = df_seed[result_col].min()
        return df_seed.loc[df_seed[result_col] == min_metric].iloc[-1]

    return df.groupby('omniopt_seed').apply(_latest_min_row)


def get_seed_min_series(df) -> pd.Series:
    idx_min = df["RESULT"].idxmin()
    return df.loc[idx_min]


def process_omniopt_results(
    path_experiment: pathlib.Path,
    n_processes: int = 48,
    evals_thr: int = 198

):
    df_experiment = data.read.read_hopt_experiments([path_experiment])

    path_experiment_config = next(path_experiment.glob("*.toml"))
    experiment_config = parse(path_experiment_config.read_text())
    cols_config = [d["name"] for d in experiment_config["DIMENSIONS"]]
    print(f"Dimensions: {cols_config}")

    print(f"Reading results indexed by {cols_config} ...")
    experiment_results = read_results_from_experiment(
        path_experiment, n_processes)
    experiment_results_indexed = index_results(experiment_results, cols_config)
    
    df_experiment = df_experiment.dropna(subset=cols_config)
    if "ExperimentConfig.num_epochs" in cols_config:
        print("Converting num epochs to int ...")
        df_experiment["ExperimentConfig.num_epochs"] = df_experiment["ExperimentConfig.num_epochs"].astype(
            int)

    df_experiment = df_experiment.sort_index()
    df_max_hopt_iter = df_experiment.reset_index(
        level="hopt_iter", drop=False).groupby(["omniopt_seed"])["hopt_iter"].max()
    seeds_with_sufficient_evals = df_max_hopt_iter.loc[df_max_hopt_iter >= evals_thr].index.get_level_values(
        "omniopt_seed")
    df_experiment = df_experiment.loc[pd.IndexSlice[:,
                                                    seeds_with_sufficient_evals, :evals_thr]]

    df = df_experiment.groupby(["omniopt_seed"]).apply(get_seed_min_series)
    df["N_evals"] = evals_thr
    df = df[[*cols_config, "N_evals", "RESULT"]]

    print(df.dtypes)

    print(f"Extracting test metrics ...")

    test_error_rates = []
    for oo_seed, *h_params in df[cols_config].itertuples():
        result = experiment_results_indexed[oo_seed][tuple(
            convert_idx_val(v) for v in h_params)]
        test_error_rates.append(
            1 - result["result_data"]["test_metrics"]["acc"])

    df["test_error_rate"] = test_error_rates

    print(f"total values {len(df)}")

    df.to_csv(path_experiment / "data" / "processed" /
              "hopt" / "metric_values.csv", index=True)

    return df


if __name__ == "__main__":
    process_omniopt_results(pathlib.Path(
        "/beegfs/ws/0/s4610340-isml/simpsons/experiments/11_simpsons_hopt_tpe_AdamW_seeds"))
    # process_omniopt_results("/beegfs/ws/0/s4610340-isml/simpsons/experiments/10_simpsons_hopt_tpe_SGDwlrs_seeds")

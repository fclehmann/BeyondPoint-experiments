from functools import partial
import json
import multiprocessing
import pathlib
from typing import Optional
import pandas as pd
import tqdm
import yaml


def read_experiment_hopt_results_df(file_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "Unnamed: 0" in df:
        df.rename({"Unnamed: 0": "HOpt iter"}, axis=1, inplace=True)
    try:
        df.drop("omniopt_seed.1", axis=1, inplace=True)
    except Exception as e:
        print(e)
        print(
            f"This is a bug caused by results extraction script. Omniopt seed is in columns = {'omniopt_seed' in df.columns}.")
        pass
    return df


def read_hopt_experiments(paths_experiments: list[pathlib.Path]) -> pd.DataFrame:

    dfs = []

    for path_exp in paths_experiments:
        df_oo_results = pd.read_csv(
            path_exp / "data" / "processed" / "hopt" / "omniopt_results.csv")
        df_oo_results["experiment"] = path_exp.stem

        if "Unnamed: 0" in df_oo_results:
            df_oo_results.rename(
                {"Unnamed: 0": "hopt_iter"}, axis=1, inplace=True)

        df_oo_results.set_index(
            ["experiment", "omniopt_seed", "hopt_iter"], inplace=True)

        try:
            df_oo_results.drop("omniopt_seed.1", axis=1, inplace=True)
        except Exception as e:
            pass

        dfs.append(df_oo_results)

    return pd.concat(dfs)


def read_result_from_run(
        path: pathlib.Path,
        result_sections: Optional[list[str]] = None,
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
        if result_sections is not None:
            result = {sec: result[sec] for sec in result_sections}

    return input_config, result


def read_results_from_experiment(path_experiment: pathlib.Path, n_processes: int = 24, chunksize: int = 10, result_sections: Optional[list[str]] = None,):
    paths = list((path_experiment / "results").iterdir())

    with multiprocessing.Pool(processes=n_processes) as p:

        results = [
            (c, r) for c, r in
            tqdm.tqdm(
                p.imap(
                    partial(read_result_from_run, result_sections=result_sections),
                    paths,
                    chunksize=chunksize
                ),
                total=len(paths)
            ) if c is not None and r is not None
        ]

    return results

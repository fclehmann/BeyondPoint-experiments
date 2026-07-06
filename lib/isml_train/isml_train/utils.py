from argparse import Namespace
from dataclasses import asdict, is_dataclass
import logging
import os
from pathlib import Path
import random
import sys
import typing

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.backends.cudnn


def reset_model_weights(model: torch.nn.Module):
    """
    Resets parameters of all children layers of model

    Works when there is no sequential inside;

    :param model:
    :return: None
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def train_test_split_bootstrap(
    df: pd.DataFrame,
    n_bootstrap_samples: int,
    with_replace: bool,
    random_state: typing.Optional[int] = None,
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Splits the dataset into train and test using sampling.
    Train dataset - n_bootstrap_samples rows are randomly selected from input dataset.
    Test dataset - rows that were not selected for train dataset are collected into test df.

    If the with_replace=False and n_bootstrap_samples = len(df.index), then 1/e rows are collected into test df.

    :param df: Input dataframe
    :param n_bootstrap_samples: How many rows to sample for train df.
    :param with_replace: Sample with or without replacement
    :param random_state: Random state for resample. If None => numpy seed is used.
    :return: (Train df, Test df)
    """
    # Depends on index of original dataframe. sklearn.utils.resample does not affect pandas index.
    train_df = sklearn.utils.resample(
        df,
        replace=with_replace,
        n_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    test_df = df[~df.index.isin(train_df.index)]
    return train_df, test_df


def seed_everything(seed=42, enable_deterministic: bool = True) -> None:
    """
    Set given seed in every random number generator available.
    https://datascience.stackexchange.com/questions/66345/why-ml-model-produces-different-results-despite-random-state-defined-and-how-to
    :param seed: seed
    :return: None
    """

    # TODO: write which seeds are used in which methods

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if enable_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def flatten_dict(
    params: typing.MutableMapping[typing.Any, typing.Any],
    delimiter: str = "/",
    parent_key: str = "",
) -> dict[str, typing.Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}

    """
    result: dict[str, typing.Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if is_dataclass(v):
            v = asdict(v)
        elif isinstance(v, Namespace):
            v = vars(v)

        if isinstance(v, typing.MutableMapping):
            result = {
                **result,
                **flatten_dict(v, parent_key=new_key, delimiter=delimiter),
            }
        else:
            result[new_key] = v
    return result


def get_omniopt_seed() -> typing.Optional[float]:
    """
    Gets omniopt(->hyperopt) seed from slurm job name if possible.
    :return: serializable dict
    """

    import re

    job_name: typing.Optional[str] = os.environ.get("SLURM_JOB_NAME", None)

    if job_name is None:
        return None

    seed_regex = r"SEED_(\d*\.\d+|\d+)"
    match = re.search(seed_regex, job_name)

    if match is None:
        return None
    else:
        return float(match.group(1))


def get_slurm_job_info(
    variables=["SLURM_JOBID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_NODELIST"]
):
    return {k: os.getenv(key=k) for k in variables}


def setup_logging(log_level: typing.Any, log_file_path: Path):
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )
    logging.root.setLevel(log_level)

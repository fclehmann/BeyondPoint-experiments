import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing as t
import seaborn as sns

DF_AGGREGATED_MULTI_SEED = t.NewType("DF_AGGREGATED_MULTI_SEED", pd.DataFrame)
DF_RAW_MULTI_SEED = t.NewType("DF_RAW_MULTI_SEED", pd.DataFrame)


def plot_metric_hist(
        df: DF_AGGREGATED_MULTI_SEED,
        experiment_name: str,
        dataset_name: str,
        hopt_algorithm_name: str,
        n_iter: int,
        metric_name: str,
        ax: t.Optional[plt.Axes] = None,
        bins: t.Union[str, int] = "auto",
        result_col: str = "RESULT",
) -> tuple[t.Optional[plt.Figure], plt.Axes]:

    fig = None

    if ax is None:
        fig: plt.Figure = plt.figure(figsize=(12, 8))
        ax = plt.subplot()

    sns.histplot(df[result_col], ax=ax, bins=bins)
    ax.set_title(
        f"Dataset: {dataset_name} - empirical distribution of {metric_name}; \n"
        f"H-opt (N seeds: {len(df['loss'])}, algorithm: {hopt_algorithm_name}, n_iter: {n_iter}); \n"
        f"Experiment: {experiment_name};"
    )
    ax.set_xlabel(metric_name)

    return fig, ax


def plot_opt_iterations(
        df: DF_RAW_MULTI_SEED,
        experiment_name: str,
        dataset_name: str,
        hopt_algorithm_name: str,
        metric_name: str,
        result_col: str = "RESULT",
        title_pre: t.Optional[str] = None,
        ax: t.Optional[plt.Axes] = None
) -> tuple[t.Optional[plt.Figure], plt.Axes]:
    fig = None
    # 123
    if ax is None:
        fig: plt.Figure = plt.figure(figsize=(12, 8))
        ax = plt.subplot()

    paths = []

    for seed, df_g in df.groupby("omniopt_seed"):
        path_acc = np.minimum.accumulate(df_g[result_col].to_numpy(na_value=df_g[result_col].max()))
        paths.append(path_acc)

    paths = np.array(paths, dtype=float)
    paths_min = paths.min(axis=0)
    paths_max = paths.max(axis=0)

    ax.plot(paths_min, color="red", label="min")
    ax.plot(paths_max, color="green", label="max")

    ax.fill_between(
        x=np.arange(paths.shape[1]),
        y1=paths_min,
        y2=paths_max,
        color="orange",
        alpha=.3
    )
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Hyperopt iteration i")

    if title_pre is None:
        title_pre = f"Accumulated min applied to each hopt path. " \
                    f"Then the matrix is built where the rows are (acc.) hopt path for seed i and column is" \
                    f"state of the metric on this iteration. The min and max is then taken over columns to estimate" \
                    f"the top and bottom border."

    ax.set_title(f"{title_pre}: dataset={dataset_name};\nexperiment={experiment_name}")

    ax.legend()

    if fig is not None:
        fig.tight_layout()
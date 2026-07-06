from collections import defaultdict
from email.policy import default
from itertools import islice
from pathlib import Path

import pandas as pd
import numpy as np
import rich.console 
import rich.table
import rich.progress
import rich.style

import experiment_runner.run.experiment_config as experiment_config

def progress_with_percentage(n_1: int, n_2: int) -> str:
    return f"{n_1} / {n_2} ({(n_1 / n_2) * 100:.2f}%)"

def get_experiment_state(path: Path, show_n_seeds: int = 5):
    path_experiment_config = next(path.glob("*.toml"))
    config = experiment_config.read_experiment_config(path_experiment_config)
    
    assert isinstance(config.DATA.seed, list)
    
    n_seeds_configured = len(config.DATA.seed)
    n_iterations_configured = config.DATA.max_evals

    seed_to_stats = defaultdict(dict)

    overall_iters_done = 0
    overall_iters_with_results = 0
    n_seeds_with_results = 0
    
    seeds_with_results_dir_absent = []
    seeds_with_result_files_absent = []

    fractions = [0.25, 0.5, 0.75]

    fraction_to_quantity = defaultdict(int)

    for seed in rich.progress.track(config.DATA.seed, description=f"Scaning {config.ENV_PARAMS.base_params.experiment_name}"):
        project_name = f"{config.ENV_PARAMS.base_params.experiment_name}_SEED_{seed}"
        project_path = path / "omniopt" / "projects" / project_name
        csv_path = project_path / "csv"

        if not csv_path.exists():
            seeds_with_results_dir_absent.append(seed)
            seed_to_stats[seed]["results_dir_exists"] = False
            seed_to_stats[seed]["results_files_exist"] = False
            seed_to_stats[seed]["n_iter"] = None
            continue
        else:
            seed_to_stats[seed]["results_dir_exists"] = True

        csv_files = sorted(list(csv_path.iterdir()), reverse=True)

        seed_to_stats[seed]["results_files_exist"] = len(csv_files) > 0

        if not seed_to_stats[seed]["results_files_exist"]:
            seeds_with_result_files_absent.append(seed)
            seed_to_stats[seed]["n_iter"] = None
            continue
        
        omni_result_df: pd.DataFrame = pd.read_csv(csv_files[0], sep=";")
        seed_to_stats[seed]["n_iter"] = len(omni_result_df)

        overall_iters_done += seed_to_stats[seed]["n_iter"]
        
        if "RESULT" in omni_result_df.columns:
            seed_to_stats[seed]["n_iter_with_result"] = len(omni_result_df.loc[~omni_result_df["RESULT"].isna()])
            overall_iters_with_results += seed_to_stats[seed]["n_iter_with_result"]
            n_seeds_with_results += 1

        for frac in fractions:
            if seed_to_stats[seed]["n_iter"] < n_iterations_configured * frac:
                fraction_to_quantity[frac] += 1
        

    table = rich.table.Table(
        title=f"State of {config.ENV_PARAMS.base_params.experiment_name}",
        show_lines=True,
        expand=True,
        title_style=rich.style.Style(color="red", bold=True),
        border_style=rich.style.Style(color="bright_black")
    )

    table.add_column("Statistic")
    table.add_column("Value")
    table.add_column("Seeds")

    table.add_row(
        "N iterations/seed configured", 
        str(n_iterations_configured), 
        ""
    )

    table.add_row(
        "N iterations total", 
        progress_with_percentage(overall_iters_done, n_seeds_configured * n_iterations_configured), 
        ""
    )
    table.add_row(
        "N iterations total with non-nan result", 
        progress_with_percentage(overall_iters_with_results, n_seeds_configured * n_iterations_configured), ""
    )
    table.add_row(
        "N seeds with results", progress_with_percentage(n_seeds_with_results, n_seeds_configured), "",
    )
    table.add_row(
        "N seeds with results dir absent", 
        progress_with_percentage(len(seeds_with_results_dir_absent), n_seeds_configured), 
        f"{list(islice(seeds_with_results_dir_absent, show_n_seeds))} ..."
    )
    table.add_row(
        "N seeds with result files absent", 
        progress_with_percentage(len(seeds_with_result_files_absent), n_seeds_configured), 
        f"{list(islice(seeds_with_result_files_absent, show_n_seeds))} ..."
    )
    table.add_row(
        "N seeds with iters quantity < .25-.5-.75", "-".join([str(fraction_to_quantity[f]) for f in fractions]), ""
    )

    console = rich.console.Console()
    console.print(table)

    
if __name__ == "__main__":
    get_experiment_state(Path("/home/egor/ws/isml/simpsons/experiments/10_simpsons_hopt_tpe_SGDwlrs_seeds"))
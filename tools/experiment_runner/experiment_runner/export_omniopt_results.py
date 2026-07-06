import logging
import pathlib
import re

import pandas as pd

_logger = logging.getLogger(__name__)


def export_omniopt_results(experiment_path: pathlib.Path) -> pd.DataFrame:
    _logger.info(f"Exporting results from {experiment_path}")

    omniopt_project_regex = rf"{experiment_path.name}_SEED_(\d*\.\d+|\d+)"
    path_to_omni_projects = experiment_path / "omniopt" / "projects"

    _logger.debug(f"Path to omniopt projects: {path_to_omni_projects}")

    omni_results: list[pd.DataFrame] = []

    not_exist_count: int = 0

    for project in path_to_omni_projects.glob(f"{experiment_path.name}_SEED_*"):

        _logger.info(f"--------------------------------- {project.name} ---------------------------------")

        _logger.info(f"Processing {project} ...")

        match = re.search(omniopt_project_regex, project.name)
        seed_str: str = match.group(1)

        if seed_str.isdigit():
            seed = int(seed_str)
        else:
            seed = float(seed_str)

        _logger.info(f"Extracted seed: {seed}")

        csv_path = project / "csv"

        if not csv_path.exists():
            not_exist_count += 1
            _logger.warning(f"Csv files not found for project: {project}")
            continue

        csv_files = list(csv_path.iterdir())

        _logger.info(f"Found csv files: {csv_files}")

        if len(csv_files) > 1:
            _logger.info("Multiple csv files => sorting in descending order ...")
            csv_files = sorted(csv_files, reverse=True)
            _logger.info(f"Will export: {csv_files[0]}")

        try:
            omni_result_df: pd.DataFrame = pd.read_csv(csv_files[0], sep=";")
            omni_result_df["omniopt_seed"] = seed
            omni_results.append(omni_result_df)
        except Exception as e:
            _logger.exception(e, exc_info=True)

    _logger.info(f"Seeds count successfully extracted: {len(omni_results)}")
    _logger.info(f"Seeds count export failed: {not_exist_count}")

    return pd.concat(omni_results)

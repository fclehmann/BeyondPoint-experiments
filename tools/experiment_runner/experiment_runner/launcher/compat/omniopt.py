import logging
from multiprocessing import Value
import os
from pathlib import Path
import re
import select
import shlex
import shutil
import subprocess
from tempfile import tempdir
import tempfile
from typing import Any, Optional, Sequence, Union
import uuid
import experiment_runner
import experiment_runner.launcher
import tomlkit
import optuna
import math

from experiment_runner.launcher.brute_force_sampler import BruteForceSamplerWithoutStop

_logger = logging.getLogger(__name__)


def parse_search_space(
    dimensions: list[dict],
) -> dict[str, optuna.distributions.BaseDistribution]:
    search_space = {}

    for dimension in dimensions:
        if dimension["range_generator"] == "hp.uniformint":
            search_space[dimension["name"]] = (
                optuna.distributions.IntUniformDistribution(
                    low=dimension["min"], high=dimension["max"]
                )
            )
        elif dimension["range_generator"] == "hp.loguniform":
            search_space[dimension["name"]] = optuna.distributions.FloatDistribution(
                low=math.exp(dimension["min"]),
                high=math.exp(dimension["max"]),
                log=True,
            )
        elif dimension["range_generator"] == "hp.uniform":
            search_space[dimension["name"]] = optuna.distributions.FloatDistribution(
                low=dimension["min"], high=dimension["max"], log=False
            )
        elif dimension["range_generator"] == "hp.choice":
            search_space[dimension["name"]] = (
                optuna.distributions.CategoricalDistribution(
                    choices=dimension["options"]
                )
            )
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

    return search_space


def get_parameter_values_from_trial(
    trial: optuna.Trial, search_space: list[dict]
) -> dict[str, Any]:
    params = {}

    for dim in search_space:
        method = getattr(trial, dim["method"])
        params[dim["method_params"]["name"]] = method(**dim["method_params"])

    return params


def extract_result_variable(text: str) -> Union[int, float, None]:
    # Regular expression to find 'RESULT: {number}' where number can be int or float
    match = re.search(r"RESULT:\s*([-+]?\d*\.?\d+)", text)

    if match:
        result_str = match.group(1)

        # Convert the result to either int or float
        if "." in result_str:
            return float(result_str)
        else:
            return int(result_str)

    return None  # Return None if no match is found


def wait_for_processes_with_output(
    processes: list[subprocess.Popen],
) -> list[list[str]]:

    p_dicts = [{"process": p, "outputs": []} for p in processes]

    # Continuously check for output
    while processes:
        # Use select to wait for any output on any of the pipes
        readable_pipes = select.select(
            [proc.stdout for proc in processes if proc.stdout], [], [], 0.1
        )[0]

        # For each readable pipe, display its output
        for pipe in readable_pipes:
            output = pipe.readline()
            if output:
                if (
                    "torch/autograd/__init__.py:251"
                    in output
                ):
                    continue
                elif "Variable._execution_engine.run_backward" in output:
                    continue
                p_obj = next(p for p in p_dicts if p["process"].stdout == pipe)
                output = output.strip()
                print(output.strip())  # Print the output without extra newline
                p_obj["outputs"].append(output)

        # Remove any finished processes
        processes = [proc for proc in processes if proc.poll() is None]

    return [p["outputs"] for p in p_dicts]


class CLIObjective:
    def __init__(
        self,
        objective_program_path: Path,
        runs_path: Path,
        data_path: Optional[Path] = None,
        data_option_name: Optional[str] = None,
        data_archive_path: Optional[Path] = None,
    ) -> None:
        self._objective_path = objective_program_path
        self._runs_path = runs_path
        self._runs_path.mkdir(exist_ok=True)
        self._data_path = data_path
        self._data_option_name = data_option_name
        self._data_archive_path = data_archive_path

    def build_command(self, params: dict[str, Any]):
        c = f"{str(self._objective_path)}"
        for k, v in params.items():
            c += f' --{k}="{v}"'
        return c

    def __call__(self, params: Sequence[dict[str, Any]]) -> Sequence[Union[int, float]]: 
        if self._data_path and self._data_option_name and self._data_archive_path:
            duid = str(uuid.uuid4())
            path_data_tmp = Path(f"/tmp/{duid}")
            path_data_tmp.mkdir()
            _logger.info(f"Copying data to {path_data_tmp}")
            print(path_data_tmp)
            path_data_archive_tmp = path_data_tmp / self._data_archive_path.name
            print(path_data_archive_tmp)
            shutil.copyfile(self._data_archive_path, path_data_archive_tmp)
            subprocess.run(f"tar -xzf {path_data_archive_tmp} -C {path_data_tmp} --strip-components={len(self._data_path.parents)-1}", shell=True)
            subprocess.run(f"ls -l {path_data_tmp}", shell=True)
            for p in params:
                p[self._data_option_name] = str(path_data_tmp / self._data_path.name)
        
        processes: list[subprocess.Popen] = []

        for p in params:
            command = self.build_command(p)
            process = subprocess.Popen(
                command,
                shell=True,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
                env=os.environ,
            )
            processes.append(process)

        outputs = wait_for_processes_with_output(processes)

        if self._data_path and self._data_option_name:
            shutil.rmtree(path_data_tmp)

        results = []

        for process_outputs in outputs:
            result = extract_result_variable("\n".join(process_outputs))
            assert result is not None
            results.append(result)

        return results


def create_sampler(ec, seed: Optional[int] = None):
    oo_algo_name = ec["DATA"]["algo_name"]
    if oo_algo_name == "gridsearch":
        return BruteForceSamplerWithoutStop(seed=seed)
    elif oo_algo_name == "tpe.suggest":
        return optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f"Unsupported oo algorithm: {oo_algo_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _logger.info("Starting objective")
    objective = CLIObjective(
        objective_program_path=Path(
            "/data/horse/ws/s4610340-isml/dev/test_launcher/experiments/004_cifar10_clf_opt_augment/run_experiment.sh"
        ),
        runs_path=Path(
            "/data/horse/ws/s4610340-isml/dev/test_launcher/experiments/004_cifar10_clf_opt_augment/runs1"
        ),
    )

    objective(
        [
            {
                "ExperimentConfig.run_uid": "S_None-N_0-ID_285a71",
                "ExperimentConfig.augmentation_seed": 1518,
            },
            {
                "ExperimentConfig.run_uid": "S_None-N_1-ID_b93000",
                "ExperimentConfig.augmentation_seed": 8830,
            },
            {
                "ExperimentConfig.run_uid": "S_None-N_2-ID_27c845",
                "ExperimentConfig.augmentation_seed": 1106,
            },
        ]
    )

from functools import partial
import json
import multiprocessing
import pathlib
from typing import Optional
import yaml
import tqdm
from tomlkit import parse
import tarfile

def read_info_from_run(path: pathlib.Path, dimensions: list[str]) -> Optional[dict]:
    """Reads run config and run results file 

    :param path: _description_
    :return: config, results tuple
    """
    path_result = next(path.glob("*.json"), None)

    if path_result is None:
        return None

    path_config = next(path.glob("*.yaml"), None)

    if path_config is None:
        return None

    with open(path_config, "r") as f:
        config = yaml.safe_load(f)

    with (path_result).open("r") as file:
        result = json.load(file)

    r = {d:config["ExperimentConfig"][d] for d in dimensions}
    r["omniopt_seed"] = int(result["environment"]["omniopt_seed"])
    r["job_id"] = result["SLURM_tracking"]["SLURM_JOBID"]
    r["val_metrics"] = result["result_data"]["train_results"][0]["metrics"][-1]["val"]
    r["test_metrics"] = result["result_data"]["test_metrics"]
    r["epoch"] = result["result_data"]["train_results"][0]["metrics"][-1]["epoch"]

    return r

def gather_data_from_hopt_experiment(
        path_experiment: pathlib.Path, 
        n_processes: int = 12, 
        chunksize: int = 10
    ):
    paths = list((path_experiment / "results").iterdir())

    data = []

    path_experiment_config = next(path_experiment.glob("*.toml"))
    experiment_config = parse(path_experiment_config.read_text())
    dimensions = [d["name"].split(".")[1] for d in experiment_config["DIMENSIONS"]]

    read = partial(read_info_from_run, dimensions=dimensions)

    with multiprocessing.Pool(processes=n_processes) as p:
        for result in tqdm.tqdm(p.imap(read, paths, chunksize=chunksize), total=len(paths)):
            if result is None:
                continue
            else:
                data.append(result)
   
    return data

def gather_results_from_experiment(
    path_experiment: pathlib.Path, 
    n_processes: int = 12, 
    chunksize: int = 10
):
    name = path_experiment.stem
    d = gather_data_from_hopt_experiment(path_experiment, n_processes=n_processes, chunksize=chunksize)
    with open(f"/beegfs/ws/0/s4610340-isml/.dev/.yahor/statistics-in-machine-learning/data/simpsons/{name}.json", "w") as f:
        json.dump(d, f)

experiments = list(pathlib.Path("/beegfs/ws/0/s4610340-isml/simpsons/experiments").glob("*_hopt_*"))
#experiments = [pathlib.Path("/beegfs/ws/0/s4610340-isml/simpsons/experiments/12_simpsons_hopt_tpe_Adamax_seeds")]
print(experiments)

for experiment in experiments:
    print(experiment)
    gather_results_from_experiment(experiment, n_processes=24)
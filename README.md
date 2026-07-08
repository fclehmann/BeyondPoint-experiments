# Quantifying Uncertainty and Variability in Machine Learning

This repository contains train pipelines, schedulers, and (up to some extent) data analyses for "Quantifying Uncertainty and Variability in Machine Learning: Confidence Intervals for Quantiles in Performance Metric Distributions".

## Data

All data relevant to the publication is stored in lightweight and aggregated form in `data/{use_case}.json`.

Each `data/{use_case}.json` is an array of data objects corresponding to experiments. These data objects have at least the following keys:
- `experiment` - name of experiment
- `image` - path to singularity image (with commit hash)
- `experiments_path` - path to experiment folder on cluster
- `best_metric_decision_type` - str - max/min/last - the method used to decide how to extract the optimum of the decision_metric in an indivdual run.
- `decision_metric` -  metric used to the optimum value of individual run in the step of post-processing.
- `seed_param` - name of seed parameter which was varied
- `seeds` - list of strings - seed values
- `metric_values` - mapping of parameter or metric name to the value corresponding to individual run. In the case of hopt experiments, it contains entire hyperparameter optimization paths and has to be grouped by hopt seed to extract an optimum. To obtain the dataframe you can treat `metric_values` as mapping str -> column and append seeds as an additional column. In the case of a multi-source experiment (simpsons 20) the structure of seeds is different. 

## Implementation notes

### Folder structure
- `./use_cases` - contains use-case specific train pipeline, experiments, notebooks, etc.
  - `./use_cases/simpsons` - simpsons classifier (CNN, VGG11)
  - `./use_cases/superconduct_tc_reg` - critical temperature of superconductors regression (DNN, GBDT)
  - `./use_cases/cifar10_clf` - cifar10 classifier (VGG11)
- `./lib` - reusable code components as installable libraries
- `./tools` - tools
   - `./tools/experiment_runner` - Deploy/run experiments on taurus. Process, archive and export of experiment results.

Each use-case is a python package. Important contents are:
- `src` - source code
- `experiments` - Experiment configs associated with this use case
- `configs` - Run-specific configs.
- `pyproject.toml`
- `poetry.lock`

### Experiment environment
- Each experiment is associated with a specific commit hash in one of 3 ways (based on `env_type`):
  - `env_type="singularity"`: Pre-built singularity image of use-case is used. Docker image definition: `docker-files/Dockerfile_use-case`. Images (versioned) are available in the private image registry in our gitlab.
- Each experiment corresponds to a specific {use-case}.
- Each experiment is defined by it's configuration file (for example
  `use_cases/simpsons/experiments/10-simpsons-hopt-tpe-SGDwlrs-seeds.toml`)

### Sweeper/launcher
Initially, we started with our in-house hyperparameter optimization framework Omniopt to sweep over the seeds. In first place, we had to make sure that the CLI interface of each use-case is compatible with the one expected by Omniopt - this is handled in each use case individually, as they also use different configuration management frameworks (`hydra`, `spock`).  Secondly, some automatization was necessary to install and launch Omniopt when running an experiment given the experiment config file - this is handled within `experiment_runner.run` module.

However, at a later point some modifications of the approach for scheduling our runs on SLURM were necessary. We have built our custom launcher module `experiment_runner.launcher` (actually scripts) based on `optuna` and `submitit`. We have used it to launch cifar10_clf experiments.

- We have used Omniopt based setup to launch simpsons and superconduct experiments.
- We have used `experiment_runner.launcher` based setup
to launch cifar10 experiments.

It should be possible to launch the cifar10 experiments using
`experiment_runner.launcher` on any cluster with SLURM batch system with the little effort (mainly replacing filesystem paths and hardware requirements). However, the launch of other experiments, which were executed by omniopt would require more efforts.

For further work, it would be beneficial to write a new clean scheduling module. Most of the work is already carried out in [submitit](https://github.com/facebookincubator/submitit) and [hydra](https://github.com/facebookresearch/hydra). On the other hand `experiment_runner.launcher` module provides a starting point.

## Reproducibility / deterministic behavior

### 1. Level: Individual runs
Reproducibility / deterministic behavior was tested for each use case individually by doing several consequtive runs with same seeds, reduced number of epochs and comparing the results (Implementated in `use_cases/{use-case}/tests/test_reproducibility.py`).
In the table below the test results are summarized:

| Use-case            | number of runs | epochs | metric       | decimal places |
|---------------------|----------------|--------|--------------|----------------|
| simpsons            | 10             | 5      | val_Accuracy | all            |
| cifar10_clf         | 10             | 5      | val_Accuracy | all            |
| superconduct_tc_reg | 10             | 5      | val_RMSE     | all            |

To reproduce the results of individual runs you have 2 options:
- Pull the image of corresponding version from our gitlab registry
- Build the image by yourself:
  1. Checkout the version you need
  2. Build the docker image using `docker-files/Dockerfile_use-case` as described in [this section](#build-use-case-singularity-image)

### 2. Level: Experiments / sweeps
It is possible to reproduce the distributions of the metrics given the experiment configs (`./use_cases/{use-case}/experiments`). We recommend to write your own software to execute the experiments on your compute resources. You can use
`experiment_runner.launcher` and `experiment_runner.run` as starting point. Please refer to [this section](#sweeperlauncher) for more details.

## Usage

### Running experiments
This section describes some aspects of running experiments. If you want to reproduce the experiments please refer to [reproducibility section](#reproducibility--deterministic-behavior). 

The experiments are executed using `experiment_runner.run` module. Depending on which sweeper backend is configured (`omniopt` or `experiment_runner.launcher`) run command will either submit job(s) and exit or start master process, which will submit jobs. It is important to specify in config file where the directory containing experiments is located (`experiments_path`), the singularity image of the use case (`singularity_image_file`) and case specific run arguments like path to dataset and output path.

```
$ python -m experiment_runner run --help
                                                                                                          
 Usage: python -m experiment_runner run [OPTIONS] EXPERIMENT_CONFIG_PATH                                  
                                                                                                          
 Runs experiment defined by given experiment config.                                                      
                                                                                                          
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    experiment_config_path      PATH  [default: None] [required]                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────╮
│ --clean_workdir                 Remove * from workdir on taurus                                        │
│ --submit_jobs                   Submit jobs                                                            │
│ --log_level            INTEGER  Logging level [default: 20]                                            │
│ --help                          Show this message and exit.                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

After the jobs on cluster are finished you will see in your `experiments_path` the
directory containing the experiment, the directory name will be equal to how your EC file is named (trimmed),
this name is further denoted as `{experiment-name}`.
So in `experiments_path/{experiment-name}` you find:
- `results/` - single train outputs (json files and models)
- `omniopt/` or `hopt/` - hopt outputs.

To process the results use `experiment_runner.post_process` module:
```
$ python -m experiment_runner process --help
                                                                                                          
 Usage: python -m experiment_runner process [OPTIONS] PATH_EXPERIMENT                                     
                                                                                                          
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    path_experiment      PATH  [default: None] [required]                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────╮
│ --metric   -m      TEXT                                                                                │
│ --workers          INTEGER  [default: 32]                                                              │
│ --as_job                                                                                               │
│ --mem              TEXT     [default: 32GB]                                                            │
│ --time             TEXT     [default: 12:00:00]                                                        │
│ --help                      Show this message and exit.                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
It will produce single `./data/processed/data.json` file within experiment folder.

#### Experiment results

The results of the experiment should be stored in a workspace on cluster:
`/data/horse/ws/s4610340-isml/{use-case}/experiments/{experiment_name}`.
You just need to specify the conventional `experiments_path` setting in your experiment config and give the 
experiment config file uniq name which meets [naming conventions](#experiment-configs-naming-convention). The rest is done by experiment-runer.

Normally, an experiment produces the following data inside the experiment directory:
- `results/` - single train outputs (.json) and models (.pt).
- `omniopt/` - OmniOpt sources and HOpt data. 


### Experiment definition
Experiment is defined using experiment config file (EC for short). ECs are located
in `./use_cases/{use-case}/experiments`. Experiment name (corresponds to file name) should be in format
`{experiment number}_{use-case}_{hopt or opt}_...`. And due to limitations of omniopt (due to 04.03.2024)
contain less then 64 chars together with postfix `_SEED_{hopt seed}`.

EC is currently .ini file (#TODO: Switch to json) and has multiple sections:
1. **`[ENV_PARAMS]`** - Information about environment. Must contain following keys:
   - `env_type` - type of deployment to use, one of: `singularity`, ~~`venv`~~, ~~`pre-created-venv`~~. 
   Preferred is `singularity`.
   - `...` parameters specific to env_type:
     - for `env_type="singularity"`:
        - `singularity_image_file` - path to singularity image
   - `[ENV_PARAMS.base_params]` - common params for every env_type 
     - `experiments_path` - path to experiments directory on taurus
     - `objective_program_name` - name of generated bash script which will be executed by OmniOpt
     - `command_name` - the command which will be executed in objective program
   use-case already installed there. If not set (preferred) -
   the new venv for this experiment will be created and use-case installed.
2. **`[FIXED_EXPERIMENT_PARAMS]`** - The arguments which will be passed into `ENV_PARAMS['command_name']`.
These arguments are kept constant during the experiment.
3. **`[EXTRA_HARDWARE_PARAMS]`** - Extra hardware params.
   - `num_cpus_per_worker` - Number of cpus per worker
   - `reservation` - Reservation
4. **`[DIMENSIONS]`** - HOpt search space is defined here. Follows OmniOpt config format.
5. **`[DEBUG]`, `[MONGODB]`, `[DATA]`** - Omniopt sections, refer to OmniOpt docs.
6. **`[LAUNCHER]`** - `experiment_runner.launcher` specific options. If this section is present, then experiment runner uses `experiment_runner.launcher` instead of omniopt. 

Make sure that each param required by `ENV_PARAMS['command_name']` is listed either in 
`[FIXED_EXPERIMENT_PARAMS]` or in `[DIMENSIONS]` section. 

The OmniOpt config is "included" in EC (sections `[DIMENSIONS]`, `[DEBUG]`, `[MONGODB]`, `[DATA]`). The actual
OmniOpt config will be then generated by experiment_runner and passed to OmniOpt.


#### Setting seeds
###### OmniOpt seed(s)
In section `[DATA]` parameter `seed`.

###### Use-case specific seeds
Use case specific seeds are controlled within `[FIXED_EXPERIMENT_PARAMS]`. The exact parameter name is case-specific.

### Notes on integration of use-case
Use-case cli must accept:
- `ExperimentConfig.out_path` option 
(e.g `--ExperimentConfig.out_path="/beegfs/ws/0/s4610340-isml/lrl_sorb/experiments/001_lrl_sorb_hopt_tl_tpe_AdamW_seeds/results/"`).
This is the folder, where the train pipeline must create a unique subdirectory (for example using `uuid`) containing
all outputs of the individual run.


### Experiment configs naming convention
`{Experiment number}-{use-case}-{HOpt or Opt}-...any_other_meaningfull_description.toml`. Don't use 
any special characters except "-" and "_".

### Build docker image
From repo root execute, substituting `simpsons` with necessary use-case:
1. Build docker image 
    ```bash
    DOCKER_BUILDKIT=1 docker build --build-arg USE_CASE=simpsons --secret id=gitlab,src=$HOME/.config/pypoetry/auth.toml -f ./docker-files/Dockerfile_use-case -t gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/simpsons:$(git rev-parse --short HEAD) .
    ```


### Running train pipeline of use-case
Train pipelines are guaranteed to work for any commit hash specified in experiment config files (For example `use_cases/superconduct_tc_reg/experiments/015-superconduct_tc_reg-dnn-opt-split_cv.toml` specifies `singularity_image_file = "/data/horse/ws/s4610340-isml/sifs/superconduct_tc_reg_d14f389.sif"` which means that the image built from the commit with hash `d14f389` is used). This is particularly important for simpsons case as it is not compatible with the latest version of isml_train lib. Please refer to ./use_cases/{use-case}/README.md for guidelines on how to obtain the data and launch the train.

## Glossary

#### HOpt seed
Seed set in OmniOpt (<=> in hyperopt <=> in HOpt algorithm)

#### HOpt iteration i
In the context of HOpt this is train and evaluation of model using some hyperparameters combination 
performed by OmniOpt on i'th iteration.

#### HOpt path
Sequence of (hopt iteration i, hyperparams, metric value) for specific seed. If there is n HOpt seeds
in experiment - there is n HOpt paths.

#### Cumulative min of HOpt path
Cumulative min applied to HOpt path (more precise: applied to metric values). 
https://stackoverflow.com/questions/23622157/calculating-cumulative-minimum-with-numpy-arrays


## Note on structure

### Requirements for structure
1. Train pipelines may have incompatible dependencies
2. logically similar Things go into one directory
3. Train pipelines for different datasets are separated
4. Single-dataset-related notebooks, scripts, etc should be in the same directory as train pipeline
5. Experiment configs should be located in the same directory as corresponding train pipeline
6. Train pipelines can have shared code which must not be repeated
7. Well-defined dependencies of each train pipeline & reproducibility
8. Run pipelines easily both locally and on cluster


./lib - Needed to share code between projects. There may be many “libraries” under the ./lib. Each library is a separate poetry project. It’s installable and can be specified as (optionally editable) local dependency in other poetry projects. Example: `simpsons` depend on `isml_train`. When simpsons is installed - isml_train is also installed along with other dependencies in selected venv. If editable -> symlink and code changes immediately reflected in venv, else copied in venv as it normally done by installs of python libs.

./use_cases - contains train pipelines for different datasets. As in the case of ./lib - each project is a separate poetry project => dependencies (and other python-package-related-settings) are specified in “pyproject.toml”. Dependencies are resolved by poetry on the first install and the full list of dependencies with their EXACT versions is saved in poetry.lock and is not modified anymore. If poetry.lock already exists - the dependency versions will be taken from there.

./tools - contains tools, also as poetry projects with corresponding benefits.

Inspired by:

https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa

https://drivendata.github.io/cookiecutter-data-science/

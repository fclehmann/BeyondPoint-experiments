# Experiment runer

- Runs experiment defined by experiment config.
- Exports HOpt experiment results

## Install

### Locally:

```shell
poetry install 
```

## Usage

In the following is assumed that experiment-runner is 
already installed and the virtual environment is activated
or you run the commands as `poetry run`.

To get help execute:

```shell
experiment-runner --help
```
```shell
experiment-runner run --help
```
```shell
experiment-runner export --help
```

### Important notes

#### Boolean options

Depending on convention, boolean options can be passed in form of either 
 - implicit format: --bool_opt for true value and absence for false value
 - explicit format: --bool_opt="True" or --bool_opt="False".

Experiment runnner supports both of them. This behavior is controlled using bool parameter 
`explicit_boolean_options` in section `[ENV_PARAMS.base_params]`. When set to `False` (default), all boolean options (ones, that have value "True" or "False") under `[FIXED_EXPERIMENT_PARAMS]` will be supplied to your program in implicit format.

## Experiment environment
- Each experiment is associated with a specific commit hash in one of 3 ways (based on `env_type`):
  - `env_type="singularity"`: Pre-built singularity image of use-case is used. Docker image definition: `docker-files/Dockerfile_use-case`. Images (versioned) are available in the private image registry in our gitlab.
- Each experiment corresponds to a specific {use-case}.
- Each experiment is defined by it's configuration file (for example
  `use_cases/simpsons/experiments/10-simpsons-hopt-tpe-SGDwlrs-seeds.toml`)

## Sweeper/launcher
Initially, we started with our in-house hyperparameter optimization framework Omniopt to sweep over the seeds. In first place, we had to make sure that the CLI interface of each use-case is compatible with the one expected by Omniopt - this is handled in each use case individually, as they also use different configuration management frameworks (`hydra`, `spock`).  Secondly, some automatization was necessary to install and launch Omniopt when running an experiment given the experiment config file - this is handled within `experiment_runner.run` module.

However, at a later point some modifications of the approach for scheduling our runs on SLURM were necessary. We have built our custom launcher module `experiment_runner.launcher` (actually scripts) based on `optuna` and `submitit`. We have used it to launch cifar10_clf experiments.

- We have used Omniopt based setup to launch simpsons and superconduct experiments.
- We have used `experiment_runner.launcher` based setup
to launch cifar10 experiments.

It should be possible to launch the cifar10 experiments using
`experiment_runner.launcher` on any cluster with SLURM batch system with the little effort (mainly replacing filesystem paths and hardware requirements). However, the launch of other experiments, which were executed by omniopt would require more efforts.

For further work, it would be beneficial to write a new clean scheduling module. Most of the work is already carried out in [submitit](https://github.com/facebookincubator/submitit) and [hydra](https://github.com/facebookresearch/hydra). On the other hand `experiment_runner.launcher` module provides a starting point.

## Running experiments
This section describes some aspects of running experiments. If you want to reproduce the experiments please refer to the reproducibility section of the top-level README.

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
│ --time             TEXT    : 12:00:00]                                                        │
│ --help                      Show this message and exit.                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
It will produce single `./data/processed/data.json` file within experiment folder.

### Experiment results

The results of the experiment should be stored in a workspace on cluster:
`/data/horse/ws/s4610340-isml/{use-case}/experiments/{experiment_name}`.
You just need to specify the conventional `experiments_path` setting in your experiment config and give the
experiment config file uniq name which meets the naming conventions (see below). The rest is done by experiment-runer.

Normally, an experiment produces the following data inside the experiment directory:
- `results/` - single train outputs (.json) and models (.pt).
- `omniopt/` - OmniOpt sources and HOpt data.

## Experiment definition
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

### Setting seeds
##### OmniOpt seed(s)
In section `[DATA]` parameter `seed`.

##### Use-case specific seeds
Use case specific seeds are controlled within `[FIXED_EXPERIMENT_PARAMS]`. The exact parameter name is case-specific.

## Notes on integration of use-case
Use-case cli must accept:
- `ExperimentConfig.out_path` option
(e.g `--ExperimentConfig.out_path="/beegfs/ws/0/s4610340-isml/lrl_sorb/experiments/001_lrl_sorb_hopt_tl_tpe_AdamW_seeds/results/"`).
This is the folder, where the train pipeline must create a unique subdirectory (for example using `uuid`) containing
all outputs of the individual run.

## Experiment configs naming convention
`{Experiment number}-{use-case}-{HOpt or Opt}-...any_other_meaningfull_description.toml`. Don't use
any special characters except "-" and "_".

## Glossary

### HOpt seed
Seed set in OmniOpt (<=> in hyperopt <=> in HOpt algorithm)

### HOpt iteration i
In the context of HOpt this is train and evaluation of model using some hyperparameters combination
performed by OmniOpt on i'th iteration.

### HOpt path
Sequence of (hopt iteration i, hyperparams, metric value) for specific seed. If there is n HOpt seeds
in experiment - there is n HOpt paths.

### Cumulative min of HOpt path
Cumulative min applied to HOpt path (more precise: applied to metric values).
https://stackoverflow.com/questions/23622157/calculating-cumulative-minimum-with-numpy-arrays

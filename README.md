# Beyond Point Estimates: Distributional Uncertainty in Machine Learning Performance Evaluation

This repository contains train pipelines, schedulers, and (up to some extent) data analyses for "Beyond Point Estimates: Distributional Uncertainty in Machine Learning Performance Evaluation".

## Implementation notes

### Folder structure
- `./use_cases` - contains use-case specific train pipeline, experiments, notebooks, etc.
  - `./use_cases/simpsons` - simpsons classifier (CNN, VGG11)
  - `./use_cases/superconduct_tc_reg` - critical temperature of superconductors regression (DNN, GBDT)
  - `./use_cases/cifar10_clf` - cifar10 classifier (VGG11)
- `./data` - raw and cleaned experiment outputs used for the analysis (see `data/README.md`)
- `./evaluation` - R-based analysis pipelines that compute and compare confidence intervals on the experiment data (see `evaluation/R_scripts/*/README.md`)
- `./lib` - reusable code components as installable libraries
- `./tools` - tools
   - `./tools/experiment_runner` - Deploy/run experiments on taurus. Process, archive and export of experiment results.

Each use-case is a python package. Important contents are:
- `src` - source code
- `experiments` - Experiment configs associated with this use case
- `configs` - Run-specific configs.
- `pyproject.toml`
- `poetry.lock`

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
  2. Build the docker image using `docker-files/Dockerfile_use-case` as described in [this section](#build-docker-image)

### 2. Level: Experiments / sweeps
It is possible to reproduce the distributions of the metrics given the experiment configs (`./use_cases/{use-case}/experiments`). We recommend to write your own software to execute the experiments on your compute resources. You can use
`experiment_runner.launcher` and `experiment_runner.run` as starting point. Please refer to `tools/experiment_runner/README.md` for more details.

## Build docker image
From repo root execute, substituting `simpsons` with necessary use-case:
1. Build docker image 
    ```bash
    DOCKER_BUILDKIT=1 docker build --build-arg USE_CASE=simpsons --secret id=gitlab,src=$HOME/.config/pypoetry/auth.toml -f ./docker-files/Dockerfile_use-case -t gcr.hrz.tu-chemnitz.de/clehm--tu-dresden.de/statistics-in-machine-learning/simpsons:$(git rev-parse --short HEAD) .
    ```

## Running train pipeline of use-case
Train pipelines are guaranteed to work for any commit hash specified in experiment config files (For example `use_cases/superconduct_tc_reg/experiments/015-superconduct_tc_reg-dnn-opt-split_cv.toml` specifies `singularity_image_file = "/data/horse/ws/s4610340-isml/sifs/superconduct_tc_reg_d14f389.sif"` which means that the image built from the commit with hash `d14f389` is used). This is particularly important for simpsons case as it is not compatible with the latest version of isml_train lib. Please refer to ./use_cases/{use-case}/README.md for guidelines on how to obtain the data and launch the train.

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


./lib - Needed to share code between projects. There may be many "libraries" under the ./lib. Each library is a separate poetry project. It's installable and can be specified as (optionally editable) local dependency in other poetry projects. Example: `simpsons` depend on `isml_train`. When simpsons is installed - isml_train is also installed along with other dependencies in selected venv. If editable -> symlink and code changes immediately reflected in venv, else copied in venv as it normally done by installs of python libs.

./use_cases - contains train pipelines for different datasets. As in the case of ./lib - each project is a separate poetry project => dependencies (and other python-package-related-settings) are specified in "pyproject.toml". Dependencies are resolved by poetry on the first install and the full list of dependencies with their EXACT versions is saved in poetry.lock and is not modified anymore. If poetry.lock already exists - the dependency versions will be taken from there.

./tools - contains tools, also as poetry projects with corresponding benefits.

Inspired by:

https://medium.com/opendoor-labs/our-python-monorepo-d34028f2b6fa

https://drivendata.github.io/cookiecutter-data-science/

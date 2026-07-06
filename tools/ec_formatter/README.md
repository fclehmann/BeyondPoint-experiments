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

### Run experiment

Execute from taurus:

```shell
experiment-runner run [OPTIONS] EXPERIMENT_CONFIG_PATH
```


### Export HOpt experiment results

Execute on taurus

```shell
experiment-runner export PATH_TO_EXPERIMENT
```

Export creates 3 csv files inside of PATH_TO_EXPERIMENT directory:
- `omniopt_results.csv` -  raw HOpt results
- `omniopt_results_min_all.csv` - all parameter constellations corresponding to minimum value of target metric.
- `omniopt_results_min_latest.csv` - parameter constellation corresponding to minimum value of target metric 
with biggest Hopt iteration number.

All information is contained in `omniopt_results.csv`. Other files are just processed `omniopt_results.csv`.

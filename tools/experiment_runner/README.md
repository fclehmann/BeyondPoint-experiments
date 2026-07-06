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
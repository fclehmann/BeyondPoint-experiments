# `data/` — Experiment data

Raw and cleaned data from HPO runs for the 3 use case.

## Files

| Task | Raw data (JSON) | Analysis data (Rdata) | Metric |
|---|---|---|---|
| CIFAR-10 classification | `cifar10_clf.json` | `cifar10_clf.Rdata` | `accuracy` |
| Simpsons classification | `simpsons.json` | `simpsons.Rdata` | `accuracy` |
| Superconductor regression | `superconduct_tc_reg.json` | `superconduct_tc_reg.Rdata` | `RMSE` |

## `.json` — Raw experiment data

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

One row per experiment with full provenance: hyperparameters (`params_example`, ~85 fields), varied HPs (`parameter_values`), all train/val/test metrics per seed (`metric_values`), seed list (`seeds`), and HPO configuration (`hopt`). Number of experiments: 6 (CIFAR-10), 18 (Simpsons), unknown (Superconductor).

## `.Rdata` — Cleaned data for analysis

Each file contains **a single object `df`** in long format with three columns:

| Column | Type | Content |
|---|---|---|
| `experiment` | factor | Experiment name |
| `seeds` | integer | Seed of the individual run |
| `accuracy` / `RMSE` | numeric | Test-set metric of that run |

Row counts: 5,500 (CIFAR-10), 22,228 (Simpsons), 12,260 (Superconductor) . The metric column name is task-dependent.

```r
load("data/cifar10_clf.Rdata")   # -> df
```

**These files are the primary input for evaluation done with the R scripts in the repo.**

## Known issues

- **`superconduct_tc_reg.json`** contains `NaN` literals and cannot be parsed directly with `jsonlite::fromJSON()` . Workaround: replace `NaN` with `null` before parsing.
- **`simpsons.json`** includes individual experiments with as few as 2 seeds (range `[2, 1000]`) [[11]] — filter before statistical analyses if needed.

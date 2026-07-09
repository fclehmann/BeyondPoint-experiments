# CI evaluation on empirical regression-error data

**Location:** `evaluation/R_scripts/ci-regression-evaluation/`

## TL;DR

Four confidence-interval methods (`t_mean`, `np_exact`, `np_asymptotic`,
`bootstrap`) for the mean and selected quantiles, evaluated on
regression-error data against an *empirical* ground truth from the
full reference dataset. Dataset-agnostic: any `.Rdata` with columns
`experiment` (factor) and a numeric error metric in `[0, ∞)` works.
Dataset selection is driven entirely by the `data_file:` entry in the
YAML — currently `superconduct_tc_reg.Rdata`.

Test run from `evaluation/`:

```bash
./run-regression-pipeline.sh \
    --config config/superconduct-evaluation-test-run.yaml \
    --output output
```

`--data` defaults to `<wrapper-dir>/../data` (i.e. `<repo>/data`) and
can be overridden explicitly if needed.

Expected output per run in `output/<config-basename>/final_results/`:

- `true_statistics.Rdata` — empirical ground truth per `df$experiment`
- `bootstrap_full_aggregated.Rdata` — bootstrap percentiles merged with truth
- eight intermediate `*_CI_df[_aggregate].Rdata` files, one per CI method
  (single-CI rows and their per-experiment aggregate)
- `overview_signTrafo.Rdata` — final merged overview across all methods,
  one row per (experiment × sample_size × num_samples × statistic ×
  confidence_level × CI_type), with `emp_conf_level`, `avg_length`,
  `avg_length_ratio` and `invalid` columns

Additionally, the workspace contains provenance copies of the YAML and
the reference `.Rdata`, the per-experiment subsample files, the raw
bootstrap distributions and the extracted bootstrap percentiles.

## CI methods

- **`t_mean`** — Student-*t* CI for the mean.
- **`np_exact`** — randomised exact non-parametric CI for quantiles.
- **`np_asymptotic`** — non-parametric CI for quantiles via asymptotic
  normality of order statistics (Van der Vaart, *Asymptotic Statistics*,
  Ex. 21.8, p. 309). Uses a sign transformation for lower quantiles
  (`p < 0.5`), signalled via `negate = TRUE` in `CI_np_asymptotic()`.
- **`bootstrap`** — percentile CI from a semi-parametric bootstrap
  distribution (Hutson, *Statist. Comput.* 12, 2002).

Companion pipelines:

- `evaluation/R_scripts/ci-validation-evaluation/` — same four
  calculations against a *theoretical* ground truth from parametric
  distributions.
- `evaluation/R_scripts/ci-classification-evaluation/` — same four
  calculations on classification-accuracy data (`[0, 1]`-clipping).

The regression scope of this pipeline is expressed through two
constraints in step 04:

- errors are expected in `[0, ∞)` (RMSE, MAE, MSE, …),
- negative CI bounds are clipped to `0` and counted as `invalid`;
  the upper bound is unbounded.

## Requirements

- **R ≥ 4.3** (step 04 uses the vectorised `|`).
- R packages: `boot`, `dplyr`, `magrittr`, `tidyr`, `tibble`, `stringr`,
  `foreach`, `doParallel`, `parallel`, `optparse`, `purrr`, `yaml`,
  `tidyverse`.

## Pipeline layout

```
evaluation/R_scripts/ci-regression-evaluation/
├── 01-draw-subsamples-and-bootstrap.R
├── 02-extract-bootstrap-percentiles.R
├── 03-merge-with-empirical-truth.R
├── 04-compute-and-aggregate-cis.R
└── README.md
```

The wrapper `run-regression-pipeline.sh` lives at `evaluation/`.
Shared libs in `evaluation/R_scripts/libs/`
(`lib-semiparam-quantile-est.R`, `lib-np-exact-CI-linopt.R`,
`lib-np-asymptotic-CI.R`) are resolved relative to each script's own
file location.

## Reference data schema

The reference `.Rdata` must contain a data frame named `df`:

```
$ experiment : Factor with one level per experimental setting
$ seeds      : integer (one row per (experiment, seed) pair)
$ <metric>   : numeric in [0, ∞), e.g. RMSE, MAE, MSE
```

Additional columns are ignored. The metric column name is
task-dependent (`RMSE` for the bundled Superconductor data);
downstream scripts address it positionally as `names(df)[3]`.
Levels of `df$experiment` must not contain `-` (parsed as delimiter
in filenames by step 02).

See `<repo>/data/README.md` for the concrete datasets bundled with
this repo.

## YAML configuration

Config files live in `evaluation/config/`. Minimal skeleton:

```yaml
global_settings:
  data_file: <reference-file>.Rdata      # e.g. superconduct_tc_reg.Rdata
  q_levels: [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
  q_type: 1
  seeds: 1
  boot_percentiles_levels: [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]

experiments:
  - id: 1
    sample_size: 10
    num_samples: 1000
    num_bootsamples: 1000
  # ... one block per (sample_size, num_samples, num_bootsamples)
```

Workspace and sub-folder names are derived automatically from
`data_file` and the experiment parameters. Switching between datasets
requires only changing the `data_file:` entry — the wrapper itself
is dataset-agnostic.

## Running manually (per step)

```bash
# step 01 — once per experiment id in the YAML
for i in $(seq 1 <NUM>); do
  Rscript evaluation/R_scripts/ci-regression-evaluation/01-draw-subsamples-and-bootstrap.R \
    --output_path         output \
    --config_file         evaluation/config/<yaml>.yaml \
    --configuration_index $i \
    --datafile_path       data
done

# step 02 — basename only for --config_file (reads provenance copy)
Rscript evaluation/R_scripts/ci-regression-evaluation/02-extract-bootstrap-percentiles.R \
  --data_parent_dir output/<config-basename> \
  --config_file     <yaml>.yaml

# step 03
Rscript evaluation/R_scripts/ci-regression-evaluation/03-merge-with-empirical-truth.R \
  --parent_dir output/<config-basename>

# step 04
Rscript evaluation/R_scripts/ci-regression-evaluation/04-compute-and-aggregate-cis.R \
  --datafile_path output/<config-basename>
```

Step 01 differs from the validation pipeline in taking an extra
`--datafile_path`.

## Reproducing the reference results

```bash
cd evaluation/
./run-regression-pipeline.sh \
    --config config/superconduct-evaluation.yaml \
    --output output
```

The results in `output/superconduct-evaluation/final_results/` are the
ones used in the associated publication.

## Caveats

- **R ≥ 4.3** required (vectorised `|` in step 04).
- **Filename → metadata coupling.** Step 02 parses experiment metadata
  from bootstrap filenames via `(?<=\\=)[^\\-]+`. Levels of
  `df$experiment` must not contain `-`.
- **Regression-specific clipping** to `[0, ∞)` in step 04 (lower bound
  clipped to `0`; upper bound unbounded).
- **Frozen reference.** Only compatibility fixes are made; no further
  development.


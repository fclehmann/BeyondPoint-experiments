# CI evaluation on empirical classification-accuracy data

**Location:** `R_scripts/ci-classification-evaluation/`

## TL;DR

Four confidence-interval methods (`t_mean`, `np_exact`, `np_asymptotic`,
`bootstrap`) for the mean and selected quantiles, evaluated on
classification-accuracy data against an *empirical* ground truth from
the full reference dataset. Dataset-agnostic: any `.Rdata` with columns
`experiment` (factor) and `accuracy` (numeric in `[0, 1]`) works.

Two test runs, both from the repository root:

```bash
# Simpsons test run
./run-classification-pipeline.sh \
    --config config/simpsons-evaluation-test-run.yaml \
    --output output \
    --data   data

# CIFAR-10 test run
./run-classification-pipeline.sh \
    --config config/cifar10-evaluation-test-run.yaml \
    --output output \
    --data   data
```

Expected output per run in
`output/<config-basename>/final_results/`:

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

Companion pipeline: `R_scripts/ci-validation-simulation/` performs the
same four calculations against a *theoretical* ground truth from
parametric distributions. A future `R_scripts/ci-regression-evaluation/`
is envisaged; the classification scope of this pipeline is expressed
through two constraints in step 04:

- accuracies are expected in `[0, 1]`,
- out-of-range CI bounds are clipped to `[0, 1]` and counted as `invalid`.

## Requirements

- **R ≥ 4.3** (step 04 uses the vectorised `|`).
- R packages: `boot`, `dplyr`, `magrittr`, `tidyr`, `tibble`, `stringr`,
  `foreach`, `doParallel`, `parallel`, `optparse`, `purrr`, `yaml`,
  `tidyverse`.

## Pipeline layout

```
R_scripts/ci-classification-evaluation/
├── 01-draw-subsamples-and-bootstrap.R
├── 02-extract-bootstrap-percentiles.R
├── 03-merge-with-empirical-truth.R
├── 04-compute-and-aggregate-cis.R
└── README.md
```

The wrapper `run-classification-pipeline.sh` lives at the repository
root. Shared libs in `R_scripts/libs/` (`lib-semiparam-quantile-est.R`,
`lib-np-exact-CI-linopt.R`, `lib-np-asymptotic-CI.R`) are resolved
relative to each script's own file location.

## Reference data schema

The reference `.Rdata` must contain a data frame named `df`:

```
$ experiment : Factor with one level per experimental setting
$ accuracy   : numeric in [0, 1]
```

Additional columns are ignored. Levels of `df$experiment` must not
contain `-` (parsed as delimiter in filenames by step 02).

## YAML configuration

Config files live in `config/`. Minimal skeleton:

```yaml
global_settings:
  data_file: <reference-file>.Rdata
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
requires only changing the `data_file:` entry.

## Running manually (per step)

```bash
# step 01 — once per experiment id in the YAML
for i in $(seq 1 <NUM>); do
  Rscript R_scripts/ci-classification-evaluation/01-draw-subsamples-and-bootstrap.R \
    --output_path         output \
    --config_file         config/<yaml>.yaml \
    --configuration_index $i \
    --datafile_path       data
done

# step 02  — basename only for --config_file (reads provenance copy)
Rscript R_scripts/ci-classification-evaluation/02-extract-bootstrap-percentiles.R \
  --data_parent_dir output/<config-basename> \
  --config_file     <yaml>.yaml

# step 03
Rscript R_scripts/ci-classification-evaluation/03-merge-with-empirical-truth.R \
  --parent_dir output/<config-basename>

# step 04
Rscript R_scripts/ci-classification-evaluation/04-compute-and-aggregate-cis.R \
  --datafile_path output/<config-basename>
```

Step 01 differs from the validation-simulation pipeline in taking an
extra `--datafile_path`.

## Reproducing the reference results

```bash
./run-classification-pipeline.sh \
    --config config/sp-bt-simpsons-evaluation.yaml \
    --output output \
    --data   data
```

The results in `output/sp-bt-simpsons-evaluation/final_results/` are
the ones used in the associated publication.

## Caveats

- **R ≥ 4.3** required (vectorised `|` in step 04).
- **Filename → metadata coupling.** Step 02 parses experiment metadata
  from bootstrap filenames via `(?<=\\=)[^\\-]+`. Levels of
  `df$experiment` must not contain `-`.
- **Classification-specific clipping** to `[0, 1]` in step 04.
- **Frozen reference.** Only compatibility fixes are made; no
  further development.

# CI evaluation on simulated data with theoretical ground truth

**Location:** `evaluation/R_scripts/ci-validation-evaluation/`

Frozen reference implementation of a validation study that compares
four confidence-interval methods for the mean and for quantiles
against theoretically known ground truth. The active codebase is
maintained in Python; this R version documents the original
methodology.

## TL;DR

Four confidence-interval methods (`t_mean`, `np_exact`,
`np_asymptotic`, `bootstrap`) for the mean and selected quantiles,
evaluated against a *theoretical* ground truth from parametric
distributions (normal, log-normal, exponential, Gumbel, beta,
uniform). Data is generated inside step 01 — no external `.Rdata`
input required.

Test run from `evaluation/`:

```bash
./run-validation-pipeline.sh \
    --config config/test-run.yaml \
    --output /tmp/uml-testrun
```

Expected output per run in
`/tmp/uml-testrun/<config-basename>/final_results/`:

- `true_statistics.Rdata` — theoretical ground truth per distribution
- `bootstrap_full_aggregated.Rdata` — bootstrap percentiles merged with truth
- eight intermediate `*_CI_df[_aggregate].Rdata` files, one per CI method
  (single-CI rows and their per-experiment aggregate)
- `overview_signTrafo.Rdata` — final merged overview across all methods,
  one row per (experiment × sample_size × num_samples × statistic ×
  confidence_level × CI_type), with `emp_conf_level`, `avg_length`,
  `avg_length_ratio` and `invalid` columns

Additionally, the workspace contains a provenance copy of the YAML,
the per-variation subsample files, the raw bootstrap distributions
and the extracted bootstrap percentiles.

For the published simulation results, replace the config with
`config/validation.yaml`.

## CI methods

- **`t_mean`** — Student-*t* CI for the mean.
- **`np_exact`** — randomised exact non-parametric CI for quantiles
  (LP-based over pairs of order statistics).
- **`np_asymptotic`** — non-parametric CI for quantiles via
  asymptotic normality of order statistics (Van der Vaart,
  *Asymptotic Statistics*, Ex. 21.8, p. 309). Uses a sign
  transformation for lower quantiles (`p < 0.5`), signalled via the
  `mirror` argument of `CI_np_asymptotic()`.
- **`bootstrap`** — percentile CI from a semi-parametric bootstrap
  distribution (Hutson, *Statist. Comput.* 12, 2002).

Companion pipelines:

- `evaluation/R_scripts/ci-classification-evaluation/` — same four
  calculations on classification-accuracy data (empirical ground
  truth, `[0, 1]`-clipping).
- `evaluation/R_scripts/ci-regression-evaluation/` — same four
  calculations on regression-error data (empirical ground truth,
  `[0, ∞)`-clipping).

## Requirements

- **R ≥ 4.0** (tested with 4.3.x).
- R packages: `jsonlite`, `boot`, `dplyr`, `magrittr`, `tidyr`,
  `tibble`, `stringr`, `purrr`, `tidyverse`, `KScorrect`, `evd`,
  `foreach`, `doParallel`, `parallel`, `optparse`, `yaml`, `lpSolve`.

  - `KScorrect` provides `rmixnorm` (normal-mixture, step 01)
  - `evd` provides `rgumbel` (Gumbel, step 01)
  - `lpSolve` is used by the exact non-parametric CI (step 04)

Parallelism uses all CPU cores minus one; on SLURM the worker count
comes from `SLURM_JOB_CPUS_PER_NODE`.

## Pipeline layout

```
evaluation/R_scripts/ci-validation-evaluation/
├── 01-draw-subsamples-and-bootstrap.R
├── 02-extract-bootstrap-percentiles.R
├── 03-merge-with-true-values.R
├── 04-compute-and-aggregate-cis.R
└── Readme.md
```

The wrapper `run-validation-pipeline.sh` lives at `evaluation/`.
Shared libs in `evaluation/R_scripts/libs/`
(`lib-semiparam-quantile-est.R`, `lib-np-asymptotic-CI.R`,
`lib-np-exact-CI-linopt.R`) are resolved relative to each script's
own file location.

## YAML configuration

Two YAML files are provided:

- **`config/validation.yaml`** — production configuration for the
  published results (12 variations,
  `num_samples`/`num_bootsamples` up to 5000).
- **`config/test-run.yaml`** — reduced configuration for local
  smoke tests. Not statistically meaningful.

Minimal skeleton:

```yaml
global_settings:
  q_levels:                [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]
  q_type:                  1
  seeds:                   1
  boot_percentiles_levels: [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]

distributions:
  - name: <experiment_name>
    type: <norm | beta | gumbel | mixnorm | ...>
    positive_support: <true | false>   # optional
    params: { ... }
    theo_mean: <numeric or R expression as string, e.g. "20/23">

variations:
  - id: <int>
    sample_size:     <int>
    num_samples:     <int>
    num_bootsamples: <int>
```

## Running manually (per step)

```bash
# step 01 — once per variation id in the YAML
for i in $(seq 1 <NUM>); do
  Rscript evaluation/R_scripts/ci-validation-evaluation/01-draw-subsamples-and-bootstrap.R \
    --output_path         <workspace> \
    --config_file         evaluation/config/<yaml>.yaml \
    --configuration_index $i
done

# step 02 — basename only for --config_file (reads provenance copy)
Rscript evaluation/R_scripts/ci-validation-evaluation/02-extract-bootstrap-percentiles.R \
  --data_parent_dir <workspace>/<config-basename> \
  --config_file     <yaml>.yaml

# step 03
Rscript evaluation/R_scripts/ci-validation-evaluation/03-merge-with-true-values.R \
  --parent_dir <workspace>/<config-basename>

# step 04
Rscript evaluation/R_scripts/ci-validation-evaluation/04-compute-and-aggregate-cis.R \
  --datafile_path <workspace>/<config-basename>
```

Unlike the classification and regression pipelines, step 01 does
**not** take a `--datafile_path`: data is generated inside step 01
from the parametric distributions defined in the YAML.

## Production run

```bash
cd evaluation/
./run-validation-pipeline.sh \
    --config config/validation.yaml \
    --output ~/ws/validation
```

For large configurations, step 01 is typically distributed as SLURM
array jobs (see `sbatch/`); steps 02–04 remain inexpensive enough to
run locally after the array has completed.

## Caveats

- **R ≥ 4.0** required.
- **Interval clipping.** CIs for bounded-support distributions (all
  simulation distributions except Gumbel) are clipped to `[0, 1]`;
  for Gumbel the upper bound is `Inf`. Gumbel-typed experiments are
  identified via `str_detect(experiment, "gumbel")`, and
  additionally via the hard-coded vector `real_valued` in the
  bootstrap branch. Pre-clipping violation counts are preserved in
  the `invalid` / `invalid_values` columns.
- **Randomised exact CI.** `CI_np_exact()` returns a distribution
  over index pairs `(lower, upper)` with weights `lambda`; step 04
  samples one pair per subsample via `sample(...)`. The inner
  randomisation is seeded (default `seed = 42`); the outer sampling
  in step 04 does not set a seed, so results may vary slightly
  between runs.
- **Filename → metadata coupling.** Step 02 parses experiment
  metadata from bootstrap filenames via `(?<=\\=)[^\\-]+`.
  Distribution names in the YAML must not contain `-`.
- **Frozen reference.** Only compatibility fixes are made; no
  further development.

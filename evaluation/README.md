# `evaluation/` — Confidence-interval analysis pipelines

R-based analysis pipelines that compute and compare four families of
confidence intervals (`t_mean`, `np_exact`, `np_asymptotic`,
`bootstrap`) on the experiment data in `../data/` and on simulated
data with theoretical ground truth.

## Layout

```
evaluation/
├── config/                              # YAML configs for evaluation runs
│   ├── cifar10-evaluation[-test-run].yaml
│   ├── simpsons-evaluation[-test-run].yaml
│   ├── superconductor-evaluation[-test-run].yaml
│   └── validation-evaluation[-test-run].yaml
├── R_scripts/
│   ├── ci-classification-evaluation/    # empirical ground truth, [0, 1]-clipping
│   ├── ci-regression-evaluation/        # empirical ground truth, [0, ∞)-clipping
│   ├── ci-validation-evaluation/        # theoretical ground truth (parametric)
│   └── libs/                            # implementations of the CI methods
│                                        # (except the Student-t interval, which
│                                        # uses base R): np_exact, np_asymptotic,
│                                        # and the semi-parametric bootstrap
├── run-classification-pipeline.sh
├── run-regression-pipeline.sh
└── run-validation-pipeline.sh
```

The three files under `R_scripts/libs/` implement the CI methods used
throughout all three pipelines:

- `lib-np-exact-CI-linopt.R` — randomised exact non-parametric CI for
  quantiles (LP-based over pairs of order statistics).
- `lib-np-asymptotic-CI.R` — non-parametric CI for quantiles via
  asymptotic normality of order statistics.
- `lib-semiparam-quantile-est.R` — semi-parametric bootstrap
  distribution used by the `bootstrap` CI.

The Student-*t* interval for the mean (`t_mean`) is not implemented
here; it uses base R's `t.test()` directly inside step 04 of each
pipeline.

## Three sibling pipelines

| Pipeline | Input | Ground truth | Wrapper |
|---|---|---|---|
| `R_scripts/ci-classification-evaluation/` | `../data/cifar10_clf.Rdata`, `../data/simpsons.Rdata` | empirical (full dataset) | `run-classification-pipeline.sh` |
| `R_scripts/ci-regression-evaluation/`     | `../data/superconduct_tc_reg.Rdata`                  | empirical (full dataset) | `run-regression-pipeline.sh` |
| `R_scripts/ci-validation-evaluation/`     | synthetic (generated in step 01)                     | theoretical (parametric) | `run-validation-pipeline.sh` |

All three follow the same four-step structure:

1. draw subsamples and compute bootstrap distributions
2. extract bootstrap percentiles
3. merge with ground truth (empirical or theoretical)
4. compute and aggregate CIs

The dataset for the two empirical pipelines is selected via the
`data_file:` entry in the YAML; `--data` defaults to `../data`.

## Typical invocation

```bash
cd evaluation/

./run-classification-pipeline.sh \
    --config config/simpsons-evaluation.yaml \
    --output output

./run-regression-pipeline.sh \
    --config config/superconductor-evaluation.yaml \
    --output output

./run-validation-pipeline.sh \
    --config config/validation-evaluation.yaml \
    --output output
```

Results per run land in `<output>/<config-basename>/final_results/`.

## Where to look next

For details on YAML schema, CI methods, required R packages, and known
caveats, see the per-pipeline READMEs:

- `R_scripts/ci-classification-evaluation/README.md`
- `R_scripts/ci-regression-evaluation/README.md`
- `R_scripts/ci-validation-evaluation/README.md`

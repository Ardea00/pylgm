# pyLGM

Configuration-driven latent Gaussian models for panel data, with one compiled
model representation designed to be shared across multiple inference engines.

The current pyLGM engine is exact Gaussian only. Schema-v1 `Pipeline` runs are
conditional on the fixed hyperparameters declared in configuration. Schema-v2
`Experiment` runs estimate configured hyperparameters independently within each
candidate/origin training fold and score the resulting plug-in empirical-Bayes
optima. Neither workflow integrates hyperparameter uncertainty. Other engines
described in the design are planned extensions rather than current functionality.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

See the [approved design](docs/superpowers/specs/2026-07-22-pylgm-design.md).

## Predictive model selection

Use the local `compare` command to evaluate configured candidates over
rolling-origin, leakage-safe forecast folds. Candidates are selected from
predictive metrics; `persistence` is always reported as an autoregressive
benchmark but is never selectable. Consequently, the selected LGM can underperform
persistence. Users must inspect the benchmark rows in `metrics.parquet` rather than
interpreting selection as evidence that an LGM beat the benchmark.

```bash
pylgm compare examples/predictive_selection/config.yaml \
  examples/predictive_selection/data.csv \
  --output comparison
```

The [general example](examples/predictive_selection/README.md) is committed and
domain-neutral. The [NIC-shaped configuration](examples/nic_backtest/README.md)
runs against a local Parquet source but does not redistribute any source data.
Version 0.2 supports empirical-Bayes hyperparameter fitting only. Prediction and
metric rows carry an explicit `evaluation_mode` value (`latest` or `vintage`).
Expected candidate/fold failures are exposed as immutable structured records in
`ComparisonResult.failures` and persisted with optimizer counts, numerical failure
history, and root-cause details in `failures.json`.

## Gaussian foundation example

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
```

The example combines fixed effects, an IID region effect, and an intrinsic RW1
time effect. Rows with missing responses are prediction targets. Version 0.1's
exact Gaussian engine conditions on the effect precisions and observation
standard deviation declared in configuration; it does not integrate their
uncertainty.

## Exact Gaussian reference limits

The exact Gaussian engine is a small/medium-model reference implementation. Its dense
posterior covariance requires O(p^2) memory and its dense factorization requires
O(p^3) work for latent dimension `p`. Conservative dimension and estimated-memory
preflights reject larger models before dense conversion. Advanced direct callers may
explicitly opt in with `fit_gaussian(model, allow_large_dense=True)`; `Pipeline`
deliberately remains safe by default. Schema-v2 experiments have the equivalent
strict configuration switch:

```yaml
inference:
  engine: exact_gaussian
  allow_large_dense: true
  hyperparameters: ...
```

The resolved value is recorded in experiment artifacts and is forwarded to every
preflight, optimizer evaluation, and prediction fit. It does not change or weaken
the 4,096-latent-dimension and 512 MiB default guards; it only provides an explicit
opt-in override for schema-v2 experiments.

## Atomic artifact publication

Run directories are published with operating-system no-replace primitives:
`renamex_np` on macOS, exported `renameat2` (or known-architecture syscall fallback)
on Linux, and `MoveFileExW` on Windows. Platforms without one of these atomic
primitives fail with an unsupported-operation error. pyLGM never weakens the
no-overwrite guarantee by falling back to a replacing rename.

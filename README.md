# pyLGM

General-purpose latent Gaussian models for Python, with Pandas and PySpark data
adapters.

Version 0.3's only stable inference engine is exact Gaussian. The public core
offers declarative `LGM` models, Gaussian likelihoods, fixed/IID/RW1/RW2
effects, and explicit hyperparameters and priors. The current runtime supports
Pandas input; the PySpark adapter described by the architecture is a planned
follow-up, not a 0.3 feature. Laplace, INLA, spatial, and HMC engines are also
out of scope for this release.

See the [approved Latte-parity and PySpark architecture](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md)
for the staged adapter and inference roadmap.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

See the [approved design](docs/superpowers/specs/2026-07-22-pylgm-design.md).

## Legacy forecasting utilities

The schema-v1 `Pipeline` and schema-v2 `Experiment` interfaces, including the
`fit` and `compare` commands, remain available for existing forecasting
workflows. `compare` evaluates configured candidates over rolling-origin,
leakage-safe forecast folds. `persistence` is reported as an autoregressive
benchmark but is never selectable, so inspect the benchmark rows in
`metrics.parquet` rather than interpreting selection as evidence that an LGM
beat the benchmark.

```bash
pylgm compare examples/predictive_selection/config.yaml \
  examples/predictive_selection/data.csv \
  --output comparison
```

The [general example](examples/predictive_selection/README.md) is committed and
domain-neutral. The [NIC-shaped configuration](examples/nic_backtest/README.md)
runs against a local Parquet source but does not redistribute any source data.
Legacy empirical-Bayes runs use plug-in optima rather than integrating
hyperparameter uncertainty. Prediction and metric rows carry an explicit
`evaluation_mode` value (`latest` or `vintage`). Expected candidate/fold failures
are exposed as immutable structured records in `ComparisonResult.failures` and
persisted with optimizer counts, numerical failure history, and root-cause
details in `failures.json`.

## Exact Gaussian example

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
```

The example combines fixed effects, an IID region effect, and an intrinsic RW1
time effect. Rows with missing responses are prediction targets. The exact
Gaussian engine conditions on declared effect precisions and observation
standard deviation; it does not integrate their uncertainty.

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

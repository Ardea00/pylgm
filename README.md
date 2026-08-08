# pyLGM

General-purpose latent Gaussian modeling foundations for Python.

Version 0.3 is a bounded foundation release: its only stable inference engine
is exact Gaussian, and its declarative runtime accepts Pandas data. It provides
`LGM`, Gaussian likelihoods, fixed/IID/RW1/RW2 effects, fit-row predictions,
posterior marginals, and linear combinations. Schema-v1 and schema-v2
forecasting interfaces remain available for compatibility.

See the [approved Latte-parity and PySpark architecture](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md)
for the explicitly labeled target state and staged roadmap. It is not the 0.3
runtime contract.

## Runnable declarative 0.3 example

The [general LGM example](examples/general_lgm/README.md) fits the same Pandas
data through both a Python declaration and the standalone YAML frontend:

```bash
PYTHONPATH=src python examples/general_lgm/run.py
```

The committed YAML includes the required numeric `sigma` and every structured
effect `precision`.

## 0.3 scope and explicit deferrals

The 0.3 exact engine is conditional on fixed hyperparameter values. A
declarative `Hyperparameter` contributes its validated `.initial` value during
compilation; its name, transform, and prior are not copied into the materialized
IR or evaluated by inference. `PCPrecision` and `GaussianPrior` therefore define
target-state prior vocabulary, not integrated 0.3 hyperparameter inference.

AR1 effects, `result.predict(new_data)`, PySpark input, parameterized IR
metadata, hyperparameter-aware inference, sparse production engines, Laplace,
INLA, spatial effects, and HMC are deferred to later slices. Predictions in 0.3
cover the rows supplied to `LGM.fit`; their arrays follow the caller's original
row order.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

See the [approved design](docs/superpowers/specs/2026-07-22-pylgm-design.md).

## Internal compiled IR policy

`pylgm.ir` remains importable for engine and advanced integration work, but it
is internal and unstable and is not covered by the top-level compatibility
policy. Code that previously constructed `CompiledLGM(..., sigma=value, ...)`
must migrate to
`CompiledLGM(..., likelihood=CompiledGaussian(value), ...)`, importing
`CompiledGaussian` from `pylgm.likelihoods`. The read-only `compiled.sigma`
accessor remains available for compiled Gaussian models.

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

## Release installation gate

A release must be verified from a clean environment with the declared build
backend available:

```bash
python -m pip install -e . --no-deps
pylgm --help
```

This repository currently has no CI workflow to extend cleanly. The offline
0.3 review environment also lacks `hatchling`, so a clean build or editable
installation cannot run there without downloading the build backend; that
environment uses `PYTHONPATH=src` only for tests, lint, and CLI smoke checks.
The clean install plus installed-CLI smoke remains a required release gate in a
provisioned, networked build environment.

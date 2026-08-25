# Internals and release policy

## 0.3 scope and explicit deferrals

The 0.3 exact and Laplace engines condition on plain numeric hyperparameter
values. A declarative `Hyperparameter` contributes its validated `.initial`
value during compilation and, since the ["Empirical Bayes"](empirical-bayes.md#empirical-bayes)
slice, is estimated by type-II ML rather than merely held fixed. Since the
same slice's MAP-II follow-up, a `Hyperparameter`'s `prior` — when declared —
is also consumed: its native-scale log density penalizes the marginal
likelihood, turning the estimate into a MAP-II point estimate. The prior is
still not copied into the materialized IR, and it is applied on the
hyperparameter's native scale with no Jacobian correction for `transform`.
`PCPrecision` and `GaussianPrior` drive both point estimation and, under
["INLA integration"](inla.md#inla-integration), the integrated (marginal) posterior.

Parameterized IR metadata, sparse production engines, and HMC are deferred to
later slices. The stationary `AR1` effect has shipped — see
["AR1 effect"](effects.md#ar1-effect) below. The spatial (CAR)
effect family is now complete for the dense reference regime: `Besag`
(intrinsic CAR/ICAR), `ProperCAR` (proper CAR, with its spatial-dependence
parameter `ρ` either fixed or estimated), and `BYM2` (the structured +
unstructured convolution, with its mixing parameter `φ` either fixed or
estimated) have all shipped (see ["Besag / intrinsic CAR (ICAR) spatial effect"](spatial-effects.md#besag-intrinsic-car-icar-spatial-effect),
["Proper CAR spatial effect"](spatial-effects.md#proper-car-spatial-effect),
["BYM2 spatial effect"](spatial-effects.md#bym2-spatial-effect), and
["Bounded hyperparameters"](empirical-bayes.md#bounded-hyperparameters) below). Remaining
spatial follow-ups (the augmented 2n representation, config-file spatial
effect types, graceful isolated-node handling, and sparse/large-graph
scaling) are tracked in the
[spatial roadmap](roadmap.md).
Optional PySpark input is now
supported as a data boundary (see below). Pandas fit-row predictions in 0.3
cover the rows supplied to `LGM.fit`, in the caller's original row order.
`result.predict(new_data)` extends this to out-of-sample rows by reusing the
fitted latent posterior (see ["Predicting new rows"](prediction.md#predicting-new-rows)
below); `new_data` itself must always be a Pandas DataFrame, including for
results fitted from a Spark DataFrame.

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

The [general example](https://github.com/Ardea00/pylgm/blob/main/examples/predictive_selection/README.md) is committed and
domain-neutral. The [NIC-shaped configuration](https://github.com/Ardea00/pylgm/blob/main/examples/nic_backtest/README.md)
runs against a local Parquet source but does not redistribute any source data.
Legacy empirical-Bayes runs use plug-in optima rather than integrating
hyperparameter uncertainty. Prediction and metric rows carry an explicit
`evaluation_mode` value (`latest` or `vintage`). Expected candidate/fold failures
are exposed as immutable structured records in `ComparisonResult.failures` and
persisted with optimizer counts, numerical failure history, and root-cause
details in `failures.json`.

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

CI runs this as the `release-gate` job in `.github/workflows/test.yml`: a
non-editable `python -m pip install .` (exercising the real hatchling build)
followed by `pylgm --help`. The `core` job runs the test suite and lint across
`{ubuntu, windows, macos} × {3.11, 3.12, 3.13}`. An offline environment lacking
`hatchling` can substitute `PYTHONPATH=src` for tests, lint, and CLI smoke
checks, but the clean install plus installed-CLI smoke remains the required
release gate in a provisioned, networked build environment.

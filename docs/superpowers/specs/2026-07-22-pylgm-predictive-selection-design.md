# pyLGM Predictive Selection Design

**Date:** 2026-07-22  
**Status:** approved design, pending implementation plan

## 1. Goal

Add configuration-driven candidate comparison, fold-local empirical-Bayes
hyperparameter estimation, and leakage-safe rolling backtesting to the existing
Gaussian foundation. The implementation remains domain-neutral; Italian inflation
is a reference dataset, not core-library behavior.

This milestone implements one hyperparameter strategy, `empirical_bayes`. The
internal boundary must permit later grid or nested-backtest strategies without
adding plugin loading, registries, or other speculative infrastructure.

## 2. Scope

Implement:

- `CompiledGaussianFamily`, separating compiled structure from numerical
  hyperparameter values;
- bounded empirical-Bayes optimization of observation scale and selected effect
  precisions on the log scale;
- declared model candidates with resolved configurations;
- rolling-origin evaluation for arbitrary configured positive horizons;
- expanding and bounded rolling training windows;
- latest-data evaluation and optional vintage-aware filtering;
- Gaussian predictive log score, interval coverage, RMSE, MAE, and bias;
- hierarchical candidate selection;
- a persistence benchmark evaluated on identical folds;
- fold-level diagnostics, predictions, metrics, failures, fingerprints, and final
  selection artifacts;
- a NIC regional-inflation reference backtest.

Do not implement grid search, nested-backtest optimization, automatic candidate
generation, NUTS, Laplace, INLA, spatial effects, Spark adapters, or dynamic plugin
discovery in this milestone.

## 3. Architecture

The milestone adds four focused components:

1. `CompiledGaussianFamily` stores observations, design, constraints, labels,
   unit/base block precisions, and hyperparameter bindings. It materializes an
   immutable `CompiledLGM` without rebuilding formulas or design matrices.
2. `EmpiricalBayesOptimizer` maximizes the exact Gaussian conditional log marginal
   likelihood over bounded log-transformed hyperparameters.
3. `RollingBacktester` creates leakage-safe folds and evaluates LGM candidates and
   the persistence benchmark on identical targets.
4. `Experiment` resolves configuration, coordinates candidates, selects the winner,
   and writes artifacts. It contains orchestration only.

Inference remains independent of YAML, Pandas, Spark, backtesting, and artifacts.
The optimizer consumes a small numerical hyperparameter problem rather than a
pipeline object.

## 4. Compiled Gaussian family

Compilation happens once per candidate and fold. The family stores each block's
unscaled precision contribution and the mapping from a configured parameter to
that contribution. `materialize(theta)` combines scaled contributions and the
observation scale into the existing `CompiledLGM` contract.

The optimizer may vary:

- `sigma`; and
- explicitly selected non-fixed effect precisions.

Fixed-effect prior precision remains fixed unless a later design extends the
supported parameter set. Bounds and initial values are mandatory, finite, positive,
and transformed to log space internally.

## 5. Empirical-Bayes optimization

The optimizer is deterministic and uses a bounded SciPy optimizer. Every objective
evaluation materializes a `CompiledLGM` and calls the exact Gaussian engine.
Repeated parameter vectors are cached within one fit.

Each fold is optimized only with observations available at its origin. The optimum
from the previous fold may be used as the next initial point because it contains no
future information. The cold-start value remains recorded.

Numerically invalid evaluations are recorded. They may receive an explicit invalid
objective value so the bounded optimizer can continue, but the candidate fails if
no valid optimum is obtained. There is no fallback to initial hyperparameters.

Diagnostics include convergence state, objective, evaluation count, elapsed time,
initial and final values, active bounds, and numerical failures.

## 6. Candidates and configuration

Configuration schema version 2 adds `candidates`, `inference.hyperparameters`, and
`evaluation`. Version 1 remains supported explicitly for the existing single-fit
pipeline.

Candidates inherit the base model and may override schema-approved model fields.
List overrides replace the complete list. Before execution, every candidate is
expanded into and persisted as a complete resolved model.

Only `strategy: empirical_bayes` is accepted in this release. Unknown strategies
fail configuration validation; no strategy is selected implicitly.

The initial public entry point is:

```python
experiment = Experiment.from_yaml("config.yaml")
comparison = experiment.compare(frame, output_dir)
winner = comparison.selected
```

## 7. Rolling evaluation

`RollingBacktester` operates on ordered time levels rather than assuming calendar
months. Configured horizons are arbitrary positive integers representing positions
in that ordered sequence. Origins may be explicit or select the last `n` eligible
levels. Training windows are expanding or rolling with a declared length.

For each origin and horizon, the fold records the training cutoff and exact target
time. Rows through the target may be supplied to construct latent coordinates and
approved covariates, but every response after the origin is masked. No row later
than the target is supplied to compilation or optimization.

Evaluation modes:

- `latest`: uses the latest values in the supplied panel;
- `vintage`: requires a configured vintage/as-of column and keeps only values whose
  vintage was available at the origin.

Every formula covariate has a declared availability policy: `known_future`,
`observed_with_lag`, or `unavailable_future`. A fold fails before fitting if a target
uses a covariate unavailable at that origin. Prediction outputs retain panel keys,
origin, target time, horizon, candidate, engine, mean, variance, and interval bounds.

## 8. Metrics and selection

Metrics are computed per fold, horizon, and candidate, then aggregated without
discarding the underlying predictions.

Selection is hierarchical:

1. exclude candidates with configured numerical-stability or calibration failures;
2. rank eligible candidates by mean Gaussian log predictive density;
3. use RMSE as the deterministic tie-break;
4. report MAE, bias, interval coverage, and sharpness as diagnostics.

This avoids arbitrary composite-score weights. Thresholds and tie tolerances are
resolved in configuration and persisted.

Candidate failures are isolated and recorded. An experiment fails only when no
candidate is eligible.

## 9. Benchmark

The milestone includes a persistence forecaster: each target prediction equals the
most recent response available at the origin for the same panel key. It uses the same origins,
horizons, target rows, and metrics as LGM candidates. It is not optimized and cannot
be selected as the final LGM model unless a later design explicitly permits that.

No general external-model plugin API is added now.

## 10. Artifacts

The existing transactional artifact writer is extended with versioned outputs for:

- fully resolved candidate configurations;
- fold definitions and data fingerprints;
- optimized hyperparameters and diagnostics;
- coordinate-preserving predictions;
- fold, horizon, and aggregate metrics;
- candidate failures and eligibility decisions;
- final ranking and selection rationale.

Tabular outputs use Parquet; small manifests and summaries use JSON. Publication
remains atomic and no partial experiment is presented as complete.

## 11. Errors and limits

Add typed configuration, fold-construction, covariate-availability, optimization,
and selection errors. Expected failure of one candidate is data in the comparison
result, not an exception that aborts other candidates.

Existing dense exact-Gaussian limits apply to every optimizer evaluation. The
experiment estimates maximum candidate/fold work before execution and rejects
unsafe runs unless the existing explicit dense override is supplied.

## 12. Verification

Tests cover:

- analytical Gaussian hyperparameter optima and materialization equivalence;
- deterministic bounds, caching, convergence diagnostics, and invalid objectives;
- no future-response or future-vintage leakage;
- arbitrary horizons and expanding/rolling windows;
- covariate availability enforcement;
- candidate resolution, failure isolation, eligibility, ranking, and RMSE tie-break;
- persistence alignment on identical targets;
- transactional, reproducible experiment artifacts;
- a compact NIC rolling backtest compared with persistence;
- unchanged Gaussian-foundation tests and layer-import boundaries.

The exact engine remains the numerical oracle. Optimized models must reproduce the
same result when rematerialized at the recorded optimum.

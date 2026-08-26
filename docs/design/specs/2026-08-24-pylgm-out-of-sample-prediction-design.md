# pyLGM Out-of-Sample Prediction Design

**Status:** Approved 2026-08-24

## Purpose

Add `result.predict(new_data)`: score rows that were not supplied to `LGM.fit`,
using the fitted latent posterior. Today predictions cover only the rows passed
to `fit`, so any new covariate setting requires a refit. This closes the last
major gap for applied forecasting and scenario analysis.

## What `predict` is, and what it is not

`predict` **reuses the fitted latent posterior**. It rebuilds the design for the
new rows against the *existing* latent domain and propagates the posterior
through it. It therefore **cannot create new latent columns**: a time point or
region absent from the fit has no posterior to score against.

Forecasting genuinely new index levels is a different operation — the correct
one is a joint fit — and **it already works**: a row whose response is `NaN`
contributes no likelihood but still receives a latent column and a full
posterior prediction. Verified on an RW1 model: the forecast extrapolates flat
from the last level with variance growing by exactly `1/τ` per step.

```python
# forecasting future periods: include them at fit time with a NaN response
frame = pd.concat([history, future_rows_with_nan_response])
result = model.fit(frame)
result.predictive_mean      # covers history AND the future rows
```

The two are complementary and the documentation must say so plainly:

| need | use |
| --- | --- |
| new covariate values / scenarios on **known** index levels | `result.predict(new_data)` |
| **new** time points, regions, or groups | NaN-response rows at `fit` time |

An unseen index level in `predict` is therefore an **error**, not a silent
fallback, and the message names the offending levels and points at the NaN-row
workflow.

## Components

### 1. `Prediction` (`inference/prediction.py`, new)

An immutable result mirroring the existing prediction vocabulary:

- `predictive_mean` — posterior mean of the linear predictor `η`.
- `predictive_variance` — posterior variance of `η`, following **each engine's
  existing convention** so new-row and fit-row outputs are comparable:
  the exact-Gaussian path adds the observation variance `σ²`, the Laplace path
  does not (this mirrors `GaussianResult` vs `LaplaceResult` today).
- `fitted_mean` — response-scale prediction, with **exactly** the semantics
  `fitted_mean` already has: identity for Gaussian; the exact lognormal
  expectation `exp(μ + σ²_η/2)` for the Poisson log link; and for the Bernoulli
  logit link the documented **point estimate** `logit⁻¹(μ)`, which ignores the
  linear-predictor variance.
- `keys` — the new rows' index, preserved for alignment.
- `.to_frame()` — a `DataFrame` indexed like `new_data` with columns
  `predictive_mean`, `predictive_sd`, `fitted_mean`.

Read-only arrays, matching the other result types.

### 2. `PredictionContext` (`inference/prediction.py`)

The information needed to rebuild a design that the result does not already
carry. Assembled at compile time and attached to the result.

- `fixed_specs: tuple[formulaic ModelSpec, ...]` — one per `Fixed` effect.
  **Essential**: re-running `model_matrix(formula, new_data)` can silently
  produce a *different* set of columns when a categorical's observed levels
  differ between fit and new data. `model_spec.get_model_matrix(new_data)`
  reproduces the fitted encoding exactly.
- `structured: tuple[tuple[str, str, tuple[str, ...]], ...]` — per non-fixed
  block, `(block name, index column, fitted labels)`, in fitted block order.
- `likelihood` — the compiled likelihood, for the response scale.
- `offset: str | None` — the offset column name.
- `width: int` — the fitted latent width, asserted against the rebuilt design.

The posterior `mean`, `covariance`, and `block_slices` already live on the
result and are not duplicated here.

### 3. `predict` on the result types

`GaussianResult.predict`, `LaplaceResult.predict`, and `INLAResult.predict`
share one implementation (`predict_from(context, mean, covariance, new_data)`).
`INLAResult` supplies its **integrated** mean and covariance, so predictions are
the grid-mixture posterior with no extra machinery.

Blocks are rebuilt in fitted order and horizontally stacked:

- `Fixed` → `model_spec.get_model_matrix(new_data)`.
- structured → a one-hot matrix mapping each row's index value onto the block's
  fitted labels (string-compared, matching how the builders key their domains).

Then, with `X` the `n_new × p` design, `μ` the latent mean and `Σ` its
covariance:

  `η = X μ + offset`,  `Var(η) = rowwise diag(X Σ Xᵀ)`

computed as `einsum("ij,jk,ik->i", X, Σ, X)` — never forming the `n_new × n_new`
matrix.

## Public contract

```python
result = model.fit(frame)
prediction = result.predict(new_data)

prediction.predictive_mean        # eta posterior mean, one per new row
prediction.predictive_variance
prediction.fitted_mean            # response scale
prediction.to_frame()             # DataFrame aligned to new_data
```

- `new_data` is a pandas `DataFrame`. It must contain every column the model
  reads: each effect's `index`, every variable in each `Fixed` formula, and the
  `offset` column when one is declared. It must **not** need the response.
- Every index value must be a level the model was fitted with. Spatial effects
  are keyed on **graph nodes**, so a region present in the graph but absent from
  the fit data is already fitted and predicts fine.
- Works for all three result types and both engines.

## Scope

### Included
- `Prediction` and `PredictionContext` types; `predict_from` shared helper.
- `predict()` on `GaussianResult`, `LaplaceResult`, `INLAResult`.
- Retention of each `Fixed` effect's formulaic `ModelSpec` at compile time
  (`build_fixed` currently discards it) and assembly of the context.
- Threading the context through `compile_lgm` / `compile_family` /
  `materialize` onto the fitted result, including `_rebuild_result`.
- Docs: the `predict` contract, the fit-time NaN-row forecasting workflow, and
  the table above distinguishing them.

### Excluded (deferred / roadmap)
- **Spark `new_data`** — the canonicalizer requires a response column, so
  reusing it is not free; pandas only this slice.
- **Prior fallback for unseen levels** — errors instead, per the contract above.
- Predictive **quantiles / simulation** and full predictive distributions
  beyond mean and variance.
- Response-scale variance for non-Gaussian links (the Bernoulli point-estimate
  limitation is inherited, not fixed here).
- Automatic construction of future rows (a `make_future_frame` helper).

## Architecture
- `inference/prediction.py` (new): `Prediction`, `PredictionContext`,
  `predict_from`.
- `effects/fixed.py`: `build_fixed` also returns/retains the `ModelSpec`
  (via a small carrier or a parallel accessor) without changing `LatentBlock`.
- `compiler.py`: collect the fixed specs and structured `(name, index, labels)`
  while building blocks; construct the `PredictionContext`; attach it to the
  compiled model so the engines can pass it to the result.
- `ir/model.py` / `ir/family.py`: carry the context alongside the compiled model
  (duck-typed/optional, defaulting to `None`, so nothing existing breaks).
- `inference/result.py`: `predict()` on the three result types; the context is
  an optional field, and `_rebuild_result` preserves it.
- No engine math changes.

## Errors
- Unseen index level: `ValueError` naming the block, the offending levels, and
  directing to the NaN-response workflow.
- `new_data` missing an index / offset / formula column: `ValueError` naming it
  (formulaic's own error is wrapped for the formula case).
- `new_data` empty: `ValueError`.
- Rebuilt design width ≠ fitted latent width: `ValueError` (an internal
  consistency guard).
- `predict` on a result with no context (e.g. a hand-constructed result):
  `ValueError` explaining that the result was not produced by `LGM.fit`.
- Non-finite prediction: `NumericalError`.

## Testing and validation
1. **Self-consistency oracle (the strongest check):** predicting the *fit rows
   themselves* reproduces `result.predictive_mean` and `predictive_variance`
   exactly (to floating-point tolerance) — for Gaussian, Poisson, and Bernoulli,
   and for all three result types.
2. **Categorical trap:** a model with a categorical `Fixed` term, predicted on
   new data containing only a *subset* of the fitted levels, produces the fitted
   encoding (asserted against the retained `ModelSpec`) — this fails if the
   implementation re-runs `model_matrix` on the new frame.
3. **Structured mapping:** new rows in a different order / with repeated levels
   map to the correct latent columns; spatial effects predict for graph nodes
   absent from the fit data.
4. **Unseen level:** raises, naming the level and the block, mentioning the
   NaN-row workflow.
5. **Missing column / empty frame / no context:** the errors above.
6. **Offset:** a model with an offset uses the new rows' offset values.
7. **Response scale:** `fitted_mean` matches the fit-row `fitted_mean` semantics
   for Poisson (lognormal) and Bernoulli (point estimate).
8. **INLA:** predictions use the integrated posterior; a fit-row round trip
   matches `INLAResult.predictive_mean`.
9. **`to_frame()`:** aligned to `new_data`'s index, expected columns, `sd` is
   `sqrt(variance)`.
10. **Regression:** existing results, predictions, and `_rebuild_result`
    (Pandas and Spark fit paths) unchanged; full suite green.

## Acceptance criteria
1. `result.predict(new_data)` returns a `Prediction` for all three result types
   and both engines, using the fitted latent posterior.
2. Predicting the fit rows reproduces the fit-row `predictive_mean` and
   `predictive_variance` exactly, and `fitted_mean` exactly for the identity
   link and for all plug-in / empirical-Bayes fits. For an **integrated** fit
   with a non-linear link, `fitted_mean` is a documented moment-matched
   approximation: `integrate_inla` mixes the transformed per-θ values while
   `predict` transforms the integrated moments, and Jensen's inequality
   separates them by an amount that scales with the hyperparameter
   uncertainty. Reproducing it exactly would require retaining every grid
   point's latent covariance.
3. The fitted categorical encoding is reproduced on new data via the retained
   `ModelSpec`.
4. Unseen index levels raise a directed error pointing to the NaN-response
   workflow, which is documented as the supported way to forecast new levels.
5. `.to_frame()` returns a DataFrame aligned to `new_data`.
6. No new runtime dependency; no engine math changes; full suite green.
7. Spark `new_data`, prior fallback, and predictive quantiles remain deferred
   and recorded.

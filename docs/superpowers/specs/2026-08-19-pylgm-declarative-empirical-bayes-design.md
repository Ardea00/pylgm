# pyLGM Declarative Empirical-Bayes Design

**Status:** Approved 2026-08-19

## Purpose

Make a declared `Hyperparameter` actually get *estimated* in the declarative
`LGM.fit`, by type-II maximum-likelihood (empirical Bayes) over the marginal
likelihood, for **both** the exact-Gaussian and Laplace engines. Today the
declarative compiler plugs in each `Hyperparameter.initial` and discards the
rest of its metadata; this slice makes the metadata live.

It reuses the existing `optimize_empirical_bayes` optimizer (already
engine-parameterized through `fit=`) and the existing latent-block / family /
IR machinery. It adds no new inference engine and no distributed optimization.

## Scope

### Included

- Optional `lower`/`upper` bounds on `Hyperparameter`, with scale-aware defaults.
- A likelihood-agnostic `CompiledFamily` whose `materialize(params) -> CompiledLGM`
  scales the named block precisions and builds the model's likelihood via a
  likelihood factory. Used by the declarative path for Gaussian **and**
  non-Gaussian likelihoods.
- Generalizing `optimize_empirical_bayes` to accept any family exposing
  `parameter_names` + `materialize` (a structural protocol), so it drives
  `CompiledFamily` with `fit=fit_gaussian` or `fit=fit_laplace`.
- `LGM.fit` automatically running empirical Bayes when the model declares any
  `Hyperparameter` (for `sigma` or an effect `precision`); otherwise the current
  fixed-plug-in fast path runs unchanged.
- Exposing the optimized point estimates on the returned posterior result via a
  new read-only `hyperparameters: Mapping[str, float] | None` field on
  `GaussianResult` and `LaplaceResult`, plus optimizer diagnostics.
- Docs and a runnable example.

### Excluded (explicit later slices)

- **Penalized MAP-II** — adding each `Hyperparameter` prior's `logpdf` to the
  objective (PC-prior regularization). Recorded on the roadmap; this slice is
  pure type-II ML and the `prior` stays declaration metadata (consumed by the
  future INLA slice).
- Hyperparameter *marginals* / posterior uncertainty (INLA).
- Deriving bounds from priors (explicit bounds only).
- **YAML empirical Bayes** — the YAML frontend stays numeric plug-in only; EB is
  Python-API only this slice.
- Unifying `GaussianResult`/`LaplaceResult` onto a shared base (the new
  `hyperparameters` field is another shared-base candidate) — deferred with the
  result-unification slice.
- Any change to the legacy schema-v2 `Experiment`/`CompiledGaussianFamily` EB
  path, which already performs empirical Bayes.

## Public Contract

```python
from pylgm import LGM, Gaussian, Poisson, Fixed, IID, RW1, Hyperparameter, PCPrecision

model = LGM(
    response="count",
    likelihood=Poisson(),
    predictor=Fixed("1 + x")
    + IID("region_effect", index="region",
          precision=Hyperparameter("region_precision", initial=1.0,
                                   prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="laplace")

result.hyperparameters          # {"region_precision": <optimized value>}  (else None)
result.predictive_mean          # posterior-predictive at the optimum
result.diagnostics["empirical_bayes_converged"]   # optimizer convergence
```

- **Automatic trigger.** If any `sigma` or effect `precision` is a
  `Hyperparameter`, `LGM.fit` estimates exactly those; plain floats stay fixed.
  A model with no `Hyperparameter` produces a result identical to today
  (fast path unchanged, `result.hyperparameters is None`).
- **Return type unchanged.** `LGM.fit` still returns `GaussianResult` /
  `LaplaceResult`; the estimates are attached, not returned as a different type.
- The Pandas caller-order and Spark canonical-key contracts are preserved for
  the posterior arrays exactly as in the non-EB path.

## Architecture

### Hyperparameter bounds (`parameters.py`)

Add optional `lower: float | None = None`, `upper: float | None = None`. Defaults
are scale-aware and keep `initial` centered in log space:
`lower = initial * 1e-3`, `upper = initial * 1e3` when not supplied. Validation:
each supplied bound is a finite positive real, and `0 < lower < initial < upper`
(a resolved `lower`/`upper` must bracket `initial`). These map directly to the
optimizer's `OptimizationBounds(initial, lower, upper)`.

### Likelihood-agnostic family (`ir/family.py`)

Extract the precision-scaling loop from `CompiledGaussianFamily.materialize`
into a shared module-level helper
`_materialize_blocks(blocks: tuple[ScalableBlock, ...], resolved: Mapping[str, float]) -> tuple[LatentBlock, ...]`
and refactor `CompiledGaussianFamily.materialize` to call it (behavior
identical; its 14 tests are the guard). This removes the duplication the new
family would otherwise introduce.

Add `CompiledFamily` (frozen, `init=False`):

- Fields: `_y`, `_observed`, `_offset`, `blocks: tuple[ScalableBlock, ...]`,
  `parameter_names: tuple[str, ...]`,
  `likelihood_factory: Callable[[Mapping[str, float]], object]`.
- `materialize(values) -> CompiledLGM`: validates `values` against
  `parameter_names` (reuse `_validate_parameter_mapping`), scales blocks via
  `_materialize_blocks`, builds the likelihood with
  `self.likelihood_factory(resolved)`, and assembles via the shared
  `_assemble_compiled_model`.
- Validation mirrors `CompiledGaussianFamily`'s array/block checks, but the
  parameter naming is caller-chosen (no `{block.name}.precision` convention):
  every bound block's `parameter` must appear in `parameter_names`, names are
  unique non-empty strings, and block designs align with the response.

`ScalableBlock`, `_assemble_compiled_model`, and `_validate_parameter_mapping`
are reused unchanged. The likelihood factory is the only likelihood-specific
element, so a future likelihood needs only a factory — no family change.

### Optimizer generalization (`optimization/empirical_bayes.py`)

Relax `_validate_problem`'s `isinstance(family, CompiledGaussianFamily)` to a
structural check: the family must expose a `parameter_names` tuple and a callable
`materialize`. Everything else in `optimize_empirical_bayes` already depends only
on `family.materialize(params)` and `result.log_marginal_likelihood`, so it drives
`CompiledFamily` with `fit=fit_laplace` (non-Gaussian) or the default
`fit=fit_gaussian`. Generalize `EmpiricalBayesResult.fit`'s type annotation to
`GaussianResult | LaplaceResult` (its `to_dict` reads only shared attributes).

### Declarative family building + `LGM.fit` dispatch (`compiler.py`, `model.py`)

`compiler.py` gains `compile_family(model, panel) -> CompiledFamily | None`,
returning `None` when the model declares no `Hyperparameter` (signal to use the
fast path). When hyperparameters are present it:

- Builds each latent block. For an effect whose `precision` is a `Hyperparameter`,
  build the block at **unit precision** (`build_iid`/`build_random_walk` called
  with `precision=1.0`, giving the base `eye` / `Dᵀ D` operator) and wrap it as
  `ScalableBlock(block, hyperparameter.name, scale=1.0)` so the resolved value
  is the precision. For a fixed-float precision, build at that value and wrap as
  `ScalableBlock(block, None, 1.0)`. `Fixed` effects are always unbound.
- Assembles `parameter_names` = the declared hyperparameter names (block
  precisions, plus `sigma`'s hyperparameter name when the Gaussian `sigma` is a
  `Hyperparameter`).
- Builds `likelihood_factory`: Gaussian with a `Hyperparameter` sigma →
  `lambda r: CompiledGaussian(r[sigma_name])`; Gaussian with fixed sigma →
  `lambda r: CompiledGaussian(fixed_sigma)`; Poisson/Bernoulli →
  `lambda r: model.likelihood.materialize({})`.

`LGM.fit` (both `_fit_pandas` and `_fit_spark`): after building the canonical
panel, call `compile_family`. If it returns `None`, run the existing fast path
unchanged. Otherwise build `bounds` (`OptimizationBounds` per declared
hyperparameter) and `initial` (each `.initial`), call
`optimize_empirical_bayes(family, bounds, initial=initials, fit=engine_fit,
allow_large_dense=...)`, take `result.fit` (the posterior at the optimum) and
`result.parameters` (optimized estimates keyed by hyperparameter name), attach
the estimates and optimizer diagnostics, and apply the same caller-order
realignment (Pandas) or canonical `prediction_keys` (Spark) as the non-EB path.
The engine↔likelihood dispatch and mismatch errors are unchanged; `engine_fit`
is `fit_gaussian` for a Gaussian likelihood and `fit_laplace` for Poisson/Bernoulli.

### Result surface (`inference/result.py`)

Add `hyperparameters: Mapping[str, float] | None = None` to both `GaussianResult`
and `LaplaceResult`: validated as a string-keyed float mapping when supplied,
stored as an immutable `MappingProxyType`, returned as a fresh read-only mapping,
default `None`. The prediction-alignment helper carries it through unchanged.
`hyperparameter_marginals()` still returns empty — EB yields point estimates,
not marginals. Optimizer summary (converged, evaluations, marginal likelihood at
the optimum) is added to `diagnostics` as scalar entries.

## Errors

- `Hyperparameter` bounds not bracketing `initial`, or non-positive: `ValueError`
  at construction.
- Optimizer non-convergence / all-invalid objective: the existing
  `OptimizationError` from `optimize_empirical_bayes`, carrying evaluation counts
  and numerical-failure history, propagates out of `LGM.fit` unchanged.
- Engine↔likelihood mismatch: the existing `UnsupportedEngineError` (unchanged).
- A `Hyperparameter` on an unsupported target (only `sigma` and effect
  `precision` are supported this slice) is a `CompilationError`.

No failed optimization is returned as a successful result.

## Testing and Validation

1. **Bounds.** `Hyperparameter` default and explicit bounds; rejection of
   `lower >= initial`, `upper <= initial`, non-positive.
2. **Family parity.** `CompiledFamily` (Gaussian factory) materialized at given
   parameters yields a `CompiledLGM` equal (design/precision/constraints/labels
   and `likelihood.sigma`) to the same model via `CompiledGaussianFamily`; the
   extracted `_materialize_blocks` keeps `CompiledGaussianFamily` behavior
   identical (existing 14 tests pass).
3. **Optimizer generalization.** `optimize_empirical_bayes` over a
   `CompiledFamily` with a Gaussian factory reproduces the estimate and marginal
   likelihood of the equivalent `CompiledGaussianFamily` problem; over a Laplace
   family (`fit=fit_laplace`) it converges.
4. **Closed form.** A Gaussian model with a single optimized observation-scale
   (or effect precision) recovers the known type-II ML value on a small analytic
   fixture (reuse the pattern in `tests/optimization/test_empirical_bayes.py`).
5. **End-to-end declarative EB.** `LGM.fit` with a `Hyperparameter` precision:
   Gaussian (`exact_gaussian`) and Poisson (`laplace`) both return a converged
   result whose `hyperparameters` holds the optimized estimate and whose marginal
   likelihood at the optimum is ≥ the marginal likelihood at `initial`.
6. **No-Hyperparameter invariance.** A model with only fixed floats returns a
   result byte-identical to the current fast path, with `hyperparameters is None`.
7. **Pandas/Spark contract.** EB predictions keep caller order (Pandas) and
   canonical `prediction_keys` (Spark); a Spark EB fit matches the sorted-Pandas
   EB fit.
8. **Errors.** Bad bounds, and a non-converging optimizer, raise the specified
   typed errors.

## Acceptance Criteria

1. `LGM.fit` estimates declared `Hyperparameter`s by type-II ML for both engines
   and exposes them on `result.hyperparameters`; plain-float models are unchanged
   with `hyperparameters is None`.
2. The new `CompiledFamily` reproduces `CompiledGaussianFamily`'s materialized IR
   and EB estimate on an equivalent Gaussian problem; the legacy path is
   untouched.
3. `optimize_empirical_bayes` drives the new family for Gaussian and Laplace with
   no duplication of its optimization logic.
4. No new runtime dependencies; full suite green; Pandas/Spark contracts preserved.
5. Penalized MAP-II and YAML EB are not implemented but are unblocked by the
   family/optimizer seam and recorded on the roadmap.

# pyLGM Result-Type Unification Design

**Status:** Approved 2026-08-24

## Purpose

Pay down the largest remaining piece of structural debt in `pylgm`, flagged
repeatedly by slice reviews: `inference/result.py` has grown to 993 lines
carrying three result types and three marginal types with substantial
duplication. Two of the marginal validators are **byte-identical**; the three
result types each restate the same nine read-only properties, the same four
delegating methods, and much of the same constructor validation.

While the types are being unified, also resolve the `predictive_variance`
inconsistency the prediction slice documented but deliberately did not fix.

The work is **two sequenced parts in separate commits**: a behaviour-preserving
refactor, then a deliberate, documented behaviour change. Keeping them apart is
the point — a regression must be attributable to one or the other.

## Part A — Unification (behaviour-preserving)

### A1. Marginal validators

`_skew_normal_marginal_array` and `_tabulated_marginal_array` are identical
character for character (2-D check, same message, finiteness check). Replace
both with a single `_marginal_grid_array`. The 1-D `_marginal_array` remains as
its sibling.

### A2. `LatentMarginals` protocol

`GaussianMarginals`, `SkewNormalMarginals`, and `TabulatedMarginals` share a
conceptual interface (`mean`, `variance`, `std`, `quantile`) with legitimately
different internals — a Gaussian pair, a mixture of skew-normals, and a
tabulated density cannot share implementations. Express the contract as a
`typing.Protocol` for documentation and type-checking, **not** a base class.
Zero runtime behaviour change.

### A3. `_BaseResult`

`GaussianResult`, `LaplaceResult`, and `INLAResult` share:

- nine read-only properties: `mean`, `covariance`, `predictive_mean`,
  `predictive_variance`, `prediction_keys`, `hyperparameters`,
  `prediction_context`, `engine`, `converged`;
- four delegating methods: `latent_marginals`, `hyperparameter_marginals`,
  `linear_combinations`, `predict`;
- the bulk of their constructor validation.

Introduce `_BaseResult` holding all three groups. Constructor validation is
shared through a `_init_common(...)` helper that each subclass calls before
handling its own extra fields, so each subclass keeps an explicit signature
(every call site passes keywords) while the shared validation lives once.

**Hard constraint — the three results must remain siblings.** They may inherit
`_BaseResult`; none may inherit another. `model.py:_rebuild_result` tests
`isinstance(result, LaplaceResult)` *before* `isinstance(result, INLAResult)`,
and `optimization/inla.py` tests `isinstance(reference, LaplaceResult)`, so
making `INLAResult` a subclass of `LaplaceResult` would silently misroute every
INLA rebuild. This is the single highest-risk property of the refactor.

Per-type surfaces stay where they are: `fitted_mean` and `link_name` on
`LaplaceResult`; `fitted_mean`, `latent_marginal_table`, and `criteria` on
`INLAResult`.

### A4. Verification

"The suite passes" is necessary but not sufficient for a refactor of the
public surface. Part A additionally requires an explicit **equivalence
harness**: across a matrix of fits (Gaussian/Poisson/Bernoulli ×
plug-in/optimise/integrate × several effects), compare before and after for
every public attribute and every method result, plus `repr`, `isinstance`
outcomes against all three types, immutability of returned arrays, and the
error messages raised by invalid construction. Any difference in Part A is a
defect.

## Part B — `predictive_variance` convention (deliberate change)

Today `GaussianResult.predictive_variance` includes the observation variance
`σ²` (`gaussian.py`: `einsum(...) + variance`), while
`LaplaceResult.predictive_variance` does not (`laplace.py`: `einsum(...)`).
Same attribute name, two different quantities.

**Resolution: `predictive_variance` is the linear-predictor variance
everywhere** — `Var(η_i | y)`, excluding observation noise.

Rationale, in order of weight:

1. **It is the only convention uniformly definable.** Poisson and Bernoulli
   have no observation-variance parameter to add, so "include it" cannot be
   applied to two of the three likelihoods.
2. **It matches the literature this library targets.** R-INLA reports
   `summary.linear.predictor` — the η posterior without observation noise —
   alongside separate fitted values; pylgm emulates that family.
3. **It is already the majority behaviour**: the Laplace path, every
   non-Gaussian model, and `predict()`'s Laplace branch already exclude it.

Changes required:

- `gaussian.py`: drop the `+ variance` term.
- `GaussianResult` gains **`observation_variance`** (the σ² previously folded
  in), so the old value is exactly
  `predictive_variance + observation_variance` — nothing is lost, and the
  migration is a one-liner.
- `inference/prediction.py`: `predict_from` currently adds `likelihood.variance`
  for a Gaussian likelihood to match the old fit-row convention; that addition
  goes, keeping `predict()` consistent with the fit rows (the existing fit-row
  round-trip tests enforce this).
- INLA's Gaussian path follows automatically — it aggregates conditional fits
  via the law of total variance and needs no change.

**Not affected, verified:** `DIC/WAIC/CPO/PIT`. `_model_criteria` works from the
grid's latent posteriors by Gauss–Hermite quadrature and never reads
`predictive_variance`. `fitted_mean` is also unaffected: `laplace.py` feeds the
η variance into `response_prediction`, which is already the linear-predictor
quantity.

**This is a breaking change to recorded numbers** for Gaussian models: reported
predictive variances shrink by σ² and reported predictive standard deviations
shrink accordingly. It must be documented prominently with the reconstruction
formula.

## Scope

### Included
- A1 validator consolidation, A2 protocol, A3 `_BaseResult` including shared
  constructor validation, A4 equivalence harness.
- Part B: the convention change, `observation_variance`, the `predict()`
  update, docs and migration note.
- Splitting `result.py` if it remains unwieldy after unification is permitted
  but not required; correctness first.

### Excluded (deferred / roadmap)
- Any change to the marginal types' numerical behaviour or public methods.
- Unifying the `Prediction` type into the result hierarchy — it is a different
  thing (new rows, no latent posterior of its own).
- Renaming `predictive_variance`, or adding a full posterior-predictive
  accessor for non-Gaussian likelihoods (no closed form).
- The bounded-parameter bounds-block extraction in `compiler.py` and the
  in-loop Newton decrement — separate recorded follow-ups.

## Architecture
- `inference/result.py`: `_marginal_grid_array`, `LatentMarginals` protocol,
  `_BaseResult` + `_init_common`, the three result types reduced to their
  distinctive surfaces.
- `inference/gaussian.py`: drop `+ variance`; pass the observation variance to
  `GaussianResult`.
- `inference/prediction.py`: drop the Gaussian `+ likelihood.variance`.
- No changes to `model.py`, `optimization/`, engines' math, or the marginal
  types' numerics.

## Errors
- Constructor validation messages are preserved exactly in Part A (asserted by
  the harness). Part B adds validation for `observation_variance` (finite,
  non-negative) consistent with the existing style.

## Testing and validation
1. **Equivalence harness (Part A):** every public attribute and method output
   identical before and after, across the fit matrix; plus `repr`, `isinstance`
   against all three types, array immutability, and constructor error messages.
2. **Sibling invariant:** an explicit test that no result type is an instance of
   another (`not isinstance(inla_result, LaplaceResult)` and the converse), so
   the `_rebuild_result` dispatch cannot be broken silently by a later change.
3. **`_rebuild_result` round trip** preserves type and every field for all three
   types.
4. **Part B:** `GaussianResult.predictive_variance` equals the η variance;
   `predictive_variance + observation_variance` reproduces the pre-change value
   exactly; Laplace and non-Gaussian outputs are unchanged; `predict()`'s
   fit-row round trip still matches for all three result types; criteria are
   unchanged (asserted numerically against pre-change values).
5. Full suite green with no tolerance loosened.

## Acceptance criteria
1. The duplicated validators are gone; the three result types share a base
   carrying the common properties, delegating methods, and constructor
   validation; `result.py` is materially smaller.
2. Part A changes no observable behaviour, demonstrated by the equivalence
   harness rather than by the suite alone.
3. The three result types remain siblings, with a test pinning it.
4. `predictive_variance` is the linear-predictor variance for every result type
   and for `predict()`; `observation_variance` reconstructs the old Gaussian
   value; criteria and `fitted_mean` are unchanged.
5. The breaking change is documented with the reconstruction formula.
6. No new runtime dependency; full suite green.

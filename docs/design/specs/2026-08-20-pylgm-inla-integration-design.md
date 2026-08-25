# pyLGM INLA Integration Core Design

**Status:** Approved 2026-08-20

## Purpose

Add INLA-style *integration* over hyperparameters to the declarative `LGM.fit`,
replacing the plug-in (EB/MAP) treatment with a grid quadrature of the
hyperparameter posterior. This delivers INLA's core value — latent posterior
marginals that account for hyperparameter uncertainty, plus posterior marginals
for the hyperparameters themselves and an integrated marginal likelihood.

This is sub-slice **3a** of INLA. It reuses the existing family, conditional
engines (exact-Gaussian and Laplace), `optimize_empirical_bayes`, priors, and
result helpers. It is an *integration wrapper* over the conditional fits; it adds
no new conditional engine.

## Scope

### Included
- `LGM.fit(frame, engine=..., hyperparameters="optimize" | "integrate")`;
  `"optimize"` is the default and current behavior, `"integrate"` runs INLA.
- Grid integration of the hyperparameter posterior in the optimizer's log space,
  centered at the posterior mode, scaled by the finite-difference Hessian.
- Integrated **latent marginals** (grid-weighted Gaussian mixture), populated
  **hyperparameter marginals**, and an **integrated marginal likelihood**.
- Integrated posterior-predictive quantities (mixture of the per-grid-point
  predictions), on both engines.
- An `INLAResult` exposing these through the shared result surface, with
  approximation-quality diagnostics.
- A guard on the number of hyperparameters / grid points (curse of
  dimensionality), docs, and a runnable example.

### Excluded (later INLA sub-slices)
- **3b:** model-assessment criteria — DIC, WAIC, CPO, PIT.
- **3c:** richer latent marginal strategies — simplified-Laplace and full-Laplace
  corrections. This slice uses the **Gaussian** latent approximation (the
  conditional fit's own Gaussian per θ).
- CCD (central composite design) integration; this slice uses a full grid.
- Result-type unification onto a shared base (still the deferred refactor slice;
  `INLAResult` reuses the existing module-level helpers to minimize duplication).
- YAML expression of priors / integration.

## Public Contract

```python
result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")

result.latent_marginals("trend")        # integrated (mixture) marginals
result.hyperparameter_marginals()        # {name: GaussianMarginals}, populated
result.log_marginal_likelihood           # integrated over hyperparameters
result.predictive_mean, result.predictive_variance   # integrated
result.diagnostics["inla_grid_points"], ["inla_effective_weight"], ["inla_mode"]
```

- `hyperparameters="integrate"` requires at least one declared `Hyperparameter`
  (otherwise there is nothing to integrate → `ValueError`).
- `hyperparameters="optimize"` (default) is byte-identical to today: EB/MAP when
  `Hyperparameter`s are declared, the fixed plug-in fast path otherwise.
- Priors, when declared, enter the posterior (MAP-style) exactly as in the
  empirical-Bayes/MAP-II slices; a prior-free hyperparameter is integrated
  against a locally-flat (bounded) posterior.
- Works for `engine="exact_gaussian"` and `engine="laplace"`; the Pandas
  caller-order and Spark canonical-key contracts are preserved for the integrated
  predictive arrays.

## Mathematics

Let `θ` be the hyperparameters, `u = log θ` the optimizer's working coordinates,
and

\[ s(u) \;=\; \log \pi(y \mid \theta(u)) \;+\; \log \pi(\theta(u)), \]

the (unnormalized) log hyperparameter posterior on the θ-scale, evaluated as the
conditional fit's `log_marginal_likelihood` plus the declared priors' `logpdf`.
The mode `u*` and the conditional fit at it come from `optimize_empirical_bayes`
(which already minimizes `-s(u)` in log space, with the MAP penalty). The
posterior over `θ` is `π(θ|y) ∝ exp(s(u))`, and integrals in `u`-space carry the
log-Jacobian `Σ_j u_j` (since `dθ_j = θ_j du_j`).

**Hessian.** `H = ∂²s/∂u²` at `u*`, by central finite differences. `-H` is
positive definite at a mode; eigendecompose `-H = V Λ Vᵀ`. The whitening step
directions are the columns of `V Λ^{-1/2}`. (The `Σ u_j` Jacobian is linear in
`u`, so it does not affect `H`.)

**Grid.** Over an integer lattice `z` (spacing `δ`, radius bounded), set
`u_k = u* + δ · V Λ^{-1/2} z`, `θ_k = clip(exp(u_k), lower, upper)`. Keep points
whose `s(u_k) ≥ s(u*) − δ_drop` (default `δ_drop = 2.5`). Each kept point gets a
log weight that includes the Jacobian:

\[ \log \tilde w_k \;=\; s(u_k) \;+\; \textstyle\sum_j u_{k,j}, \qquad w_k = \mathrm{softmax}(\log \tilde w_k). \]

A `max_grid_points` guard (default e.g. 4096) rejects over-large grids (too many
hyperparameters) with a clear error before doing the work.

**Integrated quantities** (accumulated incrementally over grid points, so peak
memory is one conditional fit at a time):
- latent mean: `m̄ = Σ_k w_k m_k`;
- latent covariance: `C̄ = Σ_k w_k (C_k + m_k m_kᵀ) − m̄ m̄ᵀ` (proper mixture
  covariance — this is where hyperparameter uncertainty widens the posterior);
- predictive mean/variance: mixture moments of the per-θ_k `predictive_mean`/
  `predictive_variance`; `fitted_mean` (response scale, non-Gaussian) as
  `Σ_k w_k fitted_mean_k`;
- hyperparameter marginals: per hyperparameter `j`, weighted mean and variance
  of `θ_{k,j}` returned as `GaussianMarginals` (a summary; the raw weighted grid
  is also exposed in diagnostics).

**Integrated marginal likelihood.** With grid cell volume `δ^d / sqrt(det(-H))`
in `u`-space,

\[ \log \pi(y) \;\approx\; \operatorname{logsumexp}_k(\log \tilde w_k) \;+\; d\,\log\delta \;-\; \tfrac12 \log\det(-H). \]

## Architecture

### Driver (`src/pylgm/optimization/inla.py`)

```python
def integrate_inla(
    family, bounds, *, initial=None, fit=None, penalty=None,
    allow_large_dense=False, grid_step=1.0, log_density_drop=2.5,
    max_grid_points=4096,
) -> INLAResult
```

Steps: (1) run `optimize_empirical_bayes` for `u*` and the mode fit; (2) build an
`evaluate(θ) -> (s_value, conditional_fit)` closure reusing `family.materialize` +
`fit` + `penalty`; (3) finite-difference `H`; (4) construct the grid and weights;
(5) accumulate the integrated quantities; (6) assemble the `INLAResult`. Pure
helpers `_finite_difference_hessian`, `_build_grid`, and the incremental
`_MixtureAccumulator` are separately unit-testable. All numerics run under the
same `np.errstate(...)` discipline as the other engines; a non-PD `-H`
(degenerate/boundary mode) is regularized (small ridge) and flagged in
diagnostics.

### Result (`src/pylgm/inference/result.py`)

Add `INLAResult`, parallel to `GaussianResult`/`LaplaceResult`, reusing the
module-level helpers (`_readonly_array`, `_readonly_diagnostics`,
`_validate_prediction_keys`, `latent_marginals_from`, `linear_combinations_from`,
`GaussianMarginals`). It stores the integrated `mean`/`covariance`,
`log_marginal_likelihood`, `predictive_mean`/`predictive_variance`, optional
`fitted_mean`/`link_name` (present when the conditional engine is Laplace),
`block_slices`, `prediction_keys`, immutable `diagnostics`, and the computed
hyperparameter marginals. `engine` returns `"inla"`; `converged` reflects a
successful integration; `hyperparameter_marginals()` returns the populated
`Mapping[str, GaussianMarginals]`; `latent_marginals`/`linear_combinations`
delegate to the shared helpers on the integrated mean/covariance.

(Result-type unification onto a shared base remains a separate deferred refactor;
`INLAResult` follows the existing parallel-type convention to keep this slice
focused.)

### Dispatch (`model.py`)

`LGM.fit` gains `hyperparameters: str = "optimize"` (validated: `"optimize"` or
`"integrate"`). `_fit_pandas`/`_fit_spark` branch: `"optimize"` → the existing
path unchanged; `"integrate"` → require a declared `Hyperparameter` (else
`ValueError`), build the family/bounds/initial/penalty exactly as
`_run_empirical_bayes` does, call `integrate_inla`, and attach the result through
the same caller-order realignment (Pandas) / canonical-key (Spark) machinery
(extend `_rebuild_result` to handle `INLAResult`). The engine↔likelihood pairing
(`self._engine`) is enforced unchanged and selects the conditional `fit`.

## Errors

- `hyperparameters` not in `{"optimize", "integrate"}`: `ValueError`.
- `"integrate"` with no declared `Hyperparameter`: `ValueError` explaining there
  is nothing to integrate.
- Grid exceeds `max_grid_points` (too many hyperparameters): `OptimizationError`
  naming the count and the guard.
- Optimizer non-convergence at the mode, or a non-finite objective: the existing
  `OptimizationError`/`NumericalError` paths.
- Numerical failure during integration (non-finite accumulation): `NumericalError`.
  No failed integration is returned as a successful result.

## Testing and Validation

1. **Hessian helper.** `_finite_difference_hessian` recovers the exact Hessian of
   a known quadratic `s(u)` to tight tolerance.
2. **Grid helper.** `_build_grid` centers at the mode, respects `log_density_drop`
   and `max_grid_points`, clips to bounds, and returns normalized weights summing
   to 1.
3. **Integration vs 1-D quadrature (anchor).** For a single-hyperparameter model,
   the integrated `log_marginal_likelihood`, the latent marginal mean/variance,
   and the hyperparameter marginal match a fine independent 1-D quadrature of the
   same posterior within tolerance.
4. **Uncertainty widens.** Integrated latent marginal variances are ≥ the plug-in
   (EB/MAP mode) variances on a fixture where the hyperparameter posterior is
   non-degenerate (integration adds between-θ variance).
5. **Both engines.** Gaussian (`exact_gaussian`) and Poisson (`laplace`) models
   integrate to a converged `INLAResult` with populated hyperparameter marginals.
6. **Dispatch & invariance.** `hyperparameters="optimize"` is unchanged;
   `"integrate"` with no `Hyperparameter` raises; the `max_grid_points` guard
   fires for too many hyperparameters. Pandas caller-order and a Spark/Pandas
   INLA parity check hold.
7. **Result contract.** `INLAResult` immutability, defensive copies,
   `latent_marginals`/`linear_combinations`/`hyperparameter_marginals`, and
   diagnostics.

## Acceptance Criteria

1. `LGM.fit(..., hyperparameters="integrate")` returns an `INLAResult` with
   integrated latent marginals, populated hyperparameter marginals, and an
   integrated marginal likelihood, for both engines.
2. On a one-hyperparameter model the integrated quantities match an independent
   fine 1-D quadrature within tolerance; integrated latent variances are no
   smaller than the plug-in variances when the hyperparameter posterior is
   non-degenerate.
3. `hyperparameters="optimize"` and no-`Hyperparameter` behavior are unchanged;
   invalid/empty/over-large integration requests raise the specified typed errors.
4. No new runtime dependencies; full suite green; Pandas/Spark contracts and the
   legacy path preserved.
5. DIC/WAIC/CPO/PIT (3b) and simplified/full-Laplace latent strategies (3c) are
   not implemented but are unblocked by this integration core and recorded on the
   roadmap.

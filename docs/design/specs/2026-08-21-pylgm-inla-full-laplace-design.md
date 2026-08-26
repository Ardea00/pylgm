# pyLGM INLA Full-Laplace Latent Marginals Design

**Status:** Approved 2026-08-21

## Purpose

Add the **full Laplace approximation** for latent marginals `π(x_i | y)` — the
most accurate INLA latent strategy — completing the three-tier ladder
(Gaussian → simplified Laplace → full Laplace). Faithful to Rue, Martino &
Chopin (2009) §3.2.2 (eqs 12, 13, 16, 17), using the standard eq-13
conditional-mean approximation (no per-latent nonlinear re-optimization). This is
INLA sub-slice **3d**.

**References (equations transcribed from the local PDF):**
- Rue, H., Martino, S., Chopin, N. (2009), JRSS-B 71(2):319–392 — §3.2.2
  (eqs 12–17).

## Scope boundary: unconstrained latent fields only

Full Laplace profiles `x_{−i}` and needs the `(p−1)` conditional precision
determinant using the prior precision `Q`. This is well-posed when the latent
field has **no linear constraints** (Fixed + IID effects; `Q` full-rank PD).
Intrinsic constrained effects (RW1/RW2, sum-to-zero constraints) make
single-coefficient marginalization on the constrained manifold research-grade
and are **out of scope** here: `latent_strategy="laplace"` with any latent
constraint raises `UnsupportedEngineError` (RW models remain fully supported
under `gaussian` and `simplified_laplace`). Documented as a known limitation.

## Equations (per hyperparameter grid point θ; Gaussian conditional fit gives
latent mean `μ`, covariance `Σ`; prior precision `Q = compiled.precision`,
observed-row design `X`, offset, response `y`)

For latent `i` with `σ_i = √Σ_ii`, evaluate `π̃_LA` at Gauss–Hermite abscissae
`x_i = μ_i + σ_i z_k` (eq 16). At each abscissa:
- **Conditional mean** (eq 13): `x_{−i}* = μ_{−i} + (Σ_{−i,i}/Σ_ii)(x_i − μ_i)`;
  form the full configuration `x` (value `x_i` at `i`, `x_{−i}*` elsewhere).
- **Linear predictor / log-likelihood:** `η = X x + offset`,
  `ℓ = Σ_j pointwise_log_density(η_j, y_j)`.
- **Joint:** `log π(x,θ,y) = −½ xᵀ Q x + ℓ`.
- **π̃_GG normalizer (eq 12):** with working weights `W_j = working_weights(η_j, y_j)`
  and `P_{−i} = Q_{−i,−i} + X_{−i}ᵀ diag(W) X_{−i}` (prior + data information with
  row/col `i` deleted),
  `log π̃_LA(x_i) = log π(x,θ,y) − ½ log det P_{−i}` (the additive `(p−1)/2·log 2π`
  constant is dropped — it cancels in the spline difference).
- **Gaussian reference:** `log π̃_G(x_i) = −½ z_k² − log(σ_i) − ½ log(2π)`.

**Spline correction (eq 17).** Fit a natural cubic spline `s(x_i)` through the
abscissa points `(x_i, Δ_k)` with `Δ_k = log π̃_LA(x_i) − log π̃_G(x_i)`; the
Laplace marginal is `π̃_LA(x_i | θ) ∝ N(x_i; μ_i, σ_i²) · exp(s(x_i))`,
normalized by quadrature on a fine per-latent grid. Evaluated on a common
per-latent grid `x_i ∈ [μ̄_i − R σ̄_i, μ̄_i + R σ̄_i]` (default `R=8`), extrapolating
`s` as constant beyond the outer abscissae (guards the tails).

**Gaussian likelihood is exact:** for a Gaussian likelihood the true conditional
`x|θ,y` is exactly Gaussian, so `Δ_k` is (up to the dropped constant) flat in
`x_i` and `s ≡ 0` ⇒ `π̃_LA = N(μ_i, σ_i²)` — full Laplace reduces exactly to the
Gaussian marginal (per grid point; the integrated grid mixture still carries
hyperparameter-mixture skew, as in 3c).

**Integrated marginal:** `π(x_i | y) ≈ Σ_k w_k π̃_LA(x_i | θ_k)` — a grid-weighted
mixture of the per-θ spline-corrected densities, evaluated on the common
per-latent grid and renormalized. Represented as a **tabulated density**.

## `TabulatedMarginals`

R-INLA's native marginal representation: per latent component, a strictly
increasing `x`-grid and non-negative `density` values (normalized to integrate to
1 by the trapezoidal rule). Provides:
- `mean`, `variance`, `std`, `skewness` — trapezoidal-quadrature moments.
- `pdf(x)` — linear interpolation (0 outside the grid).
- `cdf(x)` — cumulative-trapezoid interpolation (0/1 outside).
- `quantile(p)` — inverse-CDF by interpolation (`0<p<1`).
Read-only defensive copies (matching the other marginal types). Reduces to the
Gaussian values (within quadrature tolerance) when the density is `N(μ,σ²)`.

## Public Contract

```python
result = model.fit(frame, engine="laplace",
                   hyperparameters="integrate", latent_strategy="laplace")
m = result.latent_marginals("region")   # TabulatedMarginals
m.mean; m.variance; m.std; m.skewness
m.quantile(0.025); m.quantile(0.975); m.pdf(x); m.cdf(x)
```

- `latent_strategy` ∈ {`"gaussian"` (default), `"simplified_laplace"`,
  `"laplace"`}; the first two are unchanged.
- `"laplace"` requires `hyperparameters="integrate"` and an **unconstrained**
  latent field (else `ValueError` / `UnsupportedEngineError` respectively).
- Both engines; expensive (dense per-abscissa `(p−1)` determinant) — dense
  reference regime only.

## Scope

### Included
- `_fit_cubic_spline` / natural-cubic-spline evaluation helper (or SciPy
  `CubicSpline`, already available) with constant tail extrapolation.
- `_full_laplace_marginals(design, offset, y, grid, precision, *, n_abscissae=7,
  grid_radius=8.0, grid_points=...)` → per-latent common `x`-grid + mixed density,
  faithful to eqs 12/13/16/17; unconstrained models only.
- `TabulatedMarginals` type; `latent_strategy="laplace"` dispatch; INLAResult
  integration; constrained-model guard.
- Validation: Gaussian exact reduction, and a brute-force nested-quadrature
  true-marginal oracle for a tiny non-Gaussian model — full Laplace must match
  the truth at least as well as (and generally better than) simplified Laplace.

### Excluded (deferred / roadmap)
- Constrained/intrinsic effects (RW1/RW2) under full Laplace.
- Region-of-interest pruning (eq 15) and sparse fast-determinant updates — this
  slice recomputes the dense `(p−1)` determinant per abscissa.
- η-marginal full Laplace; the thick-tail spline fallback specifics.
- R-INLA parity fixtures (unavailable here; brute-force oracle is the anchor).
- Result-type unification (Gaussian/Laplace/INLA/Skew/Tabulated) — separate
  refactor.

## Architecture
- `optimization/inla.py`: `_full_laplace_marginals` (+ spline helper); wired into
  `integrate_inla` only when `latent_strategy="laplace"`, reusing the observed-row
  `design`/`offset`/`y` and per-grid `(weight, fit, likelihood)`; `precision` from
  `kept[0][4].precision`; constrained-model guard (`constraints.shape[0] > 0`).
- `inference/result.py`: `TabulatedMarginals`; `INLAResult` stores an optional
  tabulated table and returns it from `latent_marginals` under `"laplace"`;
  carried through `_rebuild_result` un-reordered. Export `TabulatedMarginals`.
- `model.py`: widen `latent_strategy` validation to include `"laplace"`; thread
  through; the unconstrained/integration guards.

## Errors
- `latent_strategy="laplace"` without `hyperparameters="integrate"`: `ValueError`.
- `latent_strategy="laplace"` with a constrained latent field:
  `UnsupportedEngineError` naming the limitation.
- Non-finite / non-PD `P_{−i}` determinant, or non-finite tabulated density:
  `NumericalError`.

## Testing and Validation
1. `TabulatedMarginals`: trapezoidal moments/quantiles vs a known analytic
   density (Normal, and a skew-normal) within tolerance; `cdf(quantile(p))≈p`;
   Gaussian reduction; immutability.
2. Spline helper: interpolates the abscissa values; constant tail extrapolation.
3. `_full_laplace_marginals` Gaussian-likelihood exactness: `s≡0`, tabulated
   density ≈ `N(μ_i,σ_i²)` (mean/variance/quantiles match the Gaussian marginal).
4. **Brute-force oracle:** tiny Poisson (and Bernoulli) unconstrained model
   (Fixed + IID, single grid point) — full-Laplace marginal mean/variance/
   skewness/central quantiles match a nested-quadrature true `π(x_i|θ,y)` at
   least as well as simplified Laplace (compare all three: Gaussian, SLA, full
   LA vs truth; full LA error ≤ SLA error on skewness/tails).
5. Constrained guard: an RW model with `latent_strategy="laplace"` raises
   `UnsupportedEngineError`.
6. Dispatch: default `"gaussian"` unchanged; `"laplace"` requires integrate;
   invalid strategy raises; both engines; `_rebuild_result` carries the table.

## Acceptance Criteria
1. `latent_strategy="laplace"` returns `TabulatedMarginals` for an integrated,
   unconstrained fit (both engines), faithful to RMC eqs 12/13/16/17.
2. Gaussian likelihood ⇒ per-grid-point full Laplace equals the Gaussian marginal.
3. On a tiny non-Gaussian model, full Laplace matches the brute-force true
   marginal at least as well as simplified Laplace (skewness + tail quantiles).
4. Constrained (RW) models raise a clear unsupported error under `"laplace"`;
   `gaussian`/`simplified_laplace` unchanged.
5. No new runtime dependency (SciPy `CubicSpline` already available); full suite
   green; predictions/criteria/optimize paths and the other latent strategies
   unchanged.
6. Region-of-interest pruning, sparse determinant updates, constrained-effect
   full Laplace, η-marginals, and R-INLA parity remain deferred/recorded.

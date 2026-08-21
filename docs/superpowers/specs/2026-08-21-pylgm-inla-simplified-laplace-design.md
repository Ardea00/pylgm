# pyLGM INLA Simplified-Laplace Latent Marginals Design

**Status:** Approved 2026-08-21

## Purpose

Add the **simplified Laplace approximation (SLA)** for the latent marginals
`π(x_i | y)` to the integrated fit, replacing the Gaussian latent strategy's
symmetric marginals with **location- and skewness-corrected** marginals for
non-Gaussian likelihoods. This is INLA sub-slice **3c**, implemented faithfully
to Rue, Martino & Chopin (2009), §3.2.3 and Appendix B (the equations are
transcribed below and cited by their paper numbers). Full Laplace and the
deferred INLA refinements remain later work.

**References (equations transcribed from the local PDFs):**
- Rue, H., Martino, S., Chopin, N. (2009), *Approximate Bayesian inference for
  latent Gaussian models by using integrated nested Laplace approximations*,
  JRSS-B 71(2):319–392 — §3.2.3 (eqs 14, 16, 18–22) and Appendix B (eqs 31–32).
- Martins, Simpson, Lindgren, Rue (2013), *Bayesian computing with INLA: New
  features*, CSDA 67:68–83 (SLA refinements).

## Mapping RMC's augmented field to pyLGM

RMC augment the latent field to include the linear predictors η, so each
observation's likelihood depends on a single latent component `x_j = η_j`. In
pyLGM the latent field `x` is the effect coefficients and `η = X x + offset`
(design `X`, `n×p`). The SLA of a coefficient marginal `x_i` therefore uses:
- `d³_j = ∂³/∂η³ log p(y_j | η_j)` — the likelihood's third derivative w.r.t. its
  linear predictor, evaluated at the posterior-mean predictor `μ_{η_j}(θ)`.
- `a_{ij}(θ) = corr_G(x_i, η_j)` — the Gaussian-approx posterior correlation
  between coefficient `x_i` and predictor `η_j` (RMC eq 14).

## Equations (per hyperparameter grid point θ; Gaussian conditional fit gives
latent mean `m`, covariance `Σ`)

For coefficient `i` (μ_i = m_i, σ_i = √Σ_ii) and observation `j`
(μ_{η_j} = offset_j + (X m)_j, σ_{η_j}² = (X Σ Xᵀ)_{jj},
`cov(x_i, η_j) = (Σ Xᵀ)_{ij}`, `a_{ij} = cov(x_i,η_j)/(σ_i σ_{η_j})`):

- Standardized variable (eq 16): `x_s = (x_i − μ_i)/σ_i`.
- Skewness correction (eq 21):
  `γ³_i = (1/σ_i³) Σ_j d³_j · cov(x_i, η_j)³`.
- Location correction (eq 21):
  `γ¹_i = (1/(2 σ_i)) Σ_j σ_{η_j}² (1 − a_{ij}²) · d³_j · cov(x_i, η_j)`.
- Series (eq 22): `log π̃_SLA(x_s) = const − ½ x_s² + γ¹_i x_s + (1/6) γ³_i x_s³`.

**Skew-normal fit** (eq 23, Appendix B eqs 31–32). Fit `SN(ξ, ω, a)` with density
`(2/ω) φ((x−ξ)/ω) Φ(a(x−ξ)/ω)`, `δ = a/√(1+a²)`, mean `ξ + ω δ √(2/π)`, variance
`ω²(1 − 2δ²/π)`, holding **mean = γ¹_i** and **variance = 1**, and imposing the
Appendix-B third-derivative-at-the-mode condition (eq 32, leading order):

`(4 − π) √2 / π^{3/2} · (a/ω)³ = γ³_i`.

Solve: `r ≡ a/ω = sign(γ³_i) · ( |γ³_i| · π^{3/2} / ((4−π) √2) )^{1/3}`; then with
`a = r ω`, `δ = a/√(1+a²)`, solve `ω²(1 − 2δ²/π) = 1` for `ω > 0` (1-D
root-find; `ω≈1` at small skew), and `ξ = γ¹_i − ω δ √(2/π)`. The **latent
marginal of `x_i`** is this skew-normal transformed by `x_i = μ_i + σ_i x_s`:
`SN(ξ_i' = μ_i + σ_i ξ, ω_i' = σ_i ω, a_i = a)`, i.e. mean `μ_i + σ_i γ¹_i`,
variance `σ_i²`, skewness set by `a`.

**Skew bound.** The skew-normal skewness is bounded to `|γ₁| < (4−π)/2·(√(2/π))³/(1−2/π)^{3/2} ≈ 0.995`.
If `γ³_i` implies a larger magnitude, clamp `r` to the achievable maximum and
flag the marginal (a per-marginal `skew_clamped` diagnostic count). Documented as
a known limit of the skew-normal representation.

**Gaussian likelihood is exact:** `d³_j = 0 ⇒ γ¹_i = γ³_i = 0 ⇒ a = 0, ω = 1,
ξ = 0 ⇒ SN = N(μ_i, σ_i²)` — the SLA reduces exactly to the Gaussian marginal.

**Integrated marginal:** `π(x_i | y) ≈ Σ_k w_k SN(ξ'_{ik}, ω'_{ik}, a_{ik})` — a
grid-weighted mixture of skew-normals (weights `w_k` from 3a).

## Scope

### Included
- `third_derivative(eta, y)` on the three compiled likelihoods (Gaussian 0;
  Poisson −exp(η); Bernoulli −p(1−p)(1−2p)).
- A `_simplified_laplace_marginals` helper computing per-coefficient
  `(ξ', ω', a)` per grid point (RMC eqs above + Appendix-B skew-normal fit).
- A new **skew-aware marginal type** representing a grid-weighted mixture of
  skew-normals: `mean`, `variance`, `std`, skewness, `quantile(p)` (via the
  mixture CDF and a bracketed root-find), and `pdf`/`cdf`. Reduces exactly to the
  Gaussian marginal when all skew parameters are zero.
- `latent_strategy="gaussian"` (default, unchanged) or `"simplified_laplace"` on
  `LGM.fit(..., hyperparameters="integrate", latent_strategy=...)`. Under SLA,
  `result.latent_marginals(block)` returns the skew-aware marginals; the
  integrated `mean`/`covariance` used by predictions and criteria are unchanged
  (SLA refines the marginal shape, not the mixture mean/covariance the rest of
  the result reports).
- Validation against (a) exact Gaussian-likelihood reduction and (b) a
  brute-force nested-quadrature *true* latent marginal for a tiny non-Gaussian
  model (SLA must be closer than the Gaussian strategy in skewness/quantiles).

### Excluded (deferred / roadmap)
- **Full Laplace** latent strategy (per-`x_i` profiling; RMC eq 10 / cubic-spline
  eq 17) and the thick-tail spline fallback (RMC's Student-t case).
- SLA for the linear-predictor (η) marginals — this slice corrects the
  **coefficient** marginals returned by `latent_marginals`.
- Region-of-interest pruning (RMC eq 15) — compute over all coefficients this
  slice (add pruning later if performance requires).
- Higher-order Appendix-B skew-normal terms beyond the leading `(a/ω)³` fit.
- The other deferred INLA follow-ups (integrand-mode grid recentering,
  randomized discrete PIT, robust discrete CPO, criteria for non-integrated
  fits) — recorded on the roadmap.
- **Validation against R-INLA** — unavailable in this environment; the spec's
  target R-INLA parity fixtures remain future work. The brute-force
  true-marginal oracle is the correctness anchor here.

## Public Contract

```python
result = model.fit(frame, engine="laplace",
                   hyperparameters="integrate", latent_strategy="simplified_laplace")
m = result.latent_marginals("region")     # skew-aware marginals
m.mean; m.variance; m.std; m.skewness      # arrays
m.quantile(0.025); m.quantile(0.975)       # skew-aware quantiles
```

- `latent_strategy` defaults to `"gaussian"` — existing integrated fits are
  byte-identical; `latent_marginals` still returns `GaussianMarginals` then.
- `latent_strategy="simplified_laplace"` requires `hyperparameters="integrate"`
  (SLA is a latent-marginal strategy within the integration); combining it with
  `hyperparameters="optimize"` raises `ValueError`.
- Works for both engines; for a Gaussian likelihood the SLA marginals equal the
  Gaussian marginals (exact), so the strategy is a no-op there.

## Architecture

- `likelihoods.py`: add `third_derivative`.
- `optimization/inla.py`: `_fit_skew_normal(gamma1, gamma3)` (Appendix-B solver)
  and `_simplified_laplace_marginals(design, offset, fits, weights, likelihoods)`
  returning per-coefficient mixture skew-normal parameters; called from
  `integrate_inla` only when `latent_strategy="simplified_laplace"`.
- `inference/result.py`: `SkewNormalMarginals` (grid-mixture skew-normal;
  `GaussianMarginals` retained for the Gaussian strategy); `INLAResult` gains a
  stored per-block skew-marginal table and returns it from `latent_marginals`
  under SLA (an immutable structure carried through `_rebuild_result`, not
  row-reordered — marginals are per latent block, not per data row).
- `model.py`: thread `latent_strategy` through `fit`/`_run_inla` with validation.

## Errors
- `latent_strategy` not in `{"gaussian","simplified_laplace"}`: `ValueError`.
- `latent_strategy="simplified_laplace"` without `hyperparameters="integrate"`:
  `ValueError`.
- Non-finite SLA quantities / skew-normal solve failure: `NumericalError`.
- Skew beyond the skew-normal bound: clamp + `skew_clamped` diagnostic (not an
  error).

## Testing and Validation
1. **third_derivative** correctness at hand-checked points (all three).
2. **Skew-normal fit** (`_fit_skew_normal`): `γ³=γ¹=0 ⇒ (ξ,ω,a)=(0,1,0)`; a
   nonzero `γ³` yields a skew-normal whose realized variance is 1, mean is `γ¹`,
   and whose log-density cubic coefficient matches `γ³` (eq 32) within tolerance;
   out-of-bound `γ³` clamps.
3. **Gaussian-likelihood exactness (anchor):** SLA marginals equal the Gaussian
   strategy's marginals (mean/variance/quantiles) to tight tolerance.
4. **Brute-force true-marginal oracle:** for a tiny Poisson (and Bernoulli)
   model, compare SLA `latent_marginals` mean/variance/skewness/central-quantiles
   to a nested-quadrature evaluation of the true `π(x_i|y)`; SLA must be closer
   than the Gaussian strategy (esp. skewness and the 0.025/0.975 quantiles), and
   recover the sign of the skew.
5. **SkewNormalMarginals** immutability, mixture mean/variance/skewness,
   monotone `quantile`, `pdf`/`cdf` consistency (cdf(quantile(p))≈p); Gaussian
   reduction when skew=0.
6. **Dispatch:** `latent_strategy="gaussian"` unchanged; SLA requires integrate;
   invalid strategy raises; both engines; `_rebuild_result` carries the skew
   marginals (Pandas/Spark).

## Acceptance Criteria
1. `latent_strategy="simplified_laplace"` returns skew-aware latent marginals for
   an integrated fit (both engines), faithfully per the RMC eqs above.
2. Gaussian-likelihood SLA equals the Gaussian marginals (exact); the default
   remains Gaussian and unchanged.
3. On a tiny non-Gaussian model the SLA marginal is closer to the brute-force
   true marginal than the Gaussian strategy (skewness + tail quantiles), with the
   correct skew sign.
4. No new runtime dependency; full suite green; predictions/criteria and the
   `optimize`/plug-in paths unchanged.
5. Full Laplace, η-marginal SLA, region-of-interest pruning, R-INLA parity
   fixtures, and the other INLA follow-ups remain deferred and recorded.

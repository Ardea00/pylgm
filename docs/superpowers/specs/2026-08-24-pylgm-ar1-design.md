# pyLGM AR1 Effect Design

**Status:** Approved 2026-08-24

## Purpose

Add the **stationary first-order autoregressive (AR1)** latent effect — the
temporal counterpart to the spatial CAR family, and the last major gap in the
effect vocabulary for temporal nowcasting. AR1 gives a *proper*, mean-reverting
temporal prior with an interpretable correlation parameter, complementing the
existing intrinsic RW1/RW2.

## Model

For ordered index levels `1..n` with marginal precision `τ > 0` and lag-1
correlation `ρ ∈ (−1, 1)`:

  `Cov(x_i, x_j) = (1/τ) · ρ^{|i−j|}`,
  `Q = τ/(1 − ρ²) · T`

with `T` tridiagonal: `T[1,1] = T[n,n] = 1`, `T[i,i] = 1 + ρ²` for interior `i`,
and `T[i,i±1] = −ρ`. Verified numerically: `Q · Cov = I` to machine precision,
and the marginal variance is exactly `1/τ` for every `(n, τ, ρ)` tried.

**τ is the marginal precision** (INLA's convention), so it is directly
comparable to the precision of an `IID` effect or BYM2's τ — not the innovation
precision. `ρ = 0` gives exactly `Q = τI` (independent levels); `ρ → 1`
approaches the intrinsic RW1 limit.

**AR1 is proper and full-rank**, so it carries **no constraints**. Unlike
RW1/RW2 it therefore works under all three latent strategies, including full
Laplace — the same property proper CAR and BYM2 have.

### Relationship to the existing machinery

AR1 is the temporal twin of proper CAR and reuses that slice's machinery:

| ρ | representation | path |
| --- | --- | --- |
| fixed float | `Q = τ·M`, `M = T/(1−ρ²)` constant | existing `ScalableBlock` |
| `Hyperparameter` | `Q` depends on both τ and ρ | `ParametricBlock` over (τ, ρ) |

An estimated ρ uses `LogitTransform(-1, 1)` with a small inset, is estimated by
empirical Bayes and integrated over by INLA jointly with τ, and must be declared
with `transform="logit"` (enforced, as for proper CAR's ρ). ρ's interval is the
**fixed** `(−1, 1)` — simpler than proper CAR's graph-derived interval, so no
interval resolution is needed at compile time.

### Index levels and spacing

Levels come from `ordered_observed_levels(frame[index])`, exactly as RW1/RW2, and
the design is one-hot over those levels. At least two levels are required.

**Regular spacing is assumed.** AR1 relates *consecutive* ordered levels, so an
absent period (a missing month, say) is silently treated as a single step rather
than a gap. This is the same assumption RW1/RW2 already make. It must be
documented prominently, together with the supported workaround: include the
missing period as a row with a **`NaN` response**, which contributes no
likelihood but creates its latent column and restores regular spacing.
Irregular-spacing support (correlation `ρ^{Δt}`) is deferred.

### Prediction

No prediction work is needed: AR1 is a structured block, so
`build_prediction_context` captures `(name, index, labels)` automatically and
`result.predict(new_data)` scores known time levels. An unseen level raises the
existing directed error pointing at the NaN-response workflow — which is also
the right answer for AR1, since forecasting a future period requires the joint
fit.

## Public contract

```python
from pylgm import AR1, Fixed, Gaussian, Hyperparameter, LGM
from pylgm.priors import PCPrecision

# fixed correlation
AR1("trend", index="t", precision=1.0, rho=0.8)

# estimated correlation, jointly with the marginal precision
rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
tau = Hyperparameter("trend.precision", initial=1.0,
                     prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(response="y",
            predictor=Fixed("1") + AR1("trend", index="t", precision=tau, rho=rho),
            likelihood=Gaussian(sigma=0.1))
eb = model.fit(frame)                                  # empirical Bayes
eb.hyperparameters["trend.rho"]
post = model.fit(frame, hyperparameters="integrate")   # INLA over (tau, rho)
post.hyperparameter_marginals()["trend.rho"].mean
```

- `AR1(name, index, precision=1.0, rho=0.5)`.
- `precision` (τ, marginal): float or `Hyperparameter`, as for every other effect.
- `rho`: a float strictly inside `(−1, 1)`, or a `Hyperparameter` with
  `transform="logit"`.
- Composes with `+`; participates in plug-in, optimise, and integrate; works
  with both engines and all three latent strategies; `predict()` works.

## Scope

### Included
- `effects/ar1.py`: `build_ar1(frame, name, index, precision, rho) -> LatentBlock`
  — tridiagonal precision, one-hot design over ordered levels, **empty
  constraints**.
- `AR1` spec in `effects/spec.py`; exports from `pylgm.effects` and `pylgm`.
- Compiler wiring in both declarative paths: fixed ρ → `ScalableBlock`;
  `Hyperparameter` ρ → `ParametricBlock` over (τ, ρ) with `LogitTransform(-1,1)`
  bounds, `transform="logit"` enforcement, and an initial-in-interval check;
  `_model_hyperparameters` yields an AR1 ρ.
- Docs: the AR1 contract, the τ-is-marginal-precision point, the regular-spacing
  assumption and its NaN-row workaround, and the roadmap update.

### Excluded (deferred / roadmap)
- **Irregular time spacing** (`ρ^{Δt}` and its non-uniform precision).
- **Config-file (`ModelConfig`) `ar1` type** — deferred for consistency with
  every spatial effect, even though AR1 would be easier (no graph to
  serialize).
- Seasonal / higher-order AR(p) effects.
- Group-wise AR1 (a separate series per panel unit) — a single shared series
  over the index levels this slice.

## Architecture
- `effects/ar1.py` (new): `ar1_structure(level_count, rho) -> np.ndarray`
  (the `T/(1−ρ²)` matrix) and `build_ar1(...)`. Reuses
  `ordered_observed_levels`; numpy/scipy only.
- `effects/spec.py`: `AR1` dataclass, added to `EffectSpec`, both `Predictor`
  isinstance tuples, and `__all__`.
- `compiler.py`: `AR1` branches in `compile_lgm` and `compile_family`
  (mirroring `ProperCAR`), and in `_model_hyperparameters`.
- `pylgm/__init__.py`, `effects/__init__.py`: exports.
- No changes to engines, families, INLA strategies, or the prediction module.

## Errors
- `rho` float outside `(−1, 1)`: `ValueError`.
- A `Hyperparameter` `rho` without `transform="logit"`: `CompilationError`
  naming the required transform.
- A ρ `Hyperparameter` whose `initial` lies outside the inset interval:
  `CompilationError` naming the interval.
- Fewer than two ordered levels: `ValueError`.
- Non-finite assembled precision: `NumericalError`.

## Testing and validation
1. **Precision oracle:** `inv(build_ar1(...).precision)` equals
   `(1/τ)·ρ^{|i−j|}` for several `(n, τ, ρ)` including negative ρ — an
   independent closed form, not a restatement of the construction.
2. **Marginal precision:** `diag(inv(Q)) == 1/τ` exactly; `ρ = 0` gives exactly
   `τI`.
3. **Structure:** symmetric, positive definite across ρ ∈ {−0.99, 0, 0.99};
   **zero constraint rows**; one-hot design; labels aligned to ordered levels;
   fewer than two levels raises.
4. **Spec:** frozen/hashable, ρ float validation, `Hyperparameter` ρ accepted,
   composition, duplicate-name rejection.
5. **End-to-end:** Gaussian and Poisson fits with fixed ρ; ρ estimated by
   empirical Bayes and integrated by INLA jointly with τ; **ρ recovery tracks
   the simulated truth** (data simulated from an AR1 with a known ρ, multi-seed
   as for proper CAR's ρ, asserting strong-vs-weak separation).
6. **All three latent strategies** accept AR1 (contrast: RW1 is rejected by full
   Laplace for carrying a constraint).
7. **Prediction:** `predict()` scores known time levels; an unseen level raises
   the NaN-row-directed error; a fit-row round trip reproduces the fit-row
   predictions.
8. **Non-logit ρ rejected**; regression: existing effects and paths unchanged.

## Acceptance criteria
1. `AR1(name, index, precision, rho)` composes like any effect and fits under
   plug-in, optimise, and integrate, for Gaussian and non-Gaussian likelihoods,
   implementing `Q = τ/(1−ρ²)·T`. (Non-Gaussian + integrate was unreliable when
   this slice shipped, for a pre-existing reason affecting every structured
   effect; fixed by the Laplace Newton stall rescue — see
   `2026-08-24-pylgm-newton-stall-rescue-design.md`.)
2. `inv(Q)` equals `(1/τ)ρ^{|i−j|}`; τ is the marginal precision; `ρ=0` gives `τI`.
3. ρ declared as a `Hyperparameter(transform="logit")` is estimated and
   integrated jointly with τ, and recovery tracks simulated truth.
4. AR1 is unconstrained and works under all three latent strategies;
   `predict()` works, with unseen levels directed to the NaN-row workflow.
5. No new runtime dependency; no engine/family/INLA changes; full suite green.
6. Irregular spacing, the config-file `ar1` type, AR(p)/seasonal, and group-wise
   AR1 remain deferred and recorded.

# Directed-Network SAR & Dynamic Spatial Panel (SDPD) — Design

**Date:** 2026-08-29
**Roadmap item:** Next #2 — "Directed & dynamic network structure."

## Goal

Add two latent effects to pyLGM for directed economic influence that
symmetrized CAR discards:

1. **`SAR`** — a cross-sectional spatial-autoregressive field on a **directed**
   weight matrix `W` (firm ownership, interbank exposure, supply chains).
2. **`DynamicSpatialPanel`** — the dynamic spatial panel (SDPD: Yu–Lee–Anselin),
   a time-indexed sequence of directed networks `W_t` with contemporaneous
   spatial, pure-temporal, and spatio-temporal-diffusion coefficients, plus a
   forward **forecast** for future periods.

Both fit on the existing Laplace/Gaussian engine — dense **and** E-sparse — with
no engine changes.

## The one construction

Everything is `Q = τ · MᵀM` for a sparse operator `M`, giving a symmetric,
positive-definite, **proper** (full-rank, no intrinsic null space) latent
precision. This is the "square the root operator" identity already used
implicitly; here it is explicit.

### Static SAR
Model: `x = ρ W x + ε`, `ε ~ N(0, τ⁻¹ I)` ⇒ `(I − ρW) x = ε`. So

```
M = I − ρW ,   Q = τ · MᵀM = τ (I − ρW)ᵀ(I − ρW).
```

### Dynamic SDPD
Model per period:

```
(I − ρ W_t) x_t  =  γ x_{t-1}  +  η W_t x_{t-1}  +  ε_t ,   ε_t ~ N(0, τ⁻¹ I)
   contemporaneous     temporal        spatio-temporal
```

Write `A_t = I − ρW_t`, `B_t = γI + η W_t`. Then `A_t x_t − B_t x_{t-1} = ε_t`
stacks into one **block-bidiagonal** operator over the stacked field
`x = (x_1, …, x_T)`:

```
        ⎡ A_1                       ⎤
        ⎢ −B_2  A_2                 ⎥
M   =   ⎢       −B_3  A_3           ⎥          Q = τ · MᵀM   (block-tridiagonal)
        ⎢              ⋱     ⋱      ⎥
        ⎣                   −B_T  A_T⎦
```

- **Initial condition:** conditional-on-initial — the first block row is `A_1`
  only (no lag term). Standard SDPD treatment.
- `T = 1, γ = η = 0` ⇒ `M = A_1 = I − ρW` ⇒ **identical to static `SAR`**
  (verified by test, not duplicated code).
- `η = 0` ⇒ AR-only temporal link.

`M` is block-triangular, so `det M = ∏ det A_t`. `M` is nonsingular — hence `Q`
is PD — exactly when every `A_t` is nonsingular. `B_t` (γ, η) never touches the
diagonal, so it cannot make `M` singular: **γ and η are free reals**, only `ρ`
is bounded.

## Directed weight matrix

- **Row-standardized by default** (each row sums to 1) — the econometric
  standard for SAR/SDPD, and the reason `ρ ∈ (−1, 1)` is a fixed bound (a
  row-stochastic `W` has spectral radius 1, so `I − ρW` is nonsingular for
  `|ρ| < 1`) needing **no per-graph eigensolve**, unlike ProperCAR.
- A **zero-out-degree** unit (no outgoing edges) stays a zero row → its `M` row
  is `eᵢ`, still nonsingular; that unit simply has no spatial parents.
- Raw (un-normalized) directed `W`, which would need a complex-spectrum validity
  interval, is **deferred**.

`normalize_graph`/`canonical_graph` (effects/graph.py) **enforce symmetry** and
cannot be reused for directed `W`. New module `effects/directed_graph.py`:
`normalize_directed_graph(graph) -> (nodes, W)` (no symmetry check) and
`row_standardize(W)`. The one-hot `design_from_graph` (graph.py) is reused
unchanged — it never required symmetry.

## Effect specs (frozen dataclasses, effects/spec.py)

```python
SAR(name, index, graph, rho, precision=1.0)
DynamicSpatialPanel(name, unit, time, graphs, rho, gamma, eta, precision=1.0)
```

- Each of `rho, gamma, eta, precision` may be a fixed float or a
  `Hyperparameter`. `rho` fixed must satisfy `-1 < rho < 1` (mirrors `AR1`).
- `graph` (SAR) is a directed `{node: {neighbour: weight}}` (or `{node:[…]}`
  unweighted). `graphs` (panel) is `{time_value: graph}`.
- Shared internal builder core `_gram_precision(M, precision) -> τ·(Mᵀ@M)`.

## Latent grid (Decision — balanced)

`DynamicSpatialPanel`'s latent domain is the **balanced units × periods grid**
(the same unit set in every period), labels `f"{unit}@{time}"`. Observed *data*
may be unbalanced — handled by the existing `observed` (`.notna()`) mask.
Units entering/leaving the network across periods is **deferred**. This mirrors
`SpaceTime`'s full space×time grid assumption.

## Compiler wiring (compiler.py)

Two branches per effect, mirroring the AR1/ProperCAR `rho` pattern:

- **All of ρ, γ, η fixed** → `_compiled_block` + `ScalableBlock(τ)`.
- **Any a `Hyperparameter`** → `ParametricBlock(template, (τ, ρ, γ, η), build)`
  whose `build(values)` rebuilds `τ·MᵀM`. Transforms: `ρ` → Logit(−1, 1)
  (`_bounded_parameter(effect.rho, -1.0, 1.0, inset=1e-6)`, same call as AR1);
  `γ, η` → identity (unbounded, precedent: MIDASParametric exp-Almon shapes);
  `τ` → log. Plus the hyperparameter-discovery branch (the
  `isinstance(effect, ProperCAR) and isinstance(effect.rho, Hyperparameter)`
  site) extended for the new effects' ρ/γ/η.

Constraints: `np.empty((0, n))` (proper field — no scaling, no sum-to-zero),
exactly like `build_proper_car`/`build_ar1`.

## E-sparse (mostly free)

The sparse solver is size-routed (`_exceeds_dense_threshold`, gaussian.py) and
**effect-agnostic**. A SAR/SDPD block with the incidence-style
`design_from_graph` design is automatically sparse-eligible; selected-inverse
marginals, predictive/linear-combination variances, and INLA grid mixing all
work generically off the sparse SPD precision. No new sparse machinery.

**Documented limitation (pre-existing):** above the sparse guard only
`latent_strategy="gaussian"` INLA marginals are available (simplified/full
Laplace are dense-only). Not solved here.

## Prediction & forecasting

- **In-grid prediction:** `result.predict(new_data)` for units/periods already
  in the fitted latent grid (+ fixed covariates) — reuses the existing path.
- **Forecast (SDPD capstone):** extend `M` forward. Given the fitted posterior
  mean `x̂_T` and the next period's directed network `W_{T+1}` (+ covariates),

  ```
  x̂_{t+1} = A_{t+1}⁻¹ B_{t+1} x̂_t                                (mean)
  Var(x_{t+1}) = A_{t+1}⁻¹ [ B_{t+1} Var(x_t) B_{t+1}ᵀ + τ⁻¹ I ] A_{t+1}⁻ᵀ
  ```

  a Kalman-style forward pass, iterated for horizon `h ≥ 1`. `A_{t+1}` is sparse
  (solve via factorization). Above the sparse guard, variance is propagated on
  the **diagonal** (marginal) only, consistent with the gaussian latent
  strategy. Returns forecast mean + marginal variance per (unit, future-period).

## YAML frontend (Decision — SAR only in v1)

`sar` config family (directed graph inline or `.json`; validators mirror
ProperCAR + reject fields on the wrong family). `DynamicSpatialPanel` stays
**Python-API-only in v1** — per-period graph files plus three coefficients make
its YAML schema a slice of its own (and the roadmap lists a config-file effort
separately).

## Testing (TDD)

- `normalize_directed_graph`: asymmetric `W` accepted; row sums = 1;
  zero-out-degree row preserved; self-loop / unknown-neighbour rejected.
- `build_sar`: `Q = τ(I−ρW)ᵀ(I−ρW)`, symmetric PD, empty constraints;
  `|ρ| ≥ 1` rejected; design maps observed rows to nodes.
- SAR fit: EB recovers a planted `ρ`; fixed-`ρ` fit; dense path.
- `build_dynamic_spatial_panel`: `T=1` ⇒ SAR precision (equivalence);
  block-tridiagonal structure correct; `η=0` structure; PD; balanced grid.
- SDPD fit: EB recovers planted `(ρ, γ, η)`; `η=0` fit ≡ AR-link.
- **Dense-vs-sparse agreement** on a small case; a large network fits past the
  dense guard; sparse marginal variances finite.
- Prediction for in-grid rows matches the fitted latent.
- **Forecast:** 1-step mean matches `A⁻¹B x̂_T` on a seeded case; variance
  positive and sane; multi-step iterates.
- YAML round-trip for `SAR`; bad-field rejection.
- No new runtime dependencies.

## Docs

`docs/spatial-effects.md` — SAR/SDPD section (directed `W`, the `MᵀM`
construction, ρ/γ/η reading, balanced-grid + gaussian-only-sparse-marginals
notes, forecasting). A runnable `examples/` (interbank-exposure or
firm-ownership network). `docs/roadmap.md` — Next #2 → Shipped.

## Out of scope (deferred)

- Raw un-normalized directed `W` (complex-spectrum validity interval).
- Units entering/leaving the network across periods (unbalanced latent grid).
- Simplified/full-Laplace INLA marginals above the sparse guard (pre-existing).
- `DynamicSpatialPanel` YAML frontend.

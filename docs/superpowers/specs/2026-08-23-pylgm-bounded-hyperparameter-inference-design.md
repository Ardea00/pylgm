# pyLGM Bounded-Hyperparameter Inference Design

**Status:** Approved 2026-08-23

## Purpose

Generalize pyLGM's hyperparameter inference from **positive-only, log-scale**
parameters to **arbitrary bounded** parameters via a per-parameter transform
abstraction, and use it to enable **estimation of the proper-CAR spatial
dependence ρ** (empirical-Bayes optimise and INLA integrate). This is the shared
prerequisite for BYM2's mixing parameter φ, which reuses the same machinery.

Today every declared `Hyperparameter` is strictly positive and inferred in
log space: the empirical-Bayes optimiser maps `u = log θ` (empirical_bayes.py)
and the INLA grid integrates in `u = log θ` with importance-weight Jacobian
`Σ u` (inla.py:298, since `dθ/du = θ ⇒ log|dθ/du| = u`). ρ is bounded to a
graph-dependent open interval and can be negative, so it does not fit the log
map. This slice introduces a transform per parameter, threads it through the
optimiser and the INLA grid (including the Jacobian), adds a parametric
precision block (because `Q = τ(D − ρW)` with ρ free is not a scalar multiple
of a fixed matrix), and wires ρ through `ProperCAR`.

**References:**
- Rue, Martino & Chopin (2009) §3.1 — INLA grid integration over the
  hyperparameter posterior in a transformed (θ→u) parameterization.
- Gelfand & Vounatsou (2003); Banerjee, Carlin & Gelfand (2014) §4 — proper CAR
  and the validity interval for ρ.

## Scope decisions (from brainstorming)
- **Full ρ estimation**: both empirical-Bayes optimise and INLA integrate over
  `(τ, ρ)` for `ProperCAR`.
- **Scaled-logit transform** for bounded parameters (reused verbatim for BYM2 φ).
- The concrete deliverable is ρ estimation for `ProperCAR`; φ/BYM2 is a later
  slice that reuses this machinery.

## Components

### 1. Transform abstraction (`optimization/transforms.py`, new)

A scalar transform between the natural scale θ and the unconstrained internal
scale `u`:

```
Transform:
  to_internal(theta) -> u
  from_internal(u) -> theta
  log_abs_jacobian(u) -> log|d theta / d u|
  contains(theta) -> bool          # theta in the natural domain
```

- **`LogTransform`** (positive params τ, σ): `u = log θ`, `θ = exp u`,
  `log_abs_jacobian(u) = u`, domain `(0, ∞)`.
- **`LogitTransform(a, b)`** (bounded interval, `a < b`): `θ = a + (b − a)·σ(u)`,
  `u = log((θ − a)/(b − θ))`, `log_abs_jacobian(u) = log(b − a) + log σ(u) +
  log(1 − σ(u))`, domain `(a, b)`. Uses `scipy.special.expit` / `log_expit` for
  numerical stability.

`LogTransform.log_abs_jacobian(u) = u` is the **back-compatibility anchor**: the
optimiser and INLA grid reduce to their current behaviour exactly for all-log
models.

### 2. Transform-aware bounds (`optimization/empirical_bayes.py`)

`OptimizationBounds` gains a `transform: Transform = LogTransform()` field.
Validation is delegated to the transform: `transform.contains(lower)`,
`transform.contains(upper)`, `lower < initial < upper`, and
`to_internal(lower) < to_internal(upper)` finite (so the L-BFGS box is
well-posed). For `LogitTransform(a,b)`, `lower`/`upper` must be strictly inside
`(a, b)` (an internal box of `±∞` is rejected). Default `LogTransform` keeps the
current positive-only validation semantics.

### 3. Parametric precision block (`ir/family.py`)

`ScalableBlock` (scalar multiply `precision = base · θ`) cannot represent
`τ(D − ρW)` with ρ free. Add:

```
ParametricBlock:
  block: LatentBlock                 # template: design, labels, constraints,
                                     #   and a valid precision at the initial params
  parameters: tuple[str, ...]        # hyperparameter names it consumes
  build: Callable[[Mapping[str, float]], csr_matrix]   # -> precision at those params
  materialize(resolved) -> LatentBlock                 # block with build(resolved)
```

`_materialize_blocks` dispatches: `ScalableBlock` → scalar multiply (unchanged);
`ParametricBlock` → `build(resolved)`. `CompiledFamily` /
`CompiledGaussianFamily` accept a `ScalableBlock | ParametricBlock` list and
validate that each `ParametricBlock.parameters ⊆ parameter_names`. Existing
all-`ScalableBlock` families are byte-identical.

For proper CAR the builder is
`build(v) = tau_value(v) · (D − rho_value(v)·W)`, where `tau_value`/`rho_value`
read either a fixed constant or `v[name]` (so τ and ρ may each be fixed or
free). `D`, `W` are captured once at compile time.

### 4. Family carries `parameter_bounds`

`CompiledFamily` / `CompiledGaussianFamily` gain
`parameter_bounds: Mapping[str, OptimizationBounds]`, populated by the compiler.
The compiler is the only place that knows ρ's graph interval, so it builds:
- τ, σ → `OptimizationBounds(..., transform=LogTransform())` from the
  `Hyperparameter`'s `lower/initial/upper` (identical to today).
- ρ → `OptimizationBounds(lower', initial, upper', transform=LogitTransform(a,b))`
  where `(a, b)` is the graph validity interval and `lower'/upper'` are the user
  sub-bounds (if any) intersected with a small inset of `(a, b)`.

`model.py` reads `family.parameter_bounds` instead of rebuilding bounds from
`hp.lower/upper`. When a family predates this field (defensive), it falls back to
log bounds from the hyperparameters.

### 5. Optimiser + INLA generalization

- **`optimize_empirical_bayes`**: per-parameter `transform.to_internal` for the
  start/box, `transform.from_internal` in `_parameter_values`, and
  `transform.contains` for validity (replacing the hard-coded `np.log`/`np.exp`
  and `<= 0` check). The internal-space L-BFGS box and bound snapping are
  unchanged in structure.
- **`integrate_inla`**: `u_star = to_internal(θ*)`; `evaluate(u)` uses
  `from_internal`; the in-domain check uses internal bounds; and the
  importance-weight Jacobian becomes
  `log_weights = s + Σ_i transform_i.log_abs_jacobian(u_i)` (was `s + Σ u`).
  `integrated_lml`, grid construction, and the Hessian are unchanged (they
  already work in internal `u`-space).

### 6. `Hyperparameter` generalization + ρ wiring

- **`Hyperparameter`**: `transform` accepts `"logit"` in addition to
  `"log"`/`"identity"`. For non-`"log"` transforms `initial` may be any finite
  real and `lower/upper` may be `None` (interval deferred to the effect). `"log"`
  keeps the positive-only semantics and default bounds.
- **`ProperCAR`**: remove the "rho must be a float" rejection; accept
  `rho=Hyperparameter(...)` (its `transform` defaults to `"logit"` in this
  context). Plug-in ρ (a float) is unchanged.
- **`compiler.py`**: `_model_hyperparameters` also collects a `ProperCAR`'s ρ
  when it is a `Hyperparameter`. `compile_family`'s `ProperCAR` branch: if ρ is a
  `Hyperparameter`, compute `(a, b)` via proper CAR's `_validity_interval`, emit a
  `ParametricBlock` (τ fixed-or-bound, ρ bound), and register ρ's
  `OptimizationBounds(LogitTransform(a,b))` in `parameter_bounds`; else keep the
  `ScalableBlock` plug-in path. τ handling is unchanged.

## Public contract

```python
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, ProperCAR

rho = Hyperparameter("region.rho", initial=0.5, transform="logit")   # interval from graph
tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(...))
model = LGM(response="y",
            predictor=Fixed("1") + ProperCAR("region", index="region",
                                             graph=graph, rho=rho, precision=tau),
            likelihood=Gaussian(sigma=0.1))
eb = model.fit(frame)                              # empirical-Bayes over (tau, rho)
eb.hyperparameters["region.rho"]                   # estimated rho
post = model.fit(frame, hyperparameters="integrate")   # INLA over (tau, rho)
post.hyperparameter_marginals()["region.rho"]      # rho marginal
```

- `Hyperparameter(name, initial, prior=None, transform="log"|"logit"|"identity",
  lower=None, upper=None)`; `"logit"` allows any finite `initial` and a deferred
  interval.
- `ProperCAR(..., rho=<float or Hyperparameter>)`; a `Hyperparameter` ρ is
  estimated/integrated over its graph validity interval; a float ρ is plug-in
  (unchanged).
- All existing positive/log hyperparameters (σ, effect precisions) behave
  identically — the log path is the exact special case.

## Scope

### Included
- `optimization/transforms.py`: `Transform` protocol, `LogTransform`,
  `LogitTransform`.
- `OptimizationBounds.transform`; transform-aware validation.
- `ParametricBlock` in `ir/family.py`; `_materialize_blocks` + family validators
  generalized to `ScalableBlock | ParametricBlock`.
- `parameter_bounds` on the compiled families; `model.py` reads it.
- `empirical_bayes.py` and `inla.py` generalized to per-parameter transforms
  (INLA Jacobian included).
- `Hyperparameter` `"logit"` transform + non-positive/deferred bounds.
- `ProperCAR` ρ-as-Hyperparameter; `compiler.py` `_model_hyperparameters` +
  `compile_family` ρ path (ParametricBlock + logit bounds from the graph).
- Docs + roadmap.

### Excluded (deferred / roadmap)
- **BYM2** (φ mixing, structured+unstructured) — next slice; reuses this
  machinery (`LogitTransform(0,1)` for φ + its own parametric block).
- ρ estimation for models with intrinsic/constrained spatial effects combined in
  ways beyond proper CAR.
- A fully general parametric-block *framework* (multiple unrelated parametric
  effects, shared base class) — extract only when BYM2 provides the second real
  instance.
- Non-scalar / correlated transforms; the `"identity"` transform remains a
  declared-but-unused option except where a bounded transform is needed.
- Config-file (`ModelConfig`) support for ρ hyperparameters.

## Architecture
- `optimization/transforms.py` (new): the transforms. Pure, numpy/scipy only.
- `optimization/empirical_bayes.py`: `OptimizationBounds.transform`;
  transform-driven `_validate_problem`, `_parameter_values`, and the internal
  start/box.
- `optimization/inla.py`: transform-driven `evaluate`/`u_star`/in-domain; the
  Jacobian sum.
- `ir/family.py`: `ParametricBlock`; `_materialize_blocks` dispatch; family
  validators accept the union; families store `parameter_bounds`.
- `parameters.py`: `Hyperparameter` `"logit"` transform + relaxed bounds.
- `compiler.py`: collect ρ hyperparameters; build `ParametricBlock` + ρ bounds in
  `compile_family`; populate `parameter_bounds`.
- `effects/proper_car.py`: expose `_validity_interval` (or a public helper) for
  the compiler; the builder is unchanged for plug-in ρ.
- `model.py`: `_family_optimization_inputs` reads `family.parameter_bounds`.
- No engine (`fit_gaussian`/`fit_laplace`) or latent-strategy changes.

## Errors
- `LogitTransform` with `a >= b`, or `θ`/bounds outside `(a, b)`: `ValueError`.
- ρ `Hyperparameter` `initial` outside the graph interval (after resolution):
  `ValueError` naming the interval.
- `Hyperparameter(transform="logit")` used where no interval can be resolved
  (e.g. not attached to a spatial effect that supplies one): `ValueError`.
- Non-finite transform outputs / Jacobian: `NumericalError`.
- Existing positive/log validation errors unchanged.

## Testing and validation
1. **Back-compat (critical):** the full existing suite passes unchanged; add a
   targeted test asserting an all-log optimise and integrate produce results
   numerically identical (to tight tolerance) to the pre-change behaviour — the
   `LogTransform` path must be a no-op refactor. (Anchor: `log_abs_jacobian(u)=u`
   ⇒ the INLA Jacobian `Σ u` is preserved.)
2. **Transforms:** round-trip `from_internal(to_internal(θ)) == θ` for Log and
   Logit; `log_abs_jacobian` matches a finite-difference of `log|dθ/du|`;
   `LogitTransform` outputs stay in `(a, b)`; `contains` correct at/just outside
   bounds.
3. **OptimizationBounds:** transform-aware validation (Logit rejects bounds
   outside `(a,b)`; Log rejects non-positive); default is Log.
4. **ParametricBlock:** `materialize` yields `τ(D − ρW)` at given `(τ, ρ)`;
   family holds a `ParametricBlock` beside `ScalableBlock`s and materialises the
   correct block-diagonal precision; validators accept it.
5. **Optimiser with a bounded parameter:** on a synthetic family with a
   `LogitTransform` parameter, the optimiser recovers a known interior optimum
   and respects the interval.
6. **INLA with a bounded parameter:** the grid integrates a `LogitTransform`
   parameter with the correct Jacobian (compare the integrated marginal moments
   to a fine brute-force quadrature of the same 1-D posterior).
7. **ρ recovery end-to-end:** data simulated with a known strong-dependence ρ —
   `ProperCAR(rho=Hyperparameter(...))` fit by EB recovers ρ near truth; INLA
   integrate returns a ρ marginal with mean inside the interval and nonzero
   variance; τ estimated jointly; predictions/criteria populated.
8. **Interval resolution:** ρ bounds equal the graph's `(1/μ_min, 1/μ_max)`
   (inset); an out-of-interval `initial` errors; a prior on ρ penalizes the
   objective on the natural scale.
9. **Plug-in ρ unchanged:** `ProperCAR(rho=0.9)` still uses the ScalableBlock
   path and is unaffected.

## Acceptance criteria
1. A per-parameter transform abstraction (`LogTransform`, `LogitTransform`) drives
   both the empirical-Bayes optimiser and the INLA grid, with the correct
   importance-weight Jacobian; the all-log path is numerically identical to the
   current behaviour (full suite green, plus an explicit equivalence test).
2. `ProperCAR(rho=Hyperparameter(...))` is estimated by empirical Bayes and
   integrated over by INLA, jointly with τ, over the graph validity interval;
   `result.hyperparameters` / `hyperparameter_marginals()` report ρ.
3. A `ParametricBlock` represents `τ(D − ρW)` and coexists with `ScalableBlock`s
   in a compiled family; existing effects are unchanged.
4. ρ recovery on simulated data is near truth (optimise) with a sensible marginal
   (integrate); plug-in ρ and all existing hyperparameters are unaffected.
5. No new runtime dependency (scipy `expit`/`log_expit` already available); full
   suite green; engines and latent strategies unchanged.
6. BYM2 (φ), a general parametric-block framework, and config-file ρ remain
   deferred and recorded.

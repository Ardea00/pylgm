# pyLGM BYM2 Spatial Effect Design

**Status:** Approved 2026-08-23

## Purpose

Add the **BYM2** spatial latent effect (Riebler, Sørbye, Simpson & Rue 2016) —
the reparameterized Besag–York–Mollié convolution of a *structured* (scaled
ICAR) and an *unstructured* (IID) component, mixed by `φ ∈ (0,1)` with a single
marginal precision `τ` — together with the paper's **penalized-complexity (PC)
prior for φ**. This completes the spatial CAR family (Besag/ICAR → proper CAR →
BYM2).

BYM2 is the natural target now that the prerequisites exist: Besag's
Sørbye–Rue scaling supplies the structured component, `LogitTransform(0,1)`
supplies φ's bounded inference, and `ParametricBlock` carries a precision that
is not a scalar multiple of a fixed matrix.

**References:**
- Riebler, A., Sørbye, S., Simpson, D., Rue, H. (2016), *An intuitive Bayesian
  spatial model for disease mapping that accounts for scaling*, Statistical
  Methods in Medical Research 25(4):1145–1165 — the BYM2 parameterization and
  the PC prior for φ.
- Simpson, D., Rue, H., Riebler, A., Martins, T., Sørbye, S. (2017),
  *Penalising model component complexity*, Statistical Science 32(1):1–28 —
  the PC-prior construction (KLD distance, exponential on the distance scale).
- Sørbye & Rue (2014) — the ICAR scaling BYM2 requires (already shipped).

## Model

For a neighbourhood graph over `n` regions with scaled ICAR structure `R*`
(Besag's `scale=True` structure), the BYM2 effect is

  `x = τ^{-1/2} ( √(1−φ)·v + √φ·u* )`,  `v ~ N(0, I)`,  `u* ~ ICAR(R*)`,

so that `τ` is the **marginal** precision of `x` and `φ` is the fraction of the
marginal variance that is spatially structured (`φ=0` pure IID, `φ→1` pure
spatial). The Sørbye–Rue scaling is what makes `φ` interpretable and comparable
across graphs — BYM2 therefore **always** uses the scaled structure (no
`scale` flag).

### Parameterization: marginal (n-dimensional)

With `P = R*⁻` the constrained generalized inverse of the scaled structure
(`pinv`, per connected component — already computed for scaling),

  `Cov(x) = τ⁻¹ [ (1−φ)I + φP ]`,   `Q(τ, φ) = τ [ (1−φ)I + φP ]⁻¹`.

Eigendecompose once at compile time, `P = U diag(γ) Uᵀ` (symmetric PSD), so

  `Q(τ, φ) = τ · U diag( 1 / (1 − φ + φγ) ) Uᵀ`,

an **O(n²) reassembly per (τ, φ)** after a single O(n³) decomposition. `τ`
remains a pure scalar multiplier.

**This precision is full-rank (proper), so a BYM2 effect carries no
constraints** — unlike Besag it works under *all three* latent strategies,
including full Laplace, and disconnected graphs need no special handling
(the `c` null directions of `P` contribute eigenvalue `1/(1−φ)`, finite for
`φ<1`).

**Boundary.** As `φ→1` the null-direction precision `1/(1−φ)` diverges (the
`φ=1` limit forces `x` to sum to zero per component). φ is therefore bounded
strictly inside `(0,1)` with a `1e-6` relative inset, matching proper CAR's ρ
handling; users wanting a better-conditioned fit can declare a tighter `upper`.

**Deferred: the augmented (2n) representation.** R-INLA represents BYM2 with the
latent field `(x, u*)` and a coupled sparse precision, which additionally
exposes the structured component `u*` for mapping. That is the *same model* —
a different computational representation — and is recorded as a follow-on
slice; this slice ships the marginal form only.

### PC prior for φ (Riebler et al. §3.3)

The base model is `φ = 0` (no spatial structure). On the subspace orthogonal to
the `c` null directions of `P` — where both models are proper, which keeps
`d(1)` finite so `φ=1` remains attainable — with `γ_j` the **positive**
eigenvalues of `P`:

  `KLD(φ) = ½ Σ_j [ φ(γ_j − 1) − ln(1 − φ + φγ_j) ]`,   `d(φ) = √(2·KLD(φ))`.

Each term is `t − 1 − ln t ≥ 0` for `t = 1 − φ + φγ_j > 0`, so `KLD(0) = 0` and
`d` is non-negative and strictly increasing (verified numerically).

Because `d(1)` is **finite**, the PC prior is an exponential on the distance
scale **truncated** to `[0, d(1)]`:

  `π(φ) = λ e^{−λ d(φ)} d′(φ) / (1 − e^{−λ d(1)})`,
  `d′(φ) = K′(φ)/d(φ)`,  `K′(φ) = ½ Σ_j (γ_j − 1)[1 − 1/(1 − φ + φγ_j)]`.

At `φ→0` both `K′` and `d` vanish; the limit is `d′(0) = √(S/2)` with
`S = Σ_j (γ_j − 1)²` (series expansion, verified against finite differences), so
the implementation uses that closed form below a small threshold.

**Calibration.** The user declares `P(φ < U) = α` (defaults `U = 0.5`,
`α = 2/3`, matching R-INLA's BYM2 default). λ solves

  `(1 − e^{−λ d(U)}) / (1 − e^{−λ d(1)}) = α`

by 1-D root finding (no closed form, because of the truncation). The attainable
range is `α ∈ (d(U)/d(1), 1)`; an `α` at or below `d(U)/d(1)` is unattainable
for that graph and raises a `ValueError` naming the achievable range.

**Graph dependence.** `d` depends on the graph's eigenvalues, which are unknown
when the user declares the prior. `PCBYM2Phi(upper, alpha)` is therefore a
*declaration* that the compiler **binds** to the effect's graph — mirroring how
proper CAR's ρ interval is resolved at compile time. An unbound prior raises a
directed error if its `logpdf` is called.

## Public contract

```python
from pylgm import BYM2, Fixed, Gaussian, Hyperparameter, LGM, PCBYM2Phi, PCPrecision

graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}

# fixed mixing
BYM2("region", index="region", graph=graph, phi=0.7, precision=1.0)

# estimated mixing with the paper's PC prior
phi = Hyperparameter("region.phi", initial=0.5, transform="logit",
                     prior=PCBYM2Phi(upper=0.5, alpha=2/3))
tau = Hyperparameter("region.precision", initial=1.0,
                     prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(response="y",
            predictor=Fixed("1") + BYM2("region", index="region", graph=graph,
                                        phi=phi, precision=tau),
            likelihood=Gaussian(sigma=0.1))
eb = model.fit(frame)                                    # empirical Bayes over (tau, phi)
eb.hyperparameters["region.phi"]
post = model.fit(frame, hyperparameters="integrate")     # INLA over (tau, phi)
post.hyperparameter_marginals()["region.phi"].mean
```

- `BYM2(name, index, graph, precision=1.0, phi=0.5)`; `phi` is a float in
  `(0,1)` or a `Hyperparameter` with `transform="logit"` (enforced, as for ρ).
- `precision` (τ) is a float or `Hyperparameter`, exactly as for other effects.
- Adjacency input is the shared neighbour-dict / `load_graph_file` path.
- Unconstrained ⇒ works under `latent_strategy` `"gaussian"`,
  `"simplified_laplace"`, and `"laplace"`.
- `PCBYM2Phi(upper=0.5, alpha=2/3)` — the PC prior declaration for φ.

## Scope

### Included
- `priors.PCBYM2Phi`: the declaration plus `bind(gamma)` returning a bound prior
  with `logpdf`; λ calibration by root finding; the `d′(0)` series limit;
  unattainable-α and unbound-use errors.
- `effects/bym2.py`: `build_bym2(frame, name, index, graph, precision, phi)` →
  unconstrained `LatentBlock` with the marginal precision; plus a helper
  exposing `(eigenvectors, eigenvalues)` of `P` for the compiler and the prior.
- `BYM2` spec in `effects/spec.py`; exports from `pylgm.effects` and `pylgm`.
- Family `parameter_priors` (mirroring `parameter_bounds`), populated by the
  compiler with the **bound** PC prior; `model.py` prefers it when building the
  MAP-II / INLA penalty.
- Compiler wiring: fixed φ → `ScalableBlock` (τ multiplies a fixed matrix);
  `Hyperparameter` φ → `ParametricBlock` over (τ, φ) with `LogitTransform(0,1)`
  bounds (inset) and the bound PC prior.
- Docs + roadmap.

### Excluded (deferred / roadmap)
- **The augmented (2n) representation** exposing the structured component `u*`
  separately — recorded as the next spatial follow-up.
- Isolated (zero-neighbour) nodes: rejected, as for Besag (the scaled ICAR is
  undefined there); graceful handling stays deferred.
- Config-file (`ModelConfig`) `bym2` type.
- PC priors for anything beyond φ and the existing `PCPrecision`.
- Sparse/large-graph BYM2 (this slice is the dense reference regime: one O(n³)
  eigendecomposition at compile time).

## Architecture
- `priors.py`: `PCBYM2Phi` (declaration + `bind`), `_BoundPCBYM2Phi` (`logpdf`).
  Uses `scipy.optimize.brentq`; numpy only otherwise.
- `effects/bym2.py`: `bym2_spectrum(graph)` → `(U, γ)` from
  `pinv(_scaled_structure(...))`; `build_bym2(...)` assembling
  `τ·U diag(1/(1−φ+φγ)) Uᵀ`, empty constraints, shared `design_from_graph`.
  Reuses Besag's `_scaled_structure` (no duplicated scaling).
- `effects/spec.py`: `BYM2` dataclass (canonical graph storage as `Besag`/
  `ProperCAR`; φ float validated in `(0,1)` or a `Hyperparameter`).
- `ir/family.py`: `parameter_priors` field on both families (duck-typed, as
  `parameter_bounds`).
- `compiler.py`: `_model_hyperparameters` yields a BYM2 φ hyperparameter;
  `compile_family` BYM2 branch (fixed vs estimated φ), binding `PCBYM2Phi` to
  the graph spectrum and registering bounds + priors.
- `model.py`: `_family_optimization_inputs` prefers `family.parameter_priors`
  for the penalty, falling back to the hyperparameters' own priors.
- No engine or latent-strategy changes.

## Errors
- `phi` float outside `(0,1)`, or a `Hyperparameter` φ without
  `transform="logit"`: `ValueError` / `CompilationError` (mirroring ρ).
- φ `Hyperparameter` `initial` outside the inset interval: `CompilationError`
  naming the interval.
- `PCBYM2Phi` with `alpha` unattainable for the graph (`α ≤ d(U)/d(1)`):
  `ValueError` naming the achievable range; `upper` outside `(0,1)` or
  `alpha` outside `(0,1)`: `ValueError`.
- `PCBYM2Phi.logpdf` called unbound: `ValueError` directing to attach it to a
  BYM2 φ.
- Isolated nodes / observed region absent from the graph: as Besag.
- Non-finite assembled precision: `NumericalError`.

## Testing and validation
1. **PC-φ prior math** (the numerical core, validated without R-INLA):
   `KLD(0)=0`; `d` strictly increasing; `d(1)` finite; the calibrated density
   integrates to **1** over `(0,1)` by quadrature; `P(φ<U)=α` by quadrature for
   several `(U, α)`; `d′(0)` matches a finite difference; unattainable `α`
   raises naming the range; unbound `logpdf` raises.
2. **Spectrum/builder**: `build_bym2` precision equals
   `τ·[(1−φ)I + φ·pinv(R*)]⁻¹` computed directly (independent oracle); symmetric
   positive definite; **zero constraint rows**; `φ→0` recovers `τI`;
   design/labels align; disconnected graphs work; isolated nodes rejected.
3. **Marginal-variance check**: with `φ` fixed and `τ=1`, the geometric mean of
   `diag(Cov(x))` behaves as the scaling intends (φ=0 gives exactly 1).
4. **Spec**: `BYM2` frozen/hashable, canonical graph round-trip, φ float
   validation, `Hyperparameter` φ accepted, composition, duplicate-name.
5. **End-to-end**: Gaussian and Poisson BYM2 fits (plug-in φ); φ estimated by
   empirical Bayes and integrated by INLA jointly with τ; **φ recovery tracks
   the simulated truth** (strong-spatial vs pure-IID data separate clearly,
   multi-seed as for ρ); the PC prior visibly regularizes φ.
6. **All three latent strategies accept BYM2** (contrast: Besag is rejected by
   full Laplace) — including tabulated marginals under `"laplace"`.
7. **Regression**: existing effects, priors, optimise/integrate paths unchanged;
   full suite green.

## Acceptance criteria
1. `BYM2(name, index, graph, precision, phi)` composes like any effect and fits
   under plug-in, optimise, and integrate for Gaussian and non-Gaussian
   likelihoods, implementing `Q = τ[(1−φ)I + φR*⁻]⁻¹` on the Sørbye–Rue-scaled
   structure.
2. φ is estimable: declared as a `Hyperparameter(transform="logit")` it is
   estimated by empirical Bayes and integrated over by INLA jointly with τ, and
   recovery tracks simulated truth.
3. `PCBYM2Phi(upper, alpha)` implements Riebler et al.'s PC prior for φ, bound
   to the effect's graph by the compiler; its density integrates to 1 and
   satisfies `P(φ<U)=α`; unattainable calibrations raise.
4. BYM2 is unconstrained and works under all three latent strategies.
5. No new runtime dependency; full suite green; existing effects, priors, and
   inference paths unchanged.
6. The augmented (2n) representation, config-file `bym2`, and graceful
   isolated-node handling remain deferred and recorded.

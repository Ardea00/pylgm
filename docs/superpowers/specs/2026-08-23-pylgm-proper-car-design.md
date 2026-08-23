# pyLGM Proper CAR (plug-in ρ) Spatial Effect Design

**Status:** Approved 2026-08-23

## Purpose

Add the **proper conditional autoregressive (proper CAR)** latent effect —
`Q = τ(D − ρW)` — the full-rank spatial prior with an explicit spatial-dependence
parameter ρ. This is the second slice of the spatial CAR family (after Besag/ICAR).

**Scope this slice: plug-in ρ.** ρ is a fixed value supplied by the user; the
precision τ remains a normal hyperparameter (optimise/integrate) because with ρ
fixed, `Q = τ·(D − ρW)` is a scalar multiple of the *fixed* base matrix
`M = D − ρW` — exactly the existing `ScalableBlock` pattern, needing **no
family-layer changes**. Estimating ρ (which requires a bounded, possibly-negative
hyperparameter transform across the empirical-Bayes optimiser and the INLA grid)
is deferred to a dedicated **bounded-hyperparameter inference** slice that also
unlocks BYM2's mixing parameter φ.

**References:**
- Cressie (1993); Gelfand & Vounatsou (2003) — proper CAR `τ(D − ρW)` and the
  positive-definiteness range for ρ.
- Banerjee, Carlin & Gelfand (2014), *Hierarchical Modeling and Analysis for
  Spatial Data*, §4 — the CAR family and validity conditions.

## Model

For a neighbourhood graph over `n` regions with symmetric 0/1 adjacency `W` and
degree matrix `D = diag(d_i)`, the proper-CAR prior on the latent field
`x ∈ ℝⁿ` is a proper GMRF

  `π(x | τ, ρ) ∝ |τ(D − ρW)|^{1/2} · exp(−τ/2 · xᵀ(D − ρW)x)`.

Unlike the intrinsic Besag/ICAR (`D − W`, rank `n − c`), `D − ρW` is
**full-rank / positive definite** for ρ in its validity interval, so the prior is
proper and needs **no sum-to-zero constraint**. ρ = 0 gives `Q = τD` (spatially
independent, degree-weighted); ρ → 1 approaches the singular intrinsic model.

### Validity range for ρ

`D − ρW ≻ 0` iff `I − ρ N ≻ 0` where `N = D^{−1/2} W D^{−1/2}` is the symmetric
normalized adjacency (requires `d_i > 0`, i.e. no isolated nodes). With `μ_min`,
`μ_max` the extreme eigenvalues of `N` (real; `μ_max = 1` for a graph with
edges, `μ_min ∈ [−1, 0)`), the condition `1 − ρμ > 0` for all μ gives the open
interval

  `ρ ∈ (1/μ_min, 1/μ_max)`   (typically `(negative, 1)`).

The builder computes this interval from `eigvalsh(N)` and validates the supplied
ρ against it, raising a clear error that names the valid interval when ρ is out
of range. (A bare Cholesky failure would be far less actionable.)

### Base matrix and τ

The builder emits the **fixed** base precision `M = D − ρW` (times the base τ, or
τ = 1 when the precision is an optimised `Hyperparameter`). Because ρ is fixed,
`Q = τ·M` is a scalar multiple of a constant matrix, so τ flows through the
existing `ScalableBlock.materialize` (`precision = block.precision · multiplier`)
for plug-in, empirical-Bayes optimise, and INLA integrate — identical to how
IID/RW/Besag precisions are handled. No new family machinery.

**No Sørbye–Rue scaling.** Scaling to unit generalized variance is an
*intrinsic*-model device; proper CAR is proper and has a well-defined variance,
so τ is used directly. (BYM2 will scale its *ICAR* component, not this one.)

## Adjacency input

Identical to Besag: a neighbour dict `{region: [neighbours]}` or
`load_graph_file(path)` for the R-INLA/latte `.graph` format, both normalised by
the existing `normalize_graph` to `(nodes, W)`. Domain = the graph nodes
(canonical sorted order); observed regions must be a subset of the nodes.

## Public contract

```python
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, ProperCAR

graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
predictor = Fixed("1") + ProperCAR("region", index="region", graph=graph, rho=0.9)
result = model.fit(frame)                                   # plug-in τ and ρ
# τ inferred, ρ fixed:
ProperCAR("region", index="region", graph=graph, rho=0.9,
          precision=Hyperparameter("region.precision", initial=1.0, prior=...))
result = model.fit(frame, hyperparameters="integrate")     # integrates over τ
```

- `ProperCAR(name, index, graph, rho, precision=1.0)`.
- `rho`: **required float** in the validity interval (no default). A
  `Hyperparameter` for `rho` raises a directed `ValueError` (ρ estimation is a
  later slice; pass a float).
- `precision` (τ): float or `Hyperparameter`, inferred exactly like IID/RW/Besag.
- Composes with `+`; participates in predictions, criteria, and all fit paths.
- **Works under every latent strategy**, including `latent_strategy="laplace"`
  (full Laplace) — proper CAR is the first *unconstrained* spatial effect, so the
  full-Laplace constrained-effect guard passes.

## Scope

### Included
- `ProperCAR` spec (`effects/spec.py`): frozen dataclass, graph normalised to the
  canonical immutable form (as `Besag`), `rho` float validation, `Hyperparameter`
  rho rejection; added to `EffectSpec`, `Predictor`, `__all__`.
- `effects/proper_car.py`: `build_proper_car(frame, name, index, graph, rho,
  precision) -> LatentBlock` — `M = D − ρW`, ρ-range validity check via `eigvalsh`
  of the normalized adjacency, isolated-node error, one-hot design over graph
  nodes, **empty constraints**.
- Compiler wiring: `ProperCAR` in the two declarative dispatch chains
  (`compile_lgm`, `compile_family`), reusing the `ScalableBlock` scalar-multiply
  path (base precision `M`, τ multiplier), mirroring Besag.
- Exports: `ProperCAR` from `pylgm` and `pylgm.effects`; `build_proper_car` from
  `pylgm.effects`.
- Docs: README proper-CAR section; roadmap update.

### Excluded (deferred / roadmap)
- **ρ estimation** (empirical-Bayes optimise and INLA integrate over ρ) — needs a
  bounded, possibly-negative per-parameter transform threaded through the
  log-space optimiser and INLA grid. Its own **bounded-hyperparameter inference**
  slice; shared with BYM2's φ.
- **BYM2** — structured (scaled ICAR) + unstructured IID mixture; a later slice.
- Sørbye–Rue scaling of proper CAR (intrinsic-only device; not applicable).
- Config-file (`ModelConfig`) `proper_car` type (needs graph-in-config schema),
  as for Besag.
- Graceful isolated-node handling (errors this slice, as for Besag).

## Architecture

- `effects/spec.py`: `ProperCAR` dataclass + `EffectSpec`/`Predictor`/`__all__`
  updates. Reuses `normalize_graph` for validation and canonical storage.
- `effects/proper_car.py` (new): `build_proper_car`. Uses `numpy.linalg.eigvalsh`
  on the dense normalized adjacency for the ρ-range check; `scipy.sparse` for `M`.
- `effects/__init__.py`: export `ProperCAR`, `build_proper_car`.
- `compiler.py`: import `ProperCAR`/`build_proper_car`; add the branch to
  `compile_lgm` and `compile_family` (scalar-multiply `ScalableBlock` path).
- `pylgm/__init__.py`: re-export `ProperCAR`.
- No changes to `LatentBlock`, `ScalableBlock`, the family layer, engines, or INLA
  strategies.

## Errors
- `rho` not a real float (e.g. a `Hyperparameter`): `ValueError` directing to the
  later ρ-estimation slice.
- `rho` outside `(1/μ_min, 1/μ_max)`: `ValueError` naming the valid interval.
- Isolated (zero-neighbour) node: `ValueError` naming the region (as Besag).
- Observed region absent from the graph nodes: `ValueError` naming the regions.
- Non-finite / non-PD `M` after assembly: `NumericalError`.

## Testing and validation
1. **build_proper_car structure**: hand-checked `D − ρW` on a 3-node path for a
   chosen ρ; symmetric; one-hot design over nodes; labels align; **zero
   constraint rows**.
2. **ρ = 0**: `M = D` (diagonal degree matrix).
3. **Validity range**: an in-range ρ builds a positive-definite `M` (Cholesky
   succeeds); an out-of-range ρ (e.g. ρ ≥ 1, or below `1/μ_min`) raises a
   `ValueError` naming the interval; the reported interval matches
   `(1/μ_min, 1/μ_max)` from `eigvalsh` of the normalized adjacency.
4. **Isolated node / observed-not-in-graph**: directed errors (as Besag).
5. **`Hyperparameter` rho rejected**: constructing `ProperCAR(..., rho=Hyperparameter(...))`
   raises the directed error.
6. **PD of emitted precision**: `M` (and `τ·M`) is symmetric PD; `LatentBlock`
   accepts it with empty constraints.
7. **End-to-end fit**: Gaussian and Poisson models with a `ProperCAR` effect fit
   (plug-in, and τ-integrate with a `Hyperparameter` precision), producing finite
   latent marginals for every graph node; `criteria` populated.
8. **Full-Laplace accepts proper CAR**: `latent_strategy="laplace"` with a
   `ProperCAR` effect succeeds (contrast: Besag is rejected) and returns
   tabulated marginals.
9. **Compiler dispatch**: `ProperCAR` flows through plug-in, optimise, and
   integrate; composes with Fixed/IID/RW/Besag.

## Acceptance criteria
1. `ProperCAR(name, index, graph, rho, precision)` composes like any effect and
   fits under plug-in, optimise, and integrate (over τ), for Gaussian and
   non-Gaussian likelihoods, faithful to `Q = τ(D − ρW)`.
2. Adjacency accepted as a neighbour dict and via `load_graph_file` (reusing
   `normalize_graph`); domain = graph nodes; observed ⊆ nodes.
3. ρ is validated against `(1/μ_min, 1/μ_max)`; out-of-range and
   `Hyperparameter`-rho inputs raise directed errors; ρ = 0 gives `τD`.
4. Proper CAR is unconstrained and **works under full Laplace** (unlike Besag).
5. No new runtime dependency (numpy/scipy only); no family-layer/engine changes;
   full suite green; predictions, criteria, optimise/integrate, and all existing
   effects (incl. Besag) unchanged.
6. ρ estimation, BYM2, proper-CAR scaling, and the config-file type remain
   deferred and recorded.

# pyLGM Besag/ICAR Spatial Effect Design

**Status:** Approved 2026-08-23

## Purpose

Add the **Besag / intrinsic CAR (ICAR)** latent effect — the foundational
spatial smoothing prior for areal (region-indexed) data — to pyLGM. This is the
first slice of the spatial CAR family; **proper CAR** and **BYM2** follow as
their own slices (see Roadmap), with the full family as the end state.

Besag is the spatial analogue of the random walk: an intrinsic Gaussian Markov
random field whose precision is the graph Laplacian of a neighbourhood graph.
It reuses the existing effect machinery almost verbatim (declarative spec →
builder → `LatentBlock` → block-diagonal precision/design/constraints), and the
constrained-effect handling already shipped (Gaussian and simplified-Laplace
support constrained blocks; full-Laplace correctly rejects them).

**References:**
- Besag, J. (1974); Besag, York, Mollié (1991) — intrinsic CAR / BYM.
- Rue & Held (2005), *Gaussian Markov Random Fields*, §3 — intrinsic GMRFs,
  the pairwise-difference density, and rank-deficiency = number of connected
  components.
- Sørbye, S. & Rue, H. (2014), *Scaling intrinsic Gaussian Markov random field
  priors in spatial modelling*, Spatial Statistics 8:39–51 — the unit
  generalized-variance scaling this slice applies by default.

## Model

For a neighbourhood graph over `n` regions with symmetric 0/1 adjacency `W`
(`W_ij = 1` iff regions `i` and `j` are neighbours; no self-loops) and degree
matrix `D = diag(Σ_j W_ij)`, the ICAR prior on the latent field `x ∈ ℝⁿ` is the
intrinsic GMRF

  `π(x | τ) ∝ τ^{(n−c)/2} · exp( −τ/2 · xᵀ R x )`,
  `R = D − W`,   `xᵀ R x = Σ_{i~j} (x_i − x_j)²`  (each edge counted once).

`R` is the graph Laplacian: symmetric positive-semidefinite, rank `n − c` where
`c` is the number of connected components. Its null space is spanned by the `c`
component-indicator vectors (constant within each component), so the prior is
improper along those directions and is made proper by imposing **one
sum-to-zero constraint per connected component**.

`τ` is the precision hyperparameter. In pyLGM the builder emits the base
structure matrix (`precision_base · R*`, `R*` defined below), and a declared
`Hyperparameter` precision is applied downstream by
`ScalableBlock.materialize` (`precision = block.precision · multiplier`) exactly
as for IID/RW — no changes to the scaling or integration machinery.

### Scaling (Sørbye–Rue 2014; `scale=True` by default)

The raw structure `R` has a generalized variance that depends on the graph's
size and connectivity, so `τ` is not comparable across graphs nor to an IID
precision. Scaling `R` to **unit generalized variance** fixes this and is the
prerequisite the BYM2 slice needs.

Per connected component `g` with structure submatrix `R_g` (`m_g × m_g`):
- The constrained generalized inverse (covariance under sum-to-zero) is the
  Moore–Penrose pseudoinverse `Σ_g = pinv(R_g)` (numpy; for the 1-null-space
  case this is exactly the sum-to-zero-constrained covariance).
- Marginal variances `σ²_i = (Σ_g)_ii`.

The global scaling factor is the geometric mean of all marginal variances
across every non-singleton component,
`s = exp( mean_i log σ²_i )`, and the scaled structure is `R* = s · R`. After
scaling, the geometric mean of the marginal variances of `R*` is 1. With
`scale=False`, `R* = R` (raw `D − W`, classic Besag).

**Isolated nodes.** A region with zero neighbours (`D_ii = 0`, its own
singleton component) has no ICAR structure — the prior on it is flat/improper
and its marginal variance is undefined, so scaling is undefined. This slice
raises a clear error naming the region and directing the user to model it as an
IID effect. Multi-node disconnected components (island *groups*) are fully
supported (each gets its own sum-to-zero constraint and participates in
scaling). Graceful singleton handling is deferred (Roadmap).

### Relationship to RW1

A path graph (`1—2—3—…—n`) has `R = DᵀD` with `D` the first-difference
operator, i.e. RW1 *is* the 1-D ICAR on a path. Used as a validation anchor:
unscaled path-graph Besag equals the RW1 precision structure; scaled Besag
equals it up to the (positive scalar) Sørbye–Rue factor.

## Adjacency input

One internal representation everywhere: a canonically-sorted node list plus a
symmetric 0/1 `scipy.sparse` adjacency `W`. Two user-facing input forms both
converge to it.

### Neighbour dict (primary)
`graph: Mapping[label, Sequence[label]]` — `{region: [neighbour, ...], ...}`
keyed by the same labels used in the data `index` column. Requirements
(validated): non-empty; symmetric (`b ∈ graph[a]` ⇔ `a ∈ graph[b]`); no
self-loops (`a ∉ graph[a]`); neighbour labels are themselves nodes. The node
set (domain of the latent field) is the set of dict keys.

### INLA / latte graph file (loader)
`load_graph_file(path) -> dict[str, list[str]]` parses the R-INLA/`latte.jl`
text format:
```
n
id_1 k_1 nb_1_1 nb_1_2 ... nb_1_k1
id_2 k_2 ...
...
```
First line is the node count `n`; each subsequent line is a node id, its
neighbour count `k`, then `k` neighbour ids. Ids are 1-indexed integers
(INLA convention); the loader returns them as **string** keys so the result
plugs directly into `Besag(graph=...)`. The data `index` column must use the
same ids (the standard INLA/latte workflow where the region index *is* the
graph node id). Malformed files (bad counts, out-of-range ids, non-integer
tokens) raise a clear error. Whitespace/blank lines tolerated.

## Public contract

```python
from pylgm import Besag, load_graph_file

graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
predictor = Fixed("1") + Besag("region", index="region", graph=graph)
result = model.fit(frame)                      # plug-in
result = model.fit(frame, hyperparameters="integrate")  # τ integrated

# INLA graph-file workflow
graph = load_graph_file("germany.graph")       # dict[str, list[str]]
Besag("region", index="region", graph=graph, precision=Hyperparameter(...), scale=True)
```

- `Besag(name, index, graph, precision=1.0, scale=True)`; `precision` accepts a
  float or `Hyperparameter` (integrated/optimised exactly like IID/RW).
- Composes with `+` like every other effect; participates in predictions,
  criteria, plug-in / optimise / integrate paths unchanged.
- Under `latent_strategy="laplace"` (full Laplace), a Besag effect raises
  `UnsupportedEngineError` (constrained latent field) — same as RW; `gaussian`
  and `simplified_laplace` support it.

## Scope

### Included
- `Besag` spec (`effects/spec.py`): frozen dataclass, graph normalised to a
  canonical immutable form in `__post_init__` (keeps the spec frozen/hashable
  like the others); added to `EffectSpec`, `Predictor` validation, `__all__`.
- `effects/graph.py`: adjacency normalisation (dict → sorted nodes +
  symmetric sparse `W`, with symmetry / self-loop / unknown-neighbour
  validation) and `load_graph_file`.
- `effects/besag.py`: `build_besag(frame, name, index, graph, precision, scale)
  -> LatentBlock` — structure `R = D − W`, per-component sum-to-zero
  constraints, Sørbye–Rue scaling, one-hot design over graph nodes.
- Compiler wiring: `Besag` added to the three effect-dispatch chains
  (optimised-set, plug-in, scalable-family) and imports, following the RW
  branch; `_model_hyperparameters` already picks up `.precision`
  `Hyperparameter`s generically.
- Exports: `Besag`, `load_graph_file` from `pylgm` top level.
- Docs: README spatial-effect section; roadmap entries for proper CAR / BYM2.

### Excluded (deferred / roadmap)
- **Proper CAR** — `Q = τ(D − ρW)` with a spatial-dependence hyperparameter
  `ρ`; full-rank, unconstrained. Its own slice.
- **BYM2** — structured (this scaled ICAR) + unstructured IID mixture with a
  mixing parameter `φ` and PC priors. Its own slice; reuses this builder's
  scaled structure.
- Graceful handling of isolated (zero-neighbour) nodes (errors this slice).
- Shapefile/GeoJSON adjacency construction (would pull a geo dependency —
  out of scope; users build the dict or graph file externally).
- Full-Laplace support for constrained (spatial) effects — remains rejected,
  as recorded in the full-Laplace slice.

## Architecture

- `effects/spec.py`: `Besag` dataclass + union/validation/`__all__` updates.
- `effects/graph.py` (new): `normalize_graph(graph) -> (nodes, W)` and
  `load_graph_file(path) -> dict`. Pure functions, numpy/scipy only.
- `effects/besag.py` (new): `build_besag(...)`. Uses
  `scipy.sparse.csgraph.connected_components`, `numpy.linalg.pinv` (per
  component) for scaling, `ordered_observed_levels` only to validate observed ⊆
  nodes (domain is the graph nodes, canonically sorted).
- `effects/__init__.py`: export `Besag`, `build_besag`, `load_graph_file`,
  `normalize_graph`.
- `compiler.py`: import `Besag`/`build_besag`; add the branch to all three
  dispatch chains, mirroring RW (base precision `1.0` when the precision is an
  optimised `Hyperparameter`, else the resolved float).
- `pylgm/__init__.py`: re-export `Besag`, `load_graph_file`.
- No changes to `LatentBlock`, `ScalableBlock`, engines, or INLA strategies —
  Besag is just another constrained block.

## Errors
- Graph not a non-empty mapping / neighbour not a node / asymmetric / self-loop:
  `ValueError` (spec/normalisation), naming the offending entry.
- Observed region absent from the graph nodes: `ValueError` naming the missing
  regions.
- Isolated (zero-neighbour) node: `ValueError` naming the region and directing
  to IID.
- Malformed graph file (bad node count, out-of-range id, non-integer token):
  `ValueError` with the line context.
- Non-finite / singular scaling (`pinv` yields non-finite marginal variance):
  `NumericalError`.

## Testing and validation
1. **normalize_graph**: dict → sorted nodes + symmetric `W`; symmetry,
   self-loop, unknown-neighbour errors; deterministic node ordering.
2. **load_graph_file**: parses a small file to the expected dict; round-trips a
   dict through file→loader; malformed-file errors (wrong count, bad id).
3. **build_besag structure**: hand-checked `R = D − W` on a 3-node path and a
   two-island graph; symmetric, PSD, row sums zero; design is one-hot over
   nodes; labels align.
4. **Constraints**: one sum-to-zero row per connected component (1 for a
   connected graph, 2 for the two-island graph); each row is the component
   indicator; aligns with design width.
5. **Sørbye–Rue scaling**: with `scale=True` the geometric mean of the marginal
   variances (`diag(pinv(R*_g))` over non-singleton components) ≈ 1;
   `scale=False` leaves `R = D − W`; scaling factor is positive and finite.
6. **Isolated node**: a zero-neighbour region raises the directed error.
7. **RW1 parity**: unscaled path-graph Besag precision equals the RW1 structure
   matrix; scaled equals it up to a positive scalar.
8. **End-to-end fit**: Gaussian and Poisson models with a Besag effect on a
   small connected graph fit (plug-in and integrate), recover a spatially
   smooth pattern, and predict for every graph node; `criteria` populated.
9. **Full-Laplace rejection**: `latent_strategy="laplace"` with a Besag effect
   raises `UnsupportedEngineError`; `gaussian`/`simplified_laplace` succeed.
10. **Compiler dispatch**: Besag flows through plug-in, optimise, and integrate
    paths; `Hyperparameter` precision is picked up; composes with Fixed/IID/RW.

## Acceptance criteria
1. `Besag(name, index, graph, precision, scale)` composes like any effect and
   fits under plug-in, optimise, and integrate, for Gaussian and non-Gaussian
   likelihoods, faithful to the intrinsic-CAR density above.
2. Adjacency accepted as a neighbour dict and via `load_graph_file`, converging
   on one internal representation; both validated.
3. Disconnected graphs get one sum-to-zero constraint per connected component;
   isolated nodes raise a directed error.
4. `scale=True` (default) yields unit geometric-mean generalized variance
   (Sørbye–Rue); `scale=False` yields raw `D − W`.
5. Path-graph Besag matches RW1 (up to the scaling factor); full-Laplace
   rejects Besag while gaussian/simplified-Laplace support it.
6. No new runtime dependency (numpy/scipy only); full suite green; predictions,
   criteria, optimise/integrate, and all existing effects unchanged.
7. Proper CAR and BYM2 remain deferred and recorded as the next spatial slices.

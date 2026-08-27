# pyLGM Knorr-Held Space-Time Interaction (S8) — Design

> **Slice:** S8 of the economic-context expansion plan
> (`docs/design/plans/2026-08-26-pylgm-economic-expansion.md`). Adds a
> `SpaceTime` interaction effect covering Knorr-Held interaction Types I–IV.
> Depends on S4 (weighted graphs, shipped) for the spatial factor; reuses the
> RW difference operator (S1) for the temporal factor. No new inference code.

## Goal

Add the space-time **interaction** term `δ(s,t)` of a Knorr-Held model as a
single composable effect. The linear predictor of a full Knorr-Held model is

```
η(s,t) = μ + u(s) + v(t) + δ(s,t)
         └ Fixed  └ Besag  └ RW    └ SpaceTime (this slice)
```

`SpaceTime` builds **only** `δ(s,t)`: one `LatentBlock` over the `S·T` cells
with precision `τ·(K_s ⊗ K_t)` and the type-specific identifiability
constraints. The spatial (`Besag`) and temporal (`RW1`/`RW2`) main effects are
composed by the caller with `+`, keeping each effect one self-contained block
with its own variance component — the honest Knorr-Held rendering.

**Economic use:** area × period panels where local space-time departures
(regional credit-risk clusters that emerge and fade, sector shocks that
propagate through a supply-chain graph over time) are not captured by an
additive spatial map plus a common temporal trend.

## Interaction taxonomy

The interaction type selects the two Kronecker factors. `R_s` is the
Sørbye–Rue-scaled weighted Besag Laplacian; `R_t` is the Sørbye–Rue-scaled RW
structure `DᵀD` (order 1 or 2). `I` is the identity.

| Type | `K_s` | `K_t` | Reading |
|------|-------|-------|---------|
| I    | `I_s` | `I_t` | unstructured cell-wise interaction |
| II   | `I_s` | `R_t` | each area has its own independent temporal trend |
| III  | `R_s` | `I_t` | each period has its own independent spatial pattern |
| IV   | `R_s` | `R_t` | inseparable: neighbours in *both* space and time are tied |

Type I is full rank (an IID over cells); Types II–IV are intrinsic (rank
deficient) and require constraints (below).

## API

```python
SpaceTime(
    name: str,
    space: str,                       # area index column
    time: str,                        # time index column
    graph: Mapping | None = None,     # weighted neighbour graph over areas (S4 form)
    interaction: str = "IV",          # "I" | "II" | "III" | "IV"
    order: int = 1,                   # temporal RW order 1|2, used by II/IV
    precision: float | Hyperparameter = 1.0,
    scale: bool = True,               # Sørbye–Rue scale both factors
)
```

Decisions:

- **`interaction`** (not `type`, which shadows the builtin) takes Roman
  numerals matching the literature; validated against `{"I","II","III","IV"}`.
- **`order`** defaults to **1** — RW1 is the canonical Knorr-Held interaction
  temporal structure. Ignored for Types I/III (which use `I_t`); passing it
  there is silently accepted (no effect).
- **`graph`** is **required for Types III/IV** (they need `R_s`) and raises a
  `ValueError` when absent. For Types I/II it is **optional**: when omitted the
  area universe is read from the observed `space` column (the same
  `ordered_observed_levels` rule RW uses for time); when supplied its node set
  defines the area universe (the same rule Besag uses). This spares a Type-II
  user from inventing an unused neighbour graph.
- **`scale`** defaults to `True` (Sørbye–Rue). Setting `False` skips scaling of
  both factors, for parity with `Besag(scale=False)`.
- **`precision`** is a single `τ` — a float (plug-in) or a `Hyperparameter`
  (estimated by EB, integrated by INLA). The precision scales linearly in `τ`,
  so the block is a `ScalableBlock`, exactly like `Besag`.

Example (full Type IV Knorr-Held model):

```python
from pylgm import Besag, Fixed, Gaussian, LGM, RW1, SpaceTime

model = LGM(
    response="y",
    predictor=(
        Fixed("1")
        + Besag("area", index="area", graph=W)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=W, interaction="IV", order=1)
    ),
    likelihood=Gaussian(sigma=0.5),
)
result = model.fit(frame)
```

## Latent domain and design

The latent field is the full `S·T` grid: `S` area levels × `T` time levels.
Area levels come from `graph` nodes (III/IV, or I/II with a graph) or from the
observed `space` column (I/II without a graph); time levels come from
`ordered_observed_levels(frame[time])`. The cell index is **area-major**:

```
cell(s, t) = area_position[s] · T + time_position[t]
```

Each observed row contributes a single `1.0` at its cell. Cells with no
observed row still exist as latent columns (they are predicted), exactly as RW
creates a column for every level and Besag for every node. `design` is
`csr_matrix((ones, (row, cell)), shape=(n_obs, S·T))`.

Requirements: `S·T > 0`; for Types II/IV, `T > order` (RW needs more levels than
its order); the graph, when required, is validated by the existing
`normalize_graph`. Non-finite handling and unseen-level errors mirror the
existing structured builders.

## Precision and Sørbye–Rue scaling

```
precision_matrix = τ · kron(K_s, K_t)          # scipy.sparse.kron, csr
```

- `K_s`: `identity(S)` (I/II) or `besag._scaled_structure(w, nodes, scale)`
  (III/IV) — reuses the S4 weighted, component-scaled ICAR Laplacian verbatim.
- `K_t`: `identity(T)` (I/III) or the scaled RW structure (II/IV):
  `D = difference_operator(T, order)`, `R_t = DᵀD`, then Sørbye–Rue scaled.

**Sørbye–Rue scaling** standardises each intrinsic factor to geometric-mean
marginal variance 1 *before* the Kronecker product, so `τ` has a consistent
meaning across orders and grid sizes (Sørbye & Rue 2014; Riebler et al. 2016).
Besag already performs this per component inline in `_scaled_structure`; that
loop (eigendecompose, drop the null eigenvalues, geometric-mean the marginal
variances, rescale) is extracted into a shared helper

```python
def sorbye_rue_scale(structure: np.ndarray, null_dim: int) -> np.ndarray
```

in a shared module (`effects/scaling.py`), called by both `besag`
(`null_dim=1` per connected component) and `spacetime` (`null_dim=order` for
`R_t`, a single connected structure). Besag's numerical behaviour is unchanged
(same computation, refactored).

## Constraints (derived from the null space)

The constraint rows are not hardcoded per type; they are a basis of
`null(K_s ⊗ K_t) = null(K_s) ⊗ ℝ^T  ⊕  ℝ^S ⊗ null(K_t)`.

Per-factor null bases (columns):

- `I` → empty (0 columns)
- Besag `R_s` → connected-component indicator columns (via
  `connected_components`; one column for a connected graph)
- RW1 `R_t` → `[1_T]`
- RW2 `R_t` → `[1_T, k]` with `k = arange(T)` centred

Assemble `B = [ kron(N_s, I_T) | kron(I_S, N_t) ]` (shape `S·T × (n_s·T + S·n_t)`),
take an orthonormal basis of its column span by SVD (rank
`r = n_s·T + S·n_t − n_s·n_t`, dropping the `1_s ⊗ 1_t` overlap counted twice),
and transpose to `constraints` (shape `r × S·T`). This yields exactly:

| Type | order | constraint rows |
|------|-------|-----------------|
| I    | —     | 0 |
| II   | 1 / 2 | `S` / `2S` |
| III  | —     | `T` (connected graph) |
| IV   | 1     | `S + T − 1` |
| IV   | 2     | `2S + T − 2` |

These are the Schrödle–Held sum-to-zero sets, and they make `δ` orthogonal to
the `u(s)` / `v(t)` main effects, so the spatial / temporal / interaction
variance decomposition is identified.

**Latent-strategy consequence:** Type I carries no constraints and is proper,
so it runs under all latent strategies including full Laplace. Types II–IV
carry constraints, so — like `RW2` and `Besag` — they require
`hyperparameters="integrate"` and are rejected under `latent_strategy="laplace"`
by the existing model contract. No new inference code; the constraint rows flow
through the existing null-space/kriging machinery.

## Compiler wiring

Mirrors `Besag` at every site (the usual five):

1. `effects/spec.py`: new `SpaceTime` frozen dataclass with the validation
   above; added to the `EffectSpec` union and both `Predictor` isinstance
   tuples.
2. `effects/spacetime.py` (new): `build_spacetime(frame, name, space, time,
   graph, interaction, order, precision, scale) -> LatentBlock`.
3. `compiler.compile_lgm`: `elif isinstance(effect, SpaceTime)` builds the block
   with the resolved fixed precision and records `precisions[name]`.
4. `compiler.compile_family`: `SpaceTime` branch emits
   `ScalableBlock(build_spacetime(..., τ_value), parameter, 1.0)` where
   `parameter` is `τ.name` when `precision` is a `Hyperparameter` else `None`
   — identical to the Besag branch (the structure matrix is fixed; only `τ`
   scales).
5. `effects/__init__.py` + `pylgm/__init__.py`: export `SpaceTime` and
   `build_spacetime`.

**Sibling warning.** The compiler sees the whole predictor, so after collecting
effects it checks, for each `SpaceTime`, whether the predictor also contains a
spatial main effect with `index == space` (a `Besag`/`ProperCAR`/`BYM2`) **and**
a temporal main effect with `index == time` (an `RW1`/`RW2`/`AR1`). If either is
missing it emits one `warnings.warn(...)` naming the effect and the missing
companion — the interaction's sum-to-zero constraints assume the main effects
absorb the marginals, so omitting one is a modelling error, not a crash.

## Prediction

`build_prediction_context` appends
`("spacetime", (name, space, time, area_labels, time_labels))`.
`inference/prediction.py` adds a `_spacetime_block` that, for new rows, maps
each `(area, time)` pair to its area-major cell and emits the one-hot design,
raising the same hard error as `_structured_block` for an area or time level not
in the fitted domain (predict reuses the fitted posterior; it cannot mint a new
cell). Missing `space`/`time` columns and non-finite handling mirror the other
prediction blocks.

## Testing

**Builder units** (`tests/test_spacetime_builder.py`):

- `precision == τ · kron(K_s, K_t)` for each of the four types (dense compare);
- Sørbye–Rue scaling drives each factor's geometric-mean marginal variance to 1;
- constraint row count matches the table for each (type, order), the rows are
  full-rank, and each row annihilates every null vector of the precision;
- cell mapping: design is the area-major one-hot of `(space, time)`;
- validation errors: bad `interaction`, `order` not in `{1,2}`, `T ≤ order` for
  II/IV, `graph=None` for III/IV, unseen/non-finite handling.

**Effect end-to-end** (`tests/test_spacetime_effect.py`):

- each type fits with fixed `τ` and returns an `S·T` latent mean;
- estimate-`τ` EB (`result.hyperparameters` finite > 0) and integrate
  (`hyperparameter_marginals` carries the precision);
- Type I accepted under `latent_strategy="laplace"`; Types II–IV rejected
  (mirrors `test_full_laplace_rejects_besag`);
- predict on new `(area, time)` rows;
- one statistical test: simulate an inseparable field on a small grid, fit Type
  IV, assert the recovered interaction correlates with the truth and the
  space/time/interaction variance split is sane.

## Ceilings

- **Dense solve.** The latent width is `S·T`; the 4096-dim / 512 MB preflight
  guard bites fast (60×40 = 2400 fits; 100×50 = 5000 does not). This is the
  known ceiling, gated on **E-sparse** for real economic scale — documented,
  not worked around in this slice.
- **Weak identifiability of the variance split.** On short panels the
  interaction `τ_δ` and the main-effect variances trade off; a boundary `τ̂_δ`
  reads as "no identified interaction," analogous to BYM2's `φ`.
- **Type I is degenerate as a standalone** (an IID over cells); it is included
  for taxonomy completeness and as the interaction term atop structured main
  effects.

## Design decisions

- **Interaction-only block, not a bundled convenience effect** (chosen). Keeps
  `SpaceTime` one block with one job and one variance component, reuses the S4
  `Besag` and the RW effects unchanged, and keeps `result` reporting separable
  (`"area"`, `"period"`, `"st"` are three interpretable fields). A bundled
  effect would re-own three sub-blocks, three priors, and three label
  namespaces — a larger, less composable surface that cuts against the `+`
  idiom. A one-line warning recovers the "forgot a main effect" safety without
  the bundling.
- **Intrinsic factors only (Besag + RW1/RW2), single `τ`** (chosen). This is
  the canonical Knorr-Held model, keeps the block a pure `ScalableBlock`, and
  lets the null-space rule cover every type/order. Proper factors (AR1 `ρ_t`,
  ProperCAR `ρ_s`) are **rejected for this slice**: they add weakly-identified
  correlation hyperparameters and a `ParametricBlock`, and change the null
  space — a separate later modelling extension, not a correctness upgrade.
- **Constraints derived from the null space, not hardcoded** (chosen). One rule
  reproduces all type/order counts and stays correct if factors change later,
  versus five bespoke constraint constructions.
- **Sørbye–Rue scaling of both factors** (chosen). Without it `τ_δ` means
  something different for RW1 vs RW2 and across grid sizes, and Type IV (a
  Kronecker of two structures) compounds the distortion. This is the real
  correctness knob and the standard modern recommendation.

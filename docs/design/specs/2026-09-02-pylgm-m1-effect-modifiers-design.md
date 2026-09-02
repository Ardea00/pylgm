# pyLGM M1: Effect Modifiers (weights, copy, replicate, group)

**Status:** Approved 2026-09-02
**Branch:** `research-tier` (research-grade; see [research status](../../research-status.md))

## Purpose

R-INLA expresses latent terms through one function whose arguments modify a base
model:

```r
f(index, weights, model = ..., replicate = ..., group = ...,
  control.group = list(model = ...), copy = ..., hyper = ...)
```

pyLGM has the base models — `IID`, `Besag`, `AR1`, `RW1`/`RW2`, `BYM2`,
`ProperCAR`, `SAR`, `Seasonal`, `MIDAS`, `SpaceTime`, `DynamicSpatialPanel` —
but none of the modifiers, except one special case noted below. This slice adds
them as a composable surface.

This is the remaining half of atlas item M1. The other half — sharing a latent
field *across* sub-models with a fixed or estimated scaling — shipped as `Shared`
in the joint-models slice, and already covers shared-component disease mapping,
joint longitudinal-plus-survival, and joint PD/LGD.

No new dependency, no new inference regime, no MCMC.

## Two corrections to the current API this slice makes

**`AR1(group=)` is misnamed.** R-INLA draws a distinction pyLGM currently
collapses:

- **`replicate`** — *independent* copies sharing hyperparameters.
- **`group`** — *correlated* copies, with a separable structure between groups
  given by `control.group`.

`ar1_structure(..., group_count=G)` builds `I_G ⊗ T`: independent series sharing
`rho` and `precision`. That is R-INLA's **replicate**, shipped under the name
`group`. Anyone porting `f(time, model="ar1", replicate=person)` reaches for the
wrong concept. This slice introduces both under their correct names and
deprecates the old one.

**`SpaceTime` is a special case of `group`.** `build_spacetime` computes
`precision * kron(k_s, k_t)` and the Knorr-Held types I-IV are exactly the four
combinations of {iid, structured} x {iid, structured}. That is the general
Kronecker mechanism `group` + `control.group` provides, in curated form.
`SpaceTime` therefore becomes a convenience wrapper over the general primitive
rather than a second implementation of it.

## Architecture

**Every modifier produces exactly one `LatentBlock`.** Both IR invariants —
`design == hstack(blocks)` and `precision == block_diag(blocks)` — are
preserved, as they were for joint models. No inference change.

Modifiers split into two families with very different costs.

**Design-side, pure post-processing.** `Weighted` builds the inner block and
scales its design rows: `diag(z) A`. Precision, labels and constraints pass
through untouched.

**Kronecker reconstruction.** `Replicated` and `Grouped` cannot post-process: an
inner effect built against the whole frame would already consume every level.
They build the inner **structure over the level set alone** — reusing
`_levels_frame`, which exists from the joint-models slice for exactly this — and
then compose:

    Q_replicated = I_R (x) Q_E          design indexed by (replicate, level)
    Q_grouped    = Q_S (x) Q_E          design indexed by (group, level)

`_build_effect_block` gains a wrapper case: unwrap, build the inner effect,
apply the transform.

**Constraints are the hard part, and they are where `Grouped` earns its cost.**
`Besag` carries a sum-to-zero constraint. Under `Replicated` that becomes `R`
constraints, one per replicate. Under `Grouped` the Knorr-Held types have
specific null spaces, which is why `build_spacetime` carries
`_space_null_basis` and `_time_null_basis`. That machinery is generalised, not
rewritten.

## Public API

Wrappers, not modifier fields. Four classes rather than four fields on each of a
dozen effect specs, which would be roughly 48 validation paths plus implicit
field-interaction rules, and which every future effect would have to reimplement.
This also matches `Shared(effect, scale)`, already shipped and validated in this
branch.

```python
Weighted(Besag("u", index="district", graph=g), by="z")
Replicated(AR1("t", index="year"), over="firm")
Grouped(Besag("s", index="district", graph=g), over="year",
        structure=AR1Structure(rho=0.8))
Copy("u", index="j", scale=Hyperparameter("beta"))
```

- `Weighted(effect, by)` — `by` names a numeric column. Design becomes
  `diag(by) A`.
- `Replicated(effect, over)` — `over` names the replicate column. Independent
  copies sharing every hyperparameter of `effect`.
- `Grouped(effect, over, structure)` — `over` names the group column;
  `structure` is the between-group precision, mirroring `control.group`. These
  are **new, small classes** — `IIDStructure`, `AR1Structure(rho)`,
  `RW1Structure`, `RW2Structure`, `BesagStructure(graph)` — deliberately
  separate from the effect specs of similar name: an effect spec carries an
  index column and builds a design, whereas a structure carries only a
  precision matrix over the group levels and never touches the frame.
- `Copy(name, index, scale=1.0)` — references an existing block **by name** and
  adds `scale * A_index` to its columns. `scale` may be a `Hyperparameter`.
  `index` names a column whose values must all be levels of the referenced
  block's own index; a value outside that level set is an error, because a copy
  reuses an existing latent field and cannot create a new level in it. `Copy`
  may reference a wrapped block (`Weighted`, `Replicated`, `Grouped`) — the copy
  contributes to that block's columns as they were built, so a copy of a
  `Replicated` block must supply the `(replicate, level)` pair columns.

`Copy` is deliberately asymmetric: it is a term in the predictor referencing
another term, not a wrapper around an effect, and the signature shows that.

### Composition

`Weighted` touches only the design; `Replicated` and `Grouped` touch only
precision and indexing. They therefore **commute**:

    Replicated(Weighted(E, by=z), over=r) == Weighted(Replicated(E, over=r), by=z)

This is asserted by test, not left as a convention to remember.

`Weighted`, `Replicated` and `Grouped` nest freely.

`Copy` does not participate in nesting: it is neither a wrapper nor wrappable,
so `Weighted(Copy(...))` and `Copy` of a `Copy` are both rejected. It may
however *reference* a wrapped block, which is a different relationship — the
copy contributes to that block's existing columns rather than modifying how the
block was built.

### `SpaceTime` after this slice

`SpaceTime(space, time, interaction=...)` keeps its signature and its
documentation, and compiles to the general primitive:

| type | space structure | time structure |
|---|---|---|
| I | iid | iid |
| II | iid | structured (RW of `order`) |
| III | structured (Besag) | iid |
| IV | structured (Besag) | structured (RW of `order`) |

The existing `SpaceTime` tests become the guarantee that the translation is
correct.

## Prediction

Prediction entries nest the way the specs nest. A wrapper contributes
`("weighted", (inner_entry, weights_column))`, recursively, so
`Weighted(Replicated(E))` rebuilds through one rule rather than needing a case
per combination.

`Replicated` and `Grouped` index on `(r, level)` and `(g, level)` pairs, which
is what `_paired_cell_block` already does for group-wise `AR1` and `SpaceTime` —
reused directly, including its errors for levels unseen at fit time.

`Copy` adds `scale * A_index` to the referenced block's columns, the way
`_shared_design_block` does for shared fields. The fitted `scale` is substituted
at predict time when it is a `Hyperparameter`.

## Rejections (fail loud, matching the frontend's existing style)

- A `by` column that is missing, non-numeric, or contains NaN — the error names
  the effect *and* the column.
- An all-zero `by` column: the effect is inert and the model is misspecified.
  Rejected rather than silently compiling a zero block.
- `Replicated` around an effect that already carries replicate semantics
  (`AR1(replicate=)`), which would be ambiguous.
- `Grouped` whose `structure` dimension does not match the group level count.
- `Copy` naming a block that does not exist, or naming itself.
- A ragged `(r, level)` or `(g, level)` grid — reusing `_paired_cell_block`'s
  existing errors.
- `Weighted`/`Replicated`/`Grouped` around `Fixed`, which has no index.

## Testing

**Property.** `Replicated(Weighted(E))` equals `Weighted(Replicated(E))`
exactly — design, precision, labels and constraints. This is what justifies the
wrapper design; without it the commutativity claim is folklore.

**Reduction.** `Weighted(E, by=<all ones>)` equals `E`. `Replicated(E,
over=<single level>)` equals `E`. Cheap guards of the kind that caught real
defects in the joint-models slice.

**Equivalence against an existing oracle.** Two of the four primitives already
have a correct, shipped implementation to check against:

- `Replicated(AR1(...), over=g)` must equal `AR1(..., group=g)` bit-for-bit.
- `Grouped(...)` must equal `SpaceTime(interaction=...)` bit-for-bit for all
  four types.

This is what makes slice 4 safe despite rewriting a released path.

**Constraint replication.** `Replicated(Besag(...), over=r)` must produce `R`
sum-to-zero constraints, not one. This is the most likely silent error in the
slice: a single constraint over `R` replicates leaves `R-1` directions
unidentified, and the model still fits and still returns plausible numbers.

**Prediction round-trip** for each wrapper and for one nested combination.

## Slices

Four separate implementation slices, each with its own spec and plan, ordered by
cost and by constraint complexity.

| # | primitive | touches | unlocks | risk |
|---|---|---|---|---|
| 1 | `Weighted` | design only | spatially-varying coefficients (atlas M14) | low; no released path changes |
| 2 | `Copy` | design only, by reference | a field at two indices; fixed loadings | low; new surface only |
| 3 | `Replicated` | precision, replicated constraints | generalises `AR1(group=)` | medium; renames released 0.6 API |
| 4 | `Grouped` | precision, null spaces | `SpaceTime` becomes a wrapper | high; rewrites a released, tested path |

Slices 1 and 2 are independent and may proceed in parallel. Slice 3 introduces
`replicate` and deprecates `group` with a `DeprecationWarning` naming the
translation. Slice 4 depends on 3 only for naming consistency, not mechanically.

## Out of scope

- **Off-block-diagonal precision coupling** (coregionalization, atlas M2). Every
  modifier here still produces one block.
- **Copy of a copy**, and copies that cross sub-models — the latter is `Shared`.
- **A `Hyperparameter` on a `Grouped` structure's own parameters** (a
  correlated-group `rho` estimated jointly with the inner effect's
  hyperparameters). Fixed values only in this slice, matching the same
  restriction `Shared` carries today.
- **YAML frontend** for any of the four. The Python API lands first.
- **`weights` as a field on effect specs** — the shorthand from approach C. It
  can be added later as sugar over `Weighted` without changing the mechanism,
  if nesting proves annoying in practice.

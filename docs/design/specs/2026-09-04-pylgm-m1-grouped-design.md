# pyLGM M1 Slice 4: `Grouped` (correlated copies)

**Status:** Approved 2026-09-04
**Branch:** `research-tier` (research-grade; see [research status](../../research-status.md))
**Parent spec:** [M1 effect modifiers](2026-09-02-pylgm-m1-effect-modifiers-design.md) (slice 4 of 4)

## Purpose

`Grouped(effect, over, structure)` is R-INLA's `f(index, model=..., group=g,
control.group=list(model=...))`: `G` copies of an effect that are *correlated*
across groups rather than independent. Precision `Q_S (x) Q_E` instead of
`Replicated`'s `I_R (x) Q_E`.

It is the last of the four M1 modifiers and the only one that touches a
released, tested path.

## What this spec changes about the parent

The parent spec says `SpaceTime` "becomes a convenience wrapper over the
general primitive rather than a second implementation of it." That is adopted
in a narrower reading: **`SpaceTime` delegates the mechanism, not the
surface.**

The duplication worth removing is the mathematics -- the Kronecker precision
and, above all, the null-space constraint assembly, which is the part most
likely to be got subtly wrong in two places. That gets one implementation.

Collapsing the *spec surface* was considered and rejected. `SpaceTime` carries
things the general primitive should not learn:

- a YAML/config surface (`config/model.py`), which `Weighted`, `Copy` and
  `Replicated` all lack;
- a `|` label separator that is **user-visible** -- `result.labels` reads
  `('fixed:Intercept', 'st:a|0', 'st:a|1', ...)`, so changing it is a break;
- an area universe taken from the graph rather than the observed column;
- rejection of isolated areas, and a `T <= order` validation;
- a warning when the spatial/temporal main effects are absent.

Teaching `Grouped` a separator parameter and a config surface to absorb these
is the opposite of generalising. `SpaceTime` keeps them.

## The unifying observation

A block's `constraints` are a basis of its precision's null space. This holds
across the codebase and was verified rather than assumed: `RW1` carries one
constraint, `RW2` carries two, `Besag` carries one per connected component.

That makes one formula cover every case:

    null(Q_out (x) Q_in) = null(Q_out) (x) R^in  +  R^out (x) null(Q_in)

`_interaction_constraints` in `spacetime.py` already implements it. And
`Replicated` is its degenerate case: `null(I_R)` is empty, the first term
vanishes, and what remains is `kron(I_R, C)` -- exactly what
`replicate.py` builds by hand today.

## Architecture

### The kernel

One new module, `src/pylgm/effects/kronecker.py`, holding all the shared
mathematics:

```
kron_block(name,
           outer_labels, outer_precision, null_out,
           inner_labels, inner_precision, null_in,
           outer_keys, inner_keys,          # one entry per frame row
           separator, orthonormalise) -> LatentBlock

    precision   = kron(Q_out, Q_in)
    design      = one-hot at  cell = outer_pos * n_inner + inner_pos
    constraints = [kron(N_out, I_in), kron(I_out, N_in)]  transposed
    labels      = f"{outer}{separator}{inner}"
```

Pieces, not blocks: `build_spacetime` has `k_s`, `k_t` and two label tuples but
no inner `LatentBlock` to hand over, and `grouped_block` has one it can take
apart.

Callers pass **already-resolved integer positions**, one per frame row, and own
their own validation. `replicated_block` and `build_spacetime` raise different,
test-pinned messages for an unknown level, and unifying them would break those
tests for no gain. The kernel is therefore pure numpy/scipy: no pandas, no
error strings.

`separator` is `"@"` for `Replicated` and `Grouped`, `"|"` for `SpaceTime` --
the same back-compatibility reason as `orthonormalise`, and the only two
parameters that exist for it.

Three callers: `replicated_block`, the new `grouped_block`, and
`build_spacetime`.

**`orthonormalise` is a back-compatibility flag, not a modelling choice.** Both
settings span the same constrained subspace, so no fit changes either way. It
exists because two released paths already have their own basis:

- Routing `Replicated` through the SVD turns `[1, 1, 1]` into
  `[-0.577, -0.577, -0.577]`. Same row space, different rows -- and slice 3's
  headline safety property is bit-for-bit equality with the shipped
  `AR1(group=)`. So `replicated_block` and `grouped_block` pass `False` and get
  `kron(I, C)` literally.
- `build_spacetime` orthonormalises today, including for types II and III
  where only one Kronecker part is present. So it passes `True` and its output
  is unchanged.

The flag governs the **one-part** case only. When both parts are present they
overlap in `1_out (x) 1_in`, and dropping that duplicate is what the SVD is
for -- skipping it would return a rank-deficient constraint matrix. Two parts
always orthonormalise, whatever the flag says. `Replicated` always has one part
(`null(I_R)` is empty), so it is preserved exactly.

The flag is documented as preserving two released outputs, and nothing else.

**The kernel takes the null bases as arguments rather than deriving them from
the inner block.** `build_spacetime` then passes exactly what it passes today
(`_space_null_basis`, `_time_null_basis`), so refactoring a released path onto
the kernel has no numerical surface at all. `grouped_block` passes
`inner.constraints.T` and the structure's own null basis.

### The specs

`Grouped(effect, over, structure)` in `effects/spec.py`, alongside `Weighted`,
`Copy` and `Replicated`. `.name` delegates to the inner effect, as the other
wrappers do.

Five frozen dataclasses in a new `src/pylgm/effects/structures.py`:
`IIDStructure`, `AR1Structure(rho)`, `RW1Structure`, `RW2Structure`,
`BesagStructure(graph)`. Each exposes exactly two methods:

```python
precision(levels) -> csr_matrix
null_basis(levels) -> np.ndarray     # (n_levels, null_dim)
```

They never touch the frame. They are deliberately separate from the effect
specs of similar name: an effect carries an index column and builds a design,
a structure carries only a precision over the group levels.

`IIDStructure` is redundant as API -- `Grouped(E, over=g,
structure=IIDStructure())` is `Replicated(E, over=g)` -- and ships anyway,
because Knorr-Held types I, II and III each have an iid factor and the
equivalence oracle needs all four types expressible.

### Level universe and alignment

`BesagStructure` carries named nodes, so **the graph is the universe**, aligned
**by name**: an observed level outside the node set is an error, and a node
with no observations still gets its cell, which is what lets the spatial
smoothing borrow strength for it. This is what `Besag` and `build_spacetime`
already do, and `normalize_graph` returns canonically sorted string labels, so
the ordering matches `replicate_levels`' sorted convention without further
work.

The other four structures are anonymous. Their universe is the sorted observed
levels of `over`, and the structure's dimension must match that count.

Alignment is never positional. Positional alignment is the failure this project
has shipped three times, and a permuted neighbourhood structure fits happily
and returns plausible numbers.

## `SpaceTime` after this slice

Signature, documentation, `|` labels, prediction entry, YAML surface,
main-effects warning, isolated-area rejection and `T <= order` validation are
all unchanged. `build_spacetime` computes `k_s`, `k_t` and the two null bases
exactly as today, then delegates the composition to `kron_block`.

Its output is asserted bit-for-bit identical before and after, rather than
assumed.

## Prediction

`("grouped_structured", ...)` over `_paired_cell_block`, the twin of
`Replicated`'s entry, inheriting its errors for group or level values unseen at
fit time. `_prediction_entry` keeps `split("@", 1)`.

## Testing

**The oracle.** `Grouped` must reproduce `SpaceTime`'s four Knorr-Held types on
design, precision, and the *span* of the constraints:

| type | `structure` | inner effect |
|---|---|---|
| I | `IIDStructure()` | `IID` over time |
| II | `IIDStructure()` | `RW1`/`RW2` over time |
| III | `BesagStructure(graph)` | `IID` over time |
| IV | `BesagStructure(graph)` | `RW1`/`RW2` over time |

Row space, not rows: `Grouped` does not orthonormalise and `SpaceTime` does, so
the bases differ while the constrained subspace does not. Compared by rank of
the stacked matrices.

**Reduction.** A single-level structure reduces to the bare effect.

**Commutation.** `Grouped(Weighted(E))` equals `Weighted(Grouped(E))`, as
asserted for `Replicated`.

**Prediction round-trip**, including the nested `Weighted` form -- the case
that was silently wrong in slice 3 until caught.

**Hyperparameter effectiveness.** A row per new path in
`tests/test_hyperparameter_effectiveness.py`. That file exists because this
project shipped six separate instances of a declared hyperparameter having zero
effect on the fit; adding to it is not optional.

**Integer index dtype.** The guard that was emphatically commented and entirely
untested until slice 3's review. Both `Grouped` dispatch sites get one.

## Rejections

- `Grouped(Replicated(...))` and `Replicated(Grouped(...))`, naming both
  wrappers. R-INLA permits `group` and `replicate` together; pyLGM does not,
  because the labels would become `r@g@level` and both `_prediction_entry`'s
  `split("@", 1)` and the assumption of a single inner index column would have
  to be generalised. Recorded in `research-status.md` as an f() parity gap.
- `Grouped(Grouped(...))`: two group columns are one group over their cross
  product.
- `Grouped(Fixed)` and `Grouped(Copy)`, matching `Replicated`.
- A structure whose dimension does not match the group level count.
- An observed group level outside a `BesagStructure`'s node set.

## Out of scope

- **A `Hyperparameter` on a structure's own parameters.** `AR1Structure(rho)`
  takes fixed values only, as the parent spec already restricts and as `Shared`
  does today. This is also what keeps the slice from needing the hook for
  wrapper-owned hyperparameters that slice 3's review found missing in
  `_append_family_blocks` and `_effect_hyperparameters`.
- **YAML/config surface for `Grouped`.** The Python API lands first, as for the
  other three modifiers.
- **Spark.** `_required_columns` is already blind to `MIDAS`, `SpaceTime`,
  `DynamicSpatialPanel` and `AR1(replicate=)`'s replicate column; widening it
  belongs to that helper, across all of them at once.
- **Off-block-diagonal coupling** (coregionalization, atlas M2). Every modifier
  in M1 still produces exactly one block.

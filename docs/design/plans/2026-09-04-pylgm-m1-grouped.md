# M1 Slice 4: `Grouped` Effect Modifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any indexed latent effect become `G` *correlated* copies with a between-group precision — R-INLA's `f(index, model=..., group=g, control.group=list(model=...))` — and fold the three existing Kronecker compositions onto one shared kernel.

**Architecture:** `Grouped(effect, over, structure)` builds the inner effect's structure over the level set alone, then composes `Q_S ⊗ Q_E` with a design on `(group, level)` pairs. The composition itself moves into `effects/kronecker.py`, which `replicated_block`, the new `grouped_block` and `build_spacetime` all call. One `LatentBlock` out, so both `CompiledLGM` invariants hold and inference is untouched.

**Tech Stack:** Python 3.11+, numpy 2.x, scipy 1.14+ (sparse), pandas 2.2+, pytest 8.3+. No new dependencies.

**Spec:** `docs/design/specs/2026-09-04-pylgm-m1-grouped-design.md` (slice 4 of 4)

**Branch:** `research-tier`. Research-grade; see `docs/research-status.md`.

## Global Constraints

- **No new runtime dependency.** numpy / scipy / pandas / formulaic / pydantic / pyarrow / pyyaml / typer only.
- **No MCMC.** Deterministic approximation only.
- **`CompiledLGM` invariants preserved**: `design == hstack(blocks)`, `precision == block_diag(blocks)`. `Grouped` produces exactly one block.
- **No inference change.** `inference/laplace.py` and `inference/gaussian.py` are not touched.
- **Existing tests pass unchanged at every commit.** `PYTHONPATH=src python -m pytest -q`. Baseline: **1338 passed, 0 failed**.
- **Ruff clean:** `ruff check src tests`, line length 100, rules `E4,E7,E9,F`.
- **Run tests as:** `PYTHONPATH=src python -m pytest ...`
- **Frozen dataclasses** with validation in `__post_init__`.
- **Errors** raise types from `pylgm.exceptions`, or plain `ValueError`/`TypeError` in spec `__post_init__` and in `prediction.py`, matching the surrounding code.
- **Every new hyperparameter path gets a row in `tests/test_hyperparameter_effectiveness.py`.** That file exists because this project shipped six separate instances of a declared hyperparameter having zero effect on the fit. Adding to it is not optional.

## Three risks this slice carries, and where each is handled

**1. Refactoring a released path.** `build_spacetime` is shipped and tested. Task 3 moves its composition onto the kernel. The mitigation is that the kernel takes the null bases as *arguments*, so `build_spacetime` keeps passing exactly what it passes today — and Task 3 pins its output bit-for-bit **before** touching it.

**2. Constraint bases are not unique.** Routing `Replicated` through the SVD turns `[1, 1, 1]` into `[-0.577, -0.577, -0.577]`: same row space, different rows. Slice 3's headline safety property is bit-for-bit equality with `AR1(group=)`, so the kernel must not orthonormalise a single part unless asked. See Task 2 Step 3.

**3. Silent misalignment.** Aligning a `BesagStructure`'s graph to group levels by position rather than by name permutes the neighbourhood structure; the fit converges and returns plausible numbers. Task 4 aligns by name and Task 6 rejects any observed level outside the node set.

## The name that has to be freed first

`"grouped_structured"` is already a prediction entry kind — emitted by `AR1(replicate=)` (`compiler.py:1884`). It is a leftover from the old `AR1(group=)` spelling that slice 3's rename missed: `_grouped_structured_block` and `_replicated_block` are the same function apart from three message strings, and the predict error for a *replicate* reads `"group/level"`, which is precisely the R-INLA confusion slice 3 existed to remove.

Task 1 retires it. Adding `Grouped` under that name while it means the opposite would be worse than the duplication.

## File Structure

| File | Responsibility |
|---|---|
| `src/pylgm/effects/kronecker.py` (create) | `kron_block` and `kron_null_constraints` — the composition, shared by three callers. |
| `src/pylgm/effects/structures.py` (create) | The five between-group structures. |
| `src/pylgm/effects/random_walk.py` (modify) | `rw_structure` promoted out of `spacetime.py`. |
| `src/pylgm/effects/replicate.py` (modify) | `replicated_block` delegates; `grouped_block` added. |
| `src/pylgm/effects/spacetime.py` (modify) | `build_spacetime` delegates. |
| `src/pylgm/effects/spec.py` (modify) | The `Grouped` spec and its rejections. |
| `src/pylgm/compiler.py` (modify) | Wrapper cases in the four dispatch functions. |
| `src/pylgm/inference/prediction.py` (modify) | Retire the misnamed kind; `grouped_structured` in its true meaning. |
| `src/pylgm/__init__.py`, `effects/__init__.py` (modify) | Export `Grouped` and the five structures. |
| `tests/test_kronecker_kernel.py` (create) | The kernel in isolation. |
| `tests/test_structures.py` (create) | The five structures' precisions and null bases. |
| `tests/test_grouped_spec.py` (create) | Spec validation and rejections. |
| `tests/test_grouped_compile.py` (create) | Kronecker structure, constraints, alignment, reduction. |
| `tests/test_grouped_predict.py` (create) | Prediction round-trip. |
| `tests/test_grouped_spacetime_oracle.py` (create) | Knorr-Held I–IV equivalence. |

---

### Task 1: Retire the misnamed `grouped_structured` entry kind

**Files:**
- Modify: `src/pylgm/compiler.py:1880-1886`
- Modify: `src/pylgm/inference/prediction.py:51-54`, `:258-268`, `:380-381`
- Modify: `tests/test_ar1_group.py:136`

**Interfaces:**
- Consumes: nothing.
- Produces: the name `"grouped_structured"`, free for Task 7 to use in its true meaning. `AR1(replicate=)` emits `("replicated_structured", (name, replicate, index, replicate_labels, level_labels))`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ar1_group.py`:

```python
def test_ar1_replicate_emits_the_replicated_entry_kind():
    """AR1(replicate=) is a replicate, so its predict entry must say so.

    It emitted "grouped_structured" until slice 4: a leftover of the old
    AR1(group=) spelling, whose predict error told users their *replicate*
    had an unseen "group/level" -- the exact R-INLA confusion the rename
    existed to remove.
    """
    from pylgm.compiler import build_prediction_context
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _panel(groups=3, periods=4, rho=0.6, sd=0.3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + AR1("dyn", index="t", replicate="firm", precision=1.0, rho=0.6),
        likelihood=Gaussian(sigma=0.3),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="t", response="y", panel=("firm",))
    )
    context = build_prediction_context(model, panel)
    kinds = [kind for kind, _ in context.entries]
    assert "replicated_structured" in kinds
    assert "grouped_structured" not in kinds
```

Then change `tests/test_ar1_group.py:136` from `match="group/level"` to `match="replicate/level"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_ar1_group.py -q`
Expected: FAIL — `assert 'replicated_structured' in ['fixed', 'grouped_structured']`, plus the `match="replicate/level"` case failing on the old message.

If `build_prediction_context` has a different name or signature, run `grep -n "def build_prediction_context" src/pylgm/compiler.py` and adapt the call; do not change the assertions.

- [ ] **Step 3: Point AR1 at the replicated kind**

In `src/pylgm/compiler.py`, replace the `AR1`/`replicate` branch of `_prediction_entry`:

```python
    if isinstance(effect, AR1) and effect.replicate is not None:
        # "replicated_structured", not the "grouped_structured" this emitted
        # before slice 4: AR1(replicate=) is R-INLA's replicate, and the two
        # handlers were the same function apart from their message strings.
        replicate_labels = tuple(dict.fromkeys(la.split("@", 1)[0] for la in block.labels))
        level_labels = tuple(dict.fromkeys(la.split("@", 1)[1] for la in block.labels))
        return (
            "replicated_structured",
            (effect.name, effect.replicate, effect.index, replicate_labels, level_labels),
        )
```

In `src/pylgm/inference/prediction.py`, delete `_grouped_structured_block` entirely, delete the `elif kind == "grouped_structured":` branch from `_design_block_for`, and delete the `("grouped_structured", ...)` line from the module docstring at line 53.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_ar1_group.py tests/test_replicated_predict.py -q`
Expected: PASS.

Then the full suite: `PYTHONPATH=src python -m pytest -q`
Expected: `1339 passed`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(prediction): AR1(replicate=) emits the replicated entry kind

The tail of slice 3's rename. AR1(replicate=) still emitted
\"grouped_structured\", whose predict error told users their replicate had an
unseen \"group/level\" -- the R-INLA confusion the rename existed to remove.
_grouped_structured_block and _replicated_block were the same function apart
from three message strings, so this deletes a duplicate and frees the name for
slice 4's Grouped, where it means the opposite thing."
```

---

### Task 2: The Kronecker kernel

**Files:**
- Create: `src/pylgm/effects/kronecker.py`
- Test: `tests/test_kronecker_kernel.py`

**Interfaces:**
- Consumes: `LatentBlock` from `pylgm.ir.model`.
- Produces:
  - `kron_null_constraints(null_out, null_in, n_out, n_in, orthonormalise) -> np.ndarray`, shape `(k, n_out * n_in)`.
  - `kron_block(name, outer_labels, outer_precision, null_out, inner_labels, inner_precision, null_in, outer_positions, inner_positions, separator, orthonormalise) -> LatentBlock`. `outer_positions` and `inner_positions` are integer arrays with one entry per frame row; callers resolve and validate them.

**Why positions and not keys:** `replicated_block` and `build_spacetime` raise different, test-pinned messages for an unknown level. Unifying them would break those tests for no gain, so validation stays with the callers and the kernel is pure numpy/scipy.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kronecker_kernel.py`:

```python
import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity

from pylgm.effects.kronecker import kron_block, kron_null_constraints


def test_precision_is_the_kronecker_product_in_outer_major_order():
    outer = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    inner = csr_matrix(np.diag([1.0, 3.0, 5.0]))
    block = kron_block(
        "u", ("g1", "g2"), outer, np.zeros((2, 0)),
        ("a", "b", "c"), inner, np.zeros((3, 0)),
        np.array([0, 1]), np.array([2, 0]), "@", False,
    )
    assert np.allclose(block.precision.toarray(), np.kron(outer.toarray(), inner.toarray()))


def test_labels_pair_outer_major_with_the_given_separator():
    block = kron_block(
        "u", ("g1", "g2"), identity(2, format="csr"), np.zeros((2, 0)),
        ("a", "b"), identity(2, format="csr"), np.zeros((2, 0)),
        np.array([0]), np.array([0]), "|", False,
    )
    assert block.labels == ("g1|a", "g1|b", "g2|a", "g2|b")


def test_design_places_each_row_at_outer_times_inner_plus_inner():
    block = kron_block(
        "u", ("g1", "g2"), identity(2, format="csr"), np.zeros((2, 0)),
        ("a", "b", "c"), identity(3, format="csr"), np.zeros((3, 0)),
        np.array([0, 1, 1]), np.array([2, 0, 2]), "@", False,
    )
    dense = block.design.toarray()
    assert dense.shape == (3, 6)
    assert [row.argmax() for row in dense] == [0 * 3 + 2, 1 * 3 + 0, 1 * 3 + 2]
    assert np.allclose(dense.sum(axis=1), 1.0)


def test_no_null_on_either_factor_gives_no_constraints():
    got = kron_null_constraints(np.zeros((2, 0)), np.zeros((3, 0)), 2, 3, True)
    assert got.shape == (0, 6)


def test_a_single_part_is_left_alone_when_not_orthonormalising():
    """Slice 3's bit-for-bit equality with AR1(group=) depends on this.

    The SVD spans the same row space but returns different rows -- [1, 1, 1]
    comes back as [-0.577, -0.577, -0.577] -- so Replicated must get the
    literal kron(I_R, C) it has always produced.
    """
    inner_null = np.ones((3, 1))
    got = kron_null_constraints(np.zeros((2, 0)), inner_null, 2, 3, False)
    assert np.array_equal(got, np.kron(np.eye(2), inner_null.T))


def test_a_single_part_is_orthonormalised_when_asked():
    """SpaceTime types II and III have one part and orthonormalise today."""
    got = kron_null_constraints(np.zeros((2, 0)), np.ones((3, 1)), 2, 3, True)
    assert got.shape == (2, 6)
    assert np.allclose(got @ got.T, np.eye(2))


def test_two_parts_always_orthonormalise_even_when_not_asked():
    """The two spans share 1_out (x) 1_in; keeping it twice is rank-deficient.

    Stacking them raw gives 2 + 3 = 5 rows for a 4-dimensional space, and a
    rank-deficient constraint matrix breaks the constrained solve.
    """
    got = kron_null_constraints(np.ones((2, 1)), np.ones((3, 1)), 2, 3, False)
    assert got.shape == (4, 6)
    assert np.linalg.matrix_rank(got) == 4
    assert np.allclose(got @ got.T, np.eye(4))


def test_the_constraint_span_is_the_precision_null_space():
    """The invariant the whole slice rests on, checked directly."""
    outer = csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))     # null = span{1}
    inner = csr_matrix(np.diag([1.0, 2.0]))                       # proper
    got = kron_null_constraints(np.ones((2, 1)), np.zeros((2, 0)), 2, 2, False)
    q = np.kron(outer.toarray(), inner.toarray())
    assert np.allclose(q @ got.T, 0.0)
    assert got.shape[0] == q.shape[0] - np.linalg.matrix_rank(q)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_kronecker_kernel.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm.effects.kronecker'`

- [ ] **Step 3: Implement the kernel**

Create `src/pylgm/effects/kronecker.py`:

```python
# src/pylgm/effects/kronecker.py
"""The Kronecker composition shared by Replicated, Grouped and SpaceTime.

One outer factor, one inner factor, cells laid out outer-major:

    Q      = Q_out (x) Q_in
    cell   = outer_position * n_inner + inner_position
    null(Q) = null(Q_out) (x) R^in  +  R^out (x) null(Q_in)

A block's ``constraints`` are a basis of its precision's null space -- RW1
carries one, RW2 two, Besag one per connected component -- so that last line
covers every composition in the library. ``Replicated`` is its degenerate case:
``null(I_R)`` is empty, the first term vanishes, and ``kron(I_R, C)`` remains.

Callers pass already-resolved integer positions and own their own validation:
``replicated_block`` and ``build_spacetime`` raise different, test-pinned
messages for an unknown level. This module is therefore pure numpy/scipy.
"""

import numpy as np
from scipy.sparse import csr_matrix, kron

from pylgm.ir.model import LatentBlock


def kron_null_constraints(
    null_out: np.ndarray,
    null_in: np.ndarray,
    n_out: int,
    n_in: int,
    orthonormalise: bool,
) -> np.ndarray:
    """Constraint rows spanning ``null(Q_out (x) Q_in)``, shape ``(k, n_out*n_in)``.

    ``orthonormalise`` governs the **one-part** case only, and exists to
    preserve two released outputs rather than for any modelling reason -- both
    settings span the same subspace, so no fit changes either way:

    - ``False`` returns the raw ``kron(I_out, N_in).T``, which is the literal
      ``kron(I_R, C)`` ``Replicated`` has always produced and whose bit-for-bit
      equality with the shipped ``AR1(group=)`` is slice 3's safety property.
    - ``True`` matches ``build_spacetime``, which orthonormalises today
      including for types II and III where only one part is present.

    When **both** parts are present the flag is ignored. The two spans overlap
    in ``1_out (x) 1_in``, and dropping that duplicate is what the SVD is for;
    keeping it would return a rank-deficient constraint matrix.
    """
    parts = []
    if null_out.shape[1]:
        parts.append(np.kron(null_out, np.eye(n_in)))
    if null_in.shape[1]:
        parts.append(np.kron(np.eye(n_out), null_in))
    if not parts:
        return np.zeros((0, n_out * n_in))
    if len(parts) == 1 and not orthonormalise:
        return parts[0].T
    b = np.hstack(parts)
    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10)) if singular.size else 0
    return u[:, :rank].T


def kron_block(
    name: str,
    outer_labels: tuple[str, ...],
    outer_precision: csr_matrix,
    null_out: np.ndarray,
    inner_labels: tuple[str, ...],
    inner_precision: csr_matrix,
    null_in: np.ndarray,
    outer_positions: np.ndarray,
    inner_positions: np.ndarray,
    separator: str,
    orthonormalise: bool,
) -> LatentBlock:
    """Compose two factors into one outer-major ``LatentBlock``.

    ``outer_positions`` and ``inner_positions`` hold one already-validated
    index per frame row. ``separator`` is ``"@"`` for Replicated and Grouped
    and ``"|"`` for SpaceTime, whose labels are user-visible in
    ``result.labels`` and therefore cannot change.
    """
    n_out, n_in = len(outer_labels), len(inner_labels)
    width = n_out * n_in
    cells = np.asarray(outer_positions) * n_in + np.asarray(inner_positions)
    rows = len(cells)
    design = csr_matrix(
        (np.ones(rows), (np.arange(rows), cells)), shape=(rows, width)
    )
    precision = csr_matrix(kron(outer_precision, inner_precision, format="csr"))
    constraints = kron_null_constraints(null_out, null_in, n_out, n_in, orthonormalise)
    labels = tuple(f"{o}{separator}{i}" for o in outer_labels for i in inner_labels)
    return LatentBlock(name, labels, design, precision, constraints)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_kronecker_kernel.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/kronecker.py tests/test_kronecker_kernel.py
git commit -m "feat(effects): one Kronecker composition for three callers

null(Q_out (x) Q_in) = null(Q_out) (x) R^in + R^out (x) null(Q_in) covers every
composition in the library, because a block's constraints are a basis of its
precision's null space. Replicated is the degenerate case where null(I_R) is
empty.

orthonormalise is a back-compat flag, not a modelling choice: both settings
span the same subspace. It governs the one-part case only -- two parts always
orthonormalise, since their overlap would otherwise leave the constraints
rank-deficient."
```

---

### Task 3: `replicated_block` and `build_spacetime` delegate to the kernel

**Files:**
- Modify: `src/pylgm/effects/replicate.py:26-67`
- Modify: `src/pylgm/effects/spacetime.py:53-72`, `:113-142`
- Test: `tests/test_kronecker_delegation.py` (create)

**Interfaces:**
- Consumes: `kron_block`, `kron_null_constraints` from Task 2.
- Produces: no signature change. `replicated_block` and `build_spacetime` keep their exact current signatures and outputs.

**This is the risky task.** `build_spacetime` is released and tested. The refactor must be provably output-preserving, so Step 1 pins the current output *before* the code moves.

- [ ] **Step 1: Snapshot the current output, before changing anything**

Run this against the **unmodified** `build_spacetime`:

```bash
PYTHONPATH=src python - <<'EOF'
import numpy as np, pandas as pd, pathlib
from pylgm.effects.spacetime import build_spacetime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
frame = pd.DataFrame(
    [{"s": a, "t": t, "y": 0.0} for a in ("a", "b", "c", "d") for t in range(5)]
)
out = pathlib.Path("tests/data/spacetime_snapshots.npz")
out.parent.mkdir(parents=True, exist_ok=True)
arrays = {}
for interaction in ("I", "II", "III", "IV"):
    for order in (1, 2):
        block = build_spacetime(
            frame, "st", "s", "t", GRAPH, interaction, order, precision=1.5
        )
        key = f"{interaction}_{order}"
        arrays[f"{key}_q"] = block.precision.toarray()
        arrays[f"{key}_c"] = block.constraints
        arrays[f"{key}_d"] = block.design.toarray()
np.savez_compressed(out, **arrays)
print("wrote", out, len(arrays), "arrays")
EOF
```

Expected: `wrote tests/data/spacetime_snapshots.npz 24 arrays`

Then create `tests/test_kronecker_delegation.py`:

```python
"""build_spacetime's output, pinned bit-for-bit across the kernel refactor.

The snapshots were generated from the pre-refactor implementation, so a
difference here is a regression rather than an updated expectation. This is the
only guard on a released, tested path being rewritten underneath.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm.effects.spacetime import build_spacetime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
SNAPSHOTS = np.load("tests/data/spacetime_snapshots.npz")


def _frame():
    return pd.DataFrame(
        [{"s": a, "t": t, "y": 0.0} for a in ("a", "b", "c", "d") for t in range(5)]
    )


@pytest.mark.parametrize("interaction", ["I", "II", "III", "IV"])
@pytest.mark.parametrize("order", [1, 2])
def test_spacetime_output_is_unchanged_by_the_kernel_refactor(interaction, order):
    block = build_spacetime(
        _frame(), "st", "s", "t", GRAPH, interaction, order, precision=1.5
    )
    key = f"{interaction}_{order}"
    assert np.array_equal(block.precision.toarray(), SNAPSHOTS[f"{key}_q"])
    assert np.array_equal(block.constraints, SNAPSHOTS[f"{key}_c"])
    assert np.array_equal(block.design.toarray(), SNAPSHOTS[f"{key}_d"])
    assert block.labels[:3] == ("a|0", "a|1", "a|2")
```

- [ ] **Step 2: Run it to confirm it passes pre-refactor**

Run: `PYTHONPATH=src python -m pytest tests/test_kronecker_delegation.py -q`
Expected: `8 passed`, against the unmodified `build_spacetime`. This is the baseline the refactor must reproduce. Commit the snapshot file now, so the baseline is in git before any code moves.

```bash
git add tests/data/spacetime_snapshots.npz tests/test_kronecker_delegation.py
git commit -m "test(spacetime): snapshot the pre-refactor output as a regression guard"
```

- [ ] **Step 3: Delegate `replicated_block`**

In `src/pylgm/effects/replicate.py`, replace the body of `replicated_block` after the `unknown` validation with:

```python
    replicate_positions = np.array([replicate_position[str(r)] for r in frame[over]])
    level_positions = np.array([level_position[t] for t in keys])
    return kron_block(
        inner.name,
        replicates, identity(n_replicates, format="csr"), np.zeros((n_replicates, 0)),
        levels, inner.precision, inner.constraints.T,
        replicate_positions, level_positions,
        separator="@", orthonormalise=False,
    )
```

Add `from pylgm.effects.kronecker import kron_block` to the imports and drop the now-unused `kron` import if nothing else uses it.

The `orthonormalise=False` is what keeps `kron(I_R, C)` literal; see Task 2.

- [ ] **Step 4: Delegate `build_spacetime`**

In `src/pylgm/effects/spacetime.py`, delete `_interaction_constraints` entirely and replace the block from `precision_matrix = ...` to the `return` with:

```python
    area_pos = {area: i for i, area in enumerate(areas)}
    time_pos = {t: j for j, t in enumerate(times)}
    observed_area = [str(v) for v in frame[space]]
    observed_time = [str(v) for v in frame[time]]
    missing_area = sorted({v for v in observed_area if v not in area_pos})
    if missing_area:
        raise ValueError(f"observed {space!r} level(s) {missing_area!r} not in the area universe")

    # precision_scale, not a scalar folded into k_s: IEEE multiplication is not
    # associative, and kron(precision * k_s, k_t) differs from
    # precision * kron(k_s, k_t) by up to 3.6e-15 -- enough to break the
    # bit-for-bit snapshot from Step 1. The kernel applies the scalar exactly
    # where build_spacetime applies it today.
    return kron_block(
        name,
        areas, k_s, _space_null_basis(interaction, w, S),
        times, k_t, _time_null_basis(interaction, order, T),
        np.array([area_pos[a] for a in observed_area]),
        np.array([time_pos[t] for t in observed_time]),
        separator="|", orthonormalise=True, precision_scale=precision,
    )
```

Add `from pylgm.effects.kronecker import kron_block` and drop `LatentBlock` and `kron` from the imports if unused.

- [ ] **Step 5: Run the delegation, spacetime, replicated and AR1 suites**

Run: `PYTHONPATH=src python -m pytest tests/test_kronecker_delegation.py tests/test_spacetime.py tests/test_replicated_compile.py tests/test_replicated_equivalence.py tests/test_ar1_group.py -q`
Expected: all pass, with the snapshots from Step 1 still matching.

If a spacetime snapshot differs, **do not update the snapshot** — it is the only guard on this rewrite. Check `precision_scale` is being passed rather than folded into a factor, and that `_interaction_constraints` was deleted rather than left shadowing the kernel.

- [ ] **Step 6: Run the full suite and commit**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: `1347 passed`

```bash
git add -A
git commit -m "refactor(effects): replicated_block and build_spacetime share the kernel

Both compositions were the same Kronecker assembly written twice, including
the null-space constraint logic -- the part most likely to be got subtly wrong
in two places. build_spacetime's output is pinned bit-for-bit across the move
in tests/test_kronecker_delegation.py, written against the pre-refactor code."
```

---

### Task 4: The five between-group structures

**Files:**
- Create: `src/pylgm/effects/structures.py`
- Modify: `src/pylgm/effects/random_walk.py` (add `rw_structure`)
- Modify: `src/pylgm/effects/spacetime.py` (import `rw_structure`, delete `_rw_structure`)
- Test: `tests/test_structures.py`

**Interfaces:**
- Consumes: `normalize_graph` (`effects/graph.py`), `_scaled_structure` (`effects/besag.py`), `ar1_structure` (`effects/ar1.py`), `difference_operator` and the new `rw_structure` (`effects/random_walk.py`), `sorbye_rue_scale` (`effects/scaling.py`).
- Produces: `IIDStructure()`, `AR1Structure(rho)`, `RW1Structure()`, `RW2Structure()`, `BesagStructure(graph)`. Each is a frozen dataclass with:
  - `levels(observed: tuple[str, ...]) -> tuple[str, ...]` — the group universe.
  - `precision(levels: tuple[str, ...]) -> csr_matrix`
  - `null_basis(levels: tuple[str, ...]) -> np.ndarray` of shape `(len(levels), null_dim)`

**Why `levels()` and not just a dimension:** `BesagStructure` carries named nodes, and the graph is the universe — a node with no observations still gets its cell, which is what lets the spatial smoothing borrow strength for it. The other four are anonymous and return the observed levels unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_structures.py`:

```python
import numpy as np
import pytest

from pylgm.effects.structures import (
    AR1Structure, BesagStructure, IIDStructure, RW1Structure, RW2Structure,
)

LEVELS = ("g1", "g2", "g3", "g4")
GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}


def test_iid_is_the_identity_with_no_null():
    s = IIDStructure()
    assert np.allclose(s.precision(LEVELS).toarray(), np.eye(4))
    assert s.null_basis(LEVELS).shape == (4, 0)
    assert s.levels(LEVELS) == LEVELS


def test_ar1_is_proper_so_it_has_no_null():
    s = AR1Structure(rho=0.6)
    q = s.precision(LEVELS).toarray()
    assert q.shape == (4, 4)
    assert np.linalg.matrix_rank(q) == 4
    assert s.null_basis(LEVELS).shape == (4, 0)


def test_ar1_structure_matches_the_ar1_effect_builder():
    from pylgm.effects.ar1 import ar1_structure
    assert np.allclose(
        AR1Structure(rho=0.6).precision(LEVELS).toarray(), ar1_structure(4, 0.6).toarray()
    )


def test_ar1_rejects_a_rho_outside_the_stationary_range():
    with pytest.raises(ValueError, match="rho"):
        AR1Structure(rho=1.0)


@pytest.mark.parametrize("structure,null_dim", [(RW1Structure(), 1), (RW2Structure(), 2)])
def test_random_walk_null_dimension_matches_its_order(structure, null_dim):
    q = structure.precision(LEVELS).toarray()
    basis = structure.null_basis(LEVELS)
    assert basis.shape == (4, null_dim)
    assert np.allclose(q @ basis, 0.0)


def test_rw2_null_is_the_constant_and_the_centred_ramp():
    basis = RW2Structure().null_basis(LEVELS)
    assert np.allclose(basis[:, 0], np.ones(4))
    assert np.allclose(basis[:, 1], np.arange(4) - 1.5)


def test_besag_takes_its_universe_from_the_graph_not_the_observed_levels():
    """A node with no observations keeps its cell, so smoothing lends it strength."""
    s = BesagStructure(GRAPH)
    assert s.levels(("a", "b")) == ("a", "b", "c", "d")


def test_besag_precision_and_null_come_from_the_graph():
    s = BesagStructure(GRAPH)
    nodes = s.levels(("a",))
    q = s.precision(nodes).toarray()
    basis = s.null_basis(nodes)
    assert q.shape == (4, 4)
    assert basis.shape == (4, 1)          # one connected component
    assert np.allclose(q @ basis, 0.0)


def test_besag_null_has_one_column_per_connected_component():
    s = BesagStructure({"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]})
    assert s.null_basis(s.levels(())).shape == (4, 2)


def test_besag_rejects_levels_that_are_not_the_universe_it_returned():
    """Silently ignoring the argument is the permuted-neighbourhood bug."""
    s = BesagStructure(GRAPH)
    with pytest.raises(ValueError, match="graph orders nodes"):
        s.precision(("b", "a", "c", "d"))


def test_a_level_outside_the_graph_is_rejected_by_name():
    """Aligning by position instead would permute the neighbourhood silently."""
    with pytest.raises(ValueError, match="zz"):
        BesagStructure(GRAPH).levels(("a", "zz"))


def test_an_anonymous_structure_keeps_the_observed_levels():
    for s in (IIDStructure(), AR1Structure(rho=0.3), RW1Structure(), RW2Structure()):
        assert s.levels(("x", "y")) == ("x", "y")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_structures.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm.effects.structures'`

- [ ] **Step 3: Promote `rw_structure` out of `spacetime.py`**

In `src/pylgm/effects/random_walk.py`, add after `difference_operator`:

```python
def rw_structure(level_count: int, order: Literal[1, 2], scale: bool = True) -> np.ndarray:
    """Sørbye-Rue-scaled RW structure ``DᵀD`` over ``level_count`` points.

    Lives here rather than in spacetime.py, which is where it started: both the
    space-time interaction and the between-group structures need it, and both
    already depend on this module for ``difference_operator``.
    """
    difference = difference_operator(level_count, order)
    r = (difference.T @ difference).toarray()
    return sorbye_rue_scale(r, null_dim=order) if scale else r
```

Add `from pylgm.effects.scaling import sorbye_rue_scale` to that module's imports if absent.

In `src/pylgm/effects/spacetime.py`, delete `_rw_structure` and replace its one call site with `rw_structure(T, order, scale)`, importing it from `pylgm.effects.random_walk`.

- [ ] **Step 4: Implement the structures**

Create `src/pylgm/effects/structures.py`:

```python
# src/pylgm/effects/structures.py
"""Between-group precisions for ``Grouped`` -- R-INLA's ``control.group``.

Deliberately separate from the effect specs of similar name: an effect carries
an index column and builds a design, whereas a structure carries only a
precision over the group levels and never touches the frame.

Each exposes ``levels`` (the group universe), ``precision``, and
``null_basis``. The null basis is what the Kronecker kernel needs to assemble
the composed block's constraints, and it is *not* derivable from the precision
without an eigendecomposition, so every structure states its own.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import connected_components

from pylgm.effects.ar1 import ar1_structure
from pylgm.effects.besag import _scaled_structure
from pylgm.effects.graph import normalize_graph
from pylgm.effects.random_walk import rw_structure


@dataclass(frozen=True)
class IIDStructure:
    """Independent groups. ``Grouped`` with this is exactly ``Replicated``.

    Kept as API because Knorr-Held types I, II and III each have an iid factor,
    so the equivalence oracle against ``SpaceTime`` needs it expressible.
    """

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        return observed

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        return identity(len(levels), format="csr")

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        return np.zeros((len(levels), 0))


@dataclass(frozen=True)
class AR1Structure:
    """Autoregressively correlated groups -- the panel case.

    ``rho`` is fixed in this slice, matching the restriction ``Shared`` carries
    today; estimating a structure's own hyperparameter jointly with the inner
    effect's is out of scope.
    """

    rho: float

    def __post_init__(self) -> None:
        if not isinstance(self.rho, (int, float)) or isinstance(self.rho, bool):
            raise TypeError("AR1Structure rho must be a real number")
        if not -1.0 < float(self.rho) < 1.0:
            raise ValueError("AR1Structure rho must lie strictly inside (-1, 1)")

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        return observed

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        return ar1_structure(len(levels), float(self.rho))

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        # The stationary AR1 precision is proper: no null space.
        return np.zeros((len(levels), 0))


class _RandomWalkStructure:
    """Shared body of RW1Structure and RW2Structure.

    A plain mixin, not a dataclass: ``order`` is declared by each subclass, and
    a fieldless frozen dataclass reading ``self.order`` would read as a defect.
    """

    order: int

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        return observed

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        if len(levels) <= self.order:
            raise ValueError(
                f"RW{self.order}Structure needs more than {self.order} group "
                f"level(s), got {len(levels)}"
            )
        return csr_matrix(rw_structure(len(levels), self.order, scale=True))

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        count = len(levels)
        columns = [np.ones(count)]
        if self.order == 2:
            coordinate = np.arange(count, dtype=float)
            columns.append(coordinate - coordinate.mean())
        return np.column_stack(columns)


@dataclass(frozen=True)
class RW1Structure(_RandomWalkStructure):
    """First-order random walk between groups: null is the constant."""

    order: int = 1


@dataclass(frozen=True)
class RW2Structure(_RandomWalkStructure):
    """Second-order random walk between groups: null is the constant and ramp."""

    order: int = 2


@dataclass(frozen=True)
class BesagStructure:
    """Spatially structured groups (ICAR), aligned to the graph **by name**.

    The graph is the universe, not the observed levels: a node with no
    observations still gets its cell, so the spatial smoothing lends it
    strength. This matches ``Besag`` and ``build_spacetime``.

    Positional alignment would permute the neighbourhood structure silently --
    the fit would converge and return plausible numbers -- so an observed level
    outside the node set is a hard error.
    """

    graph: Mapping

    def _normalized(self):
        return normalize_graph(dict(self.graph))

    def levels(self, observed: tuple[str, ...]) -> tuple[str, ...]:
        nodes, _ = self._normalized()
        unknown = sorted({str(v) for v in observed} - set(nodes))
        if unknown:
            raise ValueError(
                f"BesagStructure graph has no node(s) {unknown!r}; a group level "
                "outside the graph cannot be aligned to a neighbourhood"
            )
        return nodes

    def _checked(self, levels: tuple[str, ...]):
        """The graph, asserting the caller passed the universe ``levels()`` gave.

        Both methods below would otherwise silently ignore their argument and
        return a matrix ordered by the graph while the caller indexed by
        something else -- the permuted-neighbourhood failure this class exists
        to prevent.
        """
        nodes, w = self._normalized()
        if tuple(levels) != nodes:
            raise ValueError(
                f"BesagStructure was given group levels {tuple(levels)!r} but its "
                f"graph orders nodes {nodes!r}; pass the tuple levels() returned"
            )
        return nodes, w

    def precision(self, levels: tuple[str, ...]) -> csr_matrix:
        nodes, w = self._checked(levels)
        return csr_matrix(_scaled_structure(w, nodes, scale=True))

    def null_basis(self, levels: tuple[str, ...]) -> np.ndarray:
        nodes, w = self._checked(levels)
        count, membership = connected_components(w, directed=False)
        basis = np.zeros((len(nodes), count))
        for component in range(count):
            basis[membership == component, component] = 1.0
        return basis
```

Export all five from `src/pylgm/effects/__init__.py` and `src/pylgm/__init__.py`, adding them to both `__all__` lists.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_structures.py tests/test_spacetime.py -q`
Expected: `test_structures.py` all pass; `test_spacetime.py` unchanged after the `rw_structure` move.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(effects): five between-group structures for Grouped

Separate from the effect specs of similar name: an effect carries an index and
builds a design, a structure carries only a precision over the group levels and
never touches the frame.

BesagStructure aligns to its graph by NAME and takes the graph as the level
universe, matching Besag and build_spacetime -- a node with no observations
keeps its cell so smoothing lends it strength, and positional alignment would
permute the neighbourhood while still fitting.

rw_structure moves from spacetime.py to random_walk.py, where
difference_operator already lives and both callers can reach it."
```

---

### Task 5: The `Grouped` spec

**Files:**
- Modify: `src/pylgm/effects/spec.py`
- Modify: `src/pylgm/__init__.py`, `src/pylgm/effects/__init__.py`
- Test: `tests/test_grouped_spec.py`

**Interfaces:**
- Consumes: `_ComposableEffect` (`spec.py:35`); `Weighted`, `Copy`, `Replicated`, already in that file; the five structures from Task 4.
- Produces: `Grouped(effect, over, structure)` — frozen dataclass with `.effect`, `.over: str`, `.structure`, and a `.name` property delegating to `effect.name`, exactly as `Weighted` and `Replicated` do.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grouped_spec.py`:

```python
import pytest

from pylgm import (
    AR1, BesagStructure, Copy, Fixed, Grouped, IID, IIDStructure, Replicated,
    RW1Structure, Weighted,
)
from pylgm.effects.spec import Predictor

GRAPH = {"a": ["b"], "b": ["a"]}


def test_grouped_delegates_its_name_to_the_inner_effect():
    assert Grouped(IID("u", index="t"), over="region", structure=RW1Structure()).name == "u"


def test_grouped_keeps_the_effect_the_column_and_the_structure():
    inner, structure = IID("u", index="t"), RW1Structure()
    wrapped = Grouped(inner, over="region", structure=structure)
    assert wrapped.effect is inner
    assert wrapped.over == "region"
    assert wrapped.structure is structure


def test_grouped_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + Grouped(IID("u", index="t"), over="r", structure=IIDStructure())
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 2


def test_grouped_rejects_an_empty_over():
    with pytest.raises(ValueError, match="over"):
        Grouped(IID("u", index="t"), over="", structure=IIDStructure())


def test_grouped_rejects_an_effect_with_no_index():
    with pytest.raises(TypeError, match="index"):
        Grouped(Fixed("1"), over="r", structure=IIDStructure())


def test_grouped_rejects_something_that_is_not_a_structure():
    with pytest.raises(TypeError, match="structure"):
        Grouped(IID("u", index="t"), over="r", structure="besag")


def test_grouped_rejects_wrapping_a_grouped():
    with pytest.raises(TypeError, match="already grouped"):
        Grouped(
            Grouped(IID("u", index="t"), over="r", structure=IIDStructure()),
            over="year", structure=IIDStructure(),
        )


def test_grouped_rejects_wrapping_a_replicated():
    """R-INLA permits group and replicate together; pyLGM does not.

    The labels would become r@g@level, and both _prediction_entry's
    split("@", 1) and the single-inner-index assumption would have to be
    generalised. Recorded as an f() parity gap, not half-implemented.
    """
    with pytest.raises(TypeError, match="replicate"):
        Grouped(
            Replicated(IID("u", index="t"), over="firm"),
            over="year", structure=IIDStructure(),
        )


def test_replicated_rejects_wrapping_a_grouped():
    with pytest.raises(TypeError, match="group"):
        Replicated(
            Grouped(IID("u", index="t"), over="r", structure=IIDStructure()),
            over="firm",
        )


def test_grouped_rejects_an_ar1_that_already_replicates_itself():
    with pytest.raises(TypeError, match="replicate"):
        Grouped(AR1("t", index="year", replicate="firm"), over="r", structure=IIDStructure())


def test_grouped_rejects_wrapping_a_copy():
    with pytest.raises(TypeError):
        Grouped(Copy("u", index="j"), over="r", structure=IIDStructure())


def test_grouped_may_wrap_a_weighted_effect():
    wrapped = Grouped(
        Weighted(IID("u", index="t"), by="z"), over="r", structure=BesagStructure(GRAPH)
    )
    assert wrapped.name == "u"


def test_weighted_may_wrap_a_grouped_effect():
    wrapped = Weighted(
        Grouped(IID("u", index="t"), over="r", structure=IIDStructure()), by="z"
    )
    assert wrapped.name == "u"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_grouped_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'Grouped' from 'pylgm'`

- [ ] **Step 3: Implement the spec**

In `src/pylgm/effects/spec.py`, add next to `Replicated`:

```python
@dataclass(frozen=True)
class Grouped(_ComposableEffect):
    """``G`` *correlated* copies of an effect, with a between-group structure.

    ``Grouped(Besag("s", index="district", graph=g), over="year",
    structure=AR1Structure(rho=0.8))`` is one spatial field per year, with the
    years tied together by an AR1. This is R-INLA's ``f(index, model=...,
    group=g, control.group=list(model=...))``.

    The precision becomes ``Q_S (x) Q_E``. Contrast ``Replicated``, whose
    copies are independent: ``I_R (x) Q_E``, the special case where the
    between-group structure is the identity.

    Constraints follow the null space of the product, which is *not* one
    constraint per group: ``null(Q_S (x) Q_E)`` picks up ``null(Q_S) (x) R^E``
    as well, and the two spans overlap.
    """

    effect: object
    over: str
    structure: object

    def __post_init__(self) -> None:
        if isinstance(self.effect, Grouped):
            raise TypeError(
                "Grouped effect is already grouped; two group columns is one "
                "group over their cross product, so combine them into a single "
                "column"
            )
        if isinstance(self.effect, Replicated):
            raise TypeError(
                "Grouped cannot wrap a Replicated: R-INLA allows `group` and "
                "`replicate` on one term, but pyLGM does not, because the "
                "labels would become 'replicate@group@level' and the predict "
                "path resolves exactly one pair. Use one or the other."
            )
        if getattr(self.effect, "replicate", None) is not None:
            raise TypeError(
                f"{type(self.effect).__name__} already replicates itself through "
                "its own `replicate` argument; combining it with a group would "
                "give two copy mechanisms on one effect with no defined "
                "interaction"
            )
        object.__setattr__(self, "over", _non_empty_string(self.over, "over"))
        if not all(
            hasattr(self.structure, method)
            for method in ("levels", "precision", "null_basis")
        ):
            raise TypeError(
                "Grouped requires a between-group structure (IIDStructure, "
                "AR1Structure, RW1Structure, RW2Structure, BesagStructure), got "
                f"{type(self.structure).__name__}"
            )
        # Resolve the index THROUGH a Weighted wrapper, for the same reason
        # Replicated does: giving Weighted an `index` of its own turns
        # joint.Shared's "wrapper, cannot be shared" guard into dead code.
        target = self.effect.effect if isinstance(self.effect, Weighted) else self.effect
        if not hasattr(target, "index"):
            raise TypeError(
                f"Grouped requires an indexed effect, got "
                f"{type(self.effect).__name__}, which has no index."
            )
        if isinstance(self.effect, Copy):
            raise TypeError(
                "Grouped cannot wrap a Copy: a copy is a term referencing "
                "another term, not an indexed effect of its own. Group the "
                "target effect instead."
            )

    @property
    def name(self) -> str:
        return self.effect.name
```

In `Replicated.__post_init__`, add the mirror rejection immediately after the existing `isinstance(self.effect, Replicated)` check:

```python
        if isinstance(self.effect, Grouped):
            raise TypeError(
                "Replicated cannot wrap a Grouped: R-INLA allows `group` and "
                "`replicate` on one term, but pyLGM does not, because the "
                "labels would become 'replicate@group@level' and the predict "
                "path resolves exactly one pair. Use one or the other."
            )
```

`Grouped` is defined after `Replicated` in the module, so reference it lazily if the forward reference bites — move `Grouped` above `Replicated`, or compare `type(self.effect).__name__ == "Grouped"`. Prefer moving the class.

Export `Grouped` from both `__init__.py` files and add it to both `__all__` lists.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_grouped_spec.py tests/test_replicated_spec.py tests/test_joint.py -q`
Expected: all pass. `test_joint.py` matters: it holds the `hasattr(effect, "index")` guard that a careless `Grouped.index` would silently disable, which is exactly what happened in slice 3.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(effects): the Grouped spec -- correlated copies

Rejects Grouped-and-Replicated on one effect in both nesting orders. R-INLA
permits it; pyLGM does not, because the labels would become
replicate@group@level and _prediction_entry resolves exactly one pair.

Resolves the index THROUGH a Weighted wrapper rather than giving Weighted an
index of its own -- slice 3 shipped that mistake and silently disabled
joint.Shared's wrapper guard."
```

---

### Task 6: Compile `Grouped`

**Files:**
- Modify: `src/pylgm/compiler.py` — `_build_effect_block`, `_append_family_blocks`, `_effect_hyperparameters`, `_prediction_entry`
- Modify: `src/pylgm/effects/replicate.py` — add `grouped_block`
- Test: `tests/test_grouped_compile.py`

**Interfaces:**
- Consumes: `kron_block` (Task 2), the structures (Task 4), the `Grouped` spec (Task 5).
- Produces: `grouped_block(inner, frame, index, over, structure) -> LatentBlock`; `Grouped` branches in all four compiler dispatch functions.

**`_prediction_entry` is in this task, not a later one.** `.fit()` always builds a `PredictionContext`, so a `Grouped` model cannot be fitted at all until that branch exists. The plans for slices 1, 2 and 3 each deferred it and each had to be corrected mid-execution.

- [ ] **Step 1: Write the failing test**

Create `tests/test_grouped_compile.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Fixed, Gaussian, Grouped, IID, IIDStructure, LGM, Poisson,
    Replicated, RW1, RW1Structure, Weighted,
)
from pylgm.compiler import _build_effect_block
from pylgm.exceptions import CompilationError
from pylgm.parameters import Hyperparameter

GRAPH = {"r1": ["r2"], "r2": ["r1", "r3"], "r3": ["r2"]}


def _frame():
    rows = []
    for region in ("r1", "r2", "r3"):
        for t in ("a", "b"):
            rows.append({"region": region, "t": t, "z": 2.0, "y": 1.0})
    frame = pd.DataFrame(rows)
    frame["row"] = range(len(frame))
    return frame


def test_precision_is_the_kronecker_product_of_structure_and_inner():
    frame = _frame()
    inner, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t", precision=2.0), over="region",
                structure=BesagStructure(GRAPH)),
        frame,
    )
    structure = BesagStructure(GRAPH).precision(("r1", "r2", "r3")).toarray()
    # inner is built over the level set alone, so its precision is 2x2 here
    assert outer.precision.shape == (6, 6)
    assert np.allclose(
        outer.precision.toarray(), np.kron(structure, inner.precision.toarray())
    )


def test_labels_are_group_major_pairs_with_the_replicated_separator():
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t"), over="region", structure=IIDStructure()), _frame()
    )
    assert outer.labels == ("r1@a", "r1@b", "r2@a", "r2@b", "r3@a", "r3@b")


def test_an_iid_structure_reduces_grouped_to_replicated():
    frame = _frame()
    grouped, _ = _build_effect_block(
        Grouped(IID("u", index="t", precision=1.5), over="region",
                structure=IIDStructure()),
        frame,
    )
    replicated, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.5), over="region"), frame
    )
    assert grouped.labels == replicated.labels
    assert np.allclose(grouped.design.toarray(), replicated.design.toarray())
    assert np.allclose(grouped.precision.toarray(), replicated.precision.toarray())


def test_a_single_group_level_reduces_to_the_bare_effect():
    frame = pd.DataFrame({"region": ["r1"] * 3, "t": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
    bare, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    grouped, _ = _build_effect_block(
        Grouped(IID("u", index="t", precision=2.0), over="region",
                structure=IIDStructure()),
        frame,
    )
    assert np.allclose(grouped.precision.toarray(), bare.precision.toarray())
    assert np.allclose(grouped.design.toarray(), bare.design.toarray())


def test_a_correlated_structure_is_not_block_diagonal():
    """The whole point of Grouped: groups are coupled, unlike Replicated."""
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
        _frame(),
    )
    dense = outer.precision.toarray()
    # r1's cells (rows 0-1) must couple to r2's (columns 2-3)
    assert not np.allclose(dense[0:2, 2:4], 0.0)


def test_constraints_span_the_null_space_of_the_composed_precision():
    outer, _ = _build_effect_block(
        Grouped(RW1("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
        _frame(),
    )
    q = outer.precision.toarray()
    assert outer.constraints.shape[0] == q.shape[0] - np.linalg.matrix_rank(q)
    assert np.allclose(q @ outer.constraints.T, 0.0)


def test_a_group_level_outside_the_graph_is_rejected():
    frame = _frame()
    frame.loc[0, "region"] = "elsewhere"
    with pytest.raises((CompilationError, ValueError), match="elsewhere"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
            frame,
        )


def test_a_missing_group_column_is_named():
    with pytest.raises((CompilationError, ValueError), match="region"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=IIDStructure()),
            pd.DataFrame({"t": ["a", "b"], "y": [1.0, 2.0]}),
        )


def test_a_structure_whose_size_disagrees_with_the_levels_is_rejected():
    with pytest.raises((CompilationError, ValueError), match="level"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=RW1Structure()),
            pd.DataFrame({"region": ["only"], "t": ["a"], "y": [1.0]}),
        )


def test_grouped_and_weighted_commute():
    frame = _frame()
    inside, _ = _build_effect_block(
        Grouped(Weighted(IID("u", index="t"), by="z"), over="region",
                structure=BesagStructure(GRAPH)),
        frame,
    )
    outside, _ = _build_effect_block(
        Weighted(Grouped(IID("u", index="t"), over="region",
                         structure=BesagStructure(GRAPH)), by="z"),
        frame,
    )
    assert inside.labels == outside.labels
    assert np.allclose(inside.design.toarray(), outside.design.toarray())
    assert np.allclose(inside.precision.toarray(), outside.precision.toarray())
    assert np.allclose(inside.constraints, outside.constraints)


def test_an_integer_index_keeps_its_numeric_level_order():
    """The dtype guard, which slice 3 shipped commented and untested."""
    rows = [
        {"region": r, "year": y, "y": 0.0}
        for r in ("r1", "r2", "r3") for y in range(1, 13)
    ]
    frame = pd.DataFrame(rows)
    outer, _ = _build_effect_block(
        Grouped(RW1("u", index="year"), over="region", structure=BesagStructure(GRAPH)),
        frame,
    )
    expected = tuple(f"{r}@{y}" for r in ("r1", "r2", "r3") for y in range(1, 13))
    assert outer.labels == expected


def test_a_grouped_model_fits_end_to_end():
    frame = _frame()
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            IID("u", index="t", precision=1.0), over="region",
            structure=BesagStructure(GRAPH),
        ),
    ).fit(frame)
    assert np.isfinite(result.log_marginal_likelihood)
    assert len(result.labels) == 1 + 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_grouped_compile.py -q`
Expected: FAIL — `_build_effect_block` falls through to its unknown-effect branch for `Grouped`.

- [ ] **Step 3: Add `grouped_block`**

In `src/pylgm/effects/replicate.py`, add beside `replicated_block`:

```python
def group_levels(frame: pd.DataFrame, name: str, over: str, structure) -> tuple[str, ...]:
    """The group universe, rejecting a missing or null column.

    The structure has the last word: a ``BesagStructure`` returns its graph's
    nodes, so a node with no observations still gets its cell.
    """
    if over not in frame.columns:
        raise ValueError(f"{name} group column {over!r} not found")
    if frame[over].isna().any():
        raise ValueError(f"{name} group column {over!r} must not contain null values")
    observed = tuple(sorted({str(value) for value in frame[over]}))
    return structure.levels(observed)


def grouped_block(
    inner: LatentBlock,
    frame: pd.DataFrame,
    index: str,
    over: str,
    groups: tuple[str, ...],
    structure,
) -> LatentBlock:
    """Compose ``inner`` into ``G`` copies correlated by ``structure``.

    ``Replicated``'s sibling: the only difference is that the outer factor is
    the structure's precision instead of the identity, and that its null space
    contributes to the constraints.
    """
    levels = inner.labels
    n_levels, n_groups = len(levels), len(groups)
    level_position = {level: column for column, level in enumerate(levels)}
    group_position = {label: row for row, label in enumerate(groups)}

    keys = frame[index].map(str)
    unknown = sorted({value for value in keys if value not in level_position})
    if unknown:
        raise ValueError(
            f"{inner.name} index {index!r} has level(s) {unknown!r} absent from the "
            "grouped block's own level set"
        )
    outer_precision = structure.precision(groups)
    if outer_precision.shape != (n_groups, n_groups):
        raise ValueError(
            f"{inner.name} between-group structure has shape "
            f"{outer_precision.shape} but there are {n_groups} group level(s)"
        )
    return kron_block(
        inner.name,
        groups, outer_precision, structure.null_basis(groups),
        levels, inner.precision, inner.constraints.T,
        np.array([group_position[str(g)] for g in frame[over]]),
        np.array([level_position[t] for t in keys]),
        separator="@", orthonormalise=False,
    )
```

- [ ] **Step 4: Add the four compiler branches**

In `src/pylgm/compiler.py`:

**`_build_effect_block`** — add a `Grouped` branch mirroring the `Replicated` one exactly, including the `Weighted` unwrap and the dtype preservation:

```python
        elif isinstance(effect, Grouped):
            # Same Weighted unwrap as the Replicated branch: a Weighted carries
            # no `index` of its own, and its `by` column lives on the real
            # frame, not on the one-row-per-level frame.
            target = inner_spec.effect if isinstance(inner_spec, Weighted) else inner_spec
            index = target.index
            # Keep the column's OWN dtype: an integer `year` would otherwise be
            # ordered 1, 10, 11, 2 by RW1/RW2/Seasonal/AR1.
            levels = tuple(frame[index].dropna().unique())
            groups = group_levels(frame, effect.name, effect.over, effect.structure)
            structural, precision = _build_effect_block(
                target, _levels_frame(index, levels, frame[index].dtype)
            )
            block = grouped_block(
                structural, frame, index, effect.over, groups, effect.structure
            )
            if isinstance(inner_spec, Weighted):
                block = _scaled_design_block(block, _weight_vector(frame, inner_spec))
            return (block, precision)
```

Bind `inner_spec = effect.effect` at the top of the branch, as the `Replicated` branch does.

**`_effect_hyperparameters`** — add beside the `Replicated` delegation:

```python
    if isinstance(effect, Grouped):
        # Delegate: groups share the inner effect's hyperparameters, and the
        # structure's own parameters are fixed in this slice. An unfound inner
        # Hyperparameter would silently pin at its initial value.
        return _effect_hyperparameters(effect.effect)
```

**`_append_family_blocks`** — add a `Grouped` branch mirroring `Replicated`'s, calling a `_grouped_family_block` helper that is `_replicated_family_block` with `kron(structure.precision(groups), inner_build(values))` in place of `kron(identity(count), ...)`, and the same `ParametricDesignBlock` rejection **above** the composition.

**`_prediction_entry`** — add beside the `Replicated` branch:

```python
    if isinstance(effect, Grouped):
        inner_spec = effect.effect
        target = inner_spec.effect if isinstance(inner_spec, Weighted) else inner_spec
        group_labels = tuple(dict.fromkeys(la.split("@", 1)[0] for la in block.labels))
        level_labels = tuple(dict.fromkeys(la.split("@", 1)[1] for la in block.labels))
        entry = (
            "grouped_structured",
            (effect.name, effect.over, target.index, group_labels, level_labels),
        )
        if isinstance(inner_spec, Weighted):
            # Same nesting Weighted(Grouped(...)) produces: build the grouped
            # entry from the unwrapped effect, then wrap it so _design_block_for's
            # recursive dispatch reapplies the weights. Slice 3 shipped this
            # dropped, and Grouped(Weighted(...)) predicted without the weights.
            entry = ("weighted", (entry, inner_spec.by))
        return entry
```

Import `Grouped` from `pylgm.effects.spec` and `grouped_block`, `group_levels` from `pylgm.effects.replicate` at the top of `compiler.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_grouped_compile.py -q`
Expected: all pass.

Then: `PYTHONPATH=src python -m pytest -q`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(compiler): compile Grouped through the Kronecker kernel

The prediction entry lands in this task, not a later one: .fit() always builds
a PredictionContext, so a Grouped model cannot be fitted at all until that
branch exists. The plans for slices 1, 2 and 3 each deferred it and each had to
be corrected mid-execution."
```

---

### Task 7: Predict, and the Knorr-Held oracle

**Files:**
- Modify: `src/pylgm/inference/prediction.py`
- Test: `tests/test_grouped_predict.py`, `tests/test_grouped_spacetime_oracle.py`

**Interfaces:**
- Consumes: the `("grouped_structured", (name, over, index, group_labels, level_labels))` entry from Task 6.
- Produces: `_grouped_block(entry, new_data) -> np.ndarray` and its `_design_block_for` branch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grouped_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Fixed, Gaussian, Grouped, IID, LGM, RW1, Weighted,
)

GRAPH = {"r1": ["r2"], "r2": ["r1", "r3"], "r3": ["r2"]}


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        {"region": r, "t": f"t{t}", "z": 1.0 + 0.1 * t}
        for r in ("r1", "r2", "r3") for t in range(6)
    ]
    frame = pd.DataFrame(rows)
    frame["y"] = rng.standard_normal(len(frame))
    return frame


def _fit(predictor, frame):
    return LGM(response="y", predictor=predictor, likelihood=Gaussian(sigma=0.5)).fit(frame)


def test_prediction_round_trips_on_the_fit_rows():
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(IID("u", index="t"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    predicted = result.predict(frame).predictive_mean
    assert np.allclose(predicted, result.fitted_values, rtol=1e-12, atol=1e-12)


def test_prediction_round_trips_with_weights_inside_the_group():
    """The form slice 3 shipped broken: Replicated(Weighted(...)) dropped weights."""
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(Weighted(IID("u", index="t"), by="z"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    predicted = result.predict(frame).predictive_mean
    assert np.allclose(predicted, result.fitted_values, rtol=1e-12, atol=1e-12)


def test_prediction_round_trips_with_weights_outside_the_group():
    frame = _frame()
    result = _fit(
        Fixed("1") + Weighted(
            Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
            by="z",
        ),
        frame,
    )
    predicted = result.predict(frame).predictive_mean
    assert np.allclose(predicted, result.fitted_values, rtol=1e-12, atol=1e-12)


def test_a_subset_of_groups_still_scores():
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(IID("u", index="t"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    subset = frame[frame["region"] == "r2"]
    assert np.isfinite(result.predict(subset).predictive_mean).all()


def test_an_unseen_level_is_rejected():
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(IID("u", index="t"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    unseen = frame.head(3).copy()
    unseen["t"] = "t99"
    with pytest.raises(ValueError, match="group/level"):
        result.predict(unseen)
```

Create `tests/test_grouped_spacetime_oracle.py`:

```python
"""Grouped reproduces SpaceTime's four Knorr-Held interaction types.

SpaceTime is a shipped, tested implementation of the same Kronecker mechanism,
which makes it an oracle predating this slice -- the same role AR1(group=)
played for Replicated.

Constraints are compared by SPAN, not row by row: SpaceTime orthonormalises and
Grouped does not, so the bases differ while the constrained subspace does not.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Grouped, IID, IIDStructure, RW1, RW2,
)
from pylgm.compiler import _build_effect_block
from pylgm.effects.spacetime import build_spacetime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}


def _frame():
    return pd.DataFrame(
        [{"s": a, "t": t, "y": 0.0} for a in ("a", "b", "c", "d") for t in range(5)]
    )


def _same_span(first: np.ndarray, second: np.ndarray) -> bool:
    """Two constraint matrices span the same row space."""
    if first.shape != second.shape:
        return False
    if first.shape[0] == 0:
        return True
    stacked = np.vstack([first, second])
    return np.linalg.matrix_rank(stacked) == np.linalg.matrix_rank(first)


def _inner(interaction, order):
    return IID("st", index="t") if interaction in ("I", "III") else (
        RW1("st", index="t") if order == 1 else RW2("st", index="t")
    )


def _structure(interaction):
    return IIDStructure() if interaction in ("I", "II") else BesagStructure(GRAPH)


@pytest.mark.parametrize("interaction,order", [
    ("I", 1), ("II", 1), ("II", 2), ("III", 1), ("IV", 1), ("IV", 2),
])
def test_grouped_reproduces_the_knorr_held_interaction(interaction, order):
    frame = _frame()
    reference = build_spacetime(
        frame, "st", "s", "t", GRAPH, interaction, order, precision=1.0
    )
    grouped, _ = _build_effect_block(
        Grouped(_inner(interaction, order), over="s", structure=_structure(interaction)),
        frame,
    )
    assert np.allclose(grouped.design.toarray(), reference.design.toarray())
    assert np.allclose(grouped.precision.toarray(), reference.precision.toarray())
    assert _same_span(grouped.constraints, reference.constraints)


def test_the_two_differ_only_in_their_label_separator():
    """SpaceTime's `|` is user-visible in result.labels and cannot change."""
    frame = _frame()
    reference = build_spacetime(frame, "st", "s", "t", GRAPH, "IV", 1, precision=1.0)
    grouped, _ = _build_effect_block(
        Grouped(RW1("st", index="t"), over="s", structure=BesagStructure(GRAPH)), frame
    )
    assert [la.replace("@", "|") for la in grouped.labels] == list(reference.labels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_grouped_predict.py tests/test_grouped_spacetime_oracle.py -q`
Expected: predict tests FAIL with `predict() context has an unknown block kind 'grouped_structured'`; oracle tests may already pass if Task 6 is correct — if any oracle test fails, the bug is in Task 6's composition, not here.

- [ ] **Step 3: Implement the predict handler**

In `src/pylgm/inference/prediction.py`, add beside `_replicated_block`:

```python
def _grouped_block(
    entry: tuple[str, str, str, tuple[str, ...], tuple[str, ...]], new_data: pd.DataFrame
) -> np.ndarray:
    """Rebuild a grouped effect's design on (group, level) pairs.

    Group-major, matching ``grouped_block``'s ``group * n_levels + level``
    layout. The correlation between groups lives entirely in the precision, so
    the design is the same shape as a replicated one.
    """
    name, over, index_column, group_labels, level_labels = entry
    return _paired_cell_block(
        name, over, index_column, group_labels, level_labels, new_data,
        block_label="grouped",
        pair_label="group/level",
        hint="To score a new group or level, include those rows at fit time "
             "with a NaN response instead.",
    )
```

Add to `_design_block_for`:

```python
    elif kind == "grouped_structured":
        return _grouped_block(payload, new_data)
```

Add the entry kind to the module docstring beside `replicated_structured`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_grouped_predict.py tests/test_grouped_spacetime_oracle.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(prediction): rebuild grouped designs, checked against SpaceTime

The Knorr-Held oracle: Grouped reproduces build_spacetime's four interaction
types on design and precision exactly, and on the constraints by span --
SpaceTime orthonormalises and Grouped does not, so the bases differ while the
constrained subspace does not."
```

---

### Task 8: Hyperparameter effectiveness, and the docs

**Files:**
- Modify: `tests/test_hyperparameter_effectiveness.py`
- Modify: `docs/research-status.md`, `docs/effects.md`
- Test: `tests/test_grouped_compile.py` (family-path additions)

**Interfaces:**
- Consumes: everything above.
- Produces: no code surface; the structural cross-check and the honest map.

- [ ] **Step 1: Add the effectiveness rows**

In `tests/test_hyperparameter_effectiveness.py`'s `MODEL_TABLE`, add three rows following the existing entries' shape:

```python
    (
        "grouped_iid_precision",
        LGM(
            response="y", likelihood=Gaussian(sigma=1.0),
            predictor=Fixed("1") + Grouped(
                IID("u", index="row", precision=Hyperparameter("u.precision", initial=1.0)),
                over="group", structure=BesagStructure(_EFFECTIVENESS_GRAPH),
            ),
        ),
    ),
    (
        "grouped_ar1_rho",
        LGM(
            response="y", likelihood=Gaussian(sigma=1.0),
            predictor=Fixed("1") + Grouped(
                AR1("u", index="row",
                    rho=Hyperparameter("u.rho", initial=0.2, transform="logit")),
                over="group", structure=BesagStructure(_EFFECTIVENESS_GRAPH),
            ),
        ),
    ),
    (
        "grouped_weighted_precision",
        LGM(
            response="y", likelihood=Gaussian(sigma=1.0),
            predictor=Fixed("1") + Grouped(
                Weighted(
                    IID("u", index="row",
                        precision=Hyperparameter("u.precision", initial=1.0)),
                    by="x",
                ),
                over="group", structure=BesagStructure(_EFFECTIVENESS_GRAPH),
            ),
        ),
    ),
```

`transform="logit"` on the AR1 `rho` is required — the plans for slice 3 omitted it and the row failed on an out-of-range candidate. Define `_EFFECTIVENESS_GRAPH` beside `PANEL`, matching whatever `group` column the module's `FRAME` already provides; if it has none, add one and leave every existing row untouched.

- [ ] **Step 2: Run and verify each row actually changes the fit**

Run: `PYTHONPATH=src python -m pytest tests/test_hyperparameter_effectiveness.py -q`
Expected: all pass.

Then prove the rows are non-vacuous: temporarily replace `_effect_hyperparameters`'s `Grouped` delegation with `return []` and re-run. Expected: the three new rows FAIL. Restore.

- [ ] **Step 3: Add the family-path test**

Add to `tests/test_grouped_compile.py`:

```python
def test_an_estimated_inner_precision_scales_every_group():
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)),
            over="region", structure=BesagStructure(GRAPH),
        ),
    )
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    family = compile_family(model, panel)
    assert family is not None and "tau" in family.parameter_names
    low = [b for b in family.materialize({"tau": 1.0}).blocks if b.name == "u"][0]
    high = [b for b in family.materialize({"tau": 50.0}).blocks if b.name == "u"][0]
    nonzero = low.precision.toarray() != 0
    assert np.allclose(high.precision.toarray()[nonzero] / low.precision.toarray()[nonzero], 50.0)
    assert low.precision.shape == (6, 6)


def test_an_integer_index_keeps_its_numeric_level_order_in_the_family_path():
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    rows = [
        {"region": r, "year": y, "y": 0.0}
        for r in ("r1", "r2", "r3") for y in range(1, 13)
    ]
    frame = pd.DataFrame(rows)
    frame["row"] = range(len(frame))
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            RW1("u", index="year", precision=Hyperparameter("tau", initial=1.0)),
            over="region", structure=BesagStructure(GRAPH),
        ),
    )
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_family(model, panel).materialize({"tau": 1.0})
    expected = tuple(
        f"u:{r}@{y}" for r in ("r1", "r2", "r3") for y in range(1, 13)
    )
    assert tuple(compiled.labels[-36:]) == expected
```

- [ ] **Step 4: Verify the dtype tests are non-vacuous**

Stringify the levels at both `_levels_frame` call sites in the `Grouped` branches — `_levels_frame(index, tuple(str(v) for v in levels), None)` — and re-run `tests/test_grouped_compile.py`. Expected: both integer-index tests FAIL with labels ordered `1, 10, 11, 12, 2, ...`. Restore.

This is exactly the check slice 3's review had to add after the fact.

- [ ] **Step 5: Write the docs**

Add a `Grouped` section to `docs/effects.md`, beside `Replicated`, with a Python example and the four Knorr-Held structures named.

Add a `Grouped` entry to `docs/research-status.md` in the shape the `Weighted`/`Copy`/`Replicated` entries use — a "what is verified" table and a "what is NOT verified" list. The NOT-verified list must include, at minimum:

- No validation against published results on real data.
- **No YAML/config surface**, unlike `SpaceTime`, which has one.
- **Not supported on Spark**: `_required_columns` reads `effect.index` unconditionally, so a `Grouped` model raises a bare `AttributeError`, and `over` is never added to the projection. Pre-existing gap in a helper already blind to `MIDAS`, `SpaceTime`, `DynamicSpatialPanel` and `AR1(replicate=)`'s replicate column.
- **`group` and `replicate` together are rejected**, which R-INLA permits — an f() parity gap, recorded rather than half-implemented.
- **A `Hyperparameter` on a structure's own parameters is not supported**; `AR1Structure(rho)` takes fixed values only.
- `Grouped` inside a `Joint` is untested.

Verify each claim by running it before writing it down, rather than asserting from the plan.

- [ ] **Step 6: Run the full suite and commit**

Run: `PYTHONPATH=src python -m pytest -q` and `ruff check src tests`
Expected: all pass, ruff clean. Then `git restore docs/img/` if any figure regenerated.

```bash
git add -A
git commit -m "test(grouped): pin hyperparameter effectiveness and the integer level order

Both checks verified non-vacuous by mutation: neutering the Grouped delegation
in _effect_hyperparameters fails the three new rows, and stringifying the
levels at either _levels_frame call site fails the dtype tests with labels
ordered 1, 10, 11, 12, 2.

research-status.md records what this slice does NOT establish, including the
f() parity gap on group-with-replicate and the pre-existing Spark blindness."
```

---

## After the slice

The whole-slice review, on the most capable model, over the packaged diff — the same final step slices 1, 2 and 3 each took. It should sweep independently for:

- a seventh instance of the silent-hyperparameter failure mode;
- any dispatch site this plan did not name (slice 1 found `data/spark.py` only after claiming four sites were the complete set; slice 4 found a fifth prediction entry kind that slice 3's rename had missed);
- whether the `orthonormalise` and `separator` flags are documented clearly enough that a future reader does not mistake them for modelling choices.

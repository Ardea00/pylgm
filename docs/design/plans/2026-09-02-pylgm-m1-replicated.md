# M1 Slice 3: `Replicated` Effect Modifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any indexed latent effect become `R` independent copies that share every hyperparameter — R-INLA's `f(index, model=..., replicate=r)` — and rename the existing `AR1(group=)`, which is this concept under the wrong name.

**Architecture:** `Replicated(effect, over)` builds the inner effect's structure **once over the level set alone**, then composes: precision `I_R ⊗ Q_E`, design indexed on `(replicate, level)` pairs in replicate-major order, constraints `I_R ⊗ C_E`. One `LatentBlock` out, so both `CompiledLGM` invariants hold and inference is untouched.

**Tech Stack:** Python 3.11+, numpy 2.x, scipy 1.14+ (sparse), pandas 2.2+, pytest 8.3+. No new dependencies.

**Spec:** `docs/design/specs/2026-09-02-pylgm-m1-effect-modifiers-design.md` (slice 3 of 4)

**Branch:** `research-tier`. Research-grade; see `docs/research-status.md`.

## Global Constraints

- **No new runtime dependency.** numpy / scipy / pandas / formulaic / pydantic / pyarrow / pyyaml / typer only.
- **No MCMC.** Deterministic approximation only.
- **`CompiledLGM` invariants preserved**: `design == hstack(blocks)`, `precision == block_diag(blocks)`. `Replicated` produces exactly one block.
- **No inference change.** `inference/laplace.py` and `inference/gaussian.py` are not touched.
- **Existing tests pass unchanged at every commit.** `PYTHONPATH=src python -m pytest -q`.
- **Ruff clean:** `ruff check src tests`, line length 100, rules `E4,E7,E9,F`.
- **Run tests as:** `PYTHONPATH=src python -m pytest ...`
- **Frozen dataclasses** with validation in `__post_init__`.
- **Errors** raise types from `pylgm.exceptions`, or plain `ValueError`/`TypeError` in spec `__post_init__` and in `prediction.py`.
- **Every new hyperparameter path gets a row in `tests/test_hyperparameter_effectiveness.py`.** That file exists because this project shipped five separate instances of a declared hyperparameter having zero effect on the fit. Adding to it is not optional.

## The named risk: constraint replication

`Replicated(Besag(...), over=r)` must produce **R sum-to-zero constraints, one per replicate — not one**. With a single constraint over `R` replicates, `R-1` directions stay unidentified, the fit still converges, and it returns plausible numbers. Nothing else in the suite would catch it.

This has never come up before because the one existing replicate-like implementation, `AR1(group=)`, builds `constraints = np.empty((0, width))` — **AR1 carries no constraints**, so the question never arose. `Besag`, `BYM2`, `RW1`/`RW2` and `Seasonal` all do carry them.

The composition is `np.kron(np.eye(R), C)`: for a `(k, L)` constraint matrix that gives `(k·R, L·R)`, where replicate `r`'s rows are `C` in columns `[r·L, (r+1)·L)` and zero elsewhere. Task 2 tests the count, the placement and the rank.

## The rename

`AR1(group=)` is R-INLA's **`replicate`** shipped under the name `group`. R-INLA's actual `group` means *correlated* copies via `control.group`, which slice 4 implements. Task 5 introduces `AR1(replicate=)`, keeps `group=` working with a `DeprecationWarning` that names the translation, and pins bit-for-bit equivalence between `Replicated(AR1(...), over=g)` and the existing `AR1(..., group=g)` — the existing implementation is the oracle that makes this rewrite safe.

There is no `DeprecationWarning` anywhere in `src/pylgm` today; this is the first, so pick the message carefully.

## File Structure

| File | Responsibility |
|---|---|
| `src/pylgm/effects/spec.py` (modify) | The `Replicated` spec; `AR1.replicate` and the `group` deprecation. |
| `src/pylgm/effects/replicate.py` (create) | `replicated_block(inner, over, frame)` — structure, design, constraints. |
| `src/pylgm/compiler.py` (modify) | Wrapper cases in the four dispatch functions. |
| `src/pylgm/inference/prediction.py` (modify) | `replicated` entry kind over `_paired_cell_block`. |
| `src/pylgm/__init__.py` (modify) | Export `Replicated`. |
| `tests/test_replicated_spec.py` (create) | Spec validation and rejections. |
| `tests/test_replicated_compile.py` (create) | Kronecker structure, constraint replication, reduction. |
| `tests/test_replicated_predict.py` (create) | Prediction round-trip. |
| `tests/test_replicated_equivalence.py` (create) | Bit-for-bit against `AR1(group=)`; the deprecation. |

A new module rather than more of `compiler.py`: the Kronecker composition is effect-agnostic and belongs beside the other builders in `src/pylgm/effects/`, and `compiler.py` is already past 2000 lines.

---

### Task 1: The `Replicated` spec

**Files:**
- Modify: `src/pylgm/effects/spec.py`
- Test: `tests/test_replicated_spec.py`

**Interfaces:**
- Consumes: `_ComposableEffect` (`spec.py:35`); `Copy` and `Weighted`, already in that file.
- Produces: `Replicated(effect, over)` — frozen dataclass with `.effect`, `.over: str`, and a `.name` property delegating to `effect.name`.

**Why `.name` delegates:** identical reasoning to `Weighted`. Compiler sites read `effect.name` to key precisions and label blocks; delegating keeps the inner effect's name so `latent_marginals("u")` still works on a replicated effect.

- [ ] **Step 1: Write the failing test**

Create `tests/test_replicated_spec.py`:

```python
import pytest

from pylgm import AR1, Besag, Copy, Fixed, IID, Replicated, Weighted
from pylgm.effects.spec import Predictor


def test_replicated_delegates_its_name_to_the_inner_effect():
    assert Replicated(IID("u", index="t"), over="firm").name == "u"


def test_replicated_keeps_the_inner_effect_and_the_over_column():
    inner = IID("u", index="t")
    wrapped = Replicated(inner, over="firm")
    assert wrapped.effect is inner
    assert wrapped.over == "firm"


def test_replicated_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + Replicated(IID("u", index="t"), over="firm")
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 2


def test_replicated_rejects_a_non_string_over():
    with pytest.raises((TypeError, ValueError), match="over"):
        Replicated(IID("u", index="t"), over=3)


def test_replicated_rejects_an_empty_over():
    with pytest.raises(ValueError, match="over"):
        Replicated(IID("u", index="t"), over="")


def test_replicated_rejects_an_effect_with_no_index():
    with pytest.raises(TypeError, match="index"):
        Replicated(Fixed("1"), over="firm")


def test_replicated_rejects_wrapping_a_replicated():
    # Two replicate columns is one replicate over their cross product, so the
    # doubled form says nothing the single form cannot.
    with pytest.raises(TypeError, match="already replicated"):
        Replicated(Replicated(IID("u", index="t"), over="firm"), over="year")


def test_replicated_rejects_wrapping_a_copy():
    # A Copy is a term referencing another term, not an indexed effect of its own.
    with pytest.raises(TypeError):
        Replicated(Copy("u", index="j"), over="firm")


def test_replicated_rejects_an_ar1_that_already_replicates_itself():
    # AR1(replicate=) is the same concept; wrapping it would give two
    # replication mechanisms on one effect with no defined interaction.
    with pytest.raises(TypeError, match="replicate"):
        Replicated(AR1("t", index="year", replicate="firm"), over="country")


def test_replicated_may_wrap_a_weighted_effect():
    # Weighted touches only the design, Replicated only precision and indexing,
    # so the two compose; the spec asserts they commute.
    wrapped = Replicated(Weighted(IID("u", index="t"), by="z"), over="firm")
    assert wrapped.name == "u"


def test_replicated_accepts_a_constrained_effect():
    graph = {"a": ["b"], "b": ["a"]}
    assert Replicated(Besag("u", index="d", graph=graph), over="firm").name == "u"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'Replicated'`

- [ ] **Step 3: Implement the spec**

In `src/pylgm/effects/spec.py`, add after `Weighted`:

```python
@dataclass(frozen=True)
class Replicated(_ComposableEffect):
    """``R`` independent copies of an effect, sharing every hyperparameter.

    ``Replicated(AR1("t", index="year"), over="firm")`` is one AR1 series per
    firm: the firms share ``rho`` and ``precision`` but not their realizations.
    This is R-INLA's ``f(index, model=..., replicate=r)``.

    The precision becomes ``I_R (x) Q``, the design is indexed on
    ``(replicate, level)`` pairs, and a constrained inner effect gets **one
    constraint per replicate** -- a single shared constraint would leave
    ``R-1`` directions unidentified while still fitting.

    Not to be confused with R-INLA's ``group``, which is *correlated* copies
    with a between-group structure; that is a separate modifier.
    """

    effect: object
    over: str

    def __post_init__(self) -> None:
        if isinstance(self.effect, Replicated):
            raise TypeError(
                "Replicated effect is already replicated; two replicate columns "
                "is one replicate over their cross product, so combine them into "
                "a single column"
            )
        if getattr(self.effect, "replicate", None) is not None:
            raise TypeError(
                f"{type(self.effect).__name__} already replicates itself through "
                "its own `replicate` argument; wrapping it would give two "
                "replication mechanisms on one effect with no defined interaction"
            )
        if not hasattr(self.effect, "index"):
            raise TypeError(
                f"Replicated requires an indexed effect, got "
                f"{type(self.effect).__name__}, which has no index."
            )
        if isinstance(self.effect, Copy):
            raise TypeError(
                "Replicated cannot wrap a Copy: a copy is a term referencing "
                "another term, not an indexed effect of its own. Replicate the "
                "target effect instead."
            )
        object.__setattr__(self, "over", _non_empty_string(self.over, "over"))

    @property
    def name(self) -> str:
        return self.effect.name
```

Define it after `Copy` and `Weighted` so both names resolve. Note the `Copy` check must come **after** the `hasattr(index)` check would pass — `Copy` has an `index`, so order it as written above.

- [ ] **Step 4: Export it**

Add `Replicated` to `src/pylgm/effects/__init__.py`, to `src/pylgm/__init__.py`'s imports and `__all__` (alphabetical: after `ProperCAR`, before `RW1`), and to the hardcoded expected set in `tests/test_package.py`.

- [ ] **Step 5: Run tests and ruff**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_spec.py tests/test_public_exports.py -q`
Expected: PASS, 11 passed plus the export tests

Note: `test_replicated_rejects_an_ar1_that_already_replicates_itself` uses `AR1(replicate=...)`, which Task 5 adds. Until then that test will fail on an unexpected keyword. **Skip it with `pytest.mark.xfail(reason="AR1.replicate arrives in Task 5", strict=False)` and remove the marker in Task 5** — do not delete the test, and do not change the spec's `getattr(self.effect, "replicate", None)` check, which is already correct for the field's arrival.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/test_replicated_spec.py tests/test_package.py
git commit -m "feat(effects): Replicated spec for independent copies sharing hyperparameters

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Build the replicated block

**Files:**
- Create: `src/pylgm/effects/replicate.py`
- Modify: `src/pylgm/compiler.py` — `_build_effect_block`, `_effect_hyperparameters`
- Test: `tests/test_replicated_compile.py`

**Interfaces:**
- Consumes: `Replicated` (Task 1); `_levels_frame(index, levels, dtype)` at `compiler.py:895`; `LatentBlock` from `pylgm.ir.model`.
- Produces: `replicated_block(block, replicates, positions) -> LatentBlock` in the new module, and a `Replicated` case in `_build_effect_block`.

**The two-stage build.** A replicated effect cannot post-process the inner block the way `Weighted` does: an inner effect built against the whole frame would consume every level and know nothing of the replicate column. Instead build the inner effect against `_levels_frame(index, levels, dtype)` — one row per level — which yields the structure `Q`, the labels, and the constraints `C` over the level set alone. Then compose.

**Layout is replicate-major**, matching `build_ar1`'s group-major convention: `cell = replicate_position * n_levels + level_position`, labels `f"{r}@{level}"`. Keeping the same convention is what makes Task 5's bit-for-bit equivalence against `AR1(group=)` possible.

- [ ] **Step 1: Write the failing test**

Create `tests/test_replicated_compile.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, IID, LGM, Poisson, Replicated, Weighted
from pylgm.compiler import _build_effect_block, _effect_hyperparameters
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.parameters import Hyperparameter


def _frame():
    return pd.DataFrame({
        "t": ["a", "b", "c", "a", "b", "c"],
        "firm": ["f1", "f1", "f1", "f2", "f2", "f2"],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "row": range(6),
    })


def test_precision_is_the_kronecker_product_of_identity_and_the_inner_structure():
    frame = _frame()
    inner, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=2.0), over="firm"), frame
    )
    expected = np.kron(np.eye(2), inner.precision.toarray())
    assert np.allclose(outer.precision.toarray(), expected)


def test_labels_are_replicate_major_pairs():
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.0), over="firm"), _frame()
    )
    assert outer.labels == ("f1@a", "f1@b", "f1@c", "f2@a", "f2@b", "f2@c")


def test_design_maps_each_row_to_its_own_replicate_cell():
    frame = _frame()
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.0), over="firm"), frame
    )
    dense = outer.design.toarray()
    assert dense.shape == (6, 6)
    # Row k belongs to firm frame["firm"][k] at level frame["t"][k]; with three
    # levels and replicate-major layout that is cell (firm_index * 3 + level).
    position = {label: k for k, label in enumerate(outer.labels)}
    for row, (firm, level) in enumerate(zip(frame["firm"], frame["t"])):
        assert dense[row, position[f"{firm}@{level}"]] == 1.0
        assert dense[row].sum() == 1.0


def test_a_constrained_inner_effect_gets_one_constraint_per_replicate():
    """The named risk of this slice. One shared constraint over R replicates
    leaves R-1 directions unidentified: the fit still converges and returns
    plausible numbers, and nothing else in the suite would catch it."""
    frame = _frame()
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    inner, _ = _build_effect_block(Besag("u", index="t", graph=graph, precision=1.0), frame)
    outer, _ = _build_effect_block(
        Replicated(Besag("u", index="t", graph=graph, precision=1.0), over="firm"), frame
    )

    assert inner.constraints.shape == (1, 3)
    assert outer.constraints.shape == (2, 6)
    assert np.allclose(outer.constraints, np.kron(np.eye(2), inner.constraints))
    # Each replicate's constraint touches only its own columns.
    assert np.allclose(outer.constraints[0], [1, 1, 1, 0, 0, 0])
    assert np.allclose(outer.constraints[1], [0, 0, 0, 1, 1, 1])
    assert np.linalg.matrix_rank(outer.constraints) == 2


def test_an_unconstrained_inner_effect_stays_unconstrained():
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.0), over="firm"), _frame()
    )
    assert outer.constraints.shape == (0, 6)


def test_a_single_replicate_reduces_to_the_inner_block():
    frame = _frame().assign(firm="only")
    inner, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=2.0), over="firm"), frame
    )
    assert np.allclose(outer.precision.toarray(), inner.precision.toarray())
    assert np.allclose(outer.design.toarray(), inner.design.toarray())
    assert outer.labels == tuple(f"only@{label}" for label in inner.labels)


def test_inner_hyperparameters_are_discovered_through_the_wrapper():
    tau = Hyperparameter("tau", initial=1.0)
    wrapped = Replicated(IID("u", index="t", precision=tau), over="firm")
    assert [hp.name for hp in _effect_hyperparameters(wrapped)] == ["tau"]


def test_replicated_commutes_with_weighted():
    """Weighted touches only the design; Replicated only precision and indexing.
    The spec asserts they commute, so this pins it rather than leaving it as a
    convention to remember."""
    frame = _frame()
    a, _ = _build_effect_block(
        Replicated(Weighted(IID("u", index="t", precision=1.0), by="z"), over="firm"), frame
    )
    b, _ = _build_effect_block(
        Weighted(Replicated(IID("u", index="t", precision=1.0), over="firm"), by="z"), frame
    )
    assert a.labels == b.labels
    assert np.allclose(a.design.toarray(), b.design.toarray())
    assert np.allclose(a.precision.toarray(), b.precision.toarray())
    assert np.allclose(a.constraints, b.constraints)


def test_missing_replicate_column_is_rejected():
    frame = _frame().drop(columns=["firm"])
    with pytest.raises((CompilationError, DataContractError), match="firm"):
        _build_effect_block(Replicated(IID("u", index="t"), over="firm"), frame)


def test_a_null_in_the_replicate_column_is_rejected():
    frame = _frame()
    frame.loc[0, "firm"] = None
    with pytest.raises((CompilationError, DataContractError), match="firm"):
        _build_effect_block(Replicated(IID("u", index="t"), over="firm"), frame)


def test_a_replicated_model_fits():
    rng = np.random.default_rng(5)
    n = 60
    frame = pd.DataFrame({
        "t": [f"t{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(IID("u", index="t", precision=1.0), over="firm"),
    ).fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert len(result.labels) == 1 + 6 * 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_compile.py -q`
Expected: FAIL — `_build_effect_block` raises `CompilationError: unsupported effect type: Replicated`

- [ ] **Step 3: Write the composition module**

Create `src/pylgm/effects/replicate.py`:

```python
"""Independent replicates of a latent effect, sharing its hyperparameters.

``R`` copies of one effect: precision ``I_R (x) Q``, design on
``(replicate, level)`` pairs, constraints ``I_R (x) C``. The layout is
replicate-major -- ``cell = replicate * n_levels + level`` -- matching
``build_ar1``'s group-major convention, which is what lets a replicated AR1
match the shipped ``AR1(group=)`` implementation bit for bit.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, kron

from pylgm.ir.model import LatentBlock


def replicate_levels(frame: pd.DataFrame, name: str, over: str) -> tuple[str, ...]:
    """The sorted replicate levels, rejecting a missing or null column."""
    if over not in frame.columns:
        raise ValueError(f"{name} replicate column {over!r} not found")
    if frame[over].isna().any():
        raise ValueError(f"{name} replicate column {over!r} must not contain null values")
    return tuple(sorted({str(value) for value in frame[over]}))


def replicated_block(
    inner: LatentBlock,
    frame: pd.DataFrame,
    index: str,
    over: str,
    replicates: tuple[str, ...],
) -> LatentBlock:
    """Compose ``inner`` -- built over the level set alone -- into ``R`` copies.

    ``inner.constraints`` is replicated per copy rather than shared: one
    constraint over ``R`` replicates would leave ``R-1`` directions
    unidentified, and the fit would still converge on plausible numbers.
    """
    levels = inner.labels
    n_levels, n_replicates = len(levels), len(replicates)
    level_position = {level: column for column, level in enumerate(levels)}
    replicate_position = {label: row for row, label in enumerate(replicates)}

    keys = frame[index].map(str)
    unknown = sorted({value for value in keys if value not in level_position})
    if unknown:
        raise ValueError(
            f"{inner.name} index {index!r} has level(s) {unknown!r} absent from the "
            "replicated block's own level set"
        )
    cells = np.array([
        replicate_position[str(r)] * n_levels + level_position[t]
        for r, t in zip(frame[over], keys)
    ])
    width = n_replicates * n_levels
    design = csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), cells)), shape=(len(frame), width)
    )
    precision = csr_matrix(
        kron(identity(n_replicates, format="csr"), inner.precision, format="csr")
    )
    if inner.constraints.shape[0]:
        constraints = np.kron(np.eye(n_replicates), inner.constraints)
    else:
        constraints = np.empty((0, width))
    labels = tuple(f"{r}@{level}" for r in replicates for level in levels)
    return LatentBlock(inner.name, labels, design, precision, constraints)
```

- [ ] **Step 4: Wire it into the compiler**

In `src/pylgm/compiler.py`, add a `Replicated` branch to `_build_effect_block`, before the `Fixed` case:

```python
        if isinstance(effect, Replicated):
            inner_spec = effect.effect
            index = inner_spec.index
            # Keep the column's OWN dtype: stringifying here and then passing the
            # real dtype to _levels_frame is contradictory, and it is exactly the
            # slice-1 bug -- an integer `year` would be ordered 1, 10, 11, 2 by
            # RW1/RW2/Seasonal/AR1. The builder sorts, so order here is irrelevant;
            # only the dtype matters.
            levels = tuple(frame[index].dropna().unique())
            replicates = replicate_levels(frame, effect.name, effect.over)
            inner, precision = _build_effect_block(
                inner_spec, _levels_frame(index, levels, frame[index].dtype)
            )
            return (
                replicated_block(inner, frame, index, effect.over, replicates),
                precision,
            )
```

and delegate hyperparameter discovery, alongside the existing `Weighted` branch in `_effect_hyperparameters`:

```python
    if isinstance(effect, Replicated):
        # Delegate: replicates share the inner effect's hyperparameters, so an
        # unfound one would silently pin at its initial value.
        return _effect_hyperparameters(effect.effect)
```

Import `Replicated` from `pylgm.effects` and `replicate_levels`, `replicated_block` from `pylgm.effects.replicate`. Re-export both from `src/pylgm/effects/__init__.py`.

**Note on `_levels_frame`'s dtype argument:** pass `frame[index].dtype`. Slice 1 shipped a bug where a fabricated string column made order-dependent builders (`RW1`, `RW2`, `Seasonal`, `AR1`) see lexicographic order — `1, 10, 11, 2` for an integer index. Passing the real dtype is what prevents it here.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_compile.py tests/test_replicated_spec.py -q`
Expected: PASS, 22 passed (one xfail from Task 1)

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/effects/replicate.py src/pylgm/effects/__init__.py src/pylgm/compiler.py tests/test_replicated_compile.py
git commit -m "feat(effects): Replicated builds I_R (x) Q with per-replicate constraints

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `Replicated` in the hyperparameter family

**Files:**
- Modify: `src/pylgm/compiler.py` — `_append_family_blocks`
- Test: `tests/test_replicated_compile.py` (append); `tests/test_hyperparameter_effectiveness.py` (append)

**Interfaces:**
- Consumes: `replicated_block`, `replicate_levels` (Task 2); `ScalableBlock`, `ParametricBlock`, `ParametricDesignBlock` from `pylgm.ir.family`.
- Produces: no new public names.

**Why the family path needs its own case.** `_append_family_blocks` has an effect chain independent of `_build_effect_block`. Task 2 makes a replicated model *compile*; without this, a replicated effect with an estimated hyperparameter falls through that chain.

**The subtlety.** The inner effect's `build` closure produces a structure over the **level set**, not the replicated space. Every rebuild must therefore be re-composed with `I_R ⊗ ·`. A `ParametricBlock` rebuilds a **precision** — kron it. A `ParametricDesignBlock` rebuilds a **design** over the level-set frame, which is the wrong row space entirely; that combination cannot be composed and must be rejected, in the same spirit as slice 2's `ParametricBlock` refusal.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_replicated_compile.py`:

```python
def test_an_estimated_inner_precision_scales_every_replicate():
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)), over="firm"
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    family = compile_family(model, panel)
    assert family is not None
    assert "tau" in family.parameter_names

    low = family.materialize({"tau": 1.0})
    high = family.materialize({"tau": 50.0})
    u_low = [b for b in low.blocks if b.name == "u"][0].precision.toarray()
    u_high = [b for b in high.blocks if b.name == "u"][0].precision.toarray()
    nonzero = u_low != 0
    assert np.allclose(u_high[nonzero] / u_low[nonzero], 50.0)
    # The scaling reaches every replicate, not just the first.
    assert u_low.shape == (6, 6)
    assert np.count_nonzero(np.diag(u_high)) == 6


def test_an_estimated_inner_rho_rebuilds_every_replicate_band():
    """AR1's rho enters the structure, not a scalar multiplier, so the rebuilt
    precision must be re-composed with I_R on every draw -- a per-draw structure
    left uncomposed would be the wrong shape and fail loudly, but one composed
    only at the template would silently freeze rho after the first draw."""
    from pylgm import AR1
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            AR1("u", index="t", precision=1.0, rho=Hyperparameter("rho", initial=0.5)),
            over="firm",
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    family = compile_family(model, panel)
    assert "rho" in family.parameter_names

    a = [b for b in family.materialize({"rho": 0.1}).blocks if b.name == "u"][0]
    b = [b for b in family.materialize({"rho": 0.8}).blocks if b.name == "u"][0]
    assert a.precision.shape == (6, 6)
    assert not np.allclose(a.precision.toarray(), b.precision.toarray())
    # Both replicate blocks change, not only the first.
    assert not np.allclose(a.precision.toarray()[3:, 3:], b.precision.toarray()[3:, 3:])
```

and append to `tests/test_hyperparameter_effectiveness.py` — read that file first and follow its existing parametrisation exactly — these three models:

```python
    ("Replicated(IID(tau))", Fixed("1") + Replicated(
        IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)), over="firm")),
    ("Replicated(AR1(rho))", Fixed("1") + Replicated(
        AR1("u", index="t", precision=1.0, rho=Hyperparameter("rho", initial=0.5)),
        over="firm")),
    ("Replicated(Weighted(IID(tau)))", Fixed("1") + Replicated(
        Weighted(IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)), by="z"),
        over="firm")),
```

That file exists because this project shipped five separate instances of a declared hyperparameter having zero effect on the fit. If its fixture frame lacks a `firm` or `z` column, extend the fixture rather than weakening the models.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_compile.py -q -k "estimated"`
Expected: FAIL — `compile_family` does not recognise `Replicated`

- [ ] **Step 3: Implement the family case**

Add to `src/pylgm/compiler.py`, above `_append_family_blocks`:

```python
def _replicated_family_block(item, frame, effect, replicates):
    """Compose one family block into R independent replicates.

    A ScalableBlock's precision is a scalar multiple, which commutes with the
    Kronecker product, so the template composes once. A ParametricBlock rebuilds
    a *structure* per draw, so each rebuild must be re-composed with I_R --
    composing only the template would silently freeze the structure's
    hyperparameter after the first draw.
    """
    index = effect.effect.index
    composed = replicated_block(item.block, frame, index, effect.over, replicates)
    count = len(replicates)

    if isinstance(item, ParametricDesignBlock):
        raise CompilationError(
            f"effect {effect.name!r} is replicated and its design is itself a "
            "function of hyperparameters; that combination is not supported, "
            "because the rebuilt design spans the level set rather than the "
            "replicated row space"
        )
    if isinstance(item, ParametricBlock):
        def build(values, inner_build=item.build, count=count):
            return csr_matrix(
                kron(identity(count, format="csr"), inner_build(values), format="csr")
            )

        return ParametricBlock(composed, item.parameters, build)
    return ScalableBlock(composed, item.parameter, item.scale)
```

and a branch at the top of `_append_family_blocks`:

```python
    if isinstance(effect, Replicated):
        index = effect.effect.index
        levels = tuple(frame[index].dropna().unique())   # own dtype; see Task 2
        replicates = replicate_levels(frame, effect.name, effect.over)
        level_frame = _levels_frame(index, levels, frame[index].dtype)
        first = len(scalable)
        _append_family_blocks(
            effect.effect, level_frame, scalable, parameter_names,
            parameter_bounds, parameter_priors,
        )
        for position in range(first, len(scalable)):
            scalable[position] = _replicated_family_block(
                scalable[position], frame, effect, replicates
            )
        return
```

Recursing on the **level frame** rather than the full frame is what gives the inner effect its level-set structure; the composition then lifts it. Import `kron` and `identity` from `scipy.sparse` in `compiler.py` if not already present.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_compile.py tests/test_hyperparameter_effectiveness.py -q`
Expected: PASS

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/compiler.py tests/test_replicated_compile.py tests/test_hyperparameter_effectiveness.py
git commit -m "feat(compiler): Replicated in the hyperparameter family path

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Prediction through a replicated effect

**Files:**
- Modify: `src/pylgm/compiler.py` — `_prediction_entry`
- Modify: `src/pylgm/inference/prediction.py`
- Test: `tests/test_replicated_predict.py`

**Interfaces:**
- Consumes: `_paired_cell_block(name, outer_column, inner_column, outer_labels, inner_labels, new_data, *, block_label, pair_label, hint)` at `prediction.py:187` — already used by group-wise `AR1` and `SpaceTime`.
- Produces: entry kind `("replicated", (name, over, index, replicate_labels, level_labels))`.

**The invariant slice 2's final review flagged, which bites here.** `_prediction_entry` passes the **outer** block to its recursive inner call, which is sound only for wrappers that preserve labels. `Replicated` re-labels to `(replicate, level)` pairs, so the inner entry must **not** be derived from the outer block's labels. Split the composite labels back apart, exactly as `build_prediction_context` already does for group-wise `AR1` (`label.split("@", 1)`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_replicated_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Replicated


def _data(seed=13, n=72):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": [f"t{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


def _fitted():
    frame = _data()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(IID("u", index="t", precision=1.0), over="firm"),
    )
    return frame, model.fit(frame, engine="laplace")


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means():
    frame, result = _fitted()
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_prediction_uses_the_replicate_column_from_new_data():
    """Reassigning rows to different replicates must move the prediction; a
    predict path that ignored the replicate column would not."""
    frame, result = _fitted()
    rotated = frame.assign(firm=frame["firm"].map({"f0": "f1", "f1": "f2",
                                                   "f2": "f3", "f3": "f0"}))
    assert not np.allclose(
        result.predict(frame).predictive_mean,
        result.predict(rotated).predictive_mean,
    )


def test_predict_rejects_new_data_missing_the_replicate_column():
    frame, result = _fitted()
    with pytest.raises(ValueError, match="firm"):
        result.predict(frame.drop(columns=["firm"]))


def test_predict_rejects_an_unseen_replicate():
    frame, result = _fitted()
    unseen = frame.copy()
    unseen.loc[0, "firm"] = "f99"
    with pytest.raises(ValueError, match="f99"):
        result.predict(unseen)


def test_predict_rejects_an_unseen_level():
    frame, result = _fitted()
    unseen = frame.copy()
    unseen.loc[0, "t"] = "t99"
    with pytest.raises(ValueError, match="t99"):
        result.predict(unseen)


def test_round_trip_holds_when_sorted_and_first_seen_level_order_differ():
    """Levels named t1..t11 sort as t1, t10, t11, t2, ... so the fitted label
    order and the frame's first-seen order genuinely diverge. This project has
    shipped two silent misalignment bugs of exactly this class."""
    rng = np.random.default_rng(21)
    n = 110
    frame = pd.DataFrame({
        "t": [f"t{(k % 11) + 1}" for k in range(n)],
        "firm": [f"f{k % 5}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(IID("u", index="t", precision=1.0), over="firm"),
    ).fit(frame, engine="laplace")
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_predict.py -q`
Expected: FAIL — `_prediction_entry` has no `Replicated` case

- [ ] **Step 3: Emit the entry**

In `src/pylgm/compiler.py`, add to `_prediction_entry`, alongside the `Weighted` branch:

```python
    if isinstance(effect, Replicated):
        # Split the composite labels rather than recursing with the outer block:
        # Replicated RE-LABELS to (replicate, level) pairs, so an inner entry
        # derived from these labels would be wrong. Same shape as the
        # group-wise AR1 entry built below.
        replicate_labels = tuple(dict.fromkeys(
            label.split("@", 1)[0] for label in block.labels
        ))
        level_labels = tuple(dict.fromkeys(
            label.split("@", 1)[1] for label in block.labels
        ))
        return (
            "replicated",
            (effect.name, effect.over, effect.effect.index, replicate_labels, level_labels),
        )
```

- [ ] **Step 4: Rebuild it at predict time**

In `src/pylgm/inference/prediction.py`, add:

```python
def _replicated_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    """Rebuild a replicated effect's design on (replicate, level) pairs."""
    name, over, index, replicate_labels, level_labels = entry
    return _paired_cell_block(
        name, over, index, replicate_labels, level_labels, new_data,
        block_label="replicated",
        pair_label="replicate/level",
        hint="To score a new replicate or level, include those rows at fit time "
             "with a NaN response instead.",
    )
```

and register `elif kind == "replicated": return _replicated_block(payload, new_data)` in `_design_block_for`. Add the kind to `PredictionContext`'s docstring, which enumerates every kind.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_predict.py -q`
Expected: PASS, 6 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/inference/prediction.py tests/test_replicated_predict.py
git commit -m "feat(prediction): rebuild replicated designs on (replicate, level) pairs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Equivalence against `AR1(group=)`, the rename, and docs

**Files:**
- Modify: `src/pylgm/effects/spec.py` — `AR1`
- Modify: `src/pylgm/effects/ar1.py`, `src/pylgm/compiler.py` — thread the new name
- Modify: `tests/test_replicated_spec.py` — remove the Task 1 xfail marker
- Create: `tests/test_replicated_equivalence.py`
- Modify: `docs/effects.md`, `docs/research-status.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `AR1(..., replicate=)` as the supported name; `group=` accepted with a `DeprecationWarning`.

**Why the equivalence test is the point of this task.** `AR1(group=)` is a shipped, tested implementation of exactly this concept. Making `Replicated(AR1(...), over=g)` match it **bit for bit** is what proves the new general machinery is correct, using code that was correct before this slice existed. Without it, the rename is a leap.

- [ ] **Step 1: Write the equivalence test**

Create `tests/test_replicated_equivalence.py`:

```python
"""`Replicated` against the shipped `AR1(group=)`, which is the same concept.

`AR1(group=)` predates this slice and is R-INLA's `replicate` under the wrong
name: `ar1_structure(..., group_count=G)` builds `I_G (x) T`, independent series
sharing rho and precision. Matching it bit for bit is what proves the general
machinery is right, using an implementation that was correct beforehand.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Fixed, LGM, Poisson, Replicated
from pylgm.compiler import _build_effect_block


def _frame(n=72):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "year": [f"y{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


@pytest.mark.parametrize("rho", [0.0, 0.3, -0.6, 0.9])
def test_replicated_ar1_matches_the_shipped_grouped_ar1_bit_for_bit(rho):
    frame = _frame()
    with pytest.warns(DeprecationWarning):
        legacy, _ = _build_effect_block(
            AR1("u", index="year", precision=2.0, rho=rho, group="firm"), frame
        )
    general, _ = _build_effect_block(
        Replicated(AR1("u", index="year", precision=2.0, rho=rho), over="firm"), frame
    )
    assert general.labels == legacy.labels
    assert np.allclose(general.design.toarray(), legacy.design.toarray())
    assert np.allclose(general.precision.toarray(), legacy.precision.toarray())
    assert np.allclose(general.constraints, legacy.constraints)


def test_replicated_ar1_and_grouped_ar1_fit_identically():
    frame = _frame()
    general = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            AR1("u", index="year", precision=2.0, rho=0.4), over="firm"),
    ).fit(frame, engine="laplace")
    with pytest.warns(DeprecationWarning):
        legacy = LGM(
            response="y", likelihood=Poisson(),
            predictor=Fixed("1") + AR1(
                "u", index="year", precision=2.0, rho=0.4, group="firm"),
        ).fit(frame, engine="laplace")
    assert general.log_marginal_likelihood == pytest.approx(
        legacy.log_marginal_likelihood, rel=1e-12
    )
    assert general.mean == pytest.approx(legacy.mean, rel=1e-10, abs=1e-12)


def test_ar1_replicate_is_the_supported_name_and_warns_for_group():
    frame = _frame()
    modern, _ = _build_effect_block(
        AR1("u", index="year", precision=2.0, rho=0.4, replicate="firm"), frame
    )
    with pytest.warns(DeprecationWarning, match="replicate"):
        legacy, _ = _build_effect_block(
            AR1("u", index="year", precision=2.0, rho=0.4, group="firm"), frame
        )
    assert modern.labels == legacy.labels
    assert np.allclose(modern.precision.toarray(), legacy.precision.toarray())


def test_ar1_rejects_both_names_at_once():
    with pytest.raises(ValueError, match="replicate"):
        AR1("u", index="year", replicate="firm", group="firm")
```

- [ ] **Step 2: Run it to see what fails**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_equivalence.py -q`
Expected: FAIL — `AR1` has no `replicate` argument and emits no warning

- [ ] **Step 3: Rename the field**

In `src/pylgm/effects/spec.py`'s `AR1`, replace the `group` field with:

```python
    replicate: str | None = None
    group: str | None = None
```

and in `__post_init__`, before the existing `group` validation:

```python
        if self.replicate is not None and self.group is not None:
            raise ValueError(
                "AR1 takes either `replicate` or the deprecated `group`, not both"
            )
        if self.group is not None:
            warnings.warn(
                "AR1(group=) is R-INLA's `replicate` -- independent series sharing "
                "hyperparameters -- under the wrong name. Use AR1(replicate=) "
                "instead. R-INLA's own `group` means correlated copies with a "
                "between-group structure, which this is not.",
                DeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "replicate", self.group)
```

Add `import warnings` at the top of the module if absent. Every downstream reader — `build_ar1`'s call sites in `compiler.py`, `_prediction_entry`'s grouped-AR1 branch, `build_prediction_context` — must now read `effect.replicate`. Grep for `\.group` across `src/pylgm` and update each site; `SpaceTime` and `DynamicSpatialPanel` have their own unrelated columns, so read each hit before changing it.

Keep `build_ar1`'s own parameter named `group` — it is a private builder, and renaming it would widen this diff without changing any public surface.

- [ ] **Step 4: Remove the Task 1 xfail**

`tests/test_replicated_spec.py::test_replicated_rejects_an_ar1_that_already_replicates_itself` carries an `xfail` marker from Task 1 because `AR1.replicate` did not exist. Remove the marker; the test must now pass on its own.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_replicated_equivalence.py tests/test_replicated_spec.py tests/test_ar1_group.py tests/test_ar1_fit.py -q`
Expected: PASS. `tests/test_ar1_group.py` uses `group=` throughout and will now emit `DeprecationWarning`; that is correct and expected. Do **not** silence it there — instead confirm those tests still pass and note in your report how many warnings the suite now emits.

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

- [ ] **Step 6: Document it**

Add a `## Replicated` section to `docs/effects.md` after the `Copy` section, covering: what `Replicated(effect, over)` does (`I_R ⊗ Q`, paired labels, one constraint per replicate); the model written out; that replicates share every hyperparameter but not their realizations; that it commutes with `Weighted`; that a `ParametricDesignBlock` inner effect is rejected; and that `AR1(group=)` is deprecated in its favour with the translation spelled out. Add `Replicated` to the effects table. Verify every example runs.

Update the `AR1` section's group-wise subsection to lead with `replicate=` and mention `group=` only as deprecated.

In `docs/research-status.md`, add `Replicated` with its verified / not-verified summary. Verified: Kronecker structure against a hand-built `np.kron`; one constraint per replicate with full rank; single-replicate reduction; commutation with `Weighted`; bit-for-bit equivalence with the shipped `AR1(group=)` across four `rho` values; prediction round-trip including a fixture where sorted and first-seen level order diverge; and hyperparameter effectiveness rows. Not verified: no published result on real data; `Replicated` inside a `Joint` is untested; and a `ParametricDesignBlock` inner effect is rejected rather than supported.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm docs tests/test_replicated_equivalence.py tests/test_replicated_spec.py
git commit -m "feat(effects): AR1(replicate=) replaces the misnamed group=, with equivalence proof

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

**The named silent failure is constraint replication.** `Replicated(Besag(...))` must give `R` sum-to-zero rows, not one. Task 2 asserts the count, the placement and the rank. If you find yourself passing the inner constraints through unchanged, re-read that test.

**The second silent failure is the level-frame dtype.** `_levels_frame(index, levels, frame[index].dtype)` — the dtype argument is load-bearing. Slice 1 shipped a bug where a fabricated string column made `RW1`/`RW2`/`Seasonal`/`AR1` order levels lexicographically, so an integer `year` smoothed `1 → 10 → 11 → 2`. Pass the real dtype and the shared level order stays correct.

**Do not trust this plan's enumeration of dispatch sites.** Slice 1 claimed four were complete and a sixth lived in `data/spark.py`. Slice 2 found `Predictor`'s unique-name check needed relaxing, which no plan had anticipated. Sweep for yourself, and if you find a site this plan does not name, report it rather than quietly patching it.

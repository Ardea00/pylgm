# M1 Slice 2: `Copy` Effect Modifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one latent field enter the same linear predictor twice at different indices, the second occurrence scaled — R-INLA's `f(j, copy="i", hyper=list(beta=...))`.

**Architecture:** `Copy(name, index, scale)` is not a wrapper and produces **no block of its own**. It adds `scale * A_index` to the columns of the block named `name`. Compilation therefore becomes two-pass: build every non-`Copy` effect's block, then fold each `Copy` into its target. When a `scale` is a `Hyperparameter` the target block becomes a `ParametricDesignBlock`, exactly as `Shared` does in the joint slice.

**Tech Stack:** Python 3.11+, numpy 2.x, scipy 1.14+ (sparse), pandas 2.2+, pytest 8.3+. No new dependencies.

**Spec:** `docs/design/specs/2026-09-02-pylgm-m1-effect-modifiers-design.md` (slice 2 of 4)

**Branch:** `research-tier`. Research-grade; see `docs/research-status.md`.

## Global Constraints

- **No new runtime dependency.** numpy / scipy / pandas / formulaic / pydantic / pyarrow / pyyaml / typer only.
- **No MCMC.** Deterministic approximation only.
- **`CompiledLGM` invariants preserved**: `design == hstack(blocks)`, `precision == block_diag(blocks)`.
- **No inference change.** `inference/laplace.py` and `inference/gaussian.py` are not touched.
- **Existing tests pass unchanged at every commit.** `PYTHONPATH=src python -m pytest -q`.
- **Ruff clean:** `ruff check src tests`, line length 100, rules `E4,E7,E9,F`.
- **Run tests as:** `PYTHONPATH=src python -m pytest ...`
- **Frozen dataclasses** with validation in `__post_init__`.
- **Errors** raise types from `pylgm.exceptions`, or plain `ValueError`/`TypeError` in spec `__post_init__` and in `prediction.py`, matching the surrounding code.

## The assumption this slice breaks

Every other effect produces exactly one block, and two places rely on it:

- `compile_lgm` (`compiler.py:471-476`) does `blocks.append(block)` once per effect.
- `build_prediction_context` (`compiler.py:1504`) appends one entry per effect and extends `implied_labels` per effect, then asserts the result equals `compiled.labels`.

`Copy` produces **zero** new blocks. It cannot produce its own, because `design == hstack(blocks)` means blocks occupy **disjoint column spans**, and a copy shares columns with its target by definition. So compilation becomes two-pass and the prediction entry for a copied-into block becomes composite. Tasks 2 and 4 are where that lands, and they are the two a reviewer should scrutinise hardest.

**A sixth dispatch site exists beyond `compiler.py`.** `src/pylgm/data/spark.py`'s `_required_columns` reads `effect.index` for every non-`Fixed` effect. Slice 1 discovered it the hard way — after claiming four sites were the complete set. `Copy` has an `index`, so it may work by accident there; Task 2 verifies rather than assumes, and adds the Spark test either way.

## File Structure

| File | Responsibility |
|---|---|
| `src/pylgm/effects/spec.py` (modify) | The `Copy` spec and its validation. |
| `src/pylgm/compiler.py` (modify) | Two-pass fold in `compile_lgm`; family path; prediction entry. |
| `src/pylgm/inference/prediction.py` (modify) | `copied` composite entry kind. |
| `src/pylgm/__init__.py` (modify) | Export `Copy`. |
| `tests/test_copy_spec.py` (create) | Spec validation and rejections. |
| `tests/test_copy_compile.py` (create) | Two-pass fold, target resolution, label invariance. |
| `tests/test_copy_predict.py` (create) | Prediction round-trip with fixed and estimated scale. |
| `tests/test_copy_model.py` (create) | End-to-end recovery of a known copy coefficient. |

---

### Task 1: The `Copy` spec

**Files:**
- Modify: `src/pylgm/effects/spec.py`
- Test: `tests/test_copy_spec.py`

**Interfaces:**
- Consumes: `_ComposableEffect` (`spec.py:35`) for `__add__`; `Hyperparameter` from `pylgm.parameters`.
- Produces: `Copy(name, index, scale=1.0)` — frozen dataclass with `.name: str` (the **target block's** name), `.index: str`, `.scale: float | Hyperparameter`.

**Why `.name` is the target's name, not a name of its own:** `Copy` contributes to the target block's columns and creates nothing of its own, so `effect.name` pointing at the target is correct — the compiler sites that read `effect.name` to key precisions or label blocks then refer to the block the copy actually feeds.

- [ ] **Step 1: Write the failing test**

Create `tests/test_copy_spec.py`:

```python
import pytest

from pylgm import Copy, Fixed, IID, Weighted
from pylgm.effects.spec import Predictor
from pylgm.parameters import Hyperparameter


def test_copy_carries_the_target_name_the_index_and_the_scale():
    copy = Copy("u", index="j", scale=2.0)
    assert copy.name == "u"
    assert copy.index == "j"
    assert copy.scale == 2.0


def test_copy_scale_defaults_to_one():
    assert Copy("u", index="j").scale == 1.0


def test_copy_accepts_a_hyperparameter_scale():
    beta = Hyperparameter("beta", initial=1.0)
    assert Copy("u", index="j", scale=beta).scale is beta


def test_copy_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + IID("u", index="i") + Copy("u", index="j")
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 3


def test_copy_rejects_an_empty_target_name():
    with pytest.raises(ValueError, match="name"):
        Copy("", index="j")


def test_copy_rejects_an_empty_index():
    with pytest.raises(ValueError, match="index"):
        Copy("u", index="")


def test_copy_rejects_a_non_numeric_non_hyperparameter_scale():
    with pytest.raises(TypeError, match="scale"):
        Copy("u", index="j", scale="2.0")


def test_copy_cannot_be_wrapped_by_weighted():
    # Copy is a term referencing another term, not an indexed effect of its own,
    # so wrapping it has no meaning: weight the target instead.
    with pytest.raises(TypeError):
        Weighted(Copy("u", index="j"), by="z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'Copy'`

- [ ] **Step 3: Implement the spec**

In `src/pylgm/effects/spec.py`, add next to `Weighted`:

```python
@dataclass(frozen=True)
class Copy(_ComposableEffect):
    """A second occurrence of an existing latent field, at a different index.

    ``Copy("u", index="j", scale=beta)`` adds ``beta * A_j`` to the columns of
    the block named ``u``, so the field ``u`` enters the predictor twice: once
    at its own index and once at ``j``, scaled. This is R-INLA's
    ``f(j, copy="u", hyper=list(beta=...))``.

    It produces no block of its own -- there is one latent field, entering
    twice -- so ``name`` is the **target** block's name, and every compiler
    site that reads ``effect.name`` then refers to the block this copy feeds.

    ``scale`` may be a ``Hyperparameter``, which makes the target block's design
    depend on it; the compiler registers it as a ParametricDesignBlock.
    """

    name: str
    index: str
    scale: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if not isinstance(self.scale, (int, float, Hyperparameter)) or isinstance(
            self.scale, bool
        ):
            raise TypeError(
                f"Copy scale must be a real number or a Hyperparameter, got "
                f"{type(self.scale).__name__}"
            )
```

`Weighted.__post_init__` already rejects anything without `.index`; `Copy` **has** an `index`, so add an explicit rejection there too:

```python
        if isinstance(self.effect, Copy):
            raise TypeError(
                "Weighted cannot wrap a Copy: a copy is a term referencing "
                "another term, not an indexed effect of its own. Weight the "
                "target effect instead."
            )
```

Place that check before the `hasattr(self.effect, "index")` check, and define `Copy` above `Weighted` so the name resolves.

- [ ] **Step 4: Export it**

Add `Copy` to `src/pylgm/effects/__init__.py`'s re-exports, to `src/pylgm/__init__.py`'s imports and `__all__` (alphabetical: after `ComparisonResult`, before `DynamicSpatialPanel`), and to the expected export set in `tests/test_package.py`.

- [ ] **Step 5: Run tests and ruff**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_spec.py tests/test_weighted_spec.py tests/test_public_exports.py -q`
Expected: PASS

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/test_copy_spec.py tests/test_package.py
git commit -m "feat(effects): Copy spec for a second occurrence of a latent field

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Two-pass compilation with a fixed scale

**Files:**
- Modify: `src/pylgm/compiler.py` — `compile_lgm` (~line 465)
- Test: `tests/test_copy_compile.py`

**Interfaces:**
- Consumes: `Copy` (Task 1); `_build_effect_block(effect, frame)`; `_resolve_scale(scale, resolved, default)` at `compiler.py:711`.
- Produces: `_copy_incidence(frame, copy, labels) -> csr_matrix` and `_fold_copies(blocks, copies, frame, resolved) -> list[LatentBlock]`, both used again by Task 3.

**The architectural change.** `compile_lgm`'s loop currently appends one block per effect. Split it: effects that are not `Copy` build blocks as today; `Copy` effects are collected. After the loop, fold each copy into its target. The block count then equals the number of non-`Copy` effects, which is what keeps `design == hstack(blocks)` true.

- [ ] **Step 1: Write the failing test**

Create `tests/test_copy_compile.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import Copy, Fixed, IID, LGM, Poisson
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.exceptions import CompilationError


def _frame():
    return pd.DataFrame({
        "i": ["a", "b", "c", "a"],
        "j": ["b", "c", "a", "c"],
        "y": [1.0, 2.0, 3.0, 4.0],
        "row": range(4),
    })


def _panel(frame):
    return CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )


def _compiled(predictor, frame=None):
    frame = _frame() if frame is None else frame
    model = LGM(response="y", likelihood=Poisson(), predictor=predictor)
    return compile_lgm(model, _panel(frame))


def test_a_copy_adds_no_block_of_its_own():
    without = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    with_copy = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j")
    )
    assert len(with_copy.blocks) == len(without.blocks)
    assert with_copy.labels == without.labels


def test_a_copy_adds_its_scaled_incidence_to_the_target_design():
    frame = _frame()
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    copied = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j", scale=2.0)
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_copy = [b for b in copied.blocks if b.name == "u"][0]

    labels = list(u_base.labels)
    position = {label: k for k, label in enumerate(labels)}
    expected = u_base.design.toarray().copy()
    for row, level in enumerate(frame["j"].astype(str)):
        expected[row, position[level]] += 2.0
    assert np.allclose(u_copy.design.toarray(), expected)


def test_precision_labels_and_constraints_are_untouched_by_a_copy():
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    copied = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j")
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_copy = [b for b in copied.blocks if b.name == "u"][0]
    assert u_copy.labels == u_base.labels
    assert np.allclose(u_copy.precision.toarray(), u_base.precision.toarray())
    assert np.allclose(u_copy.constraints, u_base.constraints)


def test_a_zero_scale_copy_leaves_the_design_unchanged():
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    copied = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j", scale=0.0)
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_copy = [b for b in copied.blocks if b.name == "u"][0]
    assert np.allclose(u_copy.design.toarray(), u_base.design.toarray())


def test_copy_naming_a_missing_block_is_rejected():
    with pytest.raises(CompilationError, match="nonexistent"):
        _compiled(Fixed("1") + IID("u", index="i") + Copy("nonexistent", index="j"))


def test_copy_of_a_copy_is_rejected():
    with pytest.raises(CompilationError, match="copy"):
        _compiled(
            Fixed("1") + IID("u", index="i") + Copy("u", index="j") + Copy("u", index="i")
        )


def test_copy_whose_index_has_a_level_outside_the_target_is_rejected():
    # A copy reuses an existing latent field; it cannot create a new level in it.
    frame = _frame()
    frame.loc[0, "j"] = "zz"
    with pytest.raises(CompilationError, match="zz"):
        _compiled(Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j"),
                  frame=frame)


def test_a_copy_can_reference_a_weighted_block():
    """The spec allows a copy to reference a wrapped block: the copy contributes
    to that block's columns as they were built. Weighted keeps the inner name,
    so `Copy("u", ...)` finds it, and the copy's incidence is NOT re-weighted --
    the weighting belongs to the occurrence that declared it."""
    from pylgm import Weighted

    frame = _frame().assign(z=[2.0, 3.0, 4.0, 5.0])
    compiled = _compiled(
        Fixed("1") + Weighted(IID("u", index="i", precision=1.0), by="z")
        + Copy("u", index="j", scale=1.0),
        frame=frame,
    )
    u = [b for b in compiled.blocks if b.name == "u"][0]
    labels = list(u.labels)
    position = {label: k for k, label in enumerate(labels)}

    expected = np.zeros((len(frame), len(labels)))
    for row, (level_i, level_j, weight) in enumerate(
        zip(frame["i"].astype(str), frame["j"].astype(str), frame["z"])
    ):
        expected[row, position[level_i]] += weight    # weighted occurrence
        expected[row, position[level_j]] += 1.0       # unweighted copy
    assert np.allclose(u.design.toarray(), expected)


def test_copy_index_column_missing_is_rejected():
    frame = _frame().drop(columns=["j"])
    with pytest.raises(CompilationError, match="j"):
        _compiled(Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j"),
                  frame=frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_compile.py -q`
Expected: FAIL — `_build_effect_block` raises `CompilationError: unsupported effect type: Copy`

- [ ] **Step 3: Implement the incidence and the fold**

In `src/pylgm/compiler.py`, add above `compile_lgm`:

```python
def _copy_incidence(frame, copy, labels: "tuple[str, ...]") -> csr_matrix:
    """The incidence A_index of a copy's index over its target block's levels.

    A copy reuses an existing latent field, so every value of its index column
    must already be a level of the target. A value outside that set would mean
    creating a new latent component, which a copy by definition cannot do.
    """
    if copy.index not in frame.columns:
        raise DataContractError(
            f"copy index column {copy.index!r} for target block {copy.name!r} not found"
        )
    position = {label: column for column, label in enumerate(labels)}
    values = frame[copy.index].astype(str)
    unknown = sorted({value for value in values if value not in position})
    if unknown:
        raise CompilationError(
            f"copy of {copy.name!r} indexes level(s) {unknown!r} through column "
            f"{copy.index!r} that the target block does not have. A copy reuses an "
            "existing latent field and cannot add a level to it."
        )
    rows = np.arange(len(frame))
    columns = np.array([position[value] for value in values])
    return csr_matrix(
        (np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(labels))
    )


def _fold_copies(blocks, copies, frame, resolved) -> "list[LatentBlock]":
    """Add each copy's scaled incidence to the columns of the block it names.

    Copies produce no block of their own: ``design == hstack(blocks)`` gives each
    block a disjoint column span, and a copy shares its target's columns by
    definition. Folding here is what keeps that invariant true.
    """
    by_name = {block.name: index for index, block in enumerate(blocks)}
    folded = list(blocks)
    for copy in copies:
        if copy.name not in by_name:
            raise CompilationError(
                f"copy targets block {copy.name!r}, which this model does not "
                f"declare. Declared blocks: {sorted(by_name)!r}"
            )
        position = by_name[copy.name]
        target = folded[position]
        scale = _resolve_scale(copy.scale, resolved, default=1.0)
        incidence = _copy_incidence(frame, copy, target.labels)
        folded[position] = LatentBlock(
            target.name,
            target.labels,
            csr_matrix(target.design + scale * incidence),
            target.precision,
            target.constraints,
        )
    return folded


def _split_copies(effects) -> "tuple[list, list]":
    """Separate ordinary effects from copies, rejecting a copy of a copy.

    A copy of a copy has no meaning here: both fold into the same target
    columns, so the second is just a third scaled incidence -- expressible as
    one copy with the summed scale, and ambiguous as written.
    """
    ordinary, copies, targets = [], [], set()
    for effect in effects:
        if isinstance(effect, Copy):
            if effect.name in targets:
                raise CompilationError(
                    f"more than one copy targets block {effect.name!r}; a copy of a "
                    "copy folds into the same columns, so express it as one copy "
                    "with the combined scale"
                )
            targets.add(effect.name)
            copies.append(effect)
        else:
            ordinary.append(effect)
    return ordinary, copies
```

Then rewrite `compile_lgm`'s loop:

```python
    ordinary, copies = _split_copies(model.predictor.effects)
    blocks: list[LatentBlock] = []
    precisions: dict[str, float] = {}
    for effect in ordinary:
        block, precision = _build_effect_block(effect, frame)
        if precision is not None:
            precisions[effect.name] = precision
        blocks.append(block)
    blocks = _fold_copies(blocks, copies, frame, {})
```

Add `Copy` to `compiler.py`'s `from pylgm.effects import (...)` block.

- [ ] **Step 4: Run the new tests**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_compile.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Verify the Spark path handles `Copy`**

`src/pylgm/data/spark.py`'s `_required_columns` reads `effect.index` for every non-`Fixed` effect. `Copy` has an `index`, so it may already work — **verify rather than assume**, because slice 1 shipped a claim that four dispatch sites were the complete set and a sixth existed in this exact file. Add to `tests/test_copy_compile.py`:

```python
def test_spark_required_columns_include_a_copy_index():
    pytest.importorskip("pyspark")
    from pylgm.data.spark import _required_columns

    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j"),
    )
    required = _required_columns(model)
    assert "i" in required and "j" in required
```

If it fails, fix `_required_columns` in the same commit and say so in your report.

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/compiler.py tests/test_copy_compile.py
git commit -m "feat(compiler): fold Copy into its target block in a second pass

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: An estimated copy scale

**Files:**
- Modify: `src/pylgm/compiler.py` — `_effect_hyperparameters` (~line 705), `compile_family` / `_append_family_blocks` (~line 1030)
- Test: `tests/test_copy_compile.py` (append)

**Interfaces:**
- Consumes: `_copy_incidence`, `_fold_copies`, `_split_copies` (Task 2); `ParametricDesignBlock` from `pylgm.ir.family`.
- Produces: no new public names.

**Why the target becomes a `ParametricDesignBlock`:** with `scale` estimated, the target block's **design** depends on a hyperparameter and must be rebuilt on every draw. That is exactly what `Shared` does in the joint slice, and the same class serves it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_copy_compile.py`:

```python
def test_an_estimated_copy_scale_is_discovered_as_a_hyperparameter():
    from pylgm.compiler import _effect_hyperparameters
    from pylgm.parameters import Hyperparameter

    beta = Hyperparameter("beta", initial=1.0)
    assert [hp.name for hp in _effect_hyperparameters(Copy("u", index="j", scale=beta))] == [
        "beta"
    ]


def test_a_model_with_an_estimated_copy_scale_fits_and_reports_beta():
    from pylgm.parameters import Hyperparameter

    rng = np.random.default_rng(4)
    n = 80
    levels = [f"L{k}" for k in range(10)]
    i = [levels[k % 10] for k in range(n)]
    j = [levels[(k * 3) % 10] for k in range(n)]
    frame = pd.DataFrame({
        "i": i, "j": j, "y": rng.poisson(3.0, n).astype(float), "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    )
    result = model.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert np.isfinite(result.hyperparameters["beta"])


def test_the_estimated_scale_is_applied_on_every_rebuild_not_only_the_template():
    """A ParametricDesignBlock rebuilds its design per draw. If the copy were
    folded only into the template, every draw after the first would silently
    lose it -- a model that fits and returns plausible numbers."""
    from pylgm.compiler import compile_family
    from pylgm.ir.family import ParametricDesignBlock
    from pylgm.parameters import Hyperparameter

    frame = _frame()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    )
    family = compile_family(model, _panel(frame))
    assert family is not None
    assert "beta" in family.parameter_names

    item = [b for b in family.blocks if b.block.name == "u"][0]
    assert isinstance(item, ParametricDesignBlock)
    at_one = item.build({"beta": 1.0}).toarray()
    at_three = item.build({"beta": 3.0}).toarray()
    # Changing beta must change the design; a template-only fold would not.
    assert not np.allclose(at_one, at_three)
    # And the difference is exactly twice the copy's incidence.
    assert np.allclose(at_three - at_one, 2.0 * (at_one - _base_design(frame)))


def _base_design(frame):
    """The u design with no copy folded in, for the difference check above."""
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0), frame=frame)
    return [b for b in base.blocks if b.name == "u"][0].design.toarray()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_compile.py -q -k "estimated or rebuild"`
Expected: FAIL — `_effect_hyperparameters` does not know `Copy`, and `compile_family` does not fold copies

- [ ] **Step 3: Discover the hyperparameter**

In `_effect_hyperparameters`, alongside the existing `Weighted` delegation:

```python
    if isinstance(effect, Copy):
        return [effect.scale] if isinstance(effect.scale, Hyperparameter) else []
```

- [ ] **Step 4: Fold copies in the family path**

`compile_family` builds `scalable` through `_append_family_blocks`. Split copies out the same way `compile_lgm` does, and after the loop wrap each targeted block:

```python
    ordinary, copies = _split_copies(model.predictor.effects)
    for effect in ordinary:
        _append_family_blocks(
            effect, frame, scalable, parameter_names, parameter_bounds, parameter_priors
        )
    for copy in copies:
        matches = [k for k, item in enumerate(scalable) if item.block.name == copy.name]
        if not matches:
            raise CompilationError(
                f"copy targets block {copy.name!r}, which this model does not "
                f"declare. Declared blocks: "
                f"{sorted(item.block.name for item in scalable)!r}"
            )
        position = matches[0]
        incidence = _copy_incidence(frame, copy, scalable[position].block.labels)
        scalable[position] = _copied_family_block(scalable[position], copy, incidence)
        if isinstance(copy.scale, Hyperparameter):
            parameter_names.append(copy.scale.name)
            parameter_bounds[copy.scale.name] = _real_bounds(copy.scale, "copy scale")
            if copy.scale.prior is not None:
                parameter_priors[copy.scale.name] = copy.scale.prior
```

with the helper:

```python
def _copied_family_block(item, copy, incidence):
    """Fold a copy into one family block, rebuilding per draw when needed.

    A fixed scale bakes into the template. An estimated scale makes the design
    a function of the hyperparameter, so it must be re-formed on every draw --
    folding only into the template would silently drop the copy from every draw
    after the first.
    """
    inner = item.block
    fixed = not isinstance(copy.scale, Hyperparameter)
    baked = LatentBlock(
        inner.name,
        inner.labels,
        csr_matrix(inner.design + _resolve_scale(copy.scale, {}, default=1.0) * incidence),
        inner.precision,
        inner.constraints,
    )
    if fixed:
        if isinstance(item, ParametricDesignBlock):
            def build(values, inner_build=item.build, incidence=incidence,
                      scale=float(copy.scale)):
                return csr_matrix(inner_build(values) + scale * incidence)

            return ParametricDesignBlock(baked, item.parameters, build)
        if isinstance(item, ParametricBlock):
            return ParametricBlock(baked, item.parameters, item.build)
        return ScalableBlock(baked, item.parameter, item.scale)

    name = copy.scale.name
    base_design = inner.design
    inner_build = item.build if isinstance(item, ParametricDesignBlock) else None
    parameters = tuple(dict.fromkeys((item.parameters if isinstance(
        item, (ParametricBlock, ParametricDesignBlock)) else ()) + (name,)))

    def build(values, base_design=base_design, incidence=incidence, name=name,
              inner_build=inner_build):
        design = inner_build(values) if inner_build is not None else base_design
        return csr_matrix(design + float(values[name]) * incidence)

    return ParametricDesignBlock(baked, parameters, build)
```

Note the `ParametricBlock` case with an estimated copy scale: that block's `build` produces a **precision**, so it cannot also carry the design. Reject it rather than silently mishandling:

```python
    if isinstance(item, ParametricBlock) and not fixed:
        raise CompilationError(
            f"copy of {copy.name!r} has an estimated scale, but that block's "
            "precision is itself a function of hyperparameters. Combining an "
            "estimated copy scale with an estimated structural parameter on the "
            "same block is not supported; fix one of them."
        )
```

Place this check immediately after `fixed` is computed and before `baked` is
built, so it fires before any design work happens. It is the only combination
this slice refuses, and refusing it loudly is the point: a `ParametricBlock`'s
`build` returns a **precision**, so it cannot also carry a design, and silently
picking one of the two would produce a model that fits and is wrong.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_compile.py -q`
Expected: PASS, 12 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py tests/test_copy_compile.py
git commit -m "feat(compiler): estimated copy scale via ParametricDesignBlock

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Prediction through a copied block

**Files:**
- Modify: `src/pylgm/compiler.py` — `build_prediction_context` (~line 1504), `_prediction_entry` (~line 1547)
- Modify: `src/pylgm/inference/prediction.py` — `_design_block_for`
- Test: `tests/test_copy_predict.py`

**Interfaces:**
- Consumes: `_design_block_for(entry, new_data)`; the `("weighted", ...)` recursion pattern from slice 1.
- Produces: entry kind `("copied", (base_entry, ((index, labels, scale_spec, fitted), ...)))`.

**Why `build_prediction_context` changes, not only `_prediction_entry`.** That function appends one entry per effect and asserts `implied_labels == compiled.labels`. A `Copy` contributes no labels of its own, so emitting an entry for it would break the assertion. Copies must instead be attached to their target's entry — which means the loop needs the same split Task 2 introduced.

- [ ] **Step 1: Write the failing test**

Create `tests/test_copy_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import Copy, Fixed, IID, LGM, Poisson
from pylgm.parameters import Hyperparameter


def _data(seed=9, n=60):
    rng = np.random.default_rng(seed)
    levels = [f"L{k}" for k in range(10)]
    return pd.DataFrame({
        "i": [levels[k % 10] for k in range(n)],
        "j": [levels[(k * 3) % 10] for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


def _fitted(scale):
    frame = _data()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=scale),
    )
    return frame, model.fit(frame, engine="laplace")


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means_fixed_scale():
    frame, result = _fitted(2.0)
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means_estimated_scale():
    frame, result = _fitted(Hyperparameter("beta", initial=1.0))
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_prediction_uses_the_copy_index_from_new_data():
    """The copy's index is data. Permuting it on new rows must move the
    prediction; a predict path that rebuilt only the base design would not."""
    frame, result = _fitted(2.0)
    permuted = frame.assign(j=frame["j"].values[::-1])
    assert not np.allclose(
        result.predict(frame).predictive_mean,
        result.predict(permuted).predictive_mean,
    )


def test_predict_rejects_new_data_missing_the_copy_index():
    frame, result = _fitted(2.0)
    with pytest.raises(ValueError, match="j"):
        result.predict(frame.drop(columns=["j"]))


def test_predict_rejects_an_unseen_copy_level():
    frame, result = _fitted(2.0)
    unseen = frame.copy()
    unseen.loc[0, "j"] = "ZZ"
    with pytest.raises(ValueError, match="ZZ"):
        result.predict(unseen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_predict.py -q`
Expected: FAIL — the fitted-mean round-trip disagrees, because predict rebuilds the base design without the copy

- [ ] **Step 3: Attach copies to their target's entry**

In `build_prediction_context`, split copies out and attach them:

```python
    ordinary, copies = _split_copies(model.predictor.effects)
    fitted = dict(result.hyperparameters or {}) if result is not None else {}
    by_target: dict[str, list] = {}
    for copy in copies:
        spec = copy.scale.name if isinstance(copy.scale, Hyperparameter) else float(copy.scale)
        value = _resolve_scale(copy.scale, fitted, default=1.0)
        by_target.setdefault(copy.name, []).append((copy.index, spec, value))
```

then, inside the per-effect loop, wrap the entry when that effect's block is a copy target:

```python
        entry = _prediction_entry(effect, model, panel, block)
        if effect.name in by_target:
            entry = ("copied", (entry, tuple(
                (index, block.labels, spec, value)
                for index, spec, value in by_target[effect.name]
            )))
        entries.append(entry)
```

`build_prediction_context` currently has no `result` parameter. Give it one, defaulting to `None`, and pass the fitted result from `LGM._fit_pandas` and `_fit_spark`, which already have it. `build_joint_prediction_contexts` already receives a `result` and passes it through.

- [ ] **Step 4: Rebuild it at predict time**

In `src/pylgm/inference/prediction.py`, add:

```python
def _copied_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    """Rebuild a block that one or more copies fold into.

    The base entry supplies the block's own design; each copy adds its scaled
    incidence over the same columns, which is what makes a copy a second
    occurrence of one field rather than a second field.
    """
    base_entry, copies = entry
    design = _design_block_for(base_entry, new_data)
    for index_column, labels, _spec, fitted in copies:
        if index_column not in new_data.columns:
            raise ValueError(
                f"predict() new_data is missing the copy index column {index_column!r}"
            )
        position = {label: k for k, label in enumerate(labels)}
        values = new_data[index_column].astype(str)
        unknown = sorted({value for value in values if value not in position})
        if unknown:
            raise ValueError(
                f"predict() cannot score rows whose copy level was not in the fitted "
                f"model: {unknown!r}"
            )
        addition = np.zeros_like(design)
        addition[np.arange(len(new_data)), values.map(position).to_numpy()] = fitted
        design = design + addition
    return design
```

and register `elif kind == "copied": return _copied_block(payload, new_data)` in `_design_block_for`.

Add the new kind to `PredictionContext`'s docstring, which enumerates every kind.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_predict.py -q`
Expected: PASS, 5 passed

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/inference/prediction.py src/pylgm/model.py tests/test_copy_predict.py
git commit -m "feat(prediction): rebuild copied blocks with their folded copies

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Recovery of a known copy scale, plus docs

**Files:**
- Create: `tests/test_copy_model.py`
- Modify: `docs/effects.md`, `docs/research-status.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the recovery evidence that this slice delivered what it claims.

**Why this task exists:** Tasks 1-4 prove the mechanism is internally consistent. A `Copy` that compiles and predicts but cannot recover a known scale from data generated with it would satisfy every earlier test and still be wrong.

- [ ] **Step 1: Write the recovery test**

Create `tests/test_copy_model.py`:

```python
"""A field entering one predictor twice, the second time scaled.

    log mu_k = alpha + u_{i(k)} + beta * u_{j(k)}

Recovering `beta` from data generated with it is what proves the copy folds
into the right columns with the right coefficient, rather than merely
compiling.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import Copy, Fixed, IID, LGM, Poisson
from pylgm.parameters import Hyperparameter

N_LEVELS, N_ROWS, TRUE_BETA, SIGMA_U = 12, 600, 1.8, 0.5


def _simulate(seed=31):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, SIGMA_U, N_LEVELS)
    i = rng.integers(0, N_LEVELS, N_ROWS)
    j = rng.integers(0, N_LEVELS, N_ROWS)
    eta = 0.3 + u[i] + TRUE_BETA * u[j]
    return u, pd.DataFrame({
        "i": [f"L{k}" for k in i],
        "j": [f"L{k}" for k in j],
        "y": rng.poisson(np.exp(eta)).astype(float),
        "row": range(N_ROWS),
    })


def test_the_copy_scale_is_recovered():
    _, frame = _simulate()
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1")
        + IID("u", index="i", precision=1.0 / SIGMA_U**2)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    ).fit(frame, engine="laplace")
    assert result.hyperparameters["beta"] == pytest.approx(TRUE_BETA, rel=0.4)


def test_a_unit_scale_copy_equals_stacking_the_index_columns_by_hand():
    """With beta = 1, u_{i(k)} + u_{j(k)} is what you would get by summing two
    one-hot designs over the same levels -- so the copy must reproduce exactly
    the design a hand-built sum gives."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    _, frame = _simulate()
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    copied = compile_lgm(
        LGM(response="y", likelihood=Poisson(),
            predictor=Fixed("1") + IID("u", index="i", precision=1.0)
            + Copy("u", index="j", scale=1.0)),
        panel,
    )
    base = compile_lgm(
        LGM(response="y", likelihood=Poisson(),
            predictor=Fixed("1") + IID("u", index="i", precision=1.0)),
        panel,
    )
    u_copied = [b for b in copied.blocks if b.name == "u"][0].design.toarray()
    u_base = [b for b in base.blocks if b.name == "u"][0].design.toarray()

    labels = [b for b in base.blocks if b.name == "u"][0].labels
    position = {label: k for k, label in enumerate(labels)}
    manual = u_base.copy()
    for row, level in enumerate(frame["j"].astype(str)):
        manual[row, position[level]] += 1.0
    assert np.allclose(u_copied, manual)
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src python -m pytest tests/test_copy_model.py -q`
Expected: PASS, 2 passed

If the recovery fails, do **not** widen the tolerance, change the seed, or grow the sample. Report the observed `beta` and investigate. A copy that compiles but cannot recover a known scale is the failure this task exists to catch.

- [ ] **Step 3: Document it**

Add a `### Copy` section to `docs/effects.md` after the `Weighted` section, covering: what `Copy(name, index, scale)` does (adds `scale * A_index` to the target block's columns); the two-index model written out; that it produces no block of its own so `result.labels` is unchanged by adding a copy; that `scale` may be a `Hyperparameter`; that the copy's index values must already be levels of the target; and that a copy of a copy, and `Weighted(Copy(...))`, are rejected. Add `Copy` to the effects table at the top of the page. Verify every example you write actually runs.

In `docs/research-status.md`, add `Copy` alongside `Weighted` with its own verified / not-verified summary. Verified: design equality against a hand-built sum, no-extra-block invariance, prediction round-trip at 1e-12 with both fixed and estimated scale, and recovery of a known `beta`. Not verified: no published result on real data; `Copy` inside a `Joint` is untested; and an estimated copy scale on a block whose precision is also estimated is rejected rather than supported.

- [ ] **Step 4: Run the full suite and ruff**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 5: Commit**

```bash
git add tests/test_copy_model.py docs/effects.md docs/research-status.md
git commit -m "test(effects): recover a known copy scale end to end

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

**The two-pass fold is the whole slice.** Everything else follows from it. If you find yourself giving `Copy` a block of its own, stop: `design == hstack(blocks)` gives each block a disjoint column span, and a copy shares its target's columns by definition.

**The silent failure mode here** is a copy folded into the *template* but not into the per-draw rebuild, so every hyperparameter draw after the first quietly loses it. The model fits and returns plausible numbers. Task 3's `test_the_estimated_scale_is_applied_on_every_rebuild_not_only_the_template` is the guard; make sure it genuinely fails without the fix rather than adjusting it.

**Do not trust any enumeration of dispatch sites, including this plan's.** Slice 1 stated four sites were complete and a sixth existed in `data/spark.py`. Task 2 Step 5 checks that file explicitly; if you find a seventh anywhere, report it rather than quietly patching it.

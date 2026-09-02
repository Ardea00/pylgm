# M1 Slice 1: `Weighted` Effect Modifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any indexed latent effect be modulated by a numeric column, so `y ~ ... + z_i * u_{s(i)}` becomes expressible — the spatially-varying coefficient (atlas M14).

**Architecture:** `Weighted(effect, by)` is a wrapper spec, not a field on every effect. It builds the inner effect's `LatentBlock` and scales the design rows by the `by` column: `diag(w) A`. Precision, labels and constraints pass through untouched, so both `CompiledLGM` invariants hold and no inference code changes.

**Tech Stack:** Python 3.11+, numpy 2.x, scipy 1.14+ (sparse), pandas 2.2+, pytest 8.3+. No new dependencies.

**Spec:** `docs/design/specs/2026-09-02-pylgm-m1-effect-modifiers-design.md` (slice 1 of 4)

**Branch:** `research-tier`. This is research-grade work; see `docs/research-status.md`.

## Global Constraints

- **No new runtime dependency.** numpy / scipy / pandas / formulaic / pydantic / pyarrow / pyyaml / typer only.
- **No MCMC.** Deterministic approximation only.
- **`CompiledLGM` invariants preserved**: `design == hstack(blocks)`, `precision == block_diag(blocks)`. `Weighted` produces exactly one block.
- **No inference change.** `inference/laplace.py` and `inference/gaussian.py` are not touched.
- **Existing tests pass unchanged at every commit.** `PYTHONPATH=src python -m pytest -q`.
- **Ruff clean:** `ruff check src tests`, line length 100, rules `E4,E7,E9,F`.
- **Run tests as:** `PYTHONPATH=src python -m pytest ...`
- **Frozen dataclasses** with validation in `__post_init__`, matching every existing spec type.
- **Errors** raise types from `pylgm.exceptions` (`CompilationError`, `DataContractError`, `ModelValidationError`) or plain `ValueError`/`TypeError` in spec `__post_init__`, matching the surrounding code.

## The dispatch problem this slice must solve

Seven sites in `src/pylgm/compiler.py` iterate `model.predictor.effects` and branch on effect type. A wrapper unhandled at any of them falls through silently. They reduce to **four functions**:

| function | line | covers sites |
|---|---|---|
| `_build_effect_block(effect, frame)` | ~358 | `compile_lgm` (471), `compile_joint` (779), `compile_joint_family` (1682) |
| `_effect_hyperparameters(effect)` | ~661 | `_model_hyperparameters` (858) |
| `compile_family`'s own effect chain | ~967 | itself |
| `_prediction_entry(effect, model, panel, block)` | ~1447 | `build_prediction_context` (1504), `build_joint_prediction_contexts` (1541) |

Tasks 2-4 fix these four. Task 5 proves nothing was missed.

## File Structure

| File | Responsibility |
|---|---|
| `src/pylgm/effects/spec.py` (modify) | Add the `Weighted` spec and its validation. |
| `src/pylgm/compiler.py` (modify) | Wrapper cases in the four dispatch functions. |
| `src/pylgm/inference/prediction.py` (modify) | Recursive `weighted` entry kind in `_design_for`. |
| `src/pylgm/__init__.py` (modify) | Export `Weighted`. |
| `tests/test_weighted_spec.py` (create) | Spec validation and rejections. |
| `tests/test_weighted_compile.py` (create) | Block construction, reduction, delegation. |
| `tests/test_weighted_predict.py` (create) | Prediction round-trip. |
| `tests/test_weighted_svc.py` (create) | End-to-end spatially-varying coefficient recovery. |

---

### Task 1: The `Weighted` spec

**Files:**
- Modify: `src/pylgm/effects/spec.py`
- Test: `tests/test_weighted_spec.py`

**Interfaces:**
- Consumes: `_ComposableEffect` (`spec.py:35`), which gives `__add__` so effects compose with `+`.
- Produces: `Weighted(effect, by)` — frozen dataclass with `.effect`, `.by: str`, and a `.name` property delegating to `effect.name`.

**Why `.name` delegates:** many compiler sites read `effect.name` to key precisions, label blocks and report errors. Delegating means a `Weighted` block keeps the inner effect's name, so `result.labels` and `latent_marginals("u")` are unchanged by wrapping.

- [ ] **Step 1: Write the failing test**

Create `tests/test_weighted_spec.py`:

```python
import pytest

from pylgm import Besag, Fixed, IID, Weighted
from pylgm.effects.spec import Predictor


def test_weighted_delegates_its_name_to_the_inner_effect():
    inner = IID("u", index="district")
    assert Weighted(inner, by="z").name == "u"


def test_weighted_keeps_the_inner_effect_accessible():
    inner = IID("u", index="district")
    wrapped = Weighted(inner, by="z")
    assert wrapped.effect is inner
    assert wrapped.by == "z"


def test_weighted_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + Weighted(IID("u", index="district"), by="z")
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 2


def test_weighted_rejects_a_non_string_by():
    with pytest.raises((TypeError, ValueError), match="by"):
        Weighted(IID("u", index="district"), by=3)


def test_weighted_rejects_an_empty_by():
    with pytest.raises(ValueError, match="by"):
        Weighted(IID("u", index="district"), by="")


def test_weighted_rejects_an_effect_with_no_index():
    # Fixed builds its design from a formula, not an index, so weighting it is
    # meaningless -- multiply the covariate into the formula instead.
    with pytest.raises(TypeError, match="index"):
        Weighted(Fixed("1"), by="z")


def test_weighted_rejects_wrapping_a_weighted():
    # Two weight columns on one block is just their product; nesting would give
    # two ways to say one thing and complicate the prediction entry.
    inner = Weighted(IID("u", index="district"), by="z")
    with pytest.raises(TypeError, match="already weighted"):
        Weighted(inner, by="w")


def test_weighted_accepts_a_graph_effect():
    graph = {"a": ["b"], "b": ["a"]}
    wrapped = Weighted(Besag("u", index="district", graph=graph), by="z")
    assert wrapped.name == "u"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_spec.py -q`
Expected: FAIL with `ImportError: cannot import name 'Weighted'`

- [ ] **Step 3: Implement the spec**

In `src/pylgm/effects/spec.py`, add after the last effect class and before `Predictor`:

```python
@dataclass(frozen=True)
class Weighted(_ComposableEffect):
    """An indexed latent effect modulated by a numeric column.

    The design becomes ``diag(by) A`` instead of the plain incidence ``A``, so
    the effect's contribution to the predictor is ``by_i * u_{index(i)}``. This
    is R-INLA's ``f(index, weights, model=...)``, and it is what a
    spatially-varying coefficient needs: a covariate whose effect varies over a
    latent field.

    Precision, labels and constraints are the inner effect's, untouched --
    weighting changes how the field enters the predictor, not the field itself.
    """

    effect: object
    by: str

    def __post_init__(self) -> None:
        if isinstance(self.effect, Weighted):
            raise TypeError(
                "Weighted effect is already weighted; two weight columns on one "
                "block is their product, so multiply them into a single column"
            )
        if not hasattr(self.effect, "index"):
            raise TypeError(
                f"Weighted requires an indexed effect, got "
                f"{type(self.effect).__name__}, which has no index. A Fixed effect "
                "builds its design from a formula -- multiply the covariate into "
                "the formula instead."
            )
        object.__setattr__(self, "by", _non_empty_string(self.by, "by"))

    @property
    def name(self) -> str:
        return self.effect.name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_spec.py -q`
Expected: PASS, 8 passed

Note: the import in the test is `from pylgm import ... Weighted`, so export it now — add `from pylgm.effects.spec import Weighted` to whatever `src/pylgm/effects/__init__.py` re-exports, then `Weighted` to `src/pylgm/__init__.py`'s imports and to `__all__` in alphabetical position (after `SpaceTime`). Also add `"Weighted"` to the expected export list in `tests/test_package.py`.

- [ ] **Step 5: Run the export test and ruff**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_spec.py tests/test_public_exports.py -q`
Expected: PASS

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/test_weighted_spec.py tests/test_package.py
git commit -m "feat(effects): Weighted spec for covariate-modulated latent effects

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Compile a `Weighted` block

**Files:**
- Modify: `src/pylgm/compiler.py` — `_build_effect_block` (~line 358) and `_effect_hyperparameters` (~line 661)
- Test: `tests/test_weighted_compile.py`

**Interfaces:**
- Consumes: `Weighted` (Task 1); `_build_effect_block(effect, frame) -> (LatentBlock, float | None)`; `_effect_hyperparameters(effect) -> list[Hyperparameter]`.
- Produces: a compiled `Weighted` block whose design is `diag(w) A`. No new public names.

**Why `_effect_hyperparameters` must recurse:** it feeds `_model_hyperparameters`, which is how `LGM.fit` discovers what to estimate. A `Weighted(IID("u", precision=Hyperparameter("tau")))` whose inner hyperparameter is not found would silently pin `tau` at its initial value — the same silent-misspecification failure the joint slice hit with `tau_u`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_weighted_compile.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted
from pylgm.compiler import _build_effect_block, _effect_hyperparameters
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.parameters import Hyperparameter


def _frame(z):
    return pd.DataFrame({"district": ["a", "b", "c", "a"], "z": z, "row": range(4)})


def test_weighted_design_is_the_inner_design_scaled_row_wise():
    z = [2.0, -1.0, 0.5, 3.0]
    frame = _frame(z)
    plain, _ = _build_effect_block(IID("u", index="district"), frame)
    weighted, _ = _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)

    expected = np.diag(z) @ plain.design.toarray()
    assert np.allclose(weighted.design.toarray(), expected)


def test_weighted_preserves_precision_labels_and_constraints():
    frame = _frame([2.0, -1.0, 0.5, 3.0])
    plain, _ = _build_effect_block(IID("u", index="district"), frame)
    weighted, _ = _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)

    assert weighted.name == plain.name == "u"
    assert weighted.labels == plain.labels
    assert np.allclose(weighted.precision.toarray(), plain.precision.toarray())
    assert np.allclose(weighted.constraints, plain.constraints)


def test_all_ones_weights_reduce_to_the_unweighted_block():
    frame = _frame([1.0, 1.0, 1.0, 1.0])
    plain, _ = _build_effect_block(IID("u", index="district"), frame)
    weighted, _ = _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)
    assert (weighted.design != plain.design).nnz == 0


def test_inner_hyperparameters_are_still_discovered_through_the_wrapper():
    tau = Hyperparameter("tau", initial=1.0)
    wrapped = Weighted(IID("u", index="district", precision=tau), by="z")
    assert [hp.name for hp in _effect_hyperparameters(wrapped)] == ["tau"]


def test_missing_weight_column_is_rejected_naming_the_effect_and_column():
    frame = _frame([1.0, 1.0, 1.0, 1.0]).drop(columns=["z"])
    with pytest.raises((CompilationError, DataContractError), match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_non_numeric_weight_column_is_rejected():
    frame = _frame(["a", "b", "c", "d"])
    with pytest.raises((CompilationError, DataContractError), match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_nan_weight_is_rejected():
    frame = _frame([1.0, np.nan, 1.0, 1.0])
    with pytest.raises((CompilationError, DataContractError), match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_all_zero_weights_are_rejected_rather_than_compiling_an_inert_block():
    frame = _frame([0.0, 0.0, 0.0, 0.0])
    with pytest.raises((CompilationError, DataContractError), match="zero"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_compile.py -q`
Expected: FAIL — `_build_effect_block` raises `CompilationError: unsupported effect type: Weighted`

- [ ] **Step 3: Implement the wrapper case**

In `src/pylgm/compiler.py`, add a helper above `_build_effect_block`:

```python
def _weight_vector(frame, effect) -> np.ndarray:
    """The validated weight column for a Weighted effect.

    Rejected rather than tolerated: a missing or non-numeric column is a data
    contract error, and an all-zero column makes the effect contribute nothing
    while still consuming latent dimensions, which fits happily and reports a
    field the data never informed.
    """
    if effect.by not in frame.columns:
        raise DataContractError(
            f"weight column {effect.by!r} for effect {effect.name!r} not found"
        )
    column = frame[effect.by]
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError(
            f"weight column {effect.by!r} for effect {effect.name!r} must be "
            "numeric and finite"
        )
    if not np.any(values):
        raise CompilationError(
            f"weight column {effect.by!r} for effect {effect.name!r} is all zero, "
            "so the effect cannot contribute to the predictor while still "
            "consuming latent dimensions"
        )
    return values
```

Then, as the **first** branch inside `_build_effect_block`'s `try:` (before the `Fixed` case):

```python
        if isinstance(effect, Weighted):
            block, precision = _build_effect_block(effect.effect, frame)
            weights = _weight_vector(frame, effect)
            return (
                LatentBlock(
                    block.name,
                    block.labels,
                    csr_matrix(diags(weights) @ block.design),
                    block.precision,
                    block.constraints,
                ),
                precision,
            )
```

And in `_effect_hyperparameters`, as its first lines:

```python
    if isinstance(effect, Weighted):
        # Delegate: an unfound inner Hyperparameter would silently pin at its
        # initial value instead of being estimated.
        return _effect_hyperparameters(effect.effect)
```

Add `Weighted` to the `from pylgm.effects import (...)` block at the top of `compiler.py`. `diags` and `csr_matrix` are already imported (line 8); `pd` is already imported.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_compile.py tests/test_weighted_spec.py -q`
Expected: PASS, 16 passed

- [ ] **Step 5: Run the full suite — this touches shared compile paths**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, no new failures. Then `git restore docs/img/` — the example tests regenerate figures that must not be committed.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py tests/test_weighted_compile.py
git commit -m "feat(compiler): compile Weighted as diag(w) on the inner design

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `Weighted` in the hyperparameter family

**Files:**
- Modify: `src/pylgm/compiler.py` — `compile_family` (~line 967)
- Test: `tests/test_weighted_compile.py` (append)

**Interfaces:**
- Consumes: Task 2's `_weight_vector`; `ScalableBlock`, `ParametricBlock`, `ParametricDesignBlock` from `pylgm.ir.family`.
- Produces: no new public names.

**Why this is a separate task:** `compile_family` has its **own** effect chain, independent of `_build_effect_block`. Task 2 makes a `Weighted` model *compile* on the fixed-hyperparameter path; a model with an **estimated** hyperparameter instead goes through `compile_family` and falls through that chain until this task. That is why the end-to-end fit test lives here and not in Task 2 — before this task it cannot pass, and a test failing for a reason its own task cannot fix breaks the TDD cycle.

**The `ParametricDesignBlock` subtlety:** a `ScalableBlock` or `ParametricBlock` varies only its precision, which weighting does not touch, so weighting the template design suffices. A `ParametricDesignBlock` rebuilds its *design* on every hyperparameter draw, so the weights must be applied to each rebuild. This mirrors `_restack_family_block` in the joint slice, which had exactly this split.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_weighted_compile.py`:

```python
def test_weighted_model_fits_and_estimates_the_inner_hyperparameter():
    rng = np.random.default_rng(3)
    n = 60
    district = [f"d{i % 12}" for i in range(n)]
    z = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({
        "district": district, "z": z,
        "y": rng.poisson(np.exp(0.2 + 0.3 * z)).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="district", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    )
    result = model.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert result.hyperparameters["tau"] > 0


def test_weighted_family_scales_every_rebuilt_design():
    """A ParametricDesignBlock rebuilds its design per draw; weights must apply
    to each rebuild, not only to the template built at the initial value."""
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.ir.family import ParametricDesignBlock

    rng = np.random.default_rng(5)
    n = 40
    frame = pd.DataFrame({
        "district": [f"d{i % 8}" for i in range(n)],
        "z": rng.normal(1.0, 0.2, n),
        "y": rng.poisson(2.0, n).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="district", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    family = compile_family(model, panel)
    assert family is not None
    assert "tau" in family.parameter_names

    # The weighted block's design must carry the weights, whichever block kind
    # it came out as.
    weighted_blocks = [b for b in family.blocks if b.block.name == "u"]
    assert len(weighted_blocks) == 1
    item = weighted_blocks[0]
    design = (
        item.build({"tau": 1.0}) if isinstance(item, ParametricDesignBlock)
        else item.block.design
    )
    row_sums = np.asarray(design.sum(axis=1)).ravel()
    assert np.allclose(row_sums, frame["z"].to_numpy())


def test_weighted_model_with_estimated_hyperparameter_matches_a_manual_weighting():
    """Weighting a column is the same as pre-multiplying it into a one-hot design.

    Fitting Weighted(IID(...), by=z) must equal fitting the same model where the
    weighting was done by hand, which is the property that makes the wrapper
    trustworthy rather than merely functional.
    """
    rng = np.random.default_rng(11)
    n = 50
    frame = pd.DataFrame({
        "district": [f"d{i % 10}" for i in range(n)],
        "z": rng.normal(1.0, 0.3, n),
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    wrapped = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(IID("u", index="district", precision=2.0), by="z"),
    ).fit(frame, engine="laplace")

    from pylgm.compiler import _build_effect_block
    plain, _ = _build_effect_block(IID("u", index="district", precision=2.0), frame)
    manual, _ = _build_effect_block(
        Weighted(IID("u", index="district", precision=2.0), by="z"), frame
    )
    assert np.allclose(
        manual.design.toarray(),
        np.diag(frame["z"].to_numpy()) @ plain.design.toarray(),
    )
    assert np.isfinite(wrapped.log_marginal_likelihood)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_compile.py -q -k "family or manual"`
Expected: FAIL — `compile_family` does not recognise `Weighted`

- [ ] **Step 3: Implement the family case**

In `src/pylgm/compiler.py`, add above `compile_family`:

```python
def _weighted_family_block(item, weights: np.ndarray):
    """Apply a Weighted effect's weights to one family block.

    ScalableBlock and ParametricBlock vary only their precision, which weighting
    leaves alone, so scaling the template design is enough.
    ParametricDesignBlock rebuilds its design per hyperparameter draw, so its
    build output must be scaled on every rebuild too.
    """
    inner = item.block
    scaled = LatentBlock(
        inner.name,
        inner.labels,
        csr_matrix(diags(weights) @ inner.design),
        inner.precision,
        inner.constraints,
    )
    if isinstance(item, ParametricDesignBlock):
        def build(values, inner_build=item.build, weights=weights):
            return csr_matrix(diags(weights) @ inner_build(values))

        return ParametricDesignBlock(scaled, item.parameters, build)
    if isinstance(item, ParametricBlock):
        return ParametricBlock(scaled, item.parameters, item.build)
    return ScalableBlock(scaled, item.parameter, item.scale)
```

**First, extract the loop body.** `compile_family`'s effect loop runs from
`compiler.py:967` to roughly `:1398` — about 430 lines of `if/elif` that append
to `scalable` and register hyperparameters. Move that body verbatim into a
module-level function, changing nothing but the indentation and the accumulator
access:

```python
def _append_family_blocks(
    effect,
    frame,
    scalable: list,
    parameter_names: list[str],
    parameter_bounds: dict,
    parameter_priors: dict,
) -> None:
    """One effect's contribution to a CompiledFamily.

    Extracted verbatim out of compile_family's loop so the Weighted branch can
    build the inner effect through the same path instead of duplicating it. The
    accumulators are mutated in place, exactly as the inline body did.
    """
    # <the former loop body, verbatim; every `continue` becomes `return`>
```

`compile_family`'s loop then becomes:

```python
    for effect in model.predictor.effects:
        _append_family_blocks(
            effect, frame, scalable, parameter_names, parameter_bounds, parameter_priors
        )
```

Every `continue` in the old body becomes `return`. Nothing else changes.

**Then add the `Weighted` branch** as the first lines of `_append_family_blocks`:

```python
    if isinstance(effect, Weighted):
        weights = _weight_vector(frame, effect)
        first = len(scalable)
        _append_family_blocks(
            effect.effect, frame, scalable, parameter_names, parameter_bounds,
            parameter_priors,
        )
        for position in range(first, len(scalable)):
            scalable[position] = _weighted_family_block(scalable[position], weights)
        return
```

Recursing into the inner effect and then weighting whatever blocks it appended
means the wrapper never needs to know which block kind the inner effect
produces, and inner hyperparameter registration happens exactly as it would
unwrapped.

- [ ] **Step 4: Prove the extraction is behaviour-preserving**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS, unchanged counts. Then `git restore docs/img/`.

- [ ] **Step 5: Run the new tests**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_compile.py -q`
Expected: PASS, 19 passed (8 from Task 2 plus 3 here)

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py tests/test_weighted_compile.py
git commit -m "feat(compiler): Weighted in the hyperparameter family path

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Prediction through a `Weighted` effect

**Files:**
- Modify: `src/pylgm/compiler.py` — `_prediction_entry` (~line 1447)
- Modify: `src/pylgm/inference/prediction.py` — `_design_for` dispatch (~line 265)
- Test: `tests/test_weighted_predict.py`

**Interfaces:**
- Consumes: `_prediction_entry(effect, model, panel, block)`; `_design_for(context, new_data)`.
- Produces: a new entry kind `("weighted", (inner_entry, by_column))`, handled recursively so later slices' wrappers nest through the same rule.

- [ ] **Step 1: Write the failing test**

Create `tests/test_weighted_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted


def _fitted():
    rng = np.random.default_rng(17)
    n = 60
    frame = pd.DataFrame({
        "district": [f"d{i % 12}" for i in range(n)],
        "z": rng.normal(1.0, 0.4, n),
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(IID("u", index="district", precision=2.0), by="z"),
    )
    return frame, model.fit(frame, engine="laplace")


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means():
    frame, result = _fitted()
    prediction = result.predict(frame)
    assert prediction.predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-6, abs=1e-8
    )


def test_prediction_uses_new_data_weights_not_the_fitted_ones():
    """The weights are data, not fitted state: doubling z on new rows must
    change the linear predictor. If predict rebuilt an unweighted design, or
    reused the fit-time weights, this would not move."""
    frame, result = _fitted()
    doubled = frame.assign(z=frame["z"] * 2.0)
    base = result.predict(frame).predictive_mean
    scaled = result.predict(doubled).predictive_mean
    assert not np.allclose(base, scaled)


def test_predict_rejects_new_data_missing_the_weight_column():
    frame, result = _fitted()
    with pytest.raises(ValueError, match="z"):
        result.predict(frame.drop(columns=["z"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_predict.py -q`
Expected: FAIL — `build_prediction_context` produces the inner entry with no weighting, so the round-trip means disagree

- [ ] **Step 3: Emit the nested entry**

In `src/pylgm/compiler.py`, as the first branch of `_prediction_entry`:

```python
    if isinstance(effect, Weighted):
        # Nest rather than special-case: later modifier wrappers reuse this same
        # rule instead of adding a case per combination.
        return ("weighted", (_prediction_entry(effect.effect, model, panel, block), effect.by))
```

- [ ] **Step 4: Handle it in `_design_for`**

In `src/pylgm/inference/prediction.py`, add above `_design_for`:

```python
def _weighted_block(entry, new_data: pd.DataFrame, build) -> np.ndarray:
    """Scale a nested entry's rebuilt design by a weight column from new_data.

    ``build`` is ``_design_block_for``, passed in so this stays a plain function
    and the recursion has one owner.
    """
    inner_entry, by_column = entry
    if by_column not in new_data.columns:
        raise ValueError(
            f"predict() new_data is missing the weight column {by_column!r}"
        )
    weights = pd.to_numeric(new_data[by_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError(
            f"predict() weight column {by_column!r} must be numeric and finite"
        )
    return weights[:, None] * build(inner_entry, new_data)
```

Refactor `_design_for`'s per-entry dispatch into a module-level
`_design_block_for(entry, new_data)` taking one `(kind, payload)` and returning
its dense block, with `_design_for` calling it in a loop. Add the recursive case:

```python
        elif kind == "weighted":
            blocks.append(_weighted_block(payload, new_data, _design_block_for))
```

The extraction must be behaviour-preserving; the existing prediction suite proves it.

- [ ] **Step 5: Run the prediction suites**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_predict.py tests/test_predict.py tests/inference/test_prediction.py tests/test_spacetime_predict.py tests/test_midas_parametric_predict.py tests/test_joint_predict.py -q`
Expected: PASS — the pre-existing suites unchanged, proving the `_design_block_for` extraction preserved behaviour

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/inference/prediction.py tests/test_weighted_predict.py
git commit -m "feat(prediction): recursive weighted entry so wrappers nest

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Spatially-varying coefficient end to end, plus docs

**Files:**
- Create: `tests/test_weighted_svc.py`
- Modify: `docs/effects.md`, `docs/research-status.md`, `mkdocs.yml` if a new page is added

**Interfaces:**
- Consumes: everything above.
- Produces: the recovery evidence that this slice delivered its stated unlock.

**Why this task exists:** Tasks 1-4 prove the mechanism is internally consistent. This proves it fits the model it was built for. A `Weighted` effect that compiles and predicts but cannot recover a known spatially-varying coefficient would satisfy every earlier test and still be useless.

- [ ] **Step 1: Write the recovery test**

Create `tests/test_weighted_svc.py`:

```python
"""Spatially-varying coefficient: the model this slice exists to enable.

    log mu_i = alpha + z_i * u_{s(i)},    u ~ IID(tau)

The effect of covariate z varies by region. Recovering the simulated u from
data is what proves the weighting enters the predictor the way it is meant to,
rather than merely compiling.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted
from pylgm.parameters import Hyperparameter

N_REGIONS, PER_REGION = 15, 40
SIGMA_U = 0.5


def _simulate(seed=23):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, SIGMA_U, N_REGIONS)
    region = np.repeat(np.arange(N_REGIONS), PER_REGION)
    z = rng.normal(0.0, 1.0, N_REGIONS * PER_REGION)
    eta = 0.5 + z * u[region]
    y = rng.poisson(np.exp(eta)).astype(float)
    return u, pd.DataFrame({
        "region": region, "z": z, "y": y, "row": range(len(y)),
    })


def test_spatially_varying_coefficient_is_recovered():
    u_true, frame = _simulate()
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="region", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    ).fit(frame, engine="laplace")

    fitted = np.array([
        dict(zip(result.labels, result.mean))[f"u:{i}"] for i in range(N_REGIONS)
    ])
    # Correlation is the honest summary: shrinkage biases magnitudes toward zero,
    # so requiring the values themselves to match would be requiring the prior
    # not to work.
    assert np.corrcoef(fitted, u_true)[0, 1] > 0.8
    assert result.hyperparameters["tau"] > 0


def test_a_constant_weight_column_collapses_to_an_ordinary_iid():
    """With z constant, z*u_s is just a rescaled ordinary IID, so the weighted
    fit must match the unweighted one on the same data up to that scale."""
    _, frame = _simulate()
    frame = frame.assign(z=1.0)
    weighted = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(IID("u", index="region", precision=2.0), by="z"),
    ).fit(frame, engine="laplace")
    plain = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="region", precision=2.0),
    ).fit(frame, engine="laplace")

    assert weighted.log_marginal_likelihood == pytest.approx(
        plain.log_marginal_likelihood, rel=1e-9
    )
    assert weighted.mean == pytest.approx(plain.mean, rel=1e-7, abs=1e-9)
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src python -m pytest tests/test_weighted_svc.py -q`
Expected: PASS, 2 passed

If the recovery correlation fails, do **not** lower the threshold. Report the observed correlation and investigate — a weighting that compiles but does not recover a known signal is the failure this task exists to catch.

- [ ] **Step 3: Document it**

Add a `### Weighted effects` section to `docs/effects.md`, after the effects table and before the MIDAS section, covering: what `Weighted(effect, by)` does (`diag(by) A`); the spatially-varying coefficient it enables, with the model written out; that precision, labels and constraints are the inner effect's, so `latent_marginals("u")` is unchanged by wrapping; that `by` must be numeric, finite and not all-zero; and that `Fixed` cannot be wrapped because it has no index. Add `Weighted` to the effects table at the top of that page.

In `docs/research-status.md`, add `Weighted` to the joint-models entry's sibling list as a second research-grade feature with its own short verified/not-verified summary: verified by design-equality against a manual weighting, all-ones reduction, and SVC recovery on simulated data; not verified against published results on real data.

- [ ] **Step 4: Run the full suite and ruff**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS. Then `git restore docs/img/`.

Run: `ruff check src tests`
Expected: no findings

- [ ] **Step 5: Commit**

```bash
git add tests/test_weighted_svc.py docs/effects.md docs/research-status.md
git commit -m "test(effects): spatially-varying coefficient recovery for Weighted

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Notes for the implementer

**Three extractions, three proofs.** Tasks 3 and 4 each extract a loop body into a helper before adding the new case (`_append_family_blocks`, `_design_block_for`). In both, run the full suite on the extraction **alone**, before the new branch, so a regression is attributable. This is the ordering the joint-models plan used for `_build_effect_block`, and it caught a real defect there.

**The silent failure mode for this slice** is a dispatch site that does not know about `Weighted` and falls through to the plain path — the effect compiles unweighted, the model fits, and every number is wrong with no error. The four functions in the table at the top are the complete set as of `cf6e76d`; if you find a fifth, that is a finding worth reporting, not a quiet fix.

**Do not add `weights` as a field on effect specs.** The spec records that shorthand as deliberately deferred; adding it here would create two mechanisms for one thing before the wrapper has been used in anger.

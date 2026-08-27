# Knorr-Held Space-Time Interaction (S8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SpaceTime` interaction effect (Knorr-Held Types I–IV) that builds the `δ(s,t)` block of a space-time model as one `LatentBlock` with precision `τ·(K_s ⊗ K_t)` and null-space-derived identifiability constraints.

**Architecture:** A new frozen spec `SpaceTime` and a builder `build_spacetime` that assembles an area-major `S·T` one-hot design, a Kronecker precision from a Sørbye–Rue-scaled spatial factor (`besag._scaled_structure`) and temporal factor (RW1/RW2 `DᵀD`), and constraint rows read off the Kronecker null space by SVD. The block is a `ScalableBlock` (precision scales linearly in the single `τ`), wired through both compiler branches exactly like `Besag`. No new inference code — constraints flow through the existing null-space/kriging machinery. A shared `sorbye_rue_scale` helper is extracted from `besag` first (behaviour-preserving).

**Tech Stack:** numpy, scipy.sparse (`kron`, `identity`, `csr_matrix`), scipy.sparse.csgraph (`connected_components`), pandas. No new runtime dependency.

**Spec:** `docs/design/specs/2026-08-26-pylgm-spacetime-knorr-held-design.md`

## Global Constraints

- No new runtime dependency; deterministic-approximation-first (no MCMC).
- Every new effect returns a `LatentBlock` and wires through **both** `compiler.compile_lgm` (fixed-hyperparam) and `compiler.compile_family` (optimisable) branches, plus the `effects/__init__.py` + `pylgm/__init__.py` exports and the `EffectSpec` union / `Predictor` isinstance tuples.
- Existing effects, engines, latent strategies, priors, and the optimise/integrate paths stay behavior-compatible; full suite green; `ruff check src tests` clean.
- Git identity must be **Ardea00** only. Never commit as the iongroup / AndreaPanozzo identity.
- Ponytail: reuse before building, shortest correct diff, mark deliberate simplifications with a `ponytail:` comment.

---

### Task 1: Extract `sorbye_rue_scale` shared helper (behaviour-preserving)

**Files:**
- Create: `src/pylgm/effects/scaling.py`
- Modify: `src/pylgm/effects/besag.py:30-47` (replace the inline per-component scaling with a call to the helper)
- Test: `tests/test_scaling.py`

**Interfaces:**
- Produces: `sorbye_rue_scale(structure: np.ndarray, null_dim: int) -> np.ndarray` — returns `structure` rescaled so the geometric mean of its generalized marginal variances (pseudo-inverse diagonal, dropping the `null_dim` smallest eigenvalues) equals 1. Raises `ValueError` on a non-positive/non-finite marginal variance.
- Consumes (by later tasks): `besag._scaled_structure` signature is unchanged; `bym2` keeps working via it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaling.py
import numpy as np

from pylgm.effects.scaling import sorbye_rue_scale


def _marginal_variances(structure, null_dim):
    eigenvalues, eigenvectors = np.linalg.eigh(structure)
    inverse = np.zeros_like(eigenvalues)
    inverse[null_dim:] = 1.0 / eigenvalues[null_dim:]
    return np.einsum("ij,j,ij->i", eigenvectors, inverse, eigenvectors)


def test_sorbye_rue_scale_sets_geometric_mean_variance_to_one():
    # RW1 structure DtD on 6 levels: null_dim = 1 (the constant).
    n = 6
    d = np.zeros((n - 1, n))
    for i in range(n - 1):
        d[i, i] = -1.0
        d[i, i + 1] = 1.0
    r = d.T @ d
    scaled = sorbye_rue_scale(r, null_dim=1)
    variances = _marginal_variances(scaled, null_dim=1)
    assert np.isclose(np.exp(np.mean(np.log(variances))), 1.0)


def test_sorbye_rue_scale_rejects_degenerate_structure():
    import pytest

    # All-zero structure has no positive eigenvalues -> non-finite variances.
    with pytest.raises(ValueError):
        sorbye_rue_scale(np.zeros((4, 4)), null_dim=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scaling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm.effects.scaling'`

- [ ] **Step 3: Write the helper**

```python
# src/pylgm/effects/scaling.py
"""Sørbye–Rue variance scaling shared by intrinsic GMRF structures."""

import numpy as np


def sorbye_rue_scale(structure: np.ndarray, null_dim: int) -> np.ndarray:
    """Rescale an intrinsic structure to geometric-mean marginal variance 1.

    ``structure`` is a symmetric PSD matrix with ``null_dim`` zero eigenvalues
    (RW1: 1, RW2: 2, one connected ICAR component: 1). The generalized marginal
    variances are the diagonal of the pseudo-inverse formed by dropping those
    ``null_dim`` smallest eigenvalues explicitly -- ``pinv``'s default tolerance
    leaks the near-zero eigenvalue back in as a huge spurious variance once the
    structure grows beyond a few nodes.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(structure)
    inverse = np.zeros_like(eigenvalues)
    inverse[null_dim:] = 1.0 / eigenvalues[null_dim:]
    variances = np.einsum("ij,j,ij->i", eigenvectors, inverse, eigenvectors)
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("Sørbye-Rue scaling produced a non-positive marginal variance")
    factor = float(np.exp(np.mean(np.log(variances))))
    return factor * structure
```

- [ ] **Step 4: Refactor `besag._scaled_structure` to call the helper**

Replace `src/pylgm/effects/besag.py:30-47` (the `scaled = r.copy()` loop body through `scaled[np.ix_(...)] = factor * block`) with:

```python
    scaled = r.copy()
    for component in range(n_components):
        index = np.where(membership == component)[0]
        block = r[np.ix_(index, index)]
        # A connected component's ICAR Laplacian has exactly one null eigenvalue
        # (the constant vector); drop it explicitly (see sorbye_rue_scale).
        scaled[np.ix_(index, index)] = sorbye_rue_scale(block, null_dim=1)
    return scaled
```

Add the import at the top of `besag.py` (after the existing `from pylgm.effects.graph ...` line):

```python
from pylgm.effects.scaling import sorbye_rue_scale
```

- [ ] **Step 5: Run scaling + besag + bym2 tests to verify behaviour preserved**

Run: `python -m pytest tests/test_scaling.py tests/test_besag_fit.py tests/test_bym2_fit.py -v`
Expected: PASS (all). If any besag/bym2 test asserts on the old `"ICAR scaling produced..."` message, keep that wording in the helper's `ValueError` instead — but no such assertion exists at time of writing.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/scaling.py src/pylgm/effects/besag.py tests/test_scaling.py
git commit -m "refactor: extract sorbye_rue_scale helper from besag"
```

---

### Task 2: `SpaceTime` spec + validation + wiring into the union/tuples/exports

**Files:**
- Modify: `src/pylgm/effects/spec.py` — append the new dataclass after the `MIDAS` dataclass (currently the last effect dataclass, ending ~`:230`); add `SpaceTime` to the `EffectSpec` union, to both `Predictor` isinstance tuples (in `__post_init__` and `__add__`, each currently ending `... BYM2, MIDAS)`), and to `__all__`.
- Modify: `src/pylgm/effects/__init__.py` (import + `__all__`)
- Modify: `src/pylgm/__init__.py` (import + `__all__`)
- Test: `tests/test_spacetime_spec.py`

Note: MIDAS merged first, so each union/tuple/`__all__` already ends with `MIDAS` — add `SpaceTime` after it (mirror exactly how MIDAS was added).

**Interfaces:**
- Produces: `SpaceTime(name, space, time, graph=None, interaction="IV", order=1, precision=1.0, scale=True)` — a frozen `_ComposableEffect` dataclass. `name` is the block name; `space`/`time` are index columns; `graph` a canonicalized neighbour mapping or `None`; `interaction` one of `"I"|"II"|"III"|"IV"`; `order` in `{1,2}`; `precision` a positive float or `Hyperparameter`; `scale` a bool. `.index` returns `space` (so the shared row-count check and prediction-context loop, which read `effect.name`, keep working; the builder reads `space`/`time` directly).
- Consumes: `canonical_graph` from `pylgm.effects.graph`; `_non_empty_string`, `_positive_precision` from this module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spacetime_spec.py
import pytest

from pylgm import SpaceTime
from pylgm.parameters import Hyperparameter


def test_defaults_and_graph_canonicalization():
    st = SpaceTime("st", space="area", time="t", graph={"a": ["b"], "b": ["a"]})
    assert st.name == "st"
    assert st.space == "area"
    assert st.time == "t"
    assert st.interaction == "IV"
    assert st.order == 1
    assert st.scale is True
    assert dict(st.graph) == {"a": ("b",), "b": ("a",)}


def test_graph_optional_for_types_i_ii():
    st = SpaceTime("st", space="area", time="t", interaction="II")
    assert st.graph is None


def test_hyperparameter_precision_allowed():
    hp = Hyperparameter("st.precision", initial=1.0)
    st = SpaceTime("st", space="area", time="t", interaction="I", precision=hp)
    assert st.precision is hp


@pytest.mark.parametrize("bad", ["V", "iv", "", 1])
def test_rejects_bad_interaction(bad):
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", graph={"a": ["b"], "b": ["a"]}, interaction=bad)


@pytest.mark.parametrize("bad", [0, 3, 1.5])
def test_rejects_bad_order(bad):
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", graph={"a": ["b"], "b": ["a"]}, order=bad)


def test_rejects_missing_graph_for_iii_iv():
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", interaction="III")
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", interaction="IV")


def test_composes_with_plus():
    from pylgm import Fixed

    predictor = Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I")
    assert predictor.effects[-1].name == "st"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spacetime_spec.py -v`
Expected: FAIL with `ImportError: cannot import name 'SpaceTime' from 'pylgm'`

- [ ] **Step 3: Add the dataclass to `spec.py`**

Insert after the `MIDAS` dataclass (the last effect dataclass, just before the `EffectSpec` union line):

```python
@dataclass(frozen=True)
class SpaceTime(_ComposableEffect):
    """A Knorr-Held space-time interaction effect (interaction types I-IV)."""

    name: str
    space: str
    time: str
    graph: Mapping | None = None
    interaction: str = "IV"
    order: int = 1
    precision: float | Hyperparameter = 1.0
    scale: bool = True

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "space", _non_empty_string(self.space, "space"))
        object.__setattr__(self, "time", _non_empty_string(self.time, "time"))
        if self.interaction not in ("I", "II", "III", "IV"):
            raise ValueError("interaction must be one of 'I', 'II', 'III', 'IV'")
        if type(self.order) is not int or self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be a boolean")
        if self.interaction in ("III", "IV") and self.graph is None:
            raise ValueError(
                f"interaction {self.interaction!r} needs a spatial neighbour graph; "
                "pass graph=..."
            )
        if self.graph is not None:
            object.__setattr__(self, "graph", canonical_graph(self.graph))

    @property
    def index(self) -> str:
        return self.space
```

- [ ] **Step 4: Wire the union, both isinstance tuples, and `__all__` in `spec.py`**

The `EffectSpec` union:

```python
EffectSpec: TypeAlias = Fixed | IID | RW1 | RW2 | AR1 | Besag | ProperCAR | BYM2 | MIDAS | SpaceTime
```

Inside `Predictor.__post_init__` and `Predictor.__add__`, replace both `(Fixed, IID, RW1, RW2, AR1, Besag, ProperCAR, BYM2, MIDAS)` tuples with:

```python
(Fixed, IID, RW1, RW2, AR1, Besag, ProperCAR, BYM2, MIDAS, SpaceTime)
```

Add `"SpaceTime"` to `__all__` (keep alphabetical among the effect names):

```python
__all__ = ["AR1", "Besag", "BYM2", "Fixed", "IID", "MIDAS", "Predictor", "ProperCAR", "RW1", "RW2", "SpaceTime"]
```

- [ ] **Step 5: Export from both `__init__.py` files**

`src/pylgm/effects/__init__.py`: extend the `from pylgm.effects.spec import ...` line (`:9`) to include `SpaceTime`, and add `"SpaceTime"` to `__all__`.

`src/pylgm/__init__.py`: extend the `from pylgm.effects import ...` line (`:3`) to include `SpaceTime`, and add `"SpaceTime"` to `__all__`.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_spacetime_spec.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/test_spacetime_spec.py
git commit -m "feat: add SpaceTime spec (Knorr-Held interaction I-IV)"
```

---

### Task 3: `build_spacetime` builder — design, Kronecker precision, null-space constraints

**Files:**
- Create: `src/pylgm/effects/spacetime.py`
- Modify: `src/pylgm/effects/__init__.py` (export `build_spacetime`)
- Test: `tests/test_spacetime_builder.py`

**Interfaces:**
- Consumes: `sorbye_rue_scale` (Task 1); `besag._scaled_structure(w, nodes, scale)`; `random_walk.difference_operator(n, order)` (available on this branch since MIDAS/S1 merged); `normalize_graph`; `ordered_observed_levels`; `LatentBlock`.
- Produces: `build_spacetime(frame, name, space, time, graph, interaction, order, precision, scale) -> LatentBlock`. `graph` is a plain neighbour `dict` (or `None`); `precision` a resolved float. The block's `labels` are area-major cell labels `f"{area}|{t}"` for area-major cell order `area_position*T + time_position`. `precision` matrix is `csr`; `constraints` a dense `(r, S·T)` array.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spacetime_builder.py
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.spacetime import build_spacetime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}  # path a-b-c, connected


def _frame(areas, times):
    rows = [(s, t) for s in areas for t in times]
    return pd.DataFrame({"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": 0.0})


def _dense(block):
    return block.precision.toarray()


def _rw_structure(T, order):
    if order == 1:
        d = np.zeros((T - 1, T))
        for i in range(T - 1):
            d[i, i], d[i, i + 1] = -1.0, 1.0
    else:
        d = np.zeros((T - 2, T))
        for i in range(T - 2):
            d[i, i], d[i, i + 1], d[i, i + 2] = 1.0, -2.0, 1.0
    return d.T @ d


def test_type_i_precision_is_identity_kron():
    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "I", 1, 2.0, True)
    S, T = 3, 4
    assert np.allclose(_dense(block), 2.0 * np.eye(S * T))
    assert block.constraints.shape == (0, S * T)


def test_type_iii_precision_is_scaled_besag_kron_identity():
    from pylgm.effects.besag import _scaled_structure
    from pylgm.effects.graph import normalize_graph

    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "III", 1, 1.0, True)
    nodes, w = normalize_graph(GRAPH)
    R_s = _scaled_structure(w, nodes, True)
    expected = np.kron(R_s, np.eye(4))
    assert np.allclose(_dense(block), expected)


def test_type_ii_marginal_variance_normalised():
    # Type II = I_s (x) R_t; each area gets the scaled temporal structure.
    frame = _frame(["a", "b", "c"], list(range(6)))
    block = build_spacetime(frame, "st", "area", "t", None, "II", 1, 1.0, True)
    S, T = 3, 6
    R_t_block = _rw_structure(T, 1)
    # scaled so geometric-mean generalized variance == 1
    eig, vec = np.linalg.eigh(R_t_block)
    inv = np.zeros_like(eig)
    inv[1:] = 1.0 / eig[1:]
    var = np.einsum("ij,j,ij->i", vec, inv, vec)
    factor = float(np.exp(np.mean(np.log(var))))
    expected = np.kron(np.eye(S), factor * R_t_block)
    assert np.allclose(_dense(block), expected)


@pytest.mark.parametrize(
    "interaction,order,expected_rows",
    [("I", 1, 0), ("II", 1, 3), ("II", 2, 6), ("III", 1, 4), ("IV", 1, 3 + 4 - 1), ("IV", 2, 6 + 4 - 2)],
)
def test_constraint_row_counts(interaction, order, expected_rows):
    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    graph = None if interaction in ("I", "II") else GRAPH
    block = build_spacetime(frame, "st", "area", "t", graph, interaction, order, 1.0, True)
    assert block.constraints.shape[0] == expected_rows


def test_constraints_annihilate_null_space_type_iv():
    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "IV", 1, 1.0, True)
    P = block.precision.toarray()
    # every null vector of P must be killed by the constraint rows
    eig, vec = np.linalg.eigh(P)
    null_vectors = vec[:, eig < 1e-9 * eig.max()]
    assert np.allclose(block.constraints @ null_vectors, 0.0, atol=1e-8)
    # and the constraint rows are full row-rank
    assert np.linalg.matrix_rank(block.constraints) == block.constraints.shape[0]


def test_design_is_area_major_one_hot():
    frame = pd.DataFrame({"area": ["a", "c"], "t": [1, 3], "y": 0.0})
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "I", 1, 1.0, True)
    # areas sorted a,b,c -> positions 0,1,2 ; times 1,3 (observed) -> but time
    # universe is observed levels {1,3}; T=2. cell(a,1)=0*2+0=0; cell(c,3)=2*2+1=5.
    design = block.design.toarray()
    assert design.shape == (2, 3 * 2)
    assert design[0, 0] == 1.0 and design[0].sum() == 1.0
    assert design[1, 5] == 1.0 and design[1].sum() == 1.0


def test_rejects_too_few_time_levels_for_rw():
    frame = _frame(["a", "b", "c"], [0, 1])  # T=2, order=2 needs T>2
    with pytest.raises(ValueError):
        build_spacetime(frame, "st", "area", "t", None, "II", 2, 1.0, True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spacetime_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm.effects.spacetime'`

- [ ] **Step 3: Write the builder**

```python
# src/pylgm/effects/spacetime.py
"""Knorr-Held space-time interaction effect (Types I-IV)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.csgraph import connected_components

from pylgm.data.scalars import ordered_observed_levels
from pylgm.effects.besag import _scaled_structure
from pylgm.effects.graph import normalize_graph
from pylgm.effects.random_walk import difference_operator
from pylgm.effects.scaling import sorbye_rue_scale
from pylgm.ir.model import LatentBlock

_SPACE_STRUCTURED = {"III", "IV"}
_TIME_STRUCTURED = {"II", "IV"}


def _rw_structure(level_count: int, order: int, scale: bool) -> np.ndarray:
    """Sørbye-Rue-scaled temporal RW structure ``DᵀD`` (order 1 or 2)."""
    # Reuse the shared difference operator (MIDAS/S1 exposed it in random_walk).
    difference = difference_operator(level_count, order)
    r = (difference.T @ difference).toarray()
    return sorbye_rue_scale(r, null_dim=order) if scale else r


def _space_null_basis(interaction: str, w, area_count: int) -> np.ndarray:
    """Columns spanning null(K_s): empty for I/II, component indicators for III/IV."""
    if interaction not in _SPACE_STRUCTURED:
        return np.zeros((area_count, 0))
    n_components, membership = connected_components(w, directed=False)
    basis = np.zeros((area_count, n_components))
    for component in range(n_components):
        basis[membership == component, component] = 1.0
    return basis


def _time_null_basis(interaction: str, order: int, time_count: int) -> np.ndarray:
    """Columns spanning null(K_t): empty for I/III, [1] for RW1, [1, k] for RW2."""
    if interaction not in _TIME_STRUCTURED:
        return np.zeros((time_count, 0))
    columns = [np.ones(time_count)]
    if order == 2:
        coordinate = np.arange(time_count, dtype=float)
        columns.append(coordinate - coordinate.mean())
    return np.column_stack(columns)


def _interaction_constraints(
    space_null: np.ndarray, time_null: np.ndarray, area_count: int, time_count: int
) -> np.ndarray:
    """Orthonormal basis of null(K_s (x) K_t), as constraint rows (r, S*T).

    null(K_s (x) K_t) = null(K_s) (x) R^T  +  R^S (x) null(K_t). Assemble the
    two Kronecker spans, take an orthonormal basis of their combined column span
    by SVD (dropping the 1_s (x) 1_t overlap counted twice), and transpose.
    """
    parts = []
    if space_null.shape[1]:
        parts.append(np.kron(space_null, np.eye(time_count)))
    if time_null.shape[1]:
        parts.append(np.kron(np.eye(area_count), time_null))
    if not parts:
        return np.zeros((0, area_count * time_count))
    b = np.hstack(parts)
    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10)) if singular.size else 0
    return u[:, :rank].T


def build_spacetime(
    frame: pd.DataFrame,
    name: str,
    space: str,
    time: str,
    graph: Mapping | None,
    interaction: str,
    order: int,
    precision: float,
    scale: bool = True,
) -> LatentBlock:
    if space not in frame.columns:
        raise ValueError(f"space column {space!r} not found")
    if time not in frame.columns:
        raise ValueError(f"time column {time!r} not found")

    # Area universe: graph nodes when a graph is given, else observed levels.
    if graph is not None:
        areas, w = normalize_graph(dict(graph))
    else:
        areas = tuple(map(str, ordered_observed_levels(frame[space])))
        w = None
    times = tuple(map(str, ordered_observed_levels(frame[time])))
    S, T = len(areas), len(times)
    if interaction in _TIME_STRUCTURED and T <= order:
        raise ValueError(f"{name} type {interaction} requires more than {order} time levels")

    # Kronecker precision factors.
    if interaction in _SPACE_STRUCTURED:
        k_s = csr_matrix(_scaled_structure(w, areas, scale))
    else:
        k_s = identity(S, format="csr")
    if interaction in _TIME_STRUCTURED:
        k_t = csr_matrix(_rw_structure(T, order, scale))
    else:
        k_t = identity(T, format="csr")
    precision_matrix = csr_matrix(precision * kron(k_s, k_t, format="csr"))

    # Area-major one-hot design; unseen area/time levels are a hard error.
    area_pos = {area: i for i, area in enumerate(areas)}
    time_pos = {t: j for j, t in enumerate(times)}
    observed_area = [str(v) for v in frame[space]]
    observed_time = [str(v) for v in frame[time]]
    missing_area = sorted({v for v in observed_area if v not in area_pos})
    if missing_area:
        raise ValueError(f"observed {space!r} level(s) {missing_area!r} not in the area universe")
    cells = np.array(
        [area_pos[a] * T + time_pos[t] for a, t in zip(observed_area, observed_time)]
    )
    design = csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), cells)), shape=(len(frame), S * T)
    )

    space_null = _space_null_basis(interaction, w, S)
    time_null = _time_null_basis(interaction, order, T)
    constraints = _interaction_constraints(space_null, time_null, S, T)

    labels = tuple(f"{area}|{t}" for area in areas for t in times)
    return LatentBlock(name, labels, design, precision_matrix, constraints)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spacetime_builder.py -v`
Expected: PASS (all)

- [ ] **Step 5: Export the builder**

`src/pylgm/effects/__init__.py`: add `from pylgm.effects.spacetime import build_spacetime` and `"build_spacetime"` to `__all__`.

- [ ] **Step 6: Run ruff and commit**

```bash
python -m ruff check src tests
git add src/pylgm/effects/spacetime.py src/pylgm/effects/__init__.py tests/test_spacetime_builder.py
git commit -m "feat: build_spacetime Kronecker precision + null-space constraints"
```

---

### Task 4: Compiler wiring — both branches + sibling warning

**Files:**
- Modify: `src/pylgm/compiler.py` — import `SpaceTime` and `build_spacetime`; add a branch in `compile_lgm` (after the `elif isinstance(effect, MIDAS)` branch, before the `else` that raises `unsupported effect type`); add a branch in `compile_family` (after the `MIDAS` branch, i.e. after its `ScalableBlock`/`ParametricBlock` appends `continue`); add the sibling warning after the effect loop in `compile_lgm` (after `blocks.append(block)`, before the `wrong_rows` check).
- Test: `tests/test_spacetime_compile.py`

Note: MIDAS merged into this branch, so `compiler.py` already imports `MIDAS`/`build_midas` and has MIDAS branches in both functions — mirror the MIDAS branch placement, and extend the existing effects-import lines to also bring in `SpaceTime` and `build_spacetime`.

**Interfaces:**
- Consumes: `build_spacetime` (Task 3); `SpaceTime` spec (Task 2); existing `_resolved_precision`, `_compiled_block`, `ScalableBlock`, `_log_bounds`.
- Produces: a compiled `SpaceTime` block flows through both fixed (`compile_lgm`) and optimisable (`compile_family`) paths as a `ScalableBlock` keyed on `τ`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spacetime_compile.py
import warnings

import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM, RW1, SpaceTime
from pylgm.priors import PCPrecision

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def _frame():
    rows = [(s, t) for s in ["a", "b", "c"] for t in range(5)]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": rng.normal(size=len(rows))}
    )


def _full_predictor():
    return (
        Fixed("1")
        + Besag("area", index="area", graph=GRAPH)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV", order=1)
    )


def test_type_i_fixed_precision_fits():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=2.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(_frame())
    assert result.latent_marginals("st").mean.shape == (3 * 5,)


def test_type_iv_estimates_tau():
    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    predictor = (
        Fixed("1")
        + Besag("area", index="area", graph=GRAPH)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV", order=1, precision=hp)
    )
    result = LGM(response="y", predictor=predictor, likelihood=Gaussian(sigma=0.5)).fit(_frame())
    assert result.hyperparameters["st.precision"] > 0.0


def test_warns_when_a_main_effect_is_missing():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV"),
        likelihood=Gaussian(sigma=0.5),
    )
    with pytest.warns(UserWarning, match="st"):
        model.fit(_frame())


def test_no_warning_when_both_main_effects_present():
    model = LGM(response="y", predictor=_full_predictor(), likelihood=Gaussian(sigma=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(_frame())  # must not raise a converted warning
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spacetime_compile.py -v`
Expected: FAIL — `CompilationError: unsupported effect type: SpaceTime` (compile_lgm raises at `:272`).

- [ ] **Step 3: Add the `compile_lgm` branch**

Import at the top of `compiler.py` (extend the existing effects/spec imports to include `SpaceTime`, and the builder import to include `build_spacetime`).

Insert after the `elif isinstance(effect, MIDAS)` branch, before the `else:` that raises `unsupported effect type`:

```python
            elif isinstance(effect, SpaceTime):
                precision = _resolved_precision(effect.precision)
                block = build_spacetime(
                    frame, effect.name, effect.space, effect.time,
                    dict(effect.graph) if effect.graph is not None else None,
                    effect.interaction, effect.order, precision, effect.scale,
                )
                precisions[effect.name] = precision
```

- [ ] **Step 4: Add the sibling warning after the effect loop**

Insert after `blocks.append(block)` completes the loop, before the `wrong_rows` check:

```python
    _warn_missing_spacetime_main_effects(model.predictor.effects)
```

And add the helper near the other module-level helpers in `compiler.py`:

```python
def _warn_missing_spacetime_main_effects(effects) -> None:
    """Warn if a SpaceTime interaction lacks its spatial/temporal main effects.

    The interaction's sum-to-zero constraints assume the main effects absorb the
    spatial and temporal marginals; omitting one is a modelling error, so warn
    (do not crash) naming the effect and the missing companion.
    """
    spatial_indices = {e.index for e in effects if isinstance(e, (Besag, ProperCAR, BYM2))}
    temporal_indices = {e.index for e in effects if isinstance(e, (RW1, RW2, AR1))}
    for effect in effects:
        if not isinstance(effect, SpaceTime):
            continue
        missing = []
        if effect.space not in spatial_indices:
            missing.append(f"a spatial main effect on {effect.space!r}")
        if effect.time not in temporal_indices:
            missing.append(f"a temporal main effect on {effect.time!r}")
        if missing:
            warnings.warn(
                f"SpaceTime effect {effect.name!r} has no {' and no '.join(missing)}; "
                "its identifiability constraints assume the main effects absorb the "
                "marginals. Add them, or the variance split is not identified.",
                UserWarning,
                stacklevel=2,
            )
```

Confirm `import warnings` is present at the top of `compiler.py` (add if absent).

- [ ] **Step 5: Add the `compile_family` branch**

Insert after the `MIDAS` branch ends (after its final `append(...)` / `continue`), alongside the other `if isinstance(effect, ...): ... continue` branches:

```python
        if isinstance(effect, SpaceTime):
            block = _compiled_block(
                effect.name, build_spacetime,
                frame, effect.name, effect.space, effect.time,
                dict(effect.graph) if effect.graph is not None else None,
                effect.interaction, effect.order, value, effect.scale,
            )
            scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            continue
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_spacetime_compile.py -v`
Expected: PASS (all four)

- [ ] **Step 7: Run ruff and commit**

```bash
python -m ruff check src tests
git add src/pylgm/compiler.py tests/test_spacetime_compile.py
git commit -m "feat: wire SpaceTime through both compiler branches + sibling warning"
```

---

### Task 5: Prediction support — `_spacetime_block` + context entry

**Files:**
- Modify: `src/pylgm/inference/prediction.py` — add `_spacetime_block` (after `_midas_block`); dispatch it in `_design_for` (add an `elif kind == "spacetime"` alongside the existing `"midas"` branch).
- Modify: `src/pylgm/compiler.py` `build_prediction_context` — emit a `("spacetime", ...)` context entry for `SpaceTime` effects, as a new `elif isinstance(effect, SpaceTime)` beside the existing `elif isinstance(effect, MIDAS)` branch, before the `else`.
- Test: `tests/test_spacetime_predict.py`

Note: MIDAS already added a `_midas_block`, a `"midas"` dispatch branch, and a `("midas", ...)` context entry — mirror that pattern exactly.

**Interfaces:**
- Consumes: the fitted block `labels` (area-major `f"{area}|{t}"` cells from Task 3).
- Produces: a prediction context entry `("spacetime", (name, space, time, area_labels, time_labels))` and a `_spacetime_block(entry, new_data) -> np.ndarray` that maps each `(area, time)` row to its area-major cell one-hot, raising `ValueError` for an unseen area or time level (predict reuses the fitted posterior; it cannot mint a new cell).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spacetime_predict.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, LGM, SpaceTime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def _frame():
    rows = [(s, t) for s in ["a", "b", "c"] for t in range(5)]
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": rng.normal(size=len(rows))}
    )


def test_predict_on_seen_cells_roundtrips():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=1.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(_frame())
    new = pd.DataFrame({"area": ["a", "c"], "t": [0, 4], "y": [0.0, 0.0]})
    pred = result.predict(new)
    assert pred.predictive_mean.shape == (2,)
    assert np.isfinite(pred.predictive_mean).all()


def test_predict_rejects_unseen_area():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=1.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(_frame())
    new = pd.DataFrame({"area": ["z"], "t": [0], "y": [0.0]})
    with pytest.raises(ValueError, match="not in the fitted"):
        result.predict(new)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spacetime_predict.py -v`
Expected: FAIL — the first test errors because `build_prediction_context` emits a `("structured", ...)` entry whose `_structured_block` reads a single `index` column and cannot rebuild the two-column area-major cell design (width mismatch or wrong cells).

- [ ] **Step 3: Emit the `spacetime` context entry**

In `src/pylgm/compiler.py`, `build_prediction_context`, split the final `else:` (the one that appends a `("structured", ...)` entry) so `SpaceTime` gets its own entry, mirroring the existing `elif isinstance(effect, MIDAS)` branch that sits just above it:

```python
        elif isinstance(effect, SpaceTime):
            area_labels = tuple(label.split("|", 1)[0] for label in block.labels)
            time_labels = tuple(label.split("|", 1)[1] for label in block.labels)
            entries.append(
                ("spacetime", (effect.name, effect.space, effect.time,
                               tuple(dict.fromkeys(area_labels)),
                               tuple(dict.fromkeys(time_labels))))
            )
        else:
            entries.append(("structured", (effect.name, effect.index, block.labels)))
```

(`dict.fromkeys` deduplicates while preserving the area-major / time order embedded in the cell labels.)

- [ ] **Step 4: Add `_spacetime_block` and dispatch it**

In `src/pylgm/inference/prediction.py`, add after `_midas_block` (the block just before `_design_for`):

```python
def _spacetime_block(
    entry: tuple[str, str, str, tuple[str, ...], tuple[str, ...]], new_data: pd.DataFrame
) -> np.ndarray:
    name, space, time, area_labels, time_labels = entry
    for column in (space, time):
        if column not in new_data.columns:
            raise ValueError(
                f"predict() new_data is missing column {column!r} required by the "
                f"{name!r} space-time block"
            )
    area_pos = {label: i for i, label in enumerate(area_labels)}
    time_pos = {label: j for j, label in enumerate(time_labels)}
    areas = new_data[space].map(str)
    times = new_data[time].map(str)
    unseen = sorted(set(areas[~areas.isin(area_pos)]) | set(times[~times.isin(time_pos)]))
    if unseen:
        raise ValueError(
            f"predict() cannot score rows whose {name!r} space/time level was not in the "
            f"fitted model: {unseen!r}. predict reuses the fitted latent posterior, so it "
            "cannot create a new latent cell. To forecast new cells, include those rows at "
            "fit time with a NaN response instead."
        )
    T = len(time_labels)
    cells = areas.map(area_pos).to_numpy() * T + times.map(time_pos).to_numpy()
    design = np.zeros((len(new_data), len(area_labels) * T))
    design[np.arange(len(new_data)), cells] = 1.0
    return design
```

Add the dispatch in `_design_for`, after the existing `elif kind == "midas"` branch:

```python
        elif kind == "spacetime":
            blocks.append(_spacetime_block(payload, new_data))
```

Update the `PredictionContext.entries` docstring (which already lists the `fixed`/`structured`/`midas` entry shapes) to mention the new `("spacetime", ...)` shape.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_spacetime_predict.py -v`
Expected: PASS (both)

- [ ] **Step 6: Run ruff and commit**

```bash
python -m ruff check src tests
git add src/pylgm/compiler.py src/pylgm/inference/prediction.py tests/test_spacetime_predict.py
git commit -m "feat: predict() support for SpaceTime cells"
```

---

### Task 6: End-to-end latent-strategy contract + statistical recovery

**Files:**
- Test: `tests/test_spacetime_effect.py`

**Interfaces:**
- Consumes: the full `SpaceTime` effect through fit/integrate/predict (Tasks 2–5).
- Produces: no source change — this task pins the modelling contract (Type I proper under full Laplace; II–IV rejected) and one statistical-recovery test.

- [ ] **Step 1: Write the latent-strategy contract tests**

```python
# tests/test_spacetime_effect.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM, RW1, SpaceTime
from pylgm.exceptions import UnsupportedEngineError
from pylgm.priors import PCPrecision

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def _frame(seed=0):
    rows = [(s, t) for s in ["a", "b", "c"] for t in range(6)]
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": rng.normal(size=len(rows))}
    )


def test_type_i_accepted_under_full_laplace():
    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=hp),
        likelihood=Gaussian(sigma=0.5),
    )
    # Type I is proper/unconstrained -> full Laplace must not reject it.
    result = model.fit(_frame(), hyperparameters="integrate", latent_strategy="laplace")
    assert result.latent_marginals("st").mean.shape == (3 * 6,)


@pytest.mark.parametrize("interaction,order", [("II", 1), ("III", 1), ("IV", 1)])
def test_types_ii_iii_iv_rejected_under_full_laplace(interaction, order):
    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    graph = None if interaction == "II" else GRAPH
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + SpaceTime("st", space="area", time="t", graph=graph, interaction=interaction, order=order, precision=hp),
        likelihood=Gaussian(sigma=0.5),
    )
    with pytest.raises(UnsupportedEngineError):
        model.fit(_frame(), hyperparameters="integrate", latent_strategy="laplace")
```

- [ ] **Step 2: Run to verify pass (contract already holds via existing machinery)**

Run: `python -m pytest tests/test_spacetime_effect.py -v`
Expected: PASS. Type I carries zero constraints so full Laplace accepts it; II–IV carry constraints and are rejected by the existing constrained-effect guard (same path as `test_full_laplace_rejects_besag`). If Type I is instead rejected, the constraint count for Type I is wrong — return to Task 3.

- [ ] **Step 3: Add the statistical-recovery test**

```python
def test_type_iv_recovers_inseparable_field():
    # Simulate a smooth inseparable space-time field on the a-b-c path x 8 periods,
    # observe it with light noise, and check Type IV recovers it.
    areas, times = ["a", "b", "c"], list(range(8))
    rng = np.random.default_rng(7)
    coords = {"a": 0.0, "b": 1.0, "c": 2.0}
    rows, truth = [], []
    for s in areas:
        for t in times:
            # a travelling smooth bump: peak position moves with time
            value = np.exp(-0.6 * (coords[s] - 0.25 * t) ** 2)
            rows.append((s, t, value + rng.normal(scale=0.05)))
            truth.append(value)
    frame = pd.DataFrame(rows, columns=["area", "t", "y"])
    truth = np.array(truth)

    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    predictor = (
        Fixed("1")
        + Besag("area", index="area", graph=GRAPH)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV", order=1, precision=hp)
    )
    result = LGM(response="y", predictor=predictor, likelihood=Gaussian(sigma=0.05)).fit(frame)

    # The fitted linear predictor (intercept + area + period + interaction) should
    # track the truth. The point is a sane fit, not exact recovery.
    eta = result.predict(frame).predictive_mean
    centred_truth = truth - truth.mean()
    centred_eta = eta - eta.mean()
    assert np.corrcoef(centred_eta, centred_truth)[0, 1] > 0.9
    assert result.hyperparameters["st.precision"] > 0.0
```

- [ ] **Step 4: Run the recovery test**

Run: `python -m pytest tests/test_spacetime_effect.py::test_type_iv_recovers_inseparable_field -v`
Expected: PASS. If the correlation is below 0.9, first confirm `sigma=0.05` matches the simulated noise and the grid is large enough; the assertion is a sanity floor, not a tight bound — loosen to `> 0.85` only if the fit is visibly sane but the small grid caps correlation.

- [ ] **Step 5: Commit**

```bash
git add tests/test_spacetime_effect.py
git commit -m "test: SpaceTime latent-strategy contract + Type IV recovery"
```

---

### Task 7: Documentation — effects guide + roadmap

**Files:**
- Modify: `docs/effects.md` (add a `## SpaceTime` section before `## Linear constraints`)
- Modify: `docs/roadmap.md:12-13` (Shipped Effects line) and the `## Next` list

- [ ] **Step 1: Add the effects-guide section**

Insert into `docs/effects.md` before the `## Linear constraints (\`extraconstr\`)` heading (`:98`):

````markdown
## SpaceTime effect (Knorr-Held interaction)

`SpaceTime(name, space, time, graph=None, interaction="IV", order=1, precision=1.0, scale=True)`
builds the **interaction** term `δ(s,t)` of a Knorr-Held space-time model — one
latent block over the `S·T` area×period cells with precision `τ·(K_s ⊗ K_t)`.
The spatial (`Besag`) and temporal (`RW1`/`RW2`) main effects are composed with
`+`; `SpaceTime` supplies only the interaction, keeping each term its own
variance component.

| `interaction` | `K_s` | `K_t` | Reading |
|------|-------|-------|---------|
| `"I"`   | `I_s` | `I_t` | unstructured cell-wise interaction (proper) |
| `"II"`  | `I_s` | `R_t` | each area its own independent temporal trend |
| `"III"` | `R_s` | `I_t` | each period its own independent spatial pattern |
| `"IV"`  | `R_s` | `R_t` | inseparable: neighbours in *both* space and time tied |

`R_s` is the (weighted) Sørbye–Rue-scaled Besag Laplacian; `R_t` the scaled RW
structure of the given `order`. `graph` is **required for III/IV** and optional
for I/II (the area universe is then read from the observed `space` column).
`precision` is a single `τ` — a float (plug-in) or a `Hyperparameter`
(estimated by EB, integrated by INLA).

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

Type I is proper and carries no constraint, so it runs under all latent
strategies including `latent_strategy="laplace"`. Types II–IV carry the
Knorr-Held / Schrödle–Held sum-to-zero constraints (derived from the Kronecker
null space), so — like `RW2` and `Besag` — they require
`hyperparameters="integrate"` and are rejected under full Laplace. The compiler
emits a one-line warning if a `SpaceTime` effect is present without its spatial
and temporal main effects, since the constraints assume those absorb the
marginals. The latent field is dense `S·T`, so the >4096-dim preflight guard
bites at large grids — real economic scale is gated on the sparse backend.
````

- [ ] **Step 2: Update the roadmap**

In `docs/roadmap.md`, extend the Shipped `**Effects**` line (`:12-13`) to mention the space-time interaction:

```markdown
- **Effects** — `Fixed`, `IID`, `RW1`/`RW2`, stationary `AR1`, and the
  Knorr-Held `SpaceTime` interaction (Types I–IV). See [effects](effects.md).
```

Remove any now-shipped space-time bullet from `## Next` if one exists; otherwise leave the section unchanged.

- [ ] **Step 3: Commit**

```bash
git add docs/effects.md docs/roadmap.md
git commit -m "docs: SpaceTime effect guide + roadmap"
```

---

### Task 8: Full-suite green + ruff

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS (entire suite, no regressions).

- [ ] **Step 2: Run ruff**

Run: `python -m ruff check src tests`
Expected: clean.

- [ ] **Step 3: Final commit if anything was touched**

```bash
git status
# commit only if a fix was needed to reach green
```

After Task 8, invoke the **superpowers:finishing-a-development-branch** skill to verify tests, present completion options, and execute the chosen workflow.

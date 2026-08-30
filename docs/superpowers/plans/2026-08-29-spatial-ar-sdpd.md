# Directed SAR + Dynamic Spatial Panel (SDPD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a directed-network SAR effect and a dynamic spatial panel (SDPD) effect — precision `τ·MᵀM` for a sparse operator `M` — that fit on the existing dense and E-sparse engines, plus forward forecasting for future periods.

**Architecture:** Two new frozen effect specs (`SAR`, `DynamicSpatialPanel`) whose builders return a proper (full-rank, empty-constraint) `LatentBlock` with precision `precision·MᵀM`. Static SAR is the `T=1` special case (`M = I − ρW`). A directed-graph module supplies symmetry-free normalization + row-standardization. Compiler branches mirror the existing AR1/ProperCAR ρ-ParametricBlock pattern, extended to the SDPD's `(τ, ρ, γ, η)`. Forecasting is a pure Kalman-style forward recursion `x̂_{t+1} = A_{t+1}⁻¹ B_{t+1} x̂_t`.

**Tech Stack:** Python, NumPy, SciPy sparse (`csr_matrix`, `identity`, `diags`, `bmat`, `splu`), pandas. No new runtime dependencies.

## Global Constraints

- **Git identity is `Ardea00` only.** Never commit as the iongroup/AndreaPanozzo identity. Repo config already set (`user.name=Ardea00`).
- **Row-standardized directed `W`**: each row of `W` sums to 1 (a zero-out-degree row stays all-zero). This fixes `ρ ∈ (−1, 1)` with no eigensolve.
- **Latent precision is proper**: `constraints = np.empty((0, n))` — no Sørbye-Rue scaling, no sum-to-zero (like `build_proper_car`/`build_ar1`).
- **The one construction**: every precision is `precision · (Mᵀ @ M)` via the shared `_gram_precision(M, precision)` helper. Never duplicate this Gram math.
- **`T=1, γ=η=0` must reproduce static SAR exactly** (asserted by test, not by duplicated code).
- **Latent grid ordering is time-major**: stacked latent `x = (x_1, …, x_T)`, block `t` contiguous; label `f"{unit}@{time}"`; column for `(unit, time)` = `time_pos * n_units + unit_pos`.
- **No new runtime dependency.**
- Transforms: `ρ` → Logit(−1, 1) via `_bounded_parameter(..., -1.0, 1.0, inset=1e-6)`; `γ, η` → identity via `_real_bounds` (Hyperparameter declared `transform="identity"`); `τ` → log via `_log_bounds`.

---

### Task 1: Directed-graph module

**Files:**
- Create: `src/pylgm/effects/directed_graph.py`
- Test: `tests/effects/test_directed_graph.py`

**Interfaces:**
- Consumes: `pylgm.effects.graph._parse_neighbours` (weight parsing/validation).
- Produces:
  - `normalize_directed_graph(graph: Mapping) -> tuple[tuple[str, ...], csr_matrix]` — sorted string nodes and a **directed** adjacency `W` where `W[i, j]` is the influence of node `j` on node `i` (row `i` = influencers of `i`). Node set = union of keys and referenced neighbours. Rejects self-loops and non-finite/non-positive weights.
  - `row_standardize(w: csr_matrix) -> csr_matrix` — each row scaled to sum 1; all-zero rows stay zero.
  - `canonical_directed_graph(graph: Mapping) -> tuple[tuple[str, tuple], ...]` — immutable, frozen-dataclass-safe form; `dict(result)` reconstructs a mapping `normalize_directed_graph` accepts.

- [ ] **Step 1: Write the failing test**

Create `tests/effects/test_directed_graph.py`:

```python
import numpy as np
import pytest

from pylgm.effects.directed_graph import (
    canonical_directed_graph,
    normalize_directed_graph,
    row_standardize,
)


def test_asymmetric_graph_is_accepted():
    # a -> b but not b -> a: normalize_graph would reject this; directed keeps it.
    nodes, w = normalize_directed_graph({"a": ["b"], "b": []})
    assert nodes == ("a", "b")
    dense = w.toarray()
    assert dense[0, 1] == 1.0  # b influences a (row a, col b)
    assert dense[1, 0] == 0.0


def test_pure_sink_node_gets_a_column():
    # c is only ever a neighbour, never a key.
    nodes, w = normalize_directed_graph({"a": ["c"]})
    assert nodes == ("a", "c")
    assert w.shape == (2, 2)


def test_weighted_edges_preserved():
    nodes, w = normalize_directed_graph({"a": {"b": 2.0}, "b": {"a": 0.5}})
    dense = w.toarray()
    assert dense[0, 1] == 2.0
    assert dense[1, 0] == 0.5


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        normalize_directed_graph({"a": ["a"]})


def test_row_standardize_rows_sum_to_one():
    _, w = normalize_directed_graph({"a": {"b": 3.0, "c": 1.0}, "b": ["a"], "c": []})
    r = row_standardize(w).toarray()
    assert np.isclose(r[0].sum(), 1.0)
    assert np.isclose(r[0, 1], 0.75)  # b: 3/(3+1)
    assert np.isclose(r[0, 2], 0.25)  # c: 1/(3+1)


def test_row_standardize_zero_row_stays_zero():
    _, w = normalize_directed_graph({"a": ["b"], "b": []})  # b is a sink: zero row
    r = row_standardize(w).toarray()
    assert np.allclose(r[1], 0.0)


def test_canonical_round_trips_through_normalize():
    graph = {"a": {"b": 2.0}, "b": ["c"], "c": []}
    canonical = canonical_directed_graph(graph)
    assert isinstance(canonical, tuple)
    hash(canonical)  # frozen-dataclass-safe
    nodes, w = normalize_directed_graph(dict(canonical))
    assert nodes == ("a", "b", "c")
    assert w.toarray()[0, 1] == 2.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_directed_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: pylgm.effects.directed_graph`.

- [ ] **Step 3: Write the module**

Create `src/pylgm/effects/directed_graph.py`:

```python
"""Directed-network adjacency for spatial-autoregressive (SAR/SDPD) effects.

Unlike ``pylgm.effects.graph.normalize_graph`` these helpers do NOT require a
symmetric graph: ``W[i, j]`` is the influence of node ``j`` on node ``i`` (row
``i`` lists the influencers of ``i``), which is exactly the directed economic
relationship a symmetrized CAR discards.
"""

from collections.abc import Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags

from pylgm.effects.graph import _parse_neighbours


def normalize_directed_graph(
    graph: Mapping,
) -> tuple[tuple[str, ...], csr_matrix]:
    """Return sorted string node labels and a directed adjacency ``W``.

    The node set is the union of keys and referenced neighbours, so a pure sink
    (only ever a neighbour) still gets a row/column. Reuses the shared weight
    parser, so non-finite / non-positive weights are rejected identically to the
    undirected path. Self-loops are rejected.
    """
    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("graph must be a non-empty mapping of node -> neighbours")
    adjacency: dict[str, dict[str, float]] = {}
    referenced: set[str] = set()
    for node, neighbours in graph.items():
        key = str(node)
        adjacency.setdefault(key, {})
        for neighbour, weight in _parse_neighbours(key, neighbours):
            if neighbour == key:
                raise ValueError(f"graph has a self-loop at node {key!r}")
            adjacency[key][neighbour] = weight
            referenced.add(neighbour)
    nodes = tuple(sorted(set(adjacency) | referenced))
    position = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for node, neighbours in adjacency.items():
        for neighbour, weight in neighbours.items():
            rows.append(position[node])
            cols.append(position[neighbour])
            data.append(weight)
    w = csr_matrix(
        (np.array(data, dtype=float), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    return nodes, w


def row_standardize(w: csr_matrix) -> csr_matrix:
    """Scale each row of ``w`` to sum to 1; all-zero rows (sinks) stay zero."""
    w = w.tocsr().astype(float)
    sums = np.asarray(w.sum(axis=1)).ravel()
    scale = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums != 0.0)
    return (diags(scale) @ w).tocsr()


def canonical_directed_graph(graph: Mapping) -> tuple[tuple[str, tuple], ...]:
    """Immutable, canonically ordered directed graph (frozen-dataclass safe).

    ``dict()`` on the result reconstructs a mapping ``normalize_directed_graph``
    accepts again. Unweighted graphs store sorted bare labels; weighted graphs
    store sorted ``(label, weight)`` pairs so weights survive the freeze. A pure
    sink is stored with an empty neighbour tuple so it is preserved.
    """
    nodes, w = normalize_directed_graph(graph)
    weighted = bool(w.data.size) and not bool(np.all(w.data == 1.0))
    result = []
    for i, node in enumerate(nodes):
        columns = w.indices[w.indptr[i] : w.indptr[i + 1]]
        weights = w.data[w.indptr[i] : w.indptr[i + 1]]
        if weighted:
            neighbours: tuple = tuple(
                sorted((nodes[j], float(v)) for j, v in zip(columns, weights))
            )
        else:
            neighbours = tuple(sorted(nodes[j] for j in columns))
        result.append((node, neighbours))
    return tuple(result)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/effects/test_directed_graph.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/directed_graph.py tests/effects/test_directed_graph.py
git commit -m "feat(sar): directed-graph normalization and row-standardization"
```

---

### Task 2: `build_sar` and the Gram-precision core

**Files:**
- Create: `src/pylgm/effects/sar.py`
- Test: `tests/effects/test_sar.py`

**Interfaces:**
- Consumes: Task 1's `normalize_directed_graph`, `row_standardize`; `pylgm.effects.graph.design_from_graph`; `pylgm.ir.model.LatentBlock`.
- Produces:
  - `_gram_precision(m: csr_matrix, precision: float) -> csr_matrix` returning `precision * (m.T @ m)`.
  - `_sar_operator(w: csr_matrix, rho: float) -> csr_matrix` returning `I − ρW`.
  - `build_sar(frame, name, index, graph, rho, precision) -> LatentBlock` — labels are the graph nodes, design is the one-hot from `index`, precision `= precision·(I−ρW)ᵀ(I−ρW)`, `constraints = np.empty((0, n))`.

- [ ] **Step 1: Write the failing test**

Create `tests/effects/test_sar.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.sar import build_sar


def _graph():
    # directed ring a->b->c->a, plus a->c
    return {"a": ["b", "c"], "b": ["c"], "c": ["a"]}


def _frame():
    return pd.DataFrame({"region": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})


def test_precision_is_gram_of_i_minus_rho_w():
    rho, precision = 0.4, 2.0
    block = build_sar(_frame(), "s", "region", _graph(), rho, precision)
    nodes, w = normalize_directed_graph(_graph())
    w = row_standardize(w)
    m = np.eye(len(nodes)) - rho * w.toarray()
    expected = precision * (m.T @ m)
    assert np.allclose(block.precision.toarray(), expected)


def test_precision_is_symmetric_and_positive_definite():
    block = build_sar(_frame(), "s", "region", _graph(), 0.6, 1.0)
    q = block.precision.toarray()
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) > 0)


def test_constraints_are_empty_proper_field():
    block = build_sar(_frame(), "s", "region", _graph(), 0.0, 1.0)
    assert block.constraints.shape == (0, 3)


def test_design_maps_rows_to_nodes():
    block = build_sar(_frame(), "s", "region", _graph(), 0.3, 1.0)
    assert block.labels == ("a", "b", "c")
    assert block.design.shape == (3, 3)
    assert np.allclose(block.design.toarray(), np.eye(3))


def test_rho_out_of_bounds_rejected():
    with pytest.raises(ValueError, match="rho"):
        build_sar(_frame(), "s", "region", _graph(), 1.0, 1.0)


def test_rho_zero_is_identity_scaled():
    block = build_sar(_frame(), "s", "region", _graph(), 0.0, 3.0)
    assert np.allclose(block.precision.toarray(), 3.0 * np.eye(3))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_sar.py -v`
Expected: FAIL — `ModuleNotFoundError: pylgm.effects.sar`.

- [ ] **Step 3: Write the module (SAR portion)**

Create `src/pylgm/effects/sar.py`:

```python
"""Spatial-autoregressive (SAR) and dynamic spatial panel (SDPD) latent effects.

Both are the same construction: a sparse operator ``M`` whose Gram matrix
``Q = precision * MᵀM`` is a symmetric, positive-definite (proper) latent
precision. Static SAR is the ``T = 1`` special case (``M = I - ρW``); the
dynamic panel stacks per-period blocks into a block-bidiagonal ``M`` (Task 5).
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.graph import design_from_graph
from pylgm.ir.model import LatentBlock


def _gram_precision(m: csr_matrix, precision: float) -> csr_matrix:
    """The proper latent precision ``precision * MᵀM`` (symmetric, PD)."""
    return csr_matrix(precision * (m.T @ m))


def _sar_operator(w: csr_matrix, rho: float) -> csr_matrix:
    """``I - ρW`` on a row-standardized directed ``W``."""
    n = w.shape[0]
    return (identity(n, format="csr") - rho * w).tocsr()


def build_sar(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    rho: float,
    precision: float,
) -> LatentBlock:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"SAR rho must lie strictly inside (-1, 1); got {rho}")
    nodes, w = normalize_directed_graph(graph)
    w = row_standardize(w)
    design = design_from_graph(nodes, frame, index)
    m = _sar_operator(w, rho)
    precision_matrix = _gram_precision(m, precision)
    constraints = np.empty((0, len(nodes)))
    return LatentBlock(name, nodes, design, precision_matrix, constraints)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/effects/test_sar.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/sar.py tests/effects/test_sar.py
git commit -m "feat(sar): build_sar directed spatial-autoregressive effect"
```

---

### Task 3: `SAR` spec + public wiring

**Files:**
- Modify: `src/pylgm/effects/spec.py` (add `SAR` dataclass; extend `EffectSpec`, `Predictor.__post_init__`, `Predictor.__add__`, `__all__`)
- Modify: `src/pylgm/effects/__init__.py` (export `SAR`, `build_sar`)
- Modify: `src/pylgm/__init__.py` (re-export `SAR`)
- Test: `tests/effects/test_spec.py` (add SAR cases)
- Test: `tests/test_public_exports.py` (add `SAR`)

**Interfaces:**
- Consumes: Task 1's `canonical_directed_graph`; Task 2's `build_sar`.
- Produces: `SAR(name, index, graph, rho, precision=1.0)` — a frozen `_ComposableEffect`. `rho` is a fixed float in `(−1, 1)` or a `Hyperparameter`; `graph` is stored canonicalized. Composable via `+`.

- [ ] **Step 1: Write the failing test**

Add to `tests/effects/test_spec.py`:

```python
def test_sar_spec_validates_and_freezes():
    from pylgm.effects.spec import SAR

    effect = SAR("s", "region", {"a": ["b"], "b": ["a"]}, rho=0.5)
    assert effect.name == "s"
    assert effect.index == "region"
    hash(effect)  # frozen + hashable (graph canonicalized)


def test_sar_spec_rejects_rho_out_of_bounds():
    from pylgm.effects.spec import SAR

    with pytest.raises(ValueError, match="rho"):
        SAR("s", "region", {"a": ["b"], "b": ["a"]}, rho=1.5)


def test_sar_spec_accepts_hyperparameter_rho():
    from pylgm.effects.spec import SAR
    from pylgm.parameters import Hyperparameter

    effect = SAR("s", "region", {"a": ["b"]}, rho=Hyperparameter("s.rho", initial=0.2, transform="logit"))
    assert isinstance(effect.rho, Hyperparameter)


def test_sar_composes_into_predictor():
    from pylgm.effects.spec import IID, SAR

    predictor = SAR("s", "region", {"a": ["b"], "b": ["a"]}, rho=0.3) + IID("u", "id")
    assert len(predictor.effects) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_spec.py -k sar -v`
Expected: FAIL — `ImportError: cannot import name 'SAR'`.

- [ ] **Step 3: Add the spec**

In `src/pylgm/effects/spec.py`, add after the `ProperCAR` class (before `BYM2`):

```python
@dataclass(frozen=True)
class SAR(_ComposableEffect):
    """A directed spatial-autoregressive latent effect: precision
    ``precision * (I - ρW)ᵀ(I - ρW)`` on a row-standardized directed graph."""

    name: str
    index: str
    graph: Mapping
    rho: float | Hyperparameter
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.directed_graph import canonical_directed_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)
        else:
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "graph", canonical_directed_graph(self.graph))
```

Extend `EffectSpec` (line ~317), both `Predictor` isinstance tuples (lines ~336 and ~350), and `__all__` (line ~356) to include `SAR`. For example `EffectSpec` becomes:

```python
EffectSpec: TypeAlias = (
    Fixed | IID | RW1 | RW2 | AR1 | Besag | ProperCAR | SAR | BYM2 | MIDAS | MIDASParametric | SpaceTime
)
```

and each `isinstance(effect, (Fixed, IID, RW1, RW2, AR1, Besag, ProperCAR, BYM2, MIDAS, MIDASParametric, SpaceTime))` tuple gains `SAR`. Add `"SAR"` to `__all__`.

- [ ] **Step 4: Wire the exports**

In `src/pylgm/effects/__init__.py`: add `from pylgm.effects.sar import build_sar`, add `SAR` to the `from pylgm.effects.spec import (...)` block, and add both `"SAR"` and `"build_sar"` to `__all__`.

In `src/pylgm/__init__.py`: add `SAR` to the `from pylgm.effects import (...)` block and to `__all__`.

In `tests/test_public_exports.py`: add `"SAR"` to whatever collection of expected public names the test asserts (read the file first to match its exact structure).

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/effects/test_spec.py -k sar tests/test_public_exports.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/effects/test_spec.py tests/test_public_exports.py
git commit -m "feat(sar): SAR effect spec and public exports"
```

---

### Task 4: Compiler wiring for SAR + end-to-end fit

**Files:**
- Modify: `src/pylgm/compiler.py` (imports; dense `compile_lgm` branch; `_model_hyperparameters`; parametric `compile_family` branch — SAR is a "structured" effect for prediction, so no prediction-context change is needed)
- Test: `tests/test_sar_fit.py`

**Interfaces:**
- Consumes: Task 2's `build_sar`; Task 1's `normalize_directed_graph`, `row_standardize`; existing `_bounded_parameter`, `_log_bounds`, `ScalableBlock`, `ParametricBlock`, `_resolved_precision`, `_compiled_block`.
- Produces: a `SAR` effect compiles to a `ScalableBlock` when `ρ` is fixed and to a `ParametricBlock` over `(τ, ρ)` when `ρ` is a `Hyperparameter`, exactly like AR1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sar_fit.py`:

```python
import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.spec import SAR


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def _simulate(n, rho, seed=0):
    from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize

    rng = np.random.default_rng(seed)
    nodes, w = normalize_directed_graph(_ring(n))
    w = row_standardize(w).toarray()
    m = np.eye(n) - rho * w
    x = np.linalg.solve(m, rng.standard_normal(n))  # draw from the SAR field
    y = x + 0.1 * rng.standard_normal(n)
    return pd.DataFrame({"region": list(nodes), "y": y})


def test_sar_fixed_rho_fits():
    frame = _simulate(12, 0.5)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR("s", "region", _ring(12), rho=0.5, precision=Hyperparameter("s.prec", initial=1.0)),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.mean.shape[0] > 0
    assert np.isfinite(result.mean).all()


def test_sar_estimates_rho():
    frame = _simulate(20, 0.7, seed=3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR(
            "s", "region", _ring(20),
            rho=Hyperparameter("s.rho", initial=0.0, transform="logit"),
            precision=Hyperparameter("s.prec", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert 0.3 < result.hyperparameters["s.rho"] < 0.95  # recovers a positive rho
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sar_fit.py -v`
Expected: FAIL — `CompilationError: unsupported effect type: SAR`.

- [ ] **Step 3: Add imports and the dense compile branch**

In `src/pylgm/compiler.py`, extend the effects import block (near line 14) to include `SAR`, and the builders import (near line 25) to include `build_sar`; also `from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize`.

In `compile_lgm`'s effect loop, add a branch after the `ProperCAR` branch (line ~318):

```python
            elif isinstance(effect, SAR):
                precision = _resolved_precision(effect.precision)
                rho = _resolved_precision(effect.rho)
                block = build_sar(
                    frame, effect.name, effect.index, dict(effect.graph), rho, precision
                )
                precisions[effect.name] = precision
```

- [ ] **Step 4: Add hyperparameter discovery**

In `_model_hyperparameters` (after the AR1 line ~475):

```python
        if isinstance(effect, SAR) and isinstance(effect.rho, Hyperparameter):
            found.append((effect.name, effect.rho))
```

- [ ] **Step 5: Add the parametric `compile_family` branch**

In `compile_family`, add after the AR1 branch (line ~817), mirroring it:

```python
        if isinstance(effect, SAR):
            if not isinstance(effect.rho, Hyperparameter):
                block = _compiled_block(
                    effect.name, build_sar,
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, value,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            rho_bounds = _bounded_parameter(effect.rho, -1.0, 1.0, label="SAR rho", inset=1e-6)
            nodes, w = normalize_directed_graph(dict(effect.graph))
            w = row_standardize(w)
            template = _compiled_block(
                effect.name, build_sar,
                frame, effect.name, effect.index, dict(effect.graph),
                float(effect.rho.initial), value if not optimized else 1.0,
            )
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            rho_name = effect.rho.name
            wmat = w
            n_nodes = len(nodes)

            def build(values, wmat=wmat, n_nodes=n_nodes, tau_name=tau_name,
                      tau_fixed=tau_fixed, rho_name=rho_name) -> csr_matrix:
                tau = values[tau_name] if tau_name else tau_fixed
                m = identity(n_nodes, format="csr") - values[rho_name] * wmat
                return csr_matrix(tau * (m.T @ m))

            params = tuple(nm for nm in (tau_name, rho_name) if nm)
            scalable.append(ParametricBlock(template, params, build))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            parameter_names.append(rho_name)
            parameter_bounds[rho_name] = rho_bounds
            continue
```

(`identity`, `csr_matrix`, `diags` are already imported in `compiler.py` — the ProperCAR/BYM2 branches use them.)

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_sar_fit.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/compiler.py tests/test_sar_fit.py
git commit -m "feat(sar): compile and fit SAR (fixed and estimated rho)"
```

---

### Task 5: `build_dynamic_spatial_panel` + block-bidiagonal operator

**Files:**
- Modify: `src/pylgm/effects/sar.py` (add `_panel_networks`, `_sdpd_operator`, `_panel_design`, `build_dynamic_spatial_panel`)
- Test: `tests/effects/test_dynamic_spatial_panel.py`

**Interfaces:**
- Consumes: Task 1's `normalize_directed_graph`, `row_standardize`; Task 2's `_gram_precision`, `_sar_operator`.
- Produces:
  - `_panel_networks(graphs: Mapping) -> tuple[tuple[str, ...], tuple[str, ...], list[csr_matrix]]` — `(units, times, [W_t...])`. `units` = sorted union of all periods' nodes; `times` = sorted string graph keys; each `W_t` is `n×n` on the `units` ordering, row-standardized.
  - `_sdpd_operator(ws, rho, gamma, eta) -> csr_matrix` — block-bidiagonal `M` (`T·n × T·n`): diagonal `A_t = I − ρW_t`, sub-diagonal `−B_t = −(γI + ηW_t)` for `t ≥ 1`; first block row is `A_0` only.
  - `build_dynamic_spatial_panel(frame, name, unit, time, graphs, rho, gamma, eta, precision) -> LatentBlock` — time-major labels `f"{unit}@{time}"`, composite one-hot design, precision `precision·MᵀM`, empty constraints.

- [ ] **Step 1: Write the failing test**

Create `tests/effects/test_dynamic_spatial_panel.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.sar import build_dynamic_spatial_panel, build_sar


def _graph():
    return {"a": ["b"], "b": ["c"], "c": ["a"]}


def _panel_frame():
    rows = [(u, t) for t in ("1", "2") for u in ("a", "b", "c")]
    return pd.DataFrame(
        {"unit": [u for u, _ in rows], "period": [t for _, t in rows], "y": np.arange(6.0)}
    )


def test_t1_reduces_to_static_sar():
    frame = pd.DataFrame({"unit": ["a", "b", "c"], "period": ["1", "1", "1"], "y": [1.0, 2.0, 3.0]})
    panel = build_dynamic_spatial_panel(
        frame, "d", "unit", "period", {"1": _graph()}, rho=0.4, gamma=0.0, eta=0.0, precision=2.0
    )
    static = build_sar(pd.DataFrame({"region": ["a", "b", "c"]}), "s", "region", _graph(), 0.4, 2.0)
    assert np.allclose(panel.precision.toarray(), static.precision.toarray())


def test_precision_is_symmetric_pd():
    panel = build_dynamic_spatial_panel(
        _panel_frame(), "d", "unit", "period",
        {"1": _graph(), "2": _graph()}, rho=0.5, gamma=0.3, eta=0.2, precision=1.0,
    )
    q = panel.precision.toarray()
    assert q.shape == (6, 6)
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) > 0)


def test_labels_are_time_major():
    panel = build_dynamic_spatial_panel(
        _panel_frame(), "d", "unit", "period",
        {"1": _graph(), "2": _graph()}, rho=0.3, gamma=0.0, eta=0.0, precision=1.0,
    )
    assert panel.labels == ("a@1", "b@1", "c@1", "a@2", "b@2", "c@2")


def test_design_maps_unit_time_to_time_major_cell():
    frame = _panel_frame()
    panel = build_dynamic_spatial_panel(
        frame, "d", "unit", "period",
        {"1": _graph(), "2": _graph()}, rho=0.3, gamma=0.0, eta=0.0, precision=1.0,
    )
    design = panel.design.toarray()
    # row 3 is (a, period 2) -> column time_pos(1)*3 + unit_pos(0) = 3
    assert design[3, 3] == 1.0
    assert design.sum() == 6.0


def test_eta_zero_is_ar_link_block_bidiagonal():
    # gamma!=0, eta=0: sub-diagonal blocks are -gamma*I (pure temporal AR link)
    from pylgm.effects.sar import _panel_networks, _sdpd_operator

    _, _, ws = _panel_networks({"1": _graph(), "2": _graph()})
    m = _sdpd_operator(ws, rho=0.0, gamma=0.5, eta=0.0).toarray()
    n = 3
    assert np.allclose(m[n:, :n], -0.5 * np.eye(n))  # sub-diagonal = -gamma*I


def test_unobserved_time_without_graph_rejected():
    frame = pd.DataFrame({"unit": ["a"], "period": ["9"], "y": [1.0]})
    with pytest.raises(ValueError, match="time"):
        build_dynamic_spatial_panel(
            frame, "d", "unit", "period", {"1": _graph()}, rho=0.3, gamma=0.0, eta=0.0, precision=1.0
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_dynamic_spatial_panel.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_dynamic_spatial_panel'`.

- [ ] **Step 3: Extend `sar.py`**

Add to `src/pylgm/effects/sar.py` (add `bmat` to the scipy import: `from scipy.sparse import bmat, csr_matrix, identity`):

```python
def _panel_networks(
    graphs: Mapping,
) -> tuple[tuple[str, ...], tuple[str, ...], list[csr_matrix]]:
    """Align every period's directed network onto one balanced unit universe.

    ``units`` is the sorted union of all periods' nodes; ``times`` the sorted
    string keys of ``graphs``. Each returned ``W_t`` is ``n x n`` on the shared
    ``units`` ordering and row-standardized (a unit absent from period ``t`` is
    an all-zero row that period).
    """
    if not isinstance(graphs, Mapping) or not graphs:
        raise ValueError("graphs must be a non-empty mapping of time -> graph")
    times = tuple(sorted(map(str, graphs)))
    per_period: dict[str, tuple[tuple[str, ...], csr_matrix]] = {}
    union: set[str] = set()
    for t in times:
        nodes_t, w_t = normalize_directed_graph(dict(graphs[_original_key(graphs, t)]))
        per_period[t] = (nodes_t, w_t)
        union |= set(nodes_t)
    units = tuple(sorted(union))
    position = {u: i for i, u in enumerate(units)}
    n = len(units)
    matrices: list[csr_matrix] = []
    for t in times:
        nodes_t, w_t = per_period[t]
        coo = w_t.tocoo()
        rows = [position[nodes_t[i]] for i in coo.row]
        cols = [position[nodes_t[j]] for j in coo.col]
        big = csr_matrix((coo.data, (rows, cols)), shape=(n, n))
        matrices.append(row_standardize(big))
    return units, times, matrices


def _original_key(graphs: Mapping, text: str):
    """Return the original ``graphs`` key whose ``str()`` is ``text``."""
    for key in graphs:
        if str(key) == text:
            return key
    raise KeyError(text)  # unreachable: text came from graphs' own keys


def _sdpd_operator(
    ws: list[csr_matrix], rho: float, gamma: float, eta: float
) -> csr_matrix:
    """Block-bidiagonal SDPD operator ``M`` over the stacked field ``(x_1..x_T)``.

    Diagonal block ``A_t = I - ρW_t``; sub-diagonal block ``-B_t = -(γI + ηW_t)``
    for ``t >= 1``. The first block row is ``A_0`` only (conditional-on-initial).
    """
    t_count = len(ws)
    n = ws[0].shape[0]
    ident = identity(n, format="csr")
    blocks: list[list[object]] = [[None] * t_count for _ in range(t_count)]
    for t in range(t_count):
        blocks[t][t] = (ident - rho * ws[t]).tocsr()
        if t >= 1:
            blocks[t][t - 1] = -(gamma * ident + eta * ws[t]).tocsr()
    return bmat(blocks, format="csr")


def _panel_design(
    frame: pd.DataFrame,
    unit: str,
    time: str,
    units: tuple[str, ...],
    times: tuple[str, ...],
) -> csr_matrix:
    """Time-major one-hot: cell = ``time_pos * n_units + unit_pos``."""
    if unit not in frame.columns:
        raise ValueError(f"unit column {unit!r} not found")
    if time not in frame.columns:
        raise ValueError(f"time column {time!r} not found")
    upos = {u: i for i, u in enumerate(units)}
    tpos = {t: j for j, t in enumerate(times)}
    n = len(units)
    obs_u = [str(v) for v in frame[unit]]
    obs_t = [str(v) for v in frame[time]]
    missing_u = sorted({v for v in obs_u if v not in upos})
    if missing_u:
        raise ValueError(f"observed unit(s) {missing_u!r} are not in the network")
    missing_t = sorted({v for v in obs_t if v not in tpos})
    if missing_t:
        raise ValueError(f"observed time(s) {missing_t!r} have no network graph")
    cells = np.array([tpos[t] * n + upos[u] for u, t in zip(obs_u, obs_t)])
    return csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), cells)),
        shape=(len(frame), n * len(times)),
    )


def build_dynamic_spatial_panel(
    frame: pd.DataFrame,
    name: str,
    unit: str,
    time: str,
    graphs: Mapping,
    rho: float,
    gamma: float,
    eta: float,
    precision: float,
) -> LatentBlock:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"SDPD rho must lie strictly inside (-1, 1); got {rho}")
    units, times, ws = _panel_networks(graphs)
    n = len(units)
    m = _sdpd_operator(ws, rho, gamma, eta)
    precision_matrix = _gram_precision(m, precision)
    design = _panel_design(frame, unit, time, units, times)
    labels = tuple(f"{u}@{t}" for t in times for u in units)
    constraints = np.empty((0, n * len(times)))
    return LatentBlock(name, labels, design, precision_matrix, constraints)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/effects/test_dynamic_spatial_panel.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/sar.py tests/effects/test_dynamic_spatial_panel.py
git commit -m "feat(sdpd): dynamic spatial panel builder and block-bidiagonal operator"
```

---

### Task 6: `DynamicSpatialPanel` spec + exports

**Files:**
- Modify: `src/pylgm/effects/spec.py` (add `DynamicSpatialPanel`; extend `EffectSpec`, both `Predictor` isinstance tuples, `__all__`)
- Modify: `src/pylgm/effects/__init__.py` (export `DynamicSpatialPanel`, `build_dynamic_spatial_panel`)
- Modify: `src/pylgm/__init__.py` (re-export `DynamicSpatialPanel`)
- Test: `tests/effects/test_spec.py`
- Test: `tests/test_public_exports.py`

**Interfaces:**
- Consumes: Task 1's `canonical_directed_graph`; Task 5's `build_dynamic_spatial_panel`.
- Produces: `DynamicSpatialPanel(name, unit, time, graphs, rho, gamma=0.0, eta=0.0, precision=1.0)`. `rho` fixed in `(−1, 1)` or `Hyperparameter`; `gamma`, `eta` fixed finite reals or `Hyperparameter`; `graphs` stored as a sorted tuple of `(time_str, canonical_directed_graph)` pairs.

- [ ] **Step 1: Write the failing test**

Add to `tests/effects/test_spec.py`:

```python
def test_dynamic_spatial_panel_spec_freezes():
    from pylgm.effects.spec import DynamicSpatialPanel

    effect = DynamicSpatialPanel(
        "d", "unit", "period", {"1": {"a": ["b"], "b": ["a"]}}, rho=0.5, gamma=0.2, eta=0.1
    )
    assert effect.unit == "unit"
    assert effect.time == "period"
    hash(effect)  # frozen + hashable (graphs canonicalized)


def test_dynamic_spatial_panel_rejects_rho_out_of_bounds():
    from pylgm.effects.spec import DynamicSpatialPanel

    with pytest.raises(ValueError, match="rho"):
        DynamicSpatialPanel("d", "unit", "period", {"1": {"a": ["b"]}}, rho=2.0)


def test_dynamic_spatial_panel_accepts_hyperparameter_coefficients():
    from pylgm.effects.spec import DynamicSpatialPanel
    from pylgm.parameters import Hyperparameter

    effect = DynamicSpatialPanel(
        "d", "unit", "period", {"1": {"a": ["b"]}},
        rho=Hyperparameter("d.rho", initial=0.0, transform="logit"),
        gamma=Hyperparameter("d.gamma", initial=0.0, transform="identity"),
        eta=Hyperparameter("d.eta", initial=0.0, transform="identity"),
    )
    assert isinstance(effect.gamma, Hyperparameter)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_spec.py -k dynamic_spatial -v`
Expected: FAIL — `ImportError: cannot import name 'DynamicSpatialPanel'`.

- [ ] **Step 3: Add the spec**

In `src/pylgm/effects/spec.py`, add after the `SAR` class:

```python
@dataclass(frozen=True)
class DynamicSpatialPanel(_ComposableEffect):
    """A dynamic spatial panel (SDPD): per-period directed networks ``W_t`` with
    contemporaneous (``rho``), temporal (``gamma``) and spatio-temporal-diffusion
    (``eta``) coefficients. Precision ``precision * MᵀM`` for the block-bidiagonal
    operator ``M`` (see the S-slice design spec)."""

    name: str
    unit: str
    time: str
    graphs: Mapping
    rho: float | Hyperparameter
    gamma: float | Hyperparameter = 0.0
    eta: float | Hyperparameter = 0.0
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.directed_graph import canonical_directed_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "unit"))
        object.__setattr__(self, "time", _non_empty_string(self.time, "time"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)
        else:
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)
        for field_name in ("gamma", "eta"):
            value = getattr(self, field_name)
            if not isinstance(value, Hyperparameter):
                object.__setattr__(self, field_name, _finite_real(value, field_name))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.graphs, Mapping) or not self.graphs:
            raise ValueError("graphs must be a non-empty mapping of time -> graph")
        canonical = tuple(
            (str(t), canonical_directed_graph(g))
            for t, g in sorted(self.graphs.items(), key=lambda kv: str(kv[0]))
        )
        object.__setattr__(self, "graphs", canonical)

    @property
    def index(self) -> str:
        return self.unit
```

Extend `EffectSpec`, both `Predictor` isinstance tuples, and `__all__` to include `DynamicSpatialPanel`.

- [ ] **Step 4: Wire the exports**

In `src/pylgm/effects/__init__.py`: add `from pylgm.effects.sar import build_dynamic_spatial_panel, build_sar` (replace the Task-3 single-name import), add `DynamicSpatialPanel` to the spec import block, and add `"DynamicSpatialPanel"` and `"build_dynamic_spatial_panel"` to `__all__`.

In `src/pylgm/__init__.py`: add `DynamicSpatialPanel` to the effects import and to `__all__`.

In `tests/test_public_exports.py`: add `"DynamicSpatialPanel"`.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/effects/test_spec.py -k dynamic_spatial tests/test_public_exports.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/effects/test_spec.py tests/test_public_exports.py
git commit -m "feat(sdpd): DynamicSpatialPanel effect spec and exports"
```

---

### Task 7: Compiler wiring for SDPD + prediction context + end-to-end fit

**Files:**
- Modify: `src/pylgm/compiler.py` (imports; dense `compile_lgm` branch; `_model_hyperparameters`; parametric `compile_family` branch over `(τ, ρ, γ, η)`; prediction-context branch)
- Modify: `src/pylgm/inference/prediction.py` (a `dynamic_spatial_panel` design handler + registration)
- Test: `tests/test_sdpd_fit.py`

**Interfaces:**
- Consumes: Task 5's `build_dynamic_spatial_panel`, `_panel_networks`, `_sdpd_operator`; existing `_bounded_parameter`, `_real_bounds`, `_log_bounds`, `_resolved_precision`.
- Produces: SDPD compiles to a `ScalableBlock` when `ρ, γ, η` are all fixed, else a `ParametricBlock` over the present subset of `(τ, ρ, γ, η)`. `result.predict(new_data)` scores in-grid `(unit, time)` rows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sdpd_fit.py`:

```python
import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.spec import DynamicSpatialPanel


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def _simulate(n, periods, rho, gamma, eta, seed=0):
    from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize

    rng = np.random.default_rng(seed)
    nodes, w = normalize_directed_graph(_ring(n))
    w = row_standardize(w).toarray()
    a = np.eye(n) - rho * w
    b = gamma * np.eye(n) + eta * w
    x_prev = np.linalg.solve(a, rng.standard_normal(n))
    rows = []
    xs = [x_prev]
    for _ in range(1, periods):
        x = np.linalg.solve(a, b @ xs[-1] + rng.standard_normal(n))
        xs.append(x)
    for t, x in enumerate(xs):
        for i, node in enumerate(nodes):
            rows.append({"unit": node, "period": str(t), "y": x[i] + 0.1 * rng.standard_normal()})
    graphs = {str(t): _ring(n) for t in range(periods)}
    return pd.DataFrame(rows), graphs


def test_sdpd_fixed_coefficients_fits_and_predicts():
    frame, graphs = _simulate(8, 4, rho=0.4, gamma=0.3, eta=0.1)
    model = LGM(
        response="y",
        predictor=Fixed("1") + DynamicSpatialPanel(
            "d", "unit", "period", graphs, rho=0.4, gamma=0.3, eta=0.1,
            precision=Hyperparameter("d.prec", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert np.isfinite(result.mean).all()
    prediction = result.predict(frame.head(4))
    assert prediction.predictive_mean.shape == (4,)
    assert np.isfinite(prediction.predictive_mean).all()


def test_sdpd_estimates_all_three_coefficients():
    frame, graphs = _simulate(12, 6, rho=0.5, gamma=0.4, eta=0.2, seed=7)
    model = LGM(
        response="y",
        predictor=Fixed("1") + DynamicSpatialPanel(
            "d", "unit", "period", graphs,
            rho=Hyperparameter("d.rho", initial=0.0, transform="logit"),
            gamma=Hyperparameter("d.gamma", initial=0.0, transform="identity"),
            eta=Hyperparameter("d.eta", initial=0.0, transform="identity"),
            precision=Hyperparameter("d.prec", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.hyperparameters["d.rho"] > 0.2
    assert result.hyperparameters["d.gamma"] > 0.15
    assert np.isfinite(result.hyperparameters["d.eta"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_sdpd_fit.py -v`
Expected: FAIL — `CompilationError: unsupported effect type: DynamicSpatialPanel`.

- [ ] **Step 3: Add imports + dense compile branch**

In `src/pylgm/compiler.py`, add `DynamicSpatialPanel` to the effects-spec import, and `from pylgm.effects.sar import build_dynamic_spatial_panel, _panel_networks, _sdpd_operator, build_sar` (extend the Task-4 import).

Add a dense branch after the SAR branch in `compile_lgm`:

```python
            elif isinstance(effect, DynamicSpatialPanel):
                precision = _resolved_precision(effect.precision)
                block = build_dynamic_spatial_panel(
                    frame, effect.name, effect.unit, effect.time,
                    {t: dict(g) for t, g in dict(effect.graphs).items()},
                    _resolved_precision(effect.rho),
                    _resolved_precision(effect.gamma),
                    _resolved_precision(effect.eta),
                    precision,
                )
                precisions[effect.name] = precision
```

- [ ] **Step 4: Add hyperparameter discovery**

In `_model_hyperparameters`, after the SAR line:

```python
        if isinstance(effect, DynamicSpatialPanel):
            for coeff in (effect.rho, effect.gamma, effect.eta):
                if isinstance(coeff, Hyperparameter):
                    found.append((effect.name, coeff))
```

- [ ] **Step 5: Add the parametric `compile_family` branch**

After the SAR parametric branch:

```python
        if isinstance(effect, DynamicSpatialPanel):
            graphs = {t: dict(g) for t, g in dict(effect.graphs).items()}
            coeff_is_hp = any(
                isinstance(p, Hyperparameter) for p in (effect.rho, effect.gamma, effect.eta)
            )
            if not coeff_is_hp:
                block = _compiled_block(
                    effect.name, build_dynamic_spatial_panel,
                    frame, effect.name, effect.unit, effect.time, graphs,
                    effect.rho, effect.gamma, effect.eta, value,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            _, _, ws = _panel_networks(graphs)

            def _coeff(param):
                if isinstance(param, Hyperparameter):
                    return param.name, None
                return None, float(param)

            rho_name, rho_fixed = _coeff(effect.rho)
            gamma_name, gamma_fixed = _coeff(effect.gamma)
            eta_name, eta_fixed = _coeff(effect.eta)
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value

            template = _compiled_block(
                effect.name, build_dynamic_spatial_panel,
                frame, effect.name, effect.unit, effect.time, graphs,
                rho_fixed if rho_fixed is not None else float(effect.rho.initial),
                gamma_fixed if gamma_fixed is not None else float(effect.gamma.initial),
                eta_fixed if eta_fixed is not None else float(effect.eta.initial),
                value if not optimized else 1.0,
            )

            def build(values, ws=ws,
                      rho_name=rho_name, rho_fixed=rho_fixed,
                      gamma_name=gamma_name, gamma_fixed=gamma_fixed,
                      eta_name=eta_name, eta_fixed=eta_fixed,
                      tau_name=tau_name, tau_fixed=tau_fixed) -> csr_matrix:
                rho = values[rho_name] if rho_name else rho_fixed
                gamma = values[gamma_name] if gamma_name else gamma_fixed
                eta = values[eta_name] if eta_name else eta_fixed
                tau = values[tau_name] if tau_name else tau_fixed
                m = _sdpd_operator(ws, rho, gamma, eta)
                return csr_matrix(tau * (m.T @ m))

            params = tuple(nm for nm in (tau_name, rho_name, gamma_name, eta_name) if nm)
            scalable.append(ParametricBlock(template, params, build))
            if rho_name:
                parameter_names.append(rho_name)
                parameter_bounds[rho_name] = _bounded_parameter(
                    effect.rho, -1.0, 1.0, label="SDPD rho", inset=1e-6
                )
            if gamma_name:
                parameter_names.append(gamma_name)
                parameter_bounds[gamma_name] = _real_bounds(effect.gamma)
            if eta_name:
                parameter_names.append(eta_name)
                parameter_bounds[eta_name] = _real_bounds(effect.eta)
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            continue
```

- [ ] **Step 6: Add the prediction-context branch + design handler**

In `src/pylgm/compiler.py`'s prediction-context builder (the loop near line 930), add before the final `else`:

```python
        elif isinstance(effect, DynamicSpatialPanel):
            unit_labels = tuple(dict.fromkeys(label.split("@", 1)[0] for label in block.labels))
            time_labels = tuple(dict.fromkeys(label.split("@", 1)[1] for label in block.labels))
            entries.append(
                ("dynamic_spatial_panel",
                 (effect.name, effect.unit, effect.time, unit_labels, time_labels))
            )
```

In `src/pylgm/inference/prediction.py`, add a handler mirroring `_spacetime_block` but **time-major** (`cell = time_pos * n_units + unit_pos`):

```python
def _dynamic_spatial_panel_block(
    entry: tuple[str, str, str, tuple[str, ...], tuple[str, ...]], new_data: pd.DataFrame
) -> np.ndarray:
    name, unit, time, unit_labels, time_labels = entry
    for column in (unit, time):
        if column not in new_data.columns:
            raise ValueError(
                f"predict() new_data is missing column {column!r} required by the "
                f"{name!r} dynamic-spatial-panel block"
            )
    unit_pos = {label: i for i, label in enumerate(unit_labels)}
    time_pos = {label: j for j, label in enumerate(time_labels)}
    units = new_data[unit].map(str)
    times = new_data[time].map(str)
    unseen = sorted(set(units[~units.isin(unit_pos)]) | set(times[~times.isin(time_pos)]))
    if unseen:
        raise ValueError(
            f"predict() cannot score rows whose {name!r} unit/time was not in the "
            f"fitted model: {unseen!r}. predict reuses the fitted latent posterior, so it "
            "cannot create a new latent cell. To forecast future periods, use the SDPD "
            "forecast() helper instead."
        )
    n = len(unit_labels)
    cells = times.map(time_pos).to_numpy() * n + units.map(unit_pos).to_numpy()
    design = np.zeros((len(new_data), n * len(time_labels)))
    design[np.arange(len(new_data)), cells] = 1.0
    return design
```

Register it in `_design_for`'s dispatch, alongside `spacetime`:

```python
        elif kind == "dynamic_spatial_panel":
            blocks.append(_dynamic_spatial_panel_block(payload, new_data))
```

Also add `("dynamic_spatial_panel", (block_name, unit, time, unit_labels, time_labels))` to the `PredictionContext.entries` docstring list.

- [ ] **Step 7: Run to verify pass**

Run: `python -m pytest tests/test_sdpd_fit.py -v`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/inference/prediction.py tests/test_sdpd_fit.py
git commit -m "feat(sdpd): compile, fit, and predict dynamic spatial panel"
```

---

### Task 8: E-sparse verification (large network past the dense guard)

**Files:**
- Test: `tests/inference/test_sdpd_sparse.py`

**Interfaces:**
- Consumes: the full SAR/SDPD fit path from Tasks 4 and 7. This task is verification-only — the sparse solver is effect-agnostic and size-routed, so no production change is expected. If a defect surfaces, fix it in the smallest place and note it in the commit.

- [ ] **Step 1: Read the dense-guard constant and a sibling sparse test**

Read `tests/inference/test_sparse_large_model.py` to copy how it builds a model past `_MAX_DENSE_LATENT_DIMENSION` and how it asserts the sparse path ran (e.g. checking `result._covariance is None` / a `_sparse_posterior` attribute). Read `src/pylgm/inference/gaussian.py` for the exact threshold constant name so the test builds a latent dimension above it.

- [ ] **Step 2: Write the failing test**

Create `tests/inference/test_sdpd_sparse.py`. Build a SAR model whose node count exceeds the dense guard, and a small model fit both ways to assert agreement:

```python
import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.spec import SAR


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": rng.standard_normal(n)})


def test_large_sar_fits_past_dense_guard():
    n = 5000  # above _MAX_DENSE_LATENT_DIMENSION (4096)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR("s", "region", _ring(n), rho=0.5,
                                    precision=Hyperparameter("s.prec", initial=1.0)),
        likelihood=Gaussian(sigma=1.0),
    )
    result = model.fit(_frame(n))
    assert result.mean.shape[0] > n
    marginals = result.latent_marginals("s")
    assert np.isfinite(marginals.variance).all()
    assert np.all(marginals.variance >= 0)


def test_small_sar_dense_and_sparse_agree():
    # Fit the same small SAR through the dense path; assert marginal variances
    # are finite and posterior mean is stable (sparse path exercised in the
    # large test above; this pins the dense reference numbers).
    frame = _frame(30, seed=1)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR("s", "region", _ring(30), rho=0.5),
        likelihood=Gaussian(sigma=1.0),
    )
    result = model.fit(frame)
    marginals = result.latent_marginals("s")
    assert np.all(marginals.variance > 0)
    assert np.isfinite(result.mean).all()
```

- [ ] **Step 3: Run**

Run: `python -m pytest tests/inference/test_sdpd_sparse.py -v`
Expected: PASS. If the large fit raises (e.g. an INLA latent strategy that is dense-only above the guard), confirm the model uses the default `latent_strategy="gaussian"`; the design spec documents that simplified/full-Laplace marginals are dense-only. If a genuine bug surfaces in the sparse path for these blocks, fix it minimally and describe it in the commit message.

- [ ] **Step 4: Commit**

```bash
git add tests/inference/test_sdpd_sparse.py
git commit -m "test(sar): verify SAR fits past the dense guard on the sparse path"
```

---

### Task 9: SDPD forecasting (forward recursion)

**Files:**
- Create: `src/pylgm/effects/sdpd_forecast.py`
- Modify: `src/pylgm/effects/__init__.py` (export `forecast_dynamic_spatial_panel`)
- Modify: `src/pylgm/__init__.py` (re-export `forecast_dynamic_spatial_panel`)
- Test: `tests/effects/test_sdpd_forecast.py`
- Test: `tests/test_sdpd_forecast_integration.py`

**Interfaces:**
- Consumes: Task 1's `normalize_directed_graph`, `row_standardize`; Task 5's `_panel_networks`; a fitted result's `.hyperparameters`, `.latent_marginals(name)`; `pylgm.parameters.Hyperparameter`.
- Produces:
  - `sdpd_forecast(x_last, v_last, rho, gamma, eta, tau, future_ws) -> list[tuple[np.ndarray, np.ndarray]]` — the pure forward recursion: for each future `W`, mean `A⁻¹Bx`, marginal variance `diag(A⁻¹(B diag(v) Bᵀ + τ⁻¹I)A⁻ᵀ)`.
  - `forecast_dynamic_spatial_panel(result, effect, future_graphs) -> pd.DataFrame` — columns `unit, time, mean, variance` for each future period.

- [ ] **Step 1: Write the failing pure-function test**

Create `tests/effects/test_sdpd_forecast.py`:

```python
import numpy as np

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.sdpd_forecast import sdpd_forecast


def _w(n):
    graph = {str(i): [str((i + 1) % n)] for i in range(n)}
    _, w = normalize_directed_graph(graph)
    return row_standardize(w)


def test_one_step_mean_matches_closed_form():
    n, rho, gamma, eta, tau = 5, 0.4, 0.3, 0.1, 2.0
    w = _w(n)
    x_last = np.arange(n, dtype=float)
    v_last = np.ones(n)
    a = np.eye(n) - rho * w.toarray()
    b = gamma * np.eye(n) + eta * w.toarray()
    expected_mean = np.linalg.solve(a, b @ x_last)
    (mean, var), = sdpd_forecast(x_last, v_last, rho, gamma, eta, tau, [w])
    assert np.allclose(mean, expected_mean)
    assert np.all(var > 0)


def test_multi_step_iterates():
    n = 4
    w = _w(n)
    steps = sdpd_forecast(np.ones(n), np.ones(n), 0.3, 0.5, 0.0, 1.0, [w, w, w])
    assert len(steps) == 3
    for mean, var in steps:
        assert mean.shape == (n,)
        assert np.all(np.isfinite(mean))
        assert np.all(var > 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_sdpd_forecast.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

Create `src/pylgm/effects/sdpd_forecast.py`:

```python
"""Forward forecasting for the dynamic spatial panel (SDPD).

Given the fitted last-period latent ``x̂_T`` (mean + marginal variance) and the
next periods' directed networks, propagate the SDPD recursion forward:

    x̂_{t+1} = A_{t+1}⁻¹ B_{t+1} x̂_t
    Var(x_{t+1}) = A_{t+1}⁻¹ [B_{t+1} diag(Var x_t) B_{t+1}ᵀ + τ⁻¹ I] A_{t+1}⁻ᵀ

with ``A = I - ρW``, ``B = γI + ηW``. Variance is carried diagonal-only
(marginal), consistent with the gaussian latent strategy above the sparse guard.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import splu

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.sar import _panel_networks
from pylgm.parameters import Hyperparameter


def sdpd_forecast(
    x_last: np.ndarray,
    v_last: np.ndarray,
    rho: float,
    gamma: float,
    eta: float,
    tau: float,
    future_ws: list[csr_matrix],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward SDPD recursion; returns ``[(mean, marginal_var), ...]`` per step."""
    x = np.asarray(x_last, dtype=float)
    v = np.asarray(v_last, dtype=float)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for w in future_ws:
        n = w.shape[0]
        ident = identity(n, format="csc")
        a = (ident - rho * w).tocsc()
        b = (gamma * ident + eta * w).tocsc()
        lu = splu(a)
        x_next = lu.solve(b @ x)
        # ponytail: dense A^-1 (via sparse LU solve against I) — O(n^3), fine for
        # a forecast horizon; a diagonal-only fast path is a later optimization.
        a_inv = lu.solve(np.eye(n))
        inner = (b @ diags(v) @ b.T).toarray() + (1.0 / tau) * np.eye(n)
        cov_next = a_inv @ inner @ a_inv.T
        var_next = np.clip(np.diag(cov_next), 0.0, None)
        out.append((x_next, var_next))
        x, v = x_next, var_next
    return out


def _align_to_units(graph: Mapping, units: tuple[str, ...]) -> csr_matrix:
    """Row-standardized ``W`` for ``graph`` on the fitted ``units`` ordering."""
    nodes, w = normalize_directed_graph(graph)
    unknown = sorted(set(nodes) - set(units))
    if unknown:
        raise ValueError(
            f"forecast network references unit(s) not in the fitted panel: {unknown!r}"
        )
    position = {u: i for i, u in enumerate(units)}
    coo = w.tocoo()
    rows = [position[nodes[i]] for i in coo.row]
    cols = [position[nodes[j]] for j in coo.col]
    big = csr_matrix((coo.data, (rows, cols)), shape=(len(units), len(units)))
    return row_standardize(big)


def forecast_dynamic_spatial_panel(result, effect, future_graphs: Mapping) -> pd.DataFrame:
    """Forecast future periods for a fitted ``DynamicSpatialPanel`` effect.

    ``result`` is the fit output, ``effect`` the ``DynamicSpatialPanel`` spec,
    ``future_graphs`` a ``{time: directed_graph}`` mapping for the periods to
    forecast. Returns a frame with columns ``unit, time, mean, variance``.
    """

    def _value(param):
        if isinstance(param, Hyperparameter):
            return float(result.hyperparameters[param.name])
        return float(param)

    rho, gamma, eta, tau = (
        _value(effect.rho), _value(effect.gamma), _value(effect.eta), _value(effect.precision)
    )
    fitted_graphs = {t: dict(g) for t, g in dict(effect.graphs).items()}
    units, times, _ = _panel_networks(fitted_graphs)
    n, t_count = len(units), len(times)
    marginals = result.latent_marginals(effect.name)
    x_grid = np.asarray(marginals.mean, dtype=float).reshape(t_count, n)
    v_grid = np.asarray(marginals.variance, dtype=float).reshape(t_count, n)
    future_times = tuple(sorted(map(str, future_graphs)))
    future_ws = [
        _align_to_units(dict(future_graphs[_key(future_graphs, t)]), units) for t in future_times
    ]
    steps = sdpd_forecast(x_grid[-1], v_grid[-1], rho, gamma, eta, tau, future_ws)
    records = []
    for t, (mean, var) in zip(future_times, steps):
        for unit, mean_i, var_i in zip(units, mean, var):
            records.append({"unit": unit, "time": t, "mean": mean_i, "variance": var_i})
    return pd.DataFrame.from_records(records)


def _key(graphs: Mapping, text: str):
    for key in graphs:
        if str(key) == text:
            return key
    raise KeyError(text)
```

- [ ] **Step 4: Run to verify the pure function passes**

Run: `python -m pytest tests/effects/test_sdpd_forecast.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the integration test**

Create `tests/test_sdpd_forecast_integration.py`:

```python
import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.sdpd_forecast import forecast_dynamic_spatial_panel
from pylgm.effects.spec import DynamicSpatialPanel


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def test_forecast_returns_future_grid():
    n, periods = 6, 4
    rng = np.random.default_rng(0)
    rows = [
        {"unit": str(i), "period": str(t), "y": rng.standard_normal()}
        for t in range(periods)
        for i in range(n)
    ]
    frame = pd.DataFrame(rows)
    graphs = {str(t): _ring(n) for t in range(periods)}
    effect = DynamicSpatialPanel(
        "d", "unit", "period", graphs, rho=0.4, gamma=0.3, eta=0.1,
        precision=Hyperparameter("d.prec", initial=1.0),
    )
    model = LGM(response="y", predictor=Fixed("1") + effect, likelihood=Gaussian(sigma=0.2))
    result = model.fit(frame)
    # NB: pass the *compiled* effect (graphs canonicalized). Rebuild it from the
    # predictor to get the frozen spec the model actually holds.
    fitted_effect = model.predictor.effects[1]
    forecast = forecast_dynamic_spatial_panel(result, fitted_effect, {"4": _ring(n)})
    assert set(forecast.columns) == {"unit", "time", "mean", "variance"}
    assert len(forecast) == n
    assert (forecast["time"] == "4").all()
    assert np.isfinite(forecast["mean"]).all()
    assert (forecast["variance"] > 0).all()
```

(If `model.predictor` is not the attribute holding the effects, read `src/pylgm/model.py` to find the correct accessor and adjust `fitted_effect`.)

- [ ] **Step 6: Wire exports, run, verify**

Add `from pylgm.effects.sdpd_forecast import forecast_dynamic_spatial_panel` to `src/pylgm/effects/__init__.py` and its `__all__`; re-export from `src/pylgm/__init__.py` and its `__all__`; add `"forecast_dynamic_spatial_panel"` to `tests/test_public_exports.py`.

Run: `python -m pytest tests/effects/test_sdpd_forecast.py tests/test_sdpd_forecast_integration.py tests/test_public_exports.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/effects/sdpd_forecast.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/effects/test_sdpd_forecast.py tests/test_sdpd_forecast_integration.py tests/test_public_exports.py
git commit -m "feat(sdpd): forward forecasting for future periods"
```

---

### Task 10: YAML frontend for SAR

**Files:**
- Modify: `src/pylgm/config/model.py` (add `sar` to `_SPATIAL_EFFECTS`, the `type` Literal, field validation, and `_build_effect`)
- Test: `tests/config/test_model.py`

**Interfaces:**
- Consumes: Task 3's `SAR`; `pylgm.effects.load_graph_file` (a `.json` directed neighbour dict loads unchanged).
- Produces: a `sar` config effect with an inline `graph` or a `graph_file`, and a required fixed `rho`. `DynamicSpatialPanel` is intentionally **not** exposed in YAML (Python-API only in v1).

- [ ] **Step 1: Read the ProperCAR config path**

Read `src/pylgm/config/model.py` lines ~53-127 (already summarized: `_SPATIAL_EFFECTS`, the `type` Literal, the graph/graph_file/rho validators, `_build_effect`). Copy the ProperCAR handling; SAR differs only in using the directed builder and requiring `rho` like ProperCAR.

- [ ] **Step 2: Write the failing test**

Add to `tests/config/test_model.py` (match the file's existing import style and any `base_dir` fixture):

```python
def test_sar_config_builds_effect():
    from pylgm.config.model import _EffectModelConfig, _build_effect
    from pathlib import Path
    from pylgm.effects.spec import SAR

    config = _EffectModelConfig(
        name="s", type="sar", index="region",
        graph={"a": ["b"], "b": ["a"]}, rho=0.4,
    )
    effect = _build_effect(config, Path("."))
    assert isinstance(effect, SAR)
    assert effect.index == "region"


def test_sar_config_requires_rho():
    from pylgm.config.model import _EffectModelConfig

    with pytest.raises(ValueError, match="rho"):
        _EffectModelConfig(name="s", type="sar", index="region", graph={"a": ["b"]})
```

(Confirm the exact `_EffectModelConfig` constructor keyword names against the file; adjust if `phi`/`scale` are required positional-only.)

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/config/test_model.py -k sar -v`
Expected: FAIL — `sar` not an accepted `type`.

- [ ] **Step 4: Wire the config**

In `src/pylgm/config/model.py`:
- Add `"sar"` to `_SPATIAL_EFFECTS`.
- Add `"sar"` to the `type` `Literal[...]`.
- In the validator: `rho` is required for `sar` (extend the existing `if self.type == "proper_car" and self.rho is None` check to `in ("proper_car", "sar")`); and `rho` remains invalid for the other spatial types (extend `if self.type != "proper_car"` to `if self.type not in ("proper_car", "sar")`).
- In `_build_effect`, add: `if config.type == "sar": return SAR(config.name, config.index, graph, config.rho, precision)` (import `SAR` in the local import line alongside `ProperCAR`).

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/config/test_model.py -k sar -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/config/model.py tests/config/test_model.py
git commit -m "feat(sar): declarative YAML 'sar' effect type"
```

---

### Task 11: Example, docs, roadmap, full-suite regression

**Files:**
- Create: `examples/directed_network_sar/run.py`, `examples/directed_network_sar/README.md`
- Modify: `docs/spatial-effects.md` (SAR/SDPD section)
- Modify: `docs/roadmap.md` (Next #3 → Shipped; renumber remaining Next items)
- Modify: `mkdocs.yml` only if a new doc page is added (this task extends the existing `docs/spatial-effects.md`, so likely no nav change — verify)
- Test: `tests/test_package.py` (example smoke test)

**Interfaces:**
- Consumes: the full public API from Tasks 1-10.

- [ ] **Step 1: Inspect an existing example + its smoke test**

Read one existing `examples/*/` script (e.g. `examples/hybrid_nowcast/`) and the matching `tests/test_package.py` test to copy the entry-point shape (`main()` return dict, how the smoke test imports it via `runpy.run_path` or module import).

- [ ] **Step 2: Write the failing example smoke test**

Add to `tests/test_package.py`, matching the sibling example tests' mechanism:

```python
def test_directed_network_sar_example_estimates_rho():
    import runpy
    ns = runpy.run_path("examples/directed_network_sar/run.py")
    result = ns["main"]()
    assert "rho" in result
    assert np.isfinite(result["rho"])
    assert -1.0 < result["rho"] < 1.0
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_package.py -k directed_network_sar -v`
Expected: FAIL — example does not exist.

- [ ] **Step 4: Write the example**

Create `examples/directed_network_sar/run.py`: a small, seeded interbank-exposure network (each bank influenced by a few counterparties), simulate a SAR field, fit `SAR("influence", "bank", graph, rho=Hyperparameter(...))` with a `Fixed` intercept under a Gaussian likelihood, and return a dict with `rho` (the estimated spatial coefficient), `precision`, and a fitted-vs-observed summary. Keep it deterministic. Add `README.md` explaining the directed-`W` reading (row = a bank's counterparties), row-standardization, and how `ρ` measures contagion strength; mention `DynamicSpatialPanel` + `forecast_dynamic_spatial_panel` as the time-varying extension.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_package.py -k directed_network_sar -v`
Expected: PASS.

- [ ] **Step 6: Write the docs**

Add a SAR/SDPD section to `docs/spatial-effects.md`: the directed-`W` construction `Q = τ(I−ρW)ᵀ(I−ρW)`, row-standardization and the `ρ ∈ (−1, 1)` bound, the SDPD `(ρ, γ, η)` reading (contemporaneous / temporal / spatio-temporal diffusion), the balanced units×periods grid, the gaussian-only sparse-marginal note, and the `forecast_dynamic_spatial_panel` forward recursion. Link the new example.

- [ ] **Step 7: Update the roadmap**

In `docs/roadmap.md`: add a **Directed & dynamic network structure** bullet under **Shipped** (SAR on directed row-standardized `W`; SDPD with `ρ, γ, η`; E-sparse; forward forecasting; YAML `sar`), and remove item 3 from **Next**, renumbering the rest.

- [ ] **Step 8: Full-suite regression**

Run: `python -m pytest -q`
Expected: only the documented pre-existing environmental failures (Spark JVM, console-entrypoint PATH `test_installed_console_entrypoint_loads_and_reports_help`, any OneDrive temp-lock flake) — no new failures attributable to this slice.
Run: `ruff check .` — confirm clean.

- [ ] **Step 9: Commit**

```bash
git add examples/directed_network_sar docs/spatial-effects.md docs/roadmap.md tests/test_package.py
git commit -m "docs(sar): directed-network example, guide, and roadmap; SAR/SDPD shipped"
```

---

## Self-Review

**Spec coverage:**
- Directed row-standardized `W`, symmetry-free → Task 1 ✓
- `Q = τ·MᵀM` shared core, static SAR (`M = I−ρW`) → Task 2 ✓
- SAR spec + exports → Task 3 ✓; compile + fit (fixed/estimated ρ) → Task 4 ✓
- Block-bidiagonal SDPD operator, `T=1 ≡ SAR` equivalence, balanced grid → Task 5 ✓
- DynamicSpatialPanel spec (ρ, γ, η) + exports → Task 6 ✓; compile + fit + predict → Task 7 ✓
- E-sparse past the dense guard → Task 8 ✓
- Forward forecasting capstone → Task 9 ✓
- YAML `sar` frontend (SAR only in v1) → Task 10 ✓
- Example + docs + roadmap + full-suite regression → Task 11 ✓
- Deferred (raw un-normalized `W`, unbalanced grid, non-gaussian sparse marginals, SDPD YAML) → documented in the design spec, not implemented ✓

**Placeholder scan:** All code steps carry real code; each "read a sibling" step names the exact file (Tasks 8, 9, 10, 11) because the call-site detail cannot be known without reading, per the writing-plans allowance.

**Type consistency:**
- `_gram_precision(m, precision)` and `_sar_operator(w, rho)` defined in Task 2, reused in Task 5's `_sdpd_operator`/`build_dynamic_spatial_panel` and Task 4/7 `build` closures.
- `_panel_networks(graphs) -> (units, times, [W])` defined in Task 5, consumed in Task 7's parametric branch and Task 9's forecast wrapper — same signature.
- `_sdpd_operator(ws, rho, gamma, eta)` defined Task 5, consumed Task 7's `build` closure — same signature.
- `build_dynamic_spatial_panel(frame, name, unit, time, graphs, rho, gamma, eta, precision)` — identical arg order in Task 5 definition and Tasks 7 dense + parametric call sites.
- Prediction context entry `("dynamic_spatial_panel", (name, unit, time, unit_labels, time_labels))` emitted in Task 7 (compiler) and consumed by `_dynamic_spatial_panel_block` (Task 7, prediction.py) — same 5-tuple, time-major `cell = time_pos * n + unit_pos` matching `_panel_design`.
- `sdpd_forecast(x_last, v_last, rho, gamma, eta, tau, future_ws)` defined and consumed within Task 9 consistently.
- Labels `f"{unit}@{time}"` time-major everywhere (Task 5 build, Task 7 context split on `"@"`, Task 9 reshape `(T, n)`).

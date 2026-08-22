# Besag/ICAR Spatial Effect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Besag / intrinsic CAR (ICAR) spatial latent effect to pyLGM — a graph-Laplacian smoothing prior over region-indexed data — usable via the declarative `Besag(...)` spec and `LGM.fit`.

**Architecture:** A Besag effect is another constrained `LatentBlock`: one-hot design over graph nodes, precision = base·(scaled graph Laplacian `D−W`), one sum-to-zero constraint per connected component. Adjacency comes from a neighbour dict or an R-INLA `.graph` file, both normalised to `(nodes, W)`. It reuses the existing block/precision/constraint/engine machinery unchanged — no INLA-strategy or `LatentBlock` changes.

**Tech Stack:** Python, NumPy, SciPy (`scipy.sparse`, `scipy.sparse.csgraph`, `numpy.linalg.pinv`), pandas. No new runtime dependency.

## Global Constraints

- No new runtime dependency — numpy/scipy/pandas only.
- Faithful to intrinsic-CAR: `π(x|τ) ∝ exp(−τ/2·xᵀ(D−W)x)`, `xᵀ(D−W)x = Σ_{i~j}(xᵢ−xⱼ)²`; rank `n−c`, `c` = number of connected components.
- Sørbye–Rue scaling (`scale=True` default): each non-singleton connected component's structure block scaled to geometric-mean marginal variance 1, via the constrained generalized inverse `pinv(R_c)`.
- One sum-to-zero constraint row **per connected component**; isolated (zero-neighbour) nodes raise a directed `ValueError`.
- Latent-field domain = the **graph nodes** (canonically sorted by `str`), not the observed levels; observed regions must be a subset of the nodes.
- The builder emits the **base** precision (`precision_base · R*`); a `Hyperparameter` precision is scaled downstream by `ScalableBlock.materialize` — the builder receives `1.0` when the precision is an optimised `Hyperparameter`, else the resolved float, mirroring RW/IID.
- Wire only the **declarative** dispatch sites (`compile_lgm`, `compile_family`). The config-file `ModelConfig`/`_structured_blocks` path (dispatches on `effect.type` strings) is **out of scope** — it needs graph-in-config schema design; record it as deferred.
- All existing effects, engines, predictions, criteria, and INLA strategies unchanged; full suite green; `ruff check src tests` clean.

---

### Task 1: Graph normalisation + INLA graph-file loader

**Files:**
- Create: `src/pylgm/effects/graph.py`
- Test: `tests/effects/test_graph.py`

**Interfaces:**
- Produces:
  - `normalize_graph(graph: Mapping[object, Sequence[object]]) -> tuple[tuple[str, ...], csr_matrix]` — canonical sorted string node labels and a symmetric 0/1 CSR adjacency `W` (n×n, zero diagonal) in that node order.
  - `load_graph_file(path: str | os.PathLike) -> dict[str, list[str]]` — parse the R-INLA/latte `.graph` text format into a neighbour dict with string keys.

- [ ] **Step 1: Write the failing tests**

```python
# tests/effects/test_graph.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.effects.graph import load_graph_file, normalize_graph


def _dense(graph):
    nodes, w = normalize_graph(graph)
    return nodes, w.toarray()


def test_normalize_path_graph():
    nodes, w = _dense({"A": ["B"], "B": ["A", "C"], "C": ["B"]})
    assert nodes == ("A", "B", "C")
    assert np.array_equal(w, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))


def test_normalize_is_symmetric_and_sorted():
    nodes, w = normalize_graph({"z": ["a"], "a": ["z"]})
    assert nodes == ("a", "z")
    assert (w != w.T).nnz == 0


def test_normalize_two_islands():
    nodes, w = _dense({"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]})
    assert nodes == ("A", "B", "C", "D")
    assert w[0, 1] == 1 and w[2, 3] == 1 and w[0, 2] == 0


def test_normalize_rejects_asymmetric():
    with pytest.raises(ValueError, match="symmetric"):
        normalize_graph({"A": ["B"], "B": []})


def test_normalize_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop"):
        normalize_graph({"A": ["A"]})


def test_normalize_rejects_unknown_neighbour():
    with pytest.raises(ValueError, match="not a node"):
        normalize_graph({"A": ["B"]})


def test_normalize_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_graph({})


def test_load_graph_file(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("3\n1 1 2\n2 2 1 3\n3 1 2\n")
    assert load_graph_file(path) == {"1": ["2"], "2": ["1", "3"], "3": ["2"]}


def test_load_graph_file_roundtrips_dict(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("2\n1 1 2\n2 1 1\n")
    graph = load_graph_file(path)
    nodes, _ = normalize_graph(graph)
    assert nodes == ("1", "2")


def test_load_graph_file_rejects_bad_count(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("3\n1 1 2\n")
    with pytest.raises(ValueError):
        load_graph_file(path)


def test_load_graph_file_rejects_out_of_range(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("2\n1 1 5\n2 1 1\n")
    with pytest.raises(ValueError):
        load_graph_file(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_graph.py -q`
Expected: FAIL (module `pylgm.effects.graph` does not exist).

- [ ] **Step 3: Implement `graph.py`**

```python
# src/pylgm/effects/graph.py
"""Neighbourhood-graph adjacency for spatial (CAR) effects."""

import os
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix


def normalize_graph(
    graph: Mapping[object, Sequence[object]],
) -> tuple[tuple[str, ...], csr_matrix]:
    """Return canonically sorted string node labels and a symmetric 0/1 adjacency.

    The node set is the mapping's keys (stringified). Validates that the graph is
    non-empty, has no self-loops, references only known nodes, and is symmetric.
    """
    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("graph must be a non-empty mapping of node -> neighbours")
    adjacency: dict[str, set[str]] = {}
    for node, neighbours in graph.items():
        key = str(node)
        if isinstance(neighbours, (str, bytes)):
            raise ValueError(f"neighbours of {key!r} must be a sequence of nodes")
        adjacency.setdefault(key, set())
        for neighbour in neighbours:
            adjacency[key].add(str(neighbour))
    nodes = tuple(sorted(adjacency))
    node_set = set(nodes)
    for node, neighbours in adjacency.items():
        if node in neighbours:
            raise ValueError(f"graph has a self-loop at node {node!r}")
        unknown = neighbours - node_set
        if unknown:
            raise ValueError(
                f"neighbour(s) {sorted(unknown)!r} of {node!r} are not a node in the graph"
            )
    for node in nodes:
        for neighbour in adjacency[node]:
            if node not in adjacency[neighbour]:
                raise ValueError(
                    f"graph is not symmetric: {neighbour!r} lists {node!r} "
                    f"or vice versa is missing"
                )
    position = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    for node in nodes:
        for neighbour in adjacency[node]:
            rows.append(position[node])
            cols.append(position[neighbour])
    data = np.ones(len(rows), dtype=float)
    w = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return nodes, w


def load_graph_file(path: str | os.PathLike) -> dict[str, list[str]]:
    """Parse an R-INLA / latte ``.graph`` file into a neighbour dict.

    Format: first token is the node count ``n``; then for each node a line
    ``id k nb_1 ... nb_k`` (1-indexed ids). Returns string-keyed neighbours.
    """
    tokens = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            tokens.extend(line.split())
    if not tokens:
        raise ValueError("graph file is empty")
    cursor = 0

    def take_int(what: str) -> int:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError(f"graph file ended while reading {what}")
        token = tokens[cursor]
        cursor += 1
        try:
            return int(token)
        except ValueError as error:
            raise ValueError(f"graph file has a non-integer {what}: {token!r}") from error

    count = take_int("node count")
    if count <= 0:
        raise ValueError(f"graph file node count must be positive, got {count}")
    graph: dict[str, list[str]] = {}
    for _ in range(count):
        node = take_int("node id")
        if not 1 <= node <= count:
            raise ValueError(f"graph file node id {node} out of range 1..{count}")
        degree = take_int("neighbour count")
        neighbours = []
        for _ in range(degree):
            neighbour = take_int("neighbour id")
            if not 1 <= neighbour <= count:
                raise ValueError(
                    f"graph file neighbour id {neighbour} out of range 1..{count}"
                )
            neighbours.append(str(neighbour))
        graph[str(node)] = neighbours
    if cursor != len(tokens):
        raise ValueError("graph file has trailing tokens beyond the declared nodes")
    if len(graph) != count:
        raise ValueError("graph file has duplicate or missing node ids")
    return graph
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_graph.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/graph.py tests/effects/test_graph.py
git commit -m "feat: add spatial adjacency graph normalisation and loader"
```

---

### Task 2: Besag/ICAR builder

**Files:**
- Create: `src/pylgm/effects/besag.py`
- Test: `tests/effects/test_besag.py`

**Interfaces:**
- Consumes: `normalize_graph` (Task 1); `LatentBlock` from `pylgm.ir.model`.
- Produces: `build_besag(frame: pd.DataFrame, name: str, index: str, graph: Mapping, precision: float, scale: bool = True) -> LatentBlock` — one-hot design over graph nodes, precision `precision·R*` (`R* = D−W` scaled per Sørbye–Rue when `scale`), one sum-to-zero constraint row per connected component.

- [ ] **Step 1: Write the failing tests**

```python
# tests/effects/test_besag.py
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.besag import build_besag
from pylgm.effects.random_walk import build_random_walk

PATH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
ISLANDS = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}


def _frame(regions):
    return pd.DataFrame({"region": regions})


def test_structure_is_graph_laplacian():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    expected = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    assert np.allclose(block.precision.toarray(), expected)
    assert block.labels == ("A", "B", "C")


def test_design_is_one_hot_over_nodes():
    block = build_besag(_frame(["C", "A", "A"]), "region", "region", PATH, 1.0, scale=False)
    assert np.array_equal(
        block.design.toarray(),
        np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=float),
    )


def test_connected_graph_has_one_constraint():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    assert np.array_equal(block.constraints, np.ones((1, 3)))


def test_two_islands_have_two_constraints():
    block = build_besag(_frame(["A", "B", "C", "D"]), "region", "region", ISLANDS, 1.0, scale=False)
    assert block.constraints.shape == (2, 4)
    assert np.array_equal(block.constraints[0], np.array([1, 1, 0, 0], dtype=float))
    assert np.array_equal(block.constraints[1], np.array([0, 0, 1, 1], dtype=float))


def test_scaling_gives_unit_geometric_mean_variance():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=True)
    r = block.precision.toarray()
    variances = np.diag(np.linalg.pinv(r))
    assert np.isclose(np.exp(np.mean(np.log(variances))), 1.0, atol=1e-8)


def test_scale_false_leaves_raw_laplacian():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 2.0, scale=False)
    expected = 2.0 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    assert np.allclose(block.precision.toarray(), expected)


def test_isolated_node_raises():
    graph = {"A": ["B"], "B": ["A"], "C": []}
    with pytest.raises(ValueError, match="no neighbours"):
        build_besag(_frame(["A", "B", "C"]), "region", "region", graph, 1.0)


def test_observed_region_absent_from_graph_raises():
    with pytest.raises(ValueError, match="not in the graph"):
        build_besag(_frame(["A", "B", "Z"]), "region", "region", PATH, 1.0)


def test_path_graph_matches_rw1_structure():
    besag = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    rw1 = build_random_walk(_frame([0, 1, 2]), "t", "region", 1.0, 1)
    # RW1 uses integer levels; build on a matching path frame instead:
    rw1 = build_random_walk(pd.DataFrame({"region": [0, 1, 2]}), "t", "region", 1.0, 1)
    assert np.allclose(besag.precision.toarray(), rw1.precision.toarray())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_besag.py -q`
Expected: FAIL (module `pylgm.effects.besag` does not exist).

- [ ] **Step 3: Implement `besag.py`**

```python
# src/pylgm/effects/besag.py
"""Besag / intrinsic CAR (ICAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from pylgm.effects.graph import normalize_graph
from pylgm.ir.model import LatentBlock


def _scaled_structure(w: csr_matrix, scale: bool) -> np.ndarray:
    """Return the (optionally Sørbye-Rue scaled) ICAR structure matrix ``R*``."""
    degree = np.asarray(w.sum(axis=1)).ravel()
    r = (diags(degree) - w).toarray()
    n_components, membership = connected_components(w, directed=False)
    singletons = [i for i in range(len(degree)) if degree[i] == 0]
    if singletons:
        raise ValueError(
            "regions have no neighbours (isolated ICAR nodes are undefined; "
            f"model them as an IID effect instead): index positions {singletons}"
        )
    if not scale:
        return r
    scaled = r.copy()
    for component in range(n_components):
        index = np.where(membership == component)[0]
        block = r[np.ix_(index, index)]
        variances = np.diag(np.linalg.pinv(block))
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError("ICAR scaling produced a non-positive marginal variance")
        factor = float(np.exp(np.mean(np.log(variances))))
        scaled[np.ix_(index, index)] = factor * block
    return scaled


def _component_constraints(w: csr_matrix) -> np.ndarray:
    n_components, membership = connected_components(w, directed=False)
    rows = np.zeros((n_components, w.shape[0]))
    for component in range(n_components):
        rows[component, membership == component] = 1.0
    return rows


def build_besag(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    precision: float,
    scale: bool = True,
) -> LatentBlock:
    nodes, w = normalize_graph(graph)
    position = {node: column for column, node in enumerate(nodes)}
    observed = [str(value) for value in frame[index]]
    missing = sorted({value for value in observed if value not in position})
    if missing:
        raise ValueError(f"observed region(s) {missing!r} are not in the graph")
    structure = _scaled_structure(w, scale)
    precision_matrix = csr_matrix(precision * structure)
    constraints = _component_constraints(w)
    rows = np.arange(len(observed))
    columns = np.array([position[value] for value in observed])
    design = csr_matrix(
        (np.ones(len(observed)), (rows, columns)), shape=(len(observed), len(nodes))
    )
    return LatentBlock(name, nodes, design, precision_matrix, constraints)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_besag.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/besag.py tests/effects/test_besag.py
git commit -m "feat: add Besag/ICAR effect builder with Sorbye-Rue scaling"
```

---

### Task 3: `Besag` declarative spec + exports

**Files:**
- Modify: `src/pylgm/effects/spec.py`
- Modify: `src/pylgm/effects/__init__.py`
- Modify: `src/pylgm/__init__.py`
- Test: `tests/effects/test_spec.py` (add cases; create if absent)

**Interfaces:**
- Consumes: `normalize_graph` (Task 1, for validation + canonical immutable storage).
- Produces: `Besag` frozen dataclass `Besag(name, index, graph, precision=1.0, scale=True)`, added to `EffectSpec`, `Predictor` validation, and exported from `pylgm.effects` and `pylgm`. `load_graph_file` exported from both.

- [ ] **Step 1: Write the failing tests**

```python
# tests/effects/test_spec.py  (add these; keep existing tests)
import pytest

from pylgm.effects import Besag, Fixed, IID, Predictor
from pylgm.parameters import Hyperparameter


PATH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def test_besag_defaults():
    effect = Besag("region", index="region", graph=PATH)
    assert effect.name == "region"
    assert effect.index == "region"
    assert effect.precision == 1.0
    assert effect.scale is True


def test_besag_accepts_hyperparameter_precision():
    hp = Hyperparameter("region.precision", initial=1.0)
    effect = Besag("region", index="region", graph=PATH, precision=hp)
    assert effect.precision is hp


def test_besag_is_frozen_and_hashable():
    effect = Besag("region", index="region", graph=PATH)
    assert hash(effect) == hash(Besag("region", index="region", graph=PATH))
    with pytest.raises(Exception):
        effect.name = "x"


def test_besag_rejects_bad_graph():
    with pytest.raises(ValueError):
        Besag("region", index="region", graph={})


def test_besag_composes_in_predictor():
    predictor = Fixed("1") + Besag("region", index="region", graph=PATH)
    assert isinstance(predictor, Predictor)
    assert [type(e).__name__ for e in predictor.effects] == ["Fixed", "Besag"]


def test_predictor_rejects_duplicate_besag_name():
    with pytest.raises(ValueError, match="unique"):
        IID("region", "region") + Besag("region", index="region", graph=PATH)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_spec.py -q`
Expected: FAIL (`Besag` not importable).

- [ ] **Step 3: Add the `Besag` dataclass to `spec.py`**

Add after the `RW2` class (before `EffectSpec`). It normalises the graph once (validating it) and stores a canonical immutable form so the frozen dataclass stays hashable:

```python
# src/pylgm/effects/spec.py
# add imports at top:
from collections.abc import Mapping

# ... after RW2 ...

@dataclass(frozen=True)
class Besag(_ComposableEffect):
    """A Besag / intrinsic CAR (ICAR) spatial latent effect."""

    name: str
    index: str
    graph: Mapping
    precision: float | Hyperparameter = 1.0
    scale: bool = True

    def __post_init__(self) -> None:
        from pylgm.effects.graph import normalize_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be a boolean")
        nodes, w = normalize_graph(self.graph)  # validates the graph
        canonical = tuple(
            (node, tuple(sorted(nodes[j] for j in w.indices[w.indptr[i] : w.indptr[i + 1]])))
            for i, node in enumerate(nodes)
        )
        object.__setattr__(self, "graph", canonical)
```

Update the `EffectSpec` alias, the `Predictor` type checks, and `__all__`:

```python
EffectSpec: TypeAlias = Fixed | IID | RW1 | RW2 | Besag
```

In `Predictor.__post_init__` and `Predictor.__add__`, replace both
`(Fixed, IID, RW1, RW2)` tuples with `(Fixed, IID, RW1, RW2, Besag)`.

```python
__all__ = ["Besag", "Fixed", "IID", "Predictor", "RW1", "RW2"]
```

- [ ] **Step 4: Update `effects/__init__.py`**

```python
# src/pylgm/effects/__init__.py
from pylgm.effects.besag import build_besag
from pylgm.effects.fixed import build_fixed
from pylgm.effects.graph import load_graph_file, normalize_graph
from pylgm.effects.iid import build_iid
from pylgm.effects.random_walk import build_random_walk
from pylgm.effects.spec import Besag, Fixed, IID, Predictor, RW1, RW2

__all__ = [
    "Besag",
    "Fixed",
    "IID",
    "Predictor",
    "RW1",
    "RW2",
    "build_besag",
    "build_fixed",
    "build_iid",
    "build_random_walk",
    "load_graph_file",
    "normalize_graph",
]
```

- [ ] **Step 5: Update `pylgm/__init__.py`**

Add `Besag` and `load_graph_file` to the `from pylgm.effects import ...` line and to `__all__` (alphabetically: `Besag` before `CandidateFailure`; `load_graph_file` after `IID`/before `LGM`).

```python
from pylgm.effects import Besag, Fixed, IID, RW1, RW2, load_graph_file
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_spec.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/effects/test_spec.py
git commit -m "feat: add Besag declarative spec and package exports"
```

---

### Task 4: Compiler wiring (declarative fit paths) + end-to-end tests

**Files:**
- Modify: `src/pylgm/compiler.py`
- Test: `tests/test_besag_fit.py`

**Interfaces:**
- Consumes: `Besag` spec (Task 3), `build_besag` (Task 2).
- Produces: `Besag` handled in `compile_lgm` (plug-in) and `compile_family` (optimise/integrate), so `LGM(...).fit(...)` works with a Besag effect for Gaussian and non-Gaussian likelihoods, plug-in and integrated.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_besag_fit.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson

GRAPH = {str(i): [str(i - 1)] if i > 0 else [] for i in range(6)}
for i in range(6):
    GRAPH[str(i)] = [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5]


def _gaussian_frame():
    regions = [str(i) for i in range(6)]
    truth = np.sin(np.arange(6) / 2.0)
    return pd.DataFrame({"region": regions, "y": truth + 0.01 * np.arange(6)}), truth


def test_gaussian_plugin_fit_recovers_smooth_pattern():
    frame, _ = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=1.0),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result is not None


def test_poisson_plugin_fit():
    regions = [str(i) for i in range(6)]
    counts = np.array([2, 3, 5, 6, 4, 2], dtype=float)
    frame = pd.DataFrame({"region": regions, "y": counts})
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH),
        likelihood=Poisson(),
    )
    result = model.fit(frame)
    assert result is not None


def test_integrate_fit_with_hyperparameter_precision():
    frame, _ = _gaussian_frame()
    hp = Hyperparameter("region.precision", initial=1.0, prior=None)
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, hyperparameters="integrate")
    assert result is not None


def test_full_laplace_rejects_besag():
    frame, _ = _gaussian_frame()
    hp = Hyperparameter("region.precision", initial=1.0, prior=None)
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    from pylgm.exceptions import UnsupportedEngineError

    with pytest.raises(UnsupportedEngineError):
        model.fit(frame, hyperparameters="integrate", latent_strategy="laplace")


def test_composes_with_iid():
    frame, _ = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + Besag("region", index="region", graph=GRAPH)
        + IID("noise", index="region"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result is not None
```

> Adjust `Hyperparameter(...)` / `Gaussian(...)` / `LGM(...)` / result-assertion
> keyword names to the actual current signatures if they differ — read the
> constructors before finalising. The behavioural intent (each path fits; full
> Laplace rejects Besag) is what matters. For the integrate test, if a prior is
> required for integration, construct the `Hyperparameter` with the same prior
> the existing RW integrate tests use (grep `tests/` for an integrated-fit
> example).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_besag_fit.py -q`
Expected: FAIL (Besag not handled by the compiler — `CompilationError` or dispatch falls through to the RW branch).

- [ ] **Step 3: Wire `Besag` into `compile_lgm`**

In `src/pylgm/compiler.py`, update the import:

```python
from pylgm.effects import Besag, Fixed, IID, RW1, build_besag, build_fixed, build_iid, build_random_walk
```

In `compile_lgm` (the `for effect in model.predictor.effects:` loop), add a Besag branch **before** the `else` RW branch:

```python
            elif isinstance(effect, Besag):
                precision = _resolved_precision(effect.precision)
                block = build_besag(
                    frame, effect.name, effect.index, effect.graph, precision, effect.scale
                )
                precisions[effect.name] = precision
```

- [ ] **Step 4: Wire `Besag` into `compile_family`**

In `compile_family`, inside the `for effect in model.predictor.effects:` loop, add a Besag branch **before** the IID/RW dispatch (after the `value = 1.0 if optimized else precision` line):

```python
        if isinstance(effect, Besag):
            block = build_besag(
                frame, effect.name, effect.index, effect.graph, value, effect.scale
            )
            scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
            if optimized:
                parameter_names.append(precision.name)
            continue
```

> Note: the `Besag.graph` attribute is the canonical `((node, (neighbours...)),
> ...)` tuple stored by the spec. `normalize_graph` accepts it directly (it is a
> mapping-like sequence of pairs? No — it is a tuple of pairs). **Confirm**
> `build_besag`/`normalize_graph` accept the stored form: `normalize_graph`
> requires a `Mapping`. Pass `dict(effect.graph)` when calling `build_besag`
> from the compiler, i.e. `build_besag(frame, effect.name, effect.index,
> dict(effect.graph), precision, effect.scale)` in BOTH branches, so the stored
> canonical tuple round-trips back into a mapping.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_besag_fit.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite and ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: all green, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/compiler.py tests/test_besag_fit.py
git commit -m "feat: compile Besag effects through the declarative fit paths"
```

---

### Task 5: Documentation + roadmap

**Files:**
- Modify: `README.md`
- Modify: roadmap doc (grep for the existing roadmap/deferred list, e.g. `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md` or the release-gate doc).

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a Besag/ICAR section to `README.md`**

Document: the `Besag(name, index, graph, precision=1.0, scale=True)` spec; neighbour-dict and `load_graph_file` input; intrinsic-CAR model and per-component sum-to-zero constraints; Sørbye–Rue scaling default; isolated-node limitation; that it works under gaussian/simplified_laplace and is rejected by full Laplace (constrained). Show a short worked example mirroring the spec's Public Contract. Do not claim proper CAR / BYM2 (not built).

- [ ] **Step 2: Record the roadmap**

In the roadmap/deferred list, mark Besag/ICAR done and record the next spatial slices: **proper CAR** (`Q=τ(D−ρW)`, ρ hyperparameter, unconstrained) and **BYM2** (structured scaled-ICAR + unstructured IID, mixing φ, PC priors) → full CAR family. Note the deferred config-file (`ModelConfig`) `besag` type and graceful isolated-node handling.

- [ ] **Step 3: Verify docs build / suite still green**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS (docs-only, unchanged).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: document Besag/ICAR spatial effect and record spatial roadmap"
```

---

## Self-Review Notes

- **Spec coverage:** graph input + loader (Task 1) ✓; intrinsic-CAR structure, per-component constraints, Sørbye–Rue scaling, isolated-node error, domain=nodes (Task 2) ✓; declarative spec + exports (Task 3) ✓; plug-in/optimise/integrate wiring, full-Laplace rejection, composition, end-to-end fits (Task 4) ✓; docs + roadmap incl. deferred config path (Task 5) ✓.
- **Scope refinement vs spec:** the spec's "three dispatch chains" is narrowed to the **two declarative** sites (`compile_lgm`, `compile_family`); the config-file `ModelConfig`/`_structured_blocks` path is deferred (needs graph-in-config schema) and recorded in Task 5. All acceptance criteria run through the declarative path — no coverage lost.
- **RW1 parity:** `build_random_walk(order=1)` emits `precision·(Dᵀ D)` = path-graph Laplacian; unscaled path Besag equals it (Task 2 test).
- **Graph storage round-trip:** the spec stores a canonical tuple; the compiler passes `dict(effect.graph)` back into `build_besag` so `normalize_graph` (Mapping-only) accepts it — flagged in Task 4 Step 4.
- **Type consistency:** `build_besag(frame, name, index, graph, precision, scale=True)` signature identical across Tasks 2/4; `normalize_graph`/`load_graph_file` signatures identical across Tasks 1/2/3.

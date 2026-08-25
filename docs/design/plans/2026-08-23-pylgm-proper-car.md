# Proper CAR (plug-in ρ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the proper-CAR spatial latent effect `Q = τ(D − ρW)` with a fixed (plug-in) ρ, usable via the declarative `ProperCAR(...)` spec and `LGM.fit`.

**Architecture:** With ρ fixed, `Q = τ·(D − ρW)` is a scalar multiple of the constant base matrix `M = D − ρW`, so a proper-CAR effect is a normal unconstrained `LatentBlock` that flows through the existing `ScalableBlock` scalar-multiply machinery (τ optimise/integrate) with no family-layer changes. It reuses the Besag adjacency module (`normalize_graph`). Because it is full-rank/unconstrained, it also works under full Laplace.

**Tech Stack:** Python, NumPy (`eigvalsh`), SciPy (`scipy.sparse`), pandas. No new runtime dependency.

## Global Constraints

- No new runtime dependency — numpy/scipy/pandas only.
- Faithful to proper CAR: `Q = τ(D − ρW)`, full-rank/proper, **no constraints**.
- ρ is a fixed float in this slice; validity interval `ρ ∈ (1/μ_min, 1/μ_max)`, `μ` = eigenvalues of the symmetric normalized adjacency `N = D^{−1/2} W D^{−1/2}`. Out-of-range ρ and a `Hyperparameter` ρ both raise directed `ValueError`s.
- ρ estimation (bounded EB/INLA transform) is OUT of scope — deferred to the bounded-hyperparameter-inference slice.
- No Sørbye–Rue scaling (intrinsic-only device).
- τ (`precision`) is a float or `Hyperparameter`, handled exactly like IID/RW/Besag via `ScalableBlock` (the builder receives `1.0` when τ is an optimised `Hyperparameter`, else the resolved float).
- Latent-field domain = the graph nodes (canonical sorted `str` order); observed regions must be a subset. Isolated (zero-neighbour) nodes raise a directed error (as Besag).
- Wire only the two declarative dispatch sites (`compile_lgm`, `compile_family`). The config-file `_structured_blocks` path is out of scope.
- Reuse `pylgm.effects.graph.normalize_graph`; do not duplicate adjacency handling.
- All existing effects (incl. Besag), engines, family layer, predictions, criteria, and INLA strategies unchanged; full suite green; `ruff check src tests` clean.

---

### Task 1: Proper-CAR builder

**Files:**
- Create: `src/pylgm/effects/proper_car.py`
- Test: `tests/effects/test_proper_car.py`

**Interfaces:**
- Consumes: `normalize_graph` from `pylgm.effects.graph`; `LatentBlock` from `pylgm.ir.model`.
- Produces: `build_proper_car(frame: pd.DataFrame, name: str, index: str, graph: Mapping, rho: float, precision: float) -> LatentBlock` — one-hot design over graph nodes, precision `precision·(D − ρW)`, **empty** constraints.

- [ ] **Step 1: Write the failing tests**

```python
# tests/effects/test_proper_car.py
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from pylgm.effects.proper_car import build_proper_car

PATH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def _frame(regions):
    return pd.DataFrame({"region": regions})


def test_structure_is_d_minus_rho_w():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.5, 1.0)
    d = np.diag([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(block.precision.toarray(), d - 0.5 * w)
    assert block.labels == ("A", "B", "C")


def test_no_constraints():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.5, 1.0)
    assert block.constraints.shape == (0, 3)


def test_design_is_one_hot_over_nodes():
    block = build_proper_car(_frame(["C", "A"]), "region", "region", PATH, 0.5, 1.0)
    assert np.array_equal(
        block.design.toarray(), np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    )


def test_rho_zero_gives_degree_matrix():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.0, 1.0)
    assert np.allclose(block.precision.toarray(), np.diag([1.0, 2.0, 1.0]))


def test_precision_scales_the_structure():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.5, 3.0)
    d = np.diag([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(block.precision.toarray(), 3.0 * (d - 0.5 * w))


def test_in_range_rho_is_positive_definite():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.9, 1.0)
    cho_factor(block.precision.toarray())  # raises if not PD


def test_out_of_range_rho_raises_naming_interval():
    with pytest.raises(ValueError, match=r"rho must lie in"):
        build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 1.5, 1.0)


def test_rho_at_upper_boundary_raises():
    # rho = 1 is the singular intrinsic boundary (mu_max = 1 for a path graph).
    with pytest.raises(ValueError, match=r"rho must lie in"):
        build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, 1.0)


def test_isolated_node_raises():
    graph = {"A": ["B"], "B": ["A"], "C": []}
    with pytest.raises(ValueError, match="no neighbours"):
        build_proper_car(_frame(["A", "B", "C"]), "region", "region", graph, 0.5, 1.0)


def test_observed_region_absent_from_graph_raises():
    with pytest.raises(ValueError, match="not in the graph"):
        build_proper_car(_frame(["A", "B", "Z"]), "region", "region", PATH, 0.5, 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_proper_car.py -q`
Expected: FAIL (module `pylgm.effects.proper_car` does not exist).

- [ ] **Step 3: Implement `proper_car.py`**

```python
# src/pylgm/effects/proper_car.py
"""Proper conditional autoregressive (proper CAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from pylgm.effects.graph import normalize_graph
from pylgm.ir.model import LatentBlock


def _validity_interval(w: csr_matrix, degree: np.ndarray) -> tuple[float, float]:
    """Return the open interval (1/mu_min, 1/mu_max) of valid rho for D - rho W."""
    inv_sqrt = 1.0 / np.sqrt(degree)
    dense = w.toarray()
    normalized = (dense * inv_sqrt[:, None]) * inv_sqrt[None, :]
    normalized = 0.5 * (normalized + normalized.T)  # symmetrize against round-off
    mu = np.linalg.eigvalsh(normalized)
    mu_min = float(mu[0])
    mu_max = float(mu[-1])
    lower = 1.0 / mu_min if mu_min < 0 else float("-inf")
    upper = 1.0 / mu_max if mu_max > 0 else float("inf")
    return lower, upper


def build_proper_car(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    rho: float,
    precision: float,
) -> LatentBlock:
    nodes, w = normalize_graph(graph)
    degree = np.asarray(w.sum(axis=1)).ravel()
    isolated = [nodes[i] for i in range(len(degree)) if degree[i] == 0]
    if isolated:
        raise ValueError(
            "regions have no neighbours (proper CAR is singular at isolated nodes; "
            f"model them as an IID effect instead): {isolated!r}"
        )
    position = {node: column for column, node in enumerate(nodes)}
    observed = [str(value) for value in frame[index]]
    missing = sorted({value for value in observed if value not in position})
    if missing:
        raise ValueError(f"observed region(s) {missing!r} are not in the graph")
    lower, upper = _validity_interval(w, degree)
    if not lower < rho < upper:
        raise ValueError(
            f"rho must lie in the open interval ({lower:.6g}, {upper:.6g}) for "
            f"D - rho*W to be positive definite; got {rho}"
        )
    structure = (diags(degree) - rho * w).tocsr()
    precision_matrix = csr_matrix(precision * structure)
    constraints = np.empty((0, len(nodes)))
    rows = np.arange(len(observed))
    columns = np.array([position[value] for value in observed])
    design = csr_matrix(
        (np.ones(len(observed)), (rows, columns)), shape=(len(observed), len(nodes))
    )
    return LatentBlock(name, nodes, design, precision_matrix, constraints)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_proper_car.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/proper_car.py tests/effects/test_proper_car.py
git commit -m "feat: add proper CAR effect builder with rho validity check"
```

---

### Task 2: `ProperCAR` declarative spec + exports

**Files:**
- Modify: `src/pylgm/effects/spec.py`
- Modify: `src/pylgm/effects/__init__.py`
- Modify: `src/pylgm/__init__.py`
- Test: `tests/effects/test_spec.py` (add cases)

**Interfaces:**
- Consumes: `normalize_graph` (validation + canonical storage); `Hyperparameter`.
- Produces: `ProperCAR(name, index, graph, rho, precision=1.0)` frozen dataclass, added to `EffectSpec`, `Predictor` validation, and exported from `pylgm.effects` and `pylgm`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/effects/test_spec.py  (append; keep existing tests)
def test_proper_car_defaults():
    from pylgm.effects import ProperCAR

    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    assert effect.name == "region"
    assert effect.index == "region"
    assert effect.rho == 0.5
    assert effect.precision == 1.0


def test_proper_car_accepts_hyperparameter_precision():
    from pylgm.effects import ProperCAR

    hp = Hyperparameter("region.precision", initial=1.0)
    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5, precision=hp)
    assert effect.precision is hp


def test_proper_car_rejects_hyperparameter_rho():
    from pylgm.effects import ProperCAR

    hp = Hyperparameter("region.rho", initial=0.5)
    with pytest.raises(ValueError, match="rho"):
        ProperCAR("region", index="region", graph=PATH_GRAPH, rho=hp)


def test_proper_car_rejects_non_finite_rho():
    from pylgm.effects import ProperCAR

    with pytest.raises(ValueError, match="rho"):
        ProperCAR("region", index="region", graph=PATH_GRAPH, rho=float("inf"))


def test_proper_car_is_frozen_and_hashable():
    from pylgm.effects import ProperCAR

    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    assert hash(effect) == hash(
        ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    )
    with pytest.raises(FrozenInstanceError):
        effect.rho = 0.1


def test_proper_car_composes_in_predictor():
    from pylgm.effects import Predictor, ProperCAR

    predictor = Fixed("1") + ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    assert isinstance(predictor, Predictor)
    assert [type(e).__name__ for e in predictor.effects] == ["Fixed", "ProperCAR"]
```

> `PATH_GRAPH`, `Fixed`, `Hyperparameter`, `FrozenInstanceError` are already
> imported/defined in `test_spec.py` from the Besag task. If `PATH_GRAPH` is not
> present, add `PATH_GRAPH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}` near the
> top.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_spec.py -q`
Expected: FAIL (`ProperCAR` not importable).

- [ ] **Step 3: Add the `ProperCAR` dataclass to `spec.py`**

Add a finite-real helper near the other validators (after `_positive_real`):

```python
def _finite_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    return float(value)
```

Add the dataclass after `Besag` (before `EffectSpec`):

```python
@dataclass(frozen=True)
class ProperCAR(_ComposableEffect):
    """A proper conditional autoregressive (proper CAR) spatial latent effect."""

    name: str
    index: str
    graph: Mapping
    rho: float
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.graph import normalize_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.rho, Hyperparameter):
            raise ValueError(
                "rho estimation is not yet supported (plug-in only); pass a float. "
                "rho inference is planned in the bounded-hyperparameter slice"
            )
        object.__setattr__(self, "rho", _finite_real(self.rho, "rho"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        nodes, w = normalize_graph(self.graph)  # validates the graph
        canonical = tuple(
            (node, tuple(sorted(nodes[j] for j in w.indices[w.indptr[i] : w.indptr[i + 1]])))
            for i, node in enumerate(nodes)
        )
        object.__setattr__(self, "graph", canonical)
```

Update the `EffectSpec` alias, both `Predictor` isinstance tuples, and `__all__`:

```python
EffectSpec: TypeAlias = Fixed | IID | RW1 | RW2 | Besag | ProperCAR
```
Replace both `(Fixed, IID, RW1, RW2, Besag)` tuples in `Predictor` with
`(Fixed, IID, RW1, RW2, Besag, ProperCAR)`.
```python
__all__ = ["Besag", "Fixed", "IID", "Predictor", "ProperCAR", "RW1", "RW2"]
```

- [ ] **Step 4: Update `effects/__init__.py`**

Add `build_proper_car` to the imports and `ProperCAR` to the spec import, and both to `__all__`:

```python
from pylgm.effects.proper_car import build_proper_car
from pylgm.effects.spec import Besag, Fixed, IID, Predictor, ProperCAR, RW1, RW2
```
Add `"ProperCAR"` and `"build_proper_car"` to `__all__` (keep it sorted as the file already is).

- [ ] **Step 5: Update `pylgm/__init__.py`**

```python
from pylgm.effects import Besag, Fixed, IID, ProperCAR, RW1, RW2, load_graph_file
```
Add `"ProperCAR"` to `__all__` (after `"Poisson"`, before `"RW1"`).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/effects/test_spec.py -q`
Expected: PASS.

- [ ] **Step 7: Verify imports + full suite + ruff**

Run: `PYTHONPATH=src python -c "from pylgm import ProperCAR; print(ProperCAR('r', index='r', graph={'A':['B'],'B':['A']}, rho=0.5))"`
Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: import works; suite green; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/effects/test_spec.py
git commit -m "feat: add ProperCAR declarative spec and package exports"
```

---

### Task 3: Compiler wiring (declarative fit paths) + end-to-end tests

**Files:**
- Modify: `src/pylgm/compiler.py`
- Test: `tests/test_proper_car_fit.py`

**Interfaces:**
- Consumes: `ProperCAR` spec (Task 2), `build_proper_car` (Task 1).
- Produces: `ProperCAR` handled in `compile_lgm` and `compile_family`, so `LGM(...).fit(...)` works with a proper-CAR effect (plug-in and τ-integrate), including under full Laplace.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_proper_car_fit.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson, ProperCAR
from pylgm.priors import PCPrecision

GRAPH = {
    str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5]
    for i in range(6)
}


def _gaussian_frame():
    regions = [str(i) for i in range(6)]
    truth = np.sin(np.arange(6) / 2.0)
    return pd.DataFrame({"region": regions, "y": truth + 0.01 * np.arange(6)}), truth


def test_gaussian_plugin_fit():
    frame, truth = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + ProperCAR("region", index="region", graph=GRAPH, rho=0.9),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_poisson_plugin_fit():
    regions = [str(i) for i in range(6)]
    counts = np.array([2, 3, 5, 6, 4, 2], dtype=float)
    frame = pd.DataFrame({"region": regions, "y": counts})
    model = LGM(
        response="y",
        predictor=Fixed("1") + ProperCAR("region", index="region", graph=GRAPH, rho=0.9),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_integrate_over_tau_with_fixed_rho():
    frame, _ = _gaussian_frame()
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + ProperCAR("region", index="region", graph=GRAPH, rho=0.9, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    assert result.engine == "inla"
    assert "region.precision" in result.hyperparameter_marginals()
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)


def test_full_laplace_accepts_proper_car():
    # Proper CAR is unconstrained, so (unlike Besag) full Laplace must SUCCEED.
    frame, _ = _gaussian_frame()
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + ProperCAR("region", index="region", graph=GRAPH, rho=0.9, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(
        frame, engine="exact_gaussian", hyperparameters="integrate", latent_strategy="laplace"
    )
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_composes_with_iid():
    frame, _ = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + ProperCAR("region", index="region", graph=GRAPH, rho=0.9)
        + IID("noise", index="region"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.latent_marginals("region").mean.shape == (6,)
    assert result.latent_marginals("noise").mean.shape == (6,)
```

> Confirm signatures against the current code before finalising (as in the Besag
> fit tests). The `latent_marginals`/`hyperparameter_marginals`/`engine`
> attributes and the `PCPrecision`-prior integrate pattern are the same ones the
> Besag fit tests (`tests/test_besag_fit.py`) use — mirror them.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_proper_car_fit.py -q`
Expected: FAIL (ProperCAR not handled by the compiler).

- [ ] **Step 3: Wire `ProperCAR` into `compile_lgm`**

In `src/pylgm/compiler.py`, update the effects import to add `ProperCAR` and `build_proper_car`:

```python
from pylgm.effects import (
    Besag, Fixed, IID, ProperCAR, RW1,
    build_besag, build_fixed, build_iid, build_proper_car, build_random_walk,
)
```

In `compile_lgm`'s effect loop, add a `ProperCAR` branch alongside the `Besag` branch:

```python
            elif isinstance(effect, ProperCAR):
                precision = _resolved_precision(effect.precision)
                block = build_proper_car(
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, precision
                )
                precisions[effect.name] = precision
```

- [ ] **Step 4: Wire `ProperCAR` into `compile_family`**

In `compile_family`'s effect loop, add a `ProperCAR` branch alongside the `Besag` branch (after `value = 1.0 if optimized else precision`):

```python
        if isinstance(effect, ProperCAR):
            block = build_proper_car(
                frame, effect.name, effect.index, dict(effect.graph), effect.rho, value
            )
            scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
            if optimized:
                parameter_names.append(precision.name)
            continue
```

> As with Besag, pass `dict(effect.graph)` (the spec stores a canonical tuple;
> `build_proper_car`/`normalize_graph` need a Mapping).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_proper_car_fit.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite and ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: all green, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/compiler.py tests/test_proper_car_fit.py
git commit -m "feat: compile ProperCAR effects through the declarative fit paths"
```

---

### Task 4: Documentation + roadmap

**Files:**
- Modify: `README.md`
- Modify: the spatial roadmap note added in the Besag slice (grep `README.md` and `docs/` for "proper CAR" / "Spatial roadmap").

**Interfaces:** none (docs only).

- [ ] **Step 1: Add a proper-CAR section to `README.md`**

Document: `ProperCAR(name, index, graph, rho, precision=1.0)`; `Q = τ(D − ρW)`, full-rank/no constraints; ρ is a fixed float in the valid interval `(1/μ_min, 1/μ_max)` (this slice), with ρ estimation deferred; τ inferred like other effects; **works under full Laplace** (contrast Besag). Show a short runnable example mirroring the spec's Public Contract. Do NOT claim ρ estimation or BYM2.

- [ ] **Step 2: Update the roadmap**

Mark proper CAR (plug-in ρ) done. Record next: **bounded-hyperparameter inference** (per-parameter transforms for EB + INLA) unlocking ρ estimation AND BYM2's φ; then **BYM2**. Keep the deferred config-file type and graceful isolated-node notes.

- [ ] **Step 3: Verify the example runs and the suite is green**

Run the README example under `PYTHONPATH=src` to confirm it works; then `PYTHONPATH=src python -m pytest -q`.
Expected: example runs; suite green.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: document proper CAR effect and update spatial roadmap"
```

---

## Self-Review Notes

- **Spec coverage:** builder with `D−ρW`, ρ-range validity, isolated-node/observed errors, empty constraints (Task 1) ✓; `ProperCAR` spec + rho float/Hyperparameter validation + exports (Task 2) ✓; two-site compiler wiring + plug-in/integrate/full-Laplace-accept/composition e2e (Task 3) ✓; docs + roadmap (Task 4) ✓.
- **Reuse:** `normalize_graph` reused (no new graph code); τ uses the existing `ScalableBlock` scalar-multiply path (no family-layer changes) — the whole point of plug-in ρ.
- **Full-Laplace contrast:** proper CAR is unconstrained (`constraints.shape == (0, n)`), so the full-Laplace guard passes — Task 3 asserts it SUCCEEDS (Besag asserts it is rejected).
- **Type consistency:** `build_proper_car(frame, name, index, graph, rho, precision)` identical across Tasks 1/3; `_finite_real` added once in Task 2; `dict(effect.graph)` round-trip mirrors the Besag wiring.
- **ρ validity math:** `D − ρW ≻ 0 ⇔ ρ ∈ (1/μ_min, 1/μ_max)`, `μ = eigvalsh(D^{−1/2}WD^{−1/2})`; for a graph with edges `trace(N)=0 ⇒ μ_min<0<μ_max`, so the interval is `(negative, 1/μ_max)` — the `float('-inf')`/`float('inf')` guards cover the degenerate no-negative/no-positive cases defensively.

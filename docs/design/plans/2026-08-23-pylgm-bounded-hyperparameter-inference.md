# Bounded-Hyperparameter Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize hyperparameter inference from positive/log-only to arbitrary bounded parameters via a per-parameter transform abstraction, and use it to estimate the proper-CAR spatial dependence ρ (empirical-Bayes optimise and INLA integrate).

**Architecture:** A `Transform` (Log for positive τ/σ, scaled Logit for bounded ρ) maps the natural scale θ to the unconstrained internal scale `u` the optimiser and INLA grid work in. `OptimizationBounds` and the compiled families carry the transform; the optimiser and grid become transform-driven, including the INLA importance-weight Jacobian `Σ log|dθ/du|`. A `ParametricBlock` represents `τ(D−ρW)` (not a scalar multiple) alongside `ScalableBlock`. The `LogTransform` path reproduces today's math exactly.

**Tech Stack:** Python, NumPy, SciPy (`scipy.special.expit`/`log_expit`, `scipy.optimize`), pandas. No new runtime dependency.

## Global Constraints

- No new runtime dependency — numpy/scipy/pandas only (`scipy.special.log_expit` is available).
- **Back-compatibility is the primary risk:** the `LogTransform` path MUST be numerically identical to the current behaviour. `LogTransform.log_abs_jacobian(u) == u` (so the INLA Jacobian `Σ u` is preserved); the full existing suite must stay green, plus an explicit equivalence test.
- Transforms are scalar and per-parameter: `to_internal(θ)`, `from_internal(u)`, `log_abs_jacobian(u) = log|dθ/du|`, `contains(θ)`.
- `LogitTransform(a,b)`: `θ = a+(b−a)σ(u)`, `u = log((θ−a)/(b−θ))`, `log_abs_jacobian(u) = log(b−a)+log σ(u)+log(1−σ(u))`; domain `(a,b)`; box bounds must be strictly inside.
- `ScalableBlock` behaviour and all existing effects are unchanged; `ParametricBlock` is additive.
- ρ's validity interval `(1/μ_min, 1/μ_max)` is graph-dependent and computed by the compiler (reusing proper CAR's `_validity_interval`); the family carries `parameter_bounds`.
- Wire only the declarative path (`compile_family`); config-file path out of scope. Plug-in ρ (float) keeps the existing `ScalableBlock` path.
- Engines (`fit_gaussian`/`fit_laplace`) and latent strategies unchanged; `ruff check src tests` clean.

---

### Task 1: Transform abstraction

**Files:**
- Create: `src/pylgm/optimization/transforms.py`
- Test: `tests/optimization/test_transforms.py`

**Interfaces:**
- Produces: `Transform` (Protocol), `LogTransform`, `LogitTransform(lower, upper)` with `to_internal`, `from_internal`, `log_abs_jacobian`, `contains`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_transforms.py
import numpy as np
import pytest

from pylgm.optimization.transforms import LogitTransform, LogTransform


def _fd_log_jac(transform, u, h=1e-6):
    # finite-difference of log|d theta / d u|
    dtheta = (transform.from_internal(u + h) - transform.from_internal(u - h)) / (2 * h)
    return np.log(abs(dtheta))


def test_log_roundtrip_and_jacobian():
    t = LogTransform()
    for theta in (1e-3, 1.0, 25.0, 1e3):
        assert np.isclose(t.from_internal(t.to_internal(theta)), theta)
    for u in (-5.0, 0.0, 3.0):
        assert np.isclose(t.log_abs_jacobian(u), u)              # log path anchor
        assert np.isclose(t.log_abs_jacobian(u), _fd_log_jac(t, u), atol=1e-5)


def test_log_domain():
    t = LogTransform()
    assert t.contains(0.5)
    assert not t.contains(0.0)
    assert not t.contains(-1.0)


def test_logit_roundtrip_and_range():
    t = LogitTransform(-1.0, 1.0)
    for theta in (-0.9, -0.3, 0.0, 0.4, 0.95):
        assert np.isclose(t.from_internal(t.to_internal(theta)), theta)
    for u in (-8.0, -1.0, 0.0, 2.0, 9.0):
        theta = t.from_internal(u)
        assert -1.0 < theta < 1.0


def test_logit_jacobian_matches_finite_difference():
    t = LogitTransform(0.0, 1.0)
    for u in (-3.0, -0.5, 0.0, 1.5, 4.0):
        assert np.isclose(t.log_abs_jacobian(u), _fd_log_jac(t, u), atol=1e-5)


def test_logit_domain_and_validation():
    t = LogitTransform(-1.0, 1.0)
    assert t.contains(0.0)
    assert not t.contains(-1.0)
    assert not t.contains(1.0)
    assert not t.contains(2.0)
    with pytest.raises(ValueError):
        LogitTransform(1.0, 1.0)
    with pytest.raises(ValueError):
        LogitTransform(2.0, 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_transforms.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `transforms.py`**

```python
# src/pylgm/optimization/transforms.py
"""Per-parameter transforms between the natural and unconstrained scales."""

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import expit, log_expit


@runtime_checkable
class Transform(Protocol):
    """Maps a natural parameter theta to an unconstrained internal value u."""

    def to_internal(self, theta: float) -> float: ...
    def from_internal(self, u: float) -> float: ...
    def log_abs_jacobian(self, u: float) -> float: ...  # log|d theta / d u|
    def contains(self, theta: float) -> bool: ...


class LogTransform:
    """Positive parameters: u = log(theta), theta = exp(u)."""

    def to_internal(self, theta: float) -> float:
        return float(np.log(theta))

    def from_internal(self, u: float) -> float:
        return float(np.exp(u))

    def log_abs_jacobian(self, u: float) -> float:
        return float(u)

    def contains(self, theta: float) -> bool:
        return np.isfinite(theta) and theta > 0.0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogTransform)

    def __hash__(self) -> int:
        return hash(LogTransform)


class LogitTransform:
    """Bounded parameters on (lower, upper): theta = lower + (upper-lower)*sigmoid(u)."""

    def __init__(self, lower: float, upper: float) -> None:
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
            raise ValueError("LogitTransform requires finite lower < upper")
        self.lower = float(lower)
        self.upper = float(upper)

    @property
    def _width(self) -> float:
        return self.upper - self.lower

    def to_internal(self, theta: float) -> float:
        z = (theta - self.lower) / self._width
        return float(np.log(z) - np.log1p(-z))

    def from_internal(self, u: float) -> float:
        return float(self.lower + self._width * expit(u))

    def log_abs_jacobian(self, u: float) -> float:
        # d theta/d u = width * sigmoid(u) * (1 - sigmoid(u))
        return float(np.log(self._width) + log_expit(u) + log_expit(-u))

    def contains(self, theta: float) -> bool:
        return np.isfinite(theta) and self.lower < theta < self.upper

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LogitTransform)
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def __hash__(self) -> int:
        return hash((LogitTransform, self.lower, self.upper))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_transforms.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/optimization/transforms.py tests/optimization/test_transforms.py
git commit -m "feat: add per-parameter Log and Logit transforms"
```

---

### Task 2: Transform-aware `OptimizationBounds` + optimiser

**Files:**
- Modify: `src/pylgm/optimization/empirical_bayes.py`
- Modify: `src/pylgm/optimization/__init__.py` (export `LogTransform`/`LogitTransform` if convenient)
- Test: `tests/optimization/test_empirical_bayes.py` (add cases; do not break existing)

**Interfaces:**
- Consumes: `Transform`, `LogTransform`, `LogitTransform` (Task 1).
- Produces: `OptimizationBounds(initial, lower, upper, transform=LogTransform())`; `optimize_empirical_bayes` drives per-parameter transforms. Signature otherwise unchanged.

- [ ] **Step 1: Write the failing tests**

Add a bounded-parameter optimisation test and a back-compat assertion. A minimal synthetic family with a quadratic-in-internal objective is enough to exercise a `LogitTransform` parameter.

```python
# tests/optimization/test_empirical_bayes.py  (append)
import numpy as np

from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes
from pylgm.optimization.transforms import LogitTransform, LogTransform


class _ScalarFamily:
    """Family whose 'log marginal likelihood' peaks at a known natural value."""

    def __init__(self, name, target):
        self.parameter_names = (name,)
        self._name = name
        self._target = target

    def materialize(self, values):
        return values[self._name]

    # optimize_empirical_bayes calls fit(model); we inject a fit that reads the value.


def test_optimizer_recovers_bounded_optimum():
    name = "rho"
    target = 0.6

    class _Fit:
        def __init__(self, value):
            # concave in value, peak at target, within (-1, 1)
            self.log_marginal_likelihood = -((value - target) ** 2)

    fam = _ScalarFamily(name, target)
    bounds = {name: OptimizationBounds(0.0, -0.99, 0.99, transform=LogitTransform(-1.0, 1.0))}
    result = optimize_empirical_bayes(fam, bounds, fit=lambda m, **k: _Fit(m))
    assert np.isclose(result.parameters[name], target, atol=1e-3)


def test_optimization_bounds_defaults_to_log_and_validates():
    b = OptimizationBounds(1.0, 0.1, 10.0)
    assert isinstance(b.transform, LogTransform)
    import pytest
    with pytest.raises(ValueError):
        OptimizationBounds(1.0, -0.1, 10.0)  # negative lower invalid under Log
    # Logit bounds must sit inside the interval:
    with pytest.raises(ValueError):
        OptimizationBounds(0.0, -1.0, 0.9, transform=LogitTransform(-1.0, 1.0))
```

> Read the EXISTING `optimize_empirical_bayes` tests first; keep them all passing
> unchanged (they use the default Log path). If `_ScalarFamily.materialize`
> returning a bare float doesn't match how `fit` is called, adapt the stub so the
> injected `fit` receives the materialised value — the intent is a 1-D bounded
> optimisation with a known interior optimum.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_empirical_bayes.py -q`
Expected: new tests FAIL (no `transform` kwarg).

- [ ] **Step 3: Add `transform` to `OptimizationBounds`**

```python
# empirical_bayes.py
from dataclasses import dataclass, field
from pylgm.optimization.transforms import LogTransform, Transform

@dataclass(frozen=True)
class OptimizationBounds:
    initial: float
    lower: float
    upper: float
    transform: Transform = field(default_factory=LogTransform)

    def __post_init__(self) -> None:
        initial = float(self.initial)
        lower = float(self.lower)
        upper = float(self.upper)
        if not self.transform.contains(lower) or not self.transform.contains(upper):
            raise ValueError("bounds must lie in the transform's natural domain")
        if lower >= upper or not lower <= initial <= upper:
            raise ValueError(
                "parameter bounds must satisfy lower <= initial <= upper with lower < upper"
            )
        internal_lower = self.transform.to_internal(lower)
        internal_upper = self.transform.to_internal(upper)
        if not (np.isfinite(internal_lower) and np.isfinite(internal_upper)
                and internal_lower < internal_upper):
            raise ValueError("bounds must map to a finite internal interval")
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
```

(Delete the `_ordinary_positive`-based validation in `OptimizationBounds`; keep the module-level `_ordinary_positive` only where still used — the warm-start check below is generalised.)

- [ ] **Step 4: Generalise `_validate_problem` and `_parameter_values`**

In `_validate_problem`, replace the warm-start `_ordinary_positive(initial[name], ...)` and range check with transform-aware validation:

```python
        parameter_bounds = bounds[name]
        value = float(initial[name])
        if not np.isfinite(value):
            raise ValueError(f"initial value for {name!r} must be finite")
        if not parameter_bounds.lower <= value <= parameter_bounds.upper:
            raise ValueError(f"initial value for {name!r} must lie within its bounds")
        start[name] = value
```

Rewrite `_parameter_values` to take the per-name transforms and use them (replacing `np.exp` and the `<= 0` positivity check). The internal-space bound snapping is unchanged in structure:

```python
def _parameter_values(names, internal_parameters, transforms,
                      lower, upper, internal_lower, internal_upper):
    values = {}
    for index, name in enumerate(names):
        u = float(internal_parameters[index])
        low, high = internal_lower[index], internal_upper[index]
        if u <= low:
            theta = lower[index]
        elif u >= high:
            theta = upper[index]
        else:
            tolerance = _log_bound_tolerance(low, high)
            if abs(u - low) <= tolerance and abs(u - low) < abs(u - high):
                theta = lower[index]
            elif abs(u - high) <= tolerance and abs(u - high) < abs(u - low):
                theta = upper[index]
            else:
                theta = transforms[index].from_internal(u)
        if not np.isfinite(theta) or not transforms[index].contains(theta):
            # allow the exact closed bounds (contains is open); clip to them
            theta = float(np.clip(theta, lower[index], upper[index]))
            if not np.isfinite(theta):
                raise NumericalError("transformed parameters must be finite")
        values[name] = float(theta)
    return values
```

> Preserve the existing behaviour: the current code clips into `[lower, upper]`
> and snaps values that land within `_log_bound_tolerance` of a bound. Keep those
> semantics; only the θ computation (`from_internal` instead of `exp`) and the
> validity check (`transform.contains` instead of `> 0`) change. Since
> `transform.contains` is open at the bounds but the optimiser may sit exactly on
> a closed box bound, clipping to `[lower, upper]` at the boundary is intended.

In `optimize_empirical_bayes`, build internal bounds via the transforms:

```python
    transforms = [bounds[name].transform for name in names]
    natural_lower = np.asarray([bounds[name].lower for name in names])
    natural_upper = np.asarray([bounds[name].upper for name in names])
    lower = np.asarray([t.to_internal(b) for t, b in zip(transforms, natural_lower, strict=True)])
    upper = np.asarray([t.to_internal(b) for t, b in zip(transforms, natural_upper, strict=True)])
    start = np.asarray([t.to_internal(initial_values[name]) for t, name in zip(transforms, names, strict=True)])
```

and update the `objective`'s call to `_parameter_values(names, key, transforms, natural_lower, natural_upper, lower, upper)`. The L-BFGS box `scipy_bounds = list(zip(lower, upper))`, the `collapsed` check, snapping, and `active_bounds` are unchanged (they already work in internal space).

- [ ] **Step 5: Run the tests to verify they pass (new + existing)**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_empirical_bayes.py -q`
Expected: PASS (all existing + new).

- [ ] **Step 6: Full suite + ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: green, clean. (This proves the Log path is a no-op refactor across the whole suite.)

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/optimization/empirical_bayes.py src/pylgm/optimization/__init__.py tests/optimization/test_empirical_bayes.py
git commit -m "feat: make empirical-Bayes optimiser transform-aware"
```

---

### Task 3: INLA grid generalization (transform + Jacobian)

**Files:**
- Modify: `src/pylgm/optimization/inla.py`
- Test: `tests/optimization/test_inla.py` (add a bounded-parameter integration test; keep existing)

**Interfaces:**
- Consumes: `OptimizationBounds.transform` via `bounds` (Task 2).
- Produces: `integrate_inla` integrates in per-parameter internal space with the correct Jacobian.

- [ ] **Step 1: Write the failing test**

Integrate a single `LogitTransform` parameter and compare the marginal mean/variance to a fine 1-D brute-force quadrature of the same posterior. (Reuse the existing INLA test fixtures/harness for a tiny model; parameterise one hyperparameter with a logit transform.) Assert the integrated ρ marginal moments match the quadrature truth within tolerance, which fails if the Jacobian is wrong.

> Read `tests/optimization/test_inla.py` for the existing harness and mirror it.
> The key assertion: with a `LogitTransform` parameter, the integrated marginal
> equals a direct trapezoidal integration of `exp(s(θ)+logprior) ` over θ on a
> fine grid (moments within ~1e-2). Also add/keep a back-compat assertion that an
> all-log model's integrated result is unchanged.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_inla.py -q`
Expected: the new test FAILs (Jacobian currently hard-coded to `Σ u`).

- [ ] **Step 3: Generalise `integrate_inla`**

Replace the log-hard-coded pieces (around inla.py:249-298):

```python
    names = tuple(family.parameter_names)
    transforms = [bounds[name].transform for name in names]
    lower = np.array([bounds[name].lower for name in names])
    upper = np.array([bounds[name].upper for name in names])

    def evaluate(u):
        theta = {
            name: float(np.clip(transforms[i].from_internal(value), lower[i], upper[i]))
            for i, (name, value) in enumerate(zip(names, u, strict=True))
        }
        compiled = family.materialize(theta)
        ...
        return s_value, conditional, theta, compiled

    eb = optimize_empirical_bayes(...)
    u_star = np.array([transforms[i].to_internal(eb.parameters[name]) for i, name in enumerate(names)])
    ...
    internal_lower = np.array([transforms[i].to_internal(lower[i]) for i in range(len(names))])
    internal_upper = np.array([transforms[i].to_internal(upper[i]) for i in range(len(names))])
    in_domain = grid[np.all((grid >= internal_lower) & (grid <= internal_upper), axis=1)]
    ...
    # importance-weight Jacobian: log|d theta / d u| summed over parameters
    log_weights = np.array([
        s + float(sum(transforms[i].log_abs_jacobian(u[i]) for i in range(len(names))))
        for (u, s, _, _, _) in kept
    ])
```

Everything else (grid construction, `s_max` thresholding, moment accumulation, `integrated_lml`, diagnostics) is unchanged — it already operates in internal `u`-space. For an all-`LogTransform` model `log_abs_jacobian(u)=u`, so `log_weights` reduces to the current `s + Σ u` exactly.

- [ ] **Step 4: Run the tests (new + existing)**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_inla.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: green, clean (Log path unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/optimization/inla.py tests/optimization/test_inla.py
git commit -m "feat: integrate INLA grid in per-parameter transform space"
```

---

### Task 4: `ParametricBlock` in the family layer

**Files:**
- Modify: `src/pylgm/ir/family.py`
- Modify: `src/pylgm/ir/__init__.py` (export `ParametricBlock`)
- Test: `tests/ir/test_family.py` (add cases; create if absent — match existing test layout)

**Interfaces:**
- Produces: `ParametricBlock(block, parameters, build)` with `materialize(resolved) -> LatentBlock`; `_materialize_blocks` and family validators accept `ScalableBlock | ParametricBlock`; families gain `parameter_bounds`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ir/test_family.py  (append or create)
import numpy as np
from scipy.sparse import csr_matrix, diags

from pylgm.ir.family import ParametricBlock
from pylgm.ir.model import LatentBlock


def _car_block(name, d, w):
    template = LatentBlock(
        name, tuple(str(i) for i in range(len(d))),
        csr_matrix(np.eye(len(d))),
        csr_matrix(diags(d) - 0.5 * w),
        np.empty((0, len(d))),
    )

    def build(values):
        return csr_matrix(values[f"{name}.precision"] * (diags(d) - values[f"{name}.rho"] * w))

    return ParametricBlock(template, (f"{name}.precision", f"{name}.rho"), build)


def test_parametric_block_materializes_tau_d_minus_rho_w():
    d = np.array([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    block = _car_block("region", d, w)
    materialized = block.materialize({"region.precision": 3.0, "region.rho": 0.5})
    assert np.allclose(materialized.precision.toarray(), 3.0 * (np.diag(d) - 0.5 * w))
    assert materialized.labels == block.block.labels
    assert materialized.constraints.shape == (0, 3)
```

> Also add a family-level test: build a `CompiledGaussianFamily` (or
> `CompiledFamily`) mixing a `ScalableBlock` and a `ParametricBlock` and assert
> `materialize({...})` produces the correct block-diagonal precision and that the
> parametric block's parameter names are accepted. Read the existing family
> construction in `compiler.py`/tests to match the constructor arguments.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/ir/test_family.py -q`
Expected: FAIL (`ParametricBlock` missing).

- [ ] **Step 3: Implement `ParametricBlock` + dispatch**

```python
# ir/family.py
from collections.abc import Callable

@dataclass(frozen=True, init=False)
class ParametricBlock:
    block: LatentBlock
    parameters: tuple[str, ...]
    build: Callable[[Mapping[str, float]], csr_matrix]

    def __init__(self, block, parameters, build):
        if not isinstance(block, LatentBlock):
            raise ModelValidationError("parametric block must contain a latent block")
        names = tuple(parameters)
        if not names or any(not isinstance(n, str) or not n for n in names):
            raise ModelValidationError("parametric block parameters must be non-empty strings")
        if not callable(build):
            raise ModelValidationError("parametric block build must be callable")
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "parameters", names)
        object.__setattr__(self, "build", build)

    def materialize(self, resolved: Mapping[str, float]) -> LatentBlock:
        precision = self.build(resolved)
        if not np.isfinite(precision.data).all():
            raise NumericalError(
                f"parametric precision for block {self.block.name!r} must remain finite"
            )
        return LatentBlock(
            self.block.name, self.block.labels, self.block.design,
            precision, self.block.constraints,
        )
```

Generalise `_materialize_blocks` to dispatch:

```python
def _materialize_blocks(blocks, resolved):
    materialized = []
    for item in blocks:
        if isinstance(item, ParametricBlock):
            materialized.append(item.materialize(resolved))
            continue
        multiplier = resolved[item.parameter] if item.parameter else item.scale
        with np.errstate(over="ignore", invalid="ignore"):
            precision = item.block.precision * multiplier
        if not np.isfinite(precision.data).all():
            raise NumericalError(
                f"precision scaling for block {item.block.name!r} must remain finite"
            )
        materialized.append(LatentBlock(
            item.block.name, item.block.labels, item.block.design,
            precision, item.block.constraints,
        ))
    return tuple(materialized)
```

In `CompiledFamily.__init__` and `CompiledGaussianFamily.__init__`, accept the union: where they check `isinstance(item, ScalableBlock)`, allow `ParametricBlock` too; for each `ParametricBlock` validate `set(item.parameters) <= set(names)`; use `item.block` for the design-row/name checks. `block_slice` and `_qualified_labels` use `item.block` uniformly (works for both). Add an optional `parameter_bounds` field (see Task 6) — for this task, default it to an empty mapping so existing constructors keep working.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/ir/test_family.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: green, clean.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/ir/family.py src/pylgm/ir/__init__.py tests/ir/test_family.py
git commit -m "feat: add ParametricBlock for hyperparameter-dependent precisions"
```

---

### Task 5: `Hyperparameter` logit transform + `ProperCAR` accepts a Hyperparameter ρ

**Files:**
- Modify: `src/pylgm/parameters.py`
- Modify: `src/pylgm/effects/spec.py`
- Test: `tests/test_parameters.py` and `tests/effects/test_spec.py` (add cases)

**Interfaces:**
- Produces: `Hyperparameter(transform="logit")` allowing non-positive `initial` and `lower/upper=None` (deferred interval); `ProperCAR(rho=Hyperparameter(...))` accepted (rejection removed).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parameters.py  (append)
import pytest
from pylgm.parameters import Hyperparameter

def test_logit_hyperparameter_allows_nonpositive_initial_and_deferred_bounds():
    hp = Hyperparameter("region.rho", initial=0.0, transform="logit")
    assert hp.transform == "logit"
    assert hp.initial == 0.0
    assert hp.lower is None and hp.upper is None

def test_log_hyperparameter_still_positive_only():
    with pytest.raises(ValueError):
        Hyperparameter("tau", initial=-1.0)          # log default
    with pytest.raises(ValueError):
        Hyperparameter("tau", initial=1.0, transform="log", lower=-1.0)
```

```python
# tests/effects/test_spec.py  (append)
def test_proper_car_accepts_hyperparameter_rho():
    from pylgm.effects import ProperCAR
    hp = Hyperparameter("region.rho", initial=0.3, transform="logit")
    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=hp)
    assert effect.rho is hp
```

> Note: this REPLACES the proper-CAR slice's `test_proper_car_rejects_hyperparameter_rho`.
> Delete or invert that earlier test (grep for it) — ρ-as-Hyperparameter is now
> supported.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_parameters.py tests/effects/test_spec.py -q`
Expected: FAIL.

- [ ] **Step 3: Generalise `Hyperparameter`**

In `parameters.py`, add `"logit"` to the allowed transforms and branch validation:

```python
    transform: Literal["identity", "log", "logit"] = "log"
    ...
    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.transform not in ("identity", "log", "logit"):
            raise ValueError("transform must be 'identity', 'log', or 'logit'")
        if self.prior is not None and not callable(getattr(self.prior, "logpdf", None)):
            raise ValueError("prior must provide a logpdf method")
        if self.transform == "log":
            object.__setattr__(self, "initial", _positive_real(self.initial, "initial"))
            lower = self.initial * 1e-3 if self.lower is None else _positive_real(self.lower, "lower")
            upper = self.initial * 1e3 if self.upper is None else _positive_real(self.upper, "upper")
            if not lower < self.initial:
                raise ValueError("lower bound must be below initial")
            if not self.initial < upper:
                raise ValueError("upper bound must be above initial")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        else:
            # identity / logit: any finite initial; bounds may be deferred (None)
            if type(self.initial) not in (int, float) or not math.isfinite(self.initial):
                raise ValueError("initial must be a finite real value")
            object.__setattr__(self, "initial", float(self.initial))
            if self.lower is not None and not math.isfinite(self.lower):
                raise ValueError("lower must be a finite real value or None")
            if self.upper is not None and not math.isfinite(self.upper):
                raise ValueError("upper must be a finite real value or None")
            if self.lower is not None and self.upper is not None and not self.lower < self.upper:
                raise ValueError("lower must be below upper")
```

- [ ] **Step 4: Update `ProperCAR` in `spec.py`**

Remove the `isinstance(self.rho, Hyperparameter)` rejection. Accept a float OR a `Hyperparameter`:

```python
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)   # interval resolved at compile time
        else:
            object.__setattr__(self, "rho", _finite_real(self.rho, "rho"))
```

Update the `rho` type annotation to `float | Hyperparameter`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_parameters.py tests/effects/test_spec.py -q`
Expected: PASS (and the old rejection test is removed/inverted).

- [ ] **Step 6: Full suite + ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: green, clean.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/parameters.py src/pylgm/effects/spec.py tests/test_parameters.py tests/effects/test_spec.py
git commit -m "feat: support logit hyperparameters and Hyperparameter rho on ProperCAR"
```

---

### Task 6: Compiler wiring + `parameter_bounds` + end-to-end ρ estimation

**Files:**
- Modify: `src/pylgm/compiler.py`
- Modify: `src/pylgm/ir/family.py` (add `parameter_bounds` field, populated here)
- Modify: `src/pylgm/model.py` (`_family_optimization_inputs` reads `family.parameter_bounds`)
- Modify: `src/pylgm/effects/proper_car.py` (expose the validity interval)
- Test: `tests/test_proper_car_rho_estimation.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `ProperCAR(rho=Hyperparameter(...))` estimated (optimise) and integrated (INLA) jointly with τ.

- [ ] **Step 1: Write the failing end-to-end tests**

```python
# tests/test_proper_car_rho_estimation.py
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, ProperCAR
from pylgm.priors import PCPrecision

def _chain(n):
    return {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}

def _spatial_frame(n=12, rho_true=0.95, seed=0):
    # simulate smooth spatial signal on a chain, strong dependence
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho_true * x[i - 1] + rng.normal(scale=0.3)
    x -= x.mean()
    y = x + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": y})

def test_rho_estimated_by_empirical_bayes():
    frame = _spatial_frame()
    graph = _chain(12)
    rho = Hyperparameter("region.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(response="y",
                predictor=Fixed("1") + ProperCAR("region", index="region", graph=graph, rho=rho, precision=tau),
                likelihood=Gaussian(sigma=0.1))
    result = model.fit(frame)
    est = result.hyperparameters["region.rho"]
    assert -1.0 < est < 1.0
    assert est > 0.3           # recovers positive spatial dependence

def test_rho_integrated_by_inla():
    frame = _spatial_frame()
    graph = _chain(12)
    rho = Hyperparameter("region.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(response="y",
                predictor=Fixed("1") + ProperCAR("region", index="region", graph=graph, rho=rho, precision=tau),
                likelihood=Gaussian(sigma=0.1))
    result = model.fit(frame, hyperparameters="integrate")
    marg = result.hyperparameter_marginals()
    assert "region.rho" in marg and "region.precision" in marg
    m = marg["region.rho"]
    assert -1.0 < float(m.mean[0]) < 1.0
    assert float(m.variance[0]) > 0.0

def test_plugin_rho_still_works():
    frame = _spatial_frame()
    model = LGM(response="y",
                predictor=Fixed("1") + ProperCAR("region", index="region", graph=_chain(12), rho=0.9),
                likelihood=Gaussian(sigma=0.1))
    assert model.fit(frame) is not None
```

> Verify the result API names (`result.hyperparameters`,
> `hyperparameter_marginals()`, `.mean`/`.variance`) against the current code and
> the Besag/proper-CAR fit tests; adjust if they differ. The behavioural intent —
> ρ recovered by EB and a sensible ρ marginal from INLA — is what matters. Tune
> the recovery threshold if the tiny simulated fixture is noisy, but keep a
> non-trivial assertion (ρ clearly positive).

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_proper_car_rho_estimation.py -q`
Expected: FAIL (ρ not wired through the compiler).

- [ ] **Step 3: Expose the validity interval from `proper_car.py`**

Rename/add a public helper returning `(a, b)` for a graph so the compiler can build ρ bounds without duplicating the eigenvalue math:

```python
def car_rho_interval(graph) -> tuple[float, float]:
    nodes, w = normalize_graph(graph)
    degree = np.asarray(w.sum(axis=1)).ravel()
    if np.any(degree == 0):
        raise ValueError("proper CAR rho interval is undefined for isolated nodes")
    lower, upper, _, _ = _validity_interval(w, degree)
    return lower, upper
```

(`build_proper_car` continues to call `_validity_interval` internally.)

- [ ] **Step 4: Add `parameter_bounds` to the families**

Add an optional `parameter_bounds: Mapping[str, OptimizationBounds] = MappingProxyType({})` field to `CompiledFamily` and `CompiledGaussianFamily`, set from a constructor argument (default empty). Import `OptimizationBounds` lazily or accept a plain mapping to avoid a circular import (family.py must not import the optimiser at module load if that cycles — use a local import or duck-typing; confirm no cycle first).

- [ ] **Step 5: Wire ρ in `compile_family`**

In `compile_family`, replace the `ProperCAR` branch so that when ρ is a `Hyperparameter` it emits a `ParametricBlock` and registers ρ's bounds:

```python
        if isinstance(effect, ProperCAR):
            from pylgm.effects.graph import normalize_graph
            from pylgm.effects.proper_car import car_rho_interval
            from pylgm.optimization.empirical_bayes import OptimizationBounds
            from pylgm.optimization.transforms import LogitTransform, LogTransform
            nodes, w = normalize_graph(dict(effect.graph))
            degree = np.asarray(w.sum(axis=1)).ravel()
            structure_w = w  # captured for the builder
            rho_is_hp = isinstance(effect.rho, Hyperparameter)
            if not rho_is_hp:
                block = build_proper_car(
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, value
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                continue
            # rho is a Hyperparameter -> ParametricBlock over (tau, rho)
            a, b = car_rho_interval(dict(effect.graph))
            inset = 1e-6 * (b - a)
            rho_lower = a + inset if effect.rho.lower is None else max(effect.rho.lower, a + inset)
            rho_upper = b - inset if effect.rho.upper is None else min(effect.rho.upper, b - inset)
            template = build_proper_car(
                frame, effect.name, effect.index, dict(effect.graph),
                float(np.clip(effect.rho.initial, rho_lower, rho_upper)),
                value if not optimized else 1.0,
            )
            deg = degree
            wmat = w
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            rho_name = effect.rho.name
            def build(values, deg=deg, wmat=wmat, tau_name=tau_name, tau_fixed=tau_fixed, rho_name=rho_name):
                from scipy.sparse import csr_matrix, diags
                tau = values[tau_name] if tau_name else tau_fixed
                rho = values[rho_name]
                return csr_matrix(tau * (diags(deg) - rho * wmat))
            params = tuple(n for n in (tau_name, rho_name) if n)
            scalable.append(ParametricBlock(template, params, build))
            if optimized:
                parameter_names.append(precision.name)
            parameter_names.append(rho_name)
            parameter_bounds[rho_name] = OptimizationBounds(
                float(np.clip(effect.rho.initial, rho_lower, rho_upper)),
                rho_lower, rho_upper, transform=LogitTransform(a, b),
            )
            continue
```

Also: (a) import `ParametricBlock` and `Hyperparameter` in compiler.py; (b) build `parameter_bounds` for the Log parameters (σ and precisions) too — for every scalable/precision parameter add `OptimizationBounds(hp.initial, hp.lower, hp.upper, LogTransform())` keyed by name, matching what `model.py` used to build — so the family carries ALL bounds; (c) `_model_hyperparameters` must also yield `(effect.name, effect.rho)` when `effect` is a `ProperCAR` with a `Hyperparameter` rho (so `model.py` treats the model as having hyperparameters and the priored-penalty loop can include ρ's prior if any). Pass `parameter_bounds` into the `CompiledFamily`/`CompiledGaussianFamily` constructor.

> This is the integration crux — build it incrementally and lean on the tests.
> Keep the Log-parameter bounds identical to what `model.py` produced before, so
> existing optimise/integrate behaviour is unchanged.

- [ ] **Step 6: `model.py` reads `family.parameter_bounds`**

In `_family_optimization_inputs`, if `family.parameter_bounds` is non-empty use it directly (it already contains transforms); otherwise fall back to the current construction from `hp.lower/upper` (Log). Keep the prior `penalty` construction, extended so a ρ `Hyperparameter` with a `prior` is included (the penalty reads `values[name]` on the natural scale — works for ρ).

- [ ] **Step 7: Run the end-to-end tests**

Run: `PYTHONPATH=src python -m pytest tests/test_proper_car_rho_estimation.py -q`
Expected: PASS.

- [ ] **Step 8: Full suite + ruff**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Expected: green, clean.

- [ ] **Step 9: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/ir/family.py src/pylgm/model.py src/pylgm/effects/proper_car.py tests/test_proper_car_rho_estimation.py
git commit -m "feat: estimate and integrate proper-CAR rho as a bounded hyperparameter"
```

---

### Task 7: Documentation + roadmap

**Files:**
- Modify: `README.md`
- Modify: the spatial roadmap note (grep for "bounded" / "Spatial roadmap" / "proper CAR").

**Interfaces:** none (docs only).

- [ ] **Step 1: Document bounded hyperparameters + ρ estimation in `README.md`**

Update the proper-CAR section (and/or the hyperparameter section) to show `rho=Hyperparameter("region.rho", initial=0.0, transform="logit")` being estimated (optimise) and integrated (INLA), jointly with τ; note the interval comes from the graph; note the transform abstraction (Log/Logit) underpinning it. Do NOT claim BYM2/φ.

- [ ] **Step 2: Update the roadmap**

Mark bounded-hyperparameter inference + proper-CAR ρ estimation DONE. Record next: **BYM2** (structured scaled-ICAR + unstructured IID, mixing φ via `LogitTransform(0,1)` reusing this machinery, PC priors) → completes the CAR family. Keep deferred notes (general parametric-block framework, config-file ρ).

- [ ] **Step 3: Verify the README example runs + suite green**

Run the README example under `PYTHONPATH=src`; then `PYTHONPATH=src python -m pytest -q`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: document bounded-hyperparameter inference and proper-CAR rho estimation"
```

---

## Self-Review Notes

- **Spec coverage:** transforms (T1) ✓; transform-aware bounds + optimiser (T2) ✓; INLA transform + Jacobian (T3) ✓; ParametricBlock + family (T4) ✓; Hyperparameter logit + ProperCAR ρ accept (T5) ✓; compiler wiring + parameter_bounds + model.py + end-to-end ρ (T6) ✓; docs (T7) ✓.
- **Back-compat anchor:** `LogTransform.log_abs_jacobian(u)=u` ⇒ optimiser and INLA reduce to current math; the whole existing suite green at T2/T3/T4 proves the no-op refactor. Tasks are ordered so each keeps the suite green.
- **Coupling risk (T6):** the crux is ρ's graph interval → `LogitTransform`/bounds → `ParametricBlock` → optimiser/INLA. Bounds now live on the family (compiler-built), so `model.py` no longer needs the graph. Log-parameter bounds are built identically to before to preserve behaviour.
- **Type consistency:** `Transform` API (T1) used identically in T2/T3/T6; `ParametricBlock(block, parameters, build)` identical in T4/T6; `car_rho_interval` (T6) reuses proper CAR's `_validity_interval` (no duplicated eigenvalue math).
- **Removed test:** the proper-CAR slice's `test_proper_car_rejects_hyperparameter_rho` is inverted/removed in T5 (ρ-as-Hyperparameter is now supported) — a plan-mandated change to an existing test, flagged for the reviewer.
- **Circular-import watch (T4/T6):** `family.py` referencing `OptimizationBounds` — use duck-typing or a local import; confirm no import cycle (optimiser imports inference, not family). If a cycle appears, store `parameter_bounds` as a plain mapping typed structurally.

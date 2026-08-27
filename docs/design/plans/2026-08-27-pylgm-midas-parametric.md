# Restricted (Parametric) MIDAS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add classic restricted MIDAS — collapse `K` high-frequency lag columns into one regressor `β·Σ_k w(k;θ)·x_{t,k}` with a parametric lag-weight kernel (exp-Almon or Beta) whose shape params `θ` are estimated by EB / integrated by INLA, and a loading `β` with a fixed vague Gaussian prior.

**Architecture:** The single regressor column is itself a function of θ, so the IR gains `ParametricDesignBlock` — the mirror of the existing `ParametricBlock` (`build(θ)→design`, precision/labels/constraints frozen). Weight kernels + a design builder live in `effects/midas.py`; the compiler wires a per-θ design rebuild through the existing optimize/integrate machinery (no inference change); `predict()` rebuilds the aggregate at the fitted θ̂ via a context fixup that mirrors the existing fitted-likelihood substitution.

**Tech Stack:** Python, numpy, scipy.sparse, pandas. Spec: `docs/design/specs/2026-08-27-pylgm-midas-parametric-design.md`.

## Global Constraints

- No new runtime dependency (numpy / scipy / pandas / stdlib only).
- Deterministic approximation only (no MCMC).
- Pristine test output: no stray warnings beyond ones explicitly asserted via `pytest.warns`; all new tests pass under `-W error::UserWarning`.
- Git identity: **Ardea00 only**; never pass an `--author` override.
- Follow existing effect/compiler/prediction patterns (AR1, ProperCAR, MIDAS).
- Weights normalized to sum to 1 (β carries magnitude); weights computed via log-sum-exp softmax (overflow-safe).
- Lag convention: `columns[0]` = shallowest/contemporaneous lag, `columns[K-1]` = deepest.
- β prior is a **fixed** vague Gaussian precision (default `1e-6`, mirroring `Fixed`); the only new hyperparameters are the two shape params.

## File Structure

| File | Responsibility |
|---|---|
| `src/pylgm/optimization/transforms.py` | add `IdentityTransform` (real-line transform) |
| `src/pylgm/parameters.py` | make `Hyperparameter(transform="identity")` accept any finite initial + finite default bounds |
| `src/pylgm/ir/family.py` | `ParametricDesignBlock` + `_materialize_blocks` dispatch + both family validators |
| `src/pylgm/effects/midas.py` | `midas_weights` kernels + `build_midas_parametric` |
| `src/pylgm/effects/spec.py` | `MIDASParametric` effect dataclass + composition/validation |
| `src/pylgm/__init__.py`, `src/pylgm/effects/__init__.py` | export `MIDASParametric` |
| `src/pylgm/compiler.py` | `_real_bounds`; `compile_lgm` + `compile_family` branches; `build_prediction_context` entry |
| `src/pylgm/inference/prediction.py` | `_midas_parametric_block` + dispatch |
| `src/pylgm/model.py` | `_context_with_fitted_weights` fixup (θ̂ into the design entry) |
| `docs/effects.md`, `docs/roadmap.md` | document the effect; update shipped list |
| `tests/test_midas_parametric_*.py` | kernels, transform, spec, estimation/contract, prediction/recovery |

---

### Task 1: `IdentityTransform` + real-valued `Hyperparameter` identity semantics

**Files:**
- Modify: `src/pylgm/optimization/transforms.py`
- Modify: `src/pylgm/parameters.py:73-84` (the `elif self.transform == "identity":` branch) and the class docstring line 41
- Test: `tests/test_identity_transform.py`

**Interfaces:**
- Produces: `IdentityTransform` (class in `transforms.py`) implementing the `Transform` protocol — `to_internal`/`from_internal` identity, `log_abs_jacobian`→`0.0`, `contains`→`isfinite`, `domain_description`→`"the real line"`, with `__eq__`/`__hash__`.
- Produces: `Hyperparameter(transform="identity", initial=<any finite>, lower=None, upper=None)` now validates like `logit` (any finite initial; finite default bounds `initial ± 10.0` when `lower`/`upper` are `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_identity_transform.py
import numpy as np
import pytest

from pylgm.optimization.transforms import IdentityTransform, Transform
from pylgm.parameters import Hyperparameter


def test_identity_transform_is_a_transform_and_round_trips():
    t = IdentityTransform()
    assert isinstance(t, Transform)
    for x in (-3.5, 0.0, 2.7):
        assert t.from_internal(t.to_internal(x)) == pytest.approx(x)
        assert t.log_abs_jacobian(x) == 0.0
    assert t.contains(1.0) and not t.contains(np.inf)
    assert IdentityTransform() == IdentityTransform()
    assert hash(IdentityTransform()) == hash(IdentityTransform())


def test_identity_hyperparameter_accepts_nonpositive_initial():
    hp = Hyperparameter("k.shape2", initial=-0.1, transform="identity")
    assert hp.initial == pytest.approx(-0.1)
    assert hp.lower < hp.initial < hp.upper
    assert np.isfinite(hp.lower) and np.isfinite(hp.upper)


def test_identity_hyperparameter_accepts_zero_initial_and_custom_bounds():
    hp = Hyperparameter("k.shape1", initial=0.0, transform="identity", lower=-5.0, upper=5.0)
    assert (hp.lower, hp.initial, hp.upper) == (-5.0, 0.0, 5.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_identity_transform.py -v`
Expected: FAIL — `ImportError: cannot import name 'IdentityTransform'`, and the nonpositive-initial test fails on the current positive-only identity branch.

- [ ] **Step 3: Add `IdentityTransform` to `transforms.py`**

Append after `LogitTransform`:

```python
class IdentityTransform:
    """Unconstrained parameters on the whole real line: u = theta, theta = u."""

    def to_internal(self, theta: float) -> float:
        return float(theta)

    def from_internal(self, u: float) -> float:
        return float(u)

    def log_abs_jacobian(self, u: float) -> float:
        return 0.0

    def contains(self, theta: float) -> bool:
        return bool(np.isfinite(theta))

    def domain_description(self) -> str:
        return "the real line"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdentityTransform)

    def __hash__(self) -> int:
        return hash(IdentityTransform)
```

- [ ] **Step 4: Make the `Hyperparameter` identity branch real-valued**

In `src/pylgm/parameters.py`, replace the `elif self.transform == "identity":` block (currently lines 73-84, positive-only) with any-finite semantics and finite default bounds:

```python
        elif self.transform == "identity":
            # Real-line parameter (e.g. exp-Almon weight shape): any finite
            # initial; default to a finite symmetric window so OptimizationBounds
            # (which requires finite lower/upper under IdentityTransform) is valid.
            if type(self.initial) not in (int, float) or not math.isfinite(self.initial):
                raise ValueError("initial must be a finite real value")
            object.__setattr__(self, "initial", float(self.initial))
            lower = self.initial - 10.0 if self.lower is None else float(self.lower)
            upper = self.initial + 10.0 if self.upper is None else float(self.upper)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("lower and upper must be finite real values or None")
            if not lower < self.initial < upper:
                raise ValueError("bounds must satisfy lower < initial < upper")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
```

Then update the class docstring line 41: change `` ``"identity"`` is accepted but not wired into inference.`` to `` ``"identity"`` selects the real line (any finite value) — used by exp-Almon MIDAS weight shapes.``

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_identity_transform.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/optimization/transforms.py src/pylgm/parameters.py tests/test_identity_transform.py
git commit -m "feat: IdentityTransform + real-valued identity Hyperparameter"
```

---

### Task 2: `ParametricDesignBlock` IR seam

**Files:**
- Modify: `src/pylgm/ir/family.py` (add class after `ParametricBlock` at `:146`; `_materialize_blocks` at `:186-210`; validators at `:319-320`, `:339-346`, `:469-470`, `:492-496`)
- Test: `tests/test_parametric_design_block.py`

**Interfaces:**
- Produces: `ParametricDesignBlock(block: LatentBlock, parameters: tuple[str, ...], build: Callable[[Mapping[str,float]], csr_matrix])` with `.materialize(resolved) -> LatentBlock` that swaps in a fresh **design** (calling `build`), keeping `block.precision`, `block.labels`, `block.constraints`. Raises `NumericalError` if the design is non-finite.
- Consumes: `LatentBlock`, `NumericalError`, `csr_matrix`, `np`, `Callable`, `Mapping` — all already imported in `family.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parametric_design_block.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.exceptions import NumericalError
from pylgm.ir.family import ParametricDesignBlock
from pylgm.ir.model import LatentBlock


def _template(width_rows=4):
    design = csr_matrix(np.ones((width_rows, 1)))
    precision = csr_matrix([[1e-6]])
    return LatentBlock("m", ("m",), design, precision, np.empty((0, 1)))


def test_materialize_rebuilds_design_keeps_precision():
    template = _template()

    def build(resolved):
        return csr_matrix((resolved["theta"] * np.arange(4.0)).reshape(-1, 1))

    block = ParametricDesignBlock(template, ("theta",), build)
    out = block.materialize({"theta": 2.0})
    assert out.name == "m" and out.labels == ("m",)
    np.testing.assert_allclose(out.design.toarray().ravel(), [0.0, 2.0, 4.0, 6.0])
    # precision + constraints carried from the template, unchanged
    np.testing.assert_allclose(out.precision.toarray(), [[1e-6]])
    assert out.constraints.shape == (0, 1)


def test_materialize_rejects_nonfinite_design():
    block = ParametricDesignBlock(
        _template(), ("theta",), lambda r: csr_matrix(np.array([[np.inf]] * 4))
    )
    with pytest.raises(NumericalError):
        block.materialize({"theta": 1.0})
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_parametric_design_block.py -v`
Expected: FAIL — `ImportError: cannot import name 'ParametricDesignBlock'`.

- [ ] **Step 3: Add the class** after `ParametricBlock` (`family.py:146`)

```python
@dataclass(frozen=True, init=False)
class ParametricDesignBlock:
    """A latent block whose DESIGN is a function of hyperparameters.

    Mirror of ``ParametricBlock``: ``build`` returns a fresh design column while
    the precision, labels and constraints stay fixed. Used by restricted MIDAS,
    whose single regressor is a theta-weighted aggregate of the HF lags.
    """

    block: LatentBlock
    parameters: tuple[str, ...]
    build: Callable[[Mapping[str, float]], csr_matrix]

    def __init__(
        self,
        block: LatentBlock,
        parameters: tuple[str, ...],
        build: Callable[[Mapping[str, float]], csr_matrix],
    ) -> None:
        if not isinstance(block, LatentBlock):
            raise ModelValidationError("parametric-design block must contain a latent block")
        names = tuple(parameters)
        if not names or any(not isinstance(n, str) or not n for n in names):
            raise ModelValidationError("parametric-design block parameters must be non-empty strings")
        if not callable(build):
            raise ModelValidationError("parametric-design block build must be callable")
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "parameters", names)
        object.__setattr__(self, "build", build)

    def materialize(self, resolved: Mapping[str, float]) -> LatentBlock:
        design = self.build(resolved)
        if not np.isfinite(design.data).all():
            raise NumericalError(
                f"parametric design for block {self.block.name!r} must remain finite"
            )
        return LatentBlock(
            self.block.name,
            self.block.labels,
            design,
            self.block.precision,
            self.block.constraints,
        )
```

- [ ] **Step 4: Dispatch in `_materialize_blocks`** (`family.py:186-193`)

Change the signature hint and the parametric branch to accept both parametric types:

```python
def _materialize_blocks(
    blocks: "tuple[ScalableBlock | ParametricBlock | ParametricDesignBlock, ...]",
    resolved: Mapping[str, float],
) -> tuple[LatentBlock, ...]:
    materialized: list[LatentBlock] = []
    for item in blocks:
        if isinstance(item, (ParametricBlock, ParametricDesignBlock)):
            materialized.append(item.materialize(resolved))
            continue
        ...  # unchanged ScalableBlock path
```

- [ ] **Step 5: Accept it in both family validators**

In `CompiledGaussianFamily` (`:319`) and `CompiledFamily` (`:469`), extend the type check:

```python
        if any(not isinstance(item, (ScalableBlock, ParametricBlock, ParametricDesignBlock)) for item in block_values):
            raise ModelValidationError("family blocks must be scalable or parametric blocks")
```

And the parameters-subset check — `CompiledGaussianFamily` (`:341`):

```python
        for item in block_values:
            if isinstance(item, (ParametricBlock, ParametricDesignBlock)):
                if not set(item.parameters) <= set(names):
                    raise ModelValidationError(
                        "parametric block parameters must appear in parameter_names"
                    )
                parametric_names.update(item.parameters)
                continue
```

`CompiledFamily` (`:492`):

```python
        for item in block_values:
            if isinstance(item, (ParametricBlock, ParametricDesignBlock)) and not set(item.parameters) <= set(names):
                raise ModelValidationError(
                    "parametric block parameters must appear in parameter_names"
                )
```

- [ ] **Step 6: Run to verify pass**

Run: `python -m pytest tests/test_parametric_design_block.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/ir/family.py tests/test_parametric_design_block.py
git commit -m "feat: ParametricDesignBlock IR seam (theta-dependent design)"
```

---

### Task 3: Weight kernels + design builder

**Files:**
- Modify: `src/pylgm/effects/midas.py`
- Test: `tests/test_midas_parametric_kernels.py`

**Interfaces:**
- Produces: `midas_weights(kernel: str, n_lags: int, theta: tuple[float, float]) -> np.ndarray` — length-`n_lags`, sums to 1, softmax-normalized. `kernel in {"beta", "exp_almon"}`.
- Produces: `build_midas_parametric(frame: pd.DataFrame, name: str, columns: tuple[str, ...], kernel: str, theta: tuple[float, float], prior_precision: float) -> LatentBlock` — width-1 block: `design = (V @ w).reshape(-1, 1)` (`V` = the `(N, K)` lag matrix), `precision = [[prior_precision]]`, empty constraints, labels `(name,)`.
- Consumes: `LatentBlock`, `csr_matrix`, `np`, `pd` (already imported in `midas.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_midas_parametric_kernels.py
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.midas import build_midas_parametric, midas_weights


def test_weights_sum_to_one_both_kernels():
    for kernel, theta in (("beta", (2.0, 2.0)), ("exp_almon", (0.0, -0.5))):
        w = midas_weights(kernel, 8, theta)
        assert w.shape == (8,)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_exp_almon_decays_when_quadratic_negative():
    w = midas_weights("exp_almon", 8, (0.0, -0.3))
    assert np.all(np.diff(w) < 0)  # strictly decreasing over lag index


def test_beta_has_interior_hump_when_symmetric():
    w = midas_weights("beta", 9, (2.0, 2.0))
    assert np.argmax(w) == 4  # symmetric interior peak on a 9-point grid


def test_extreme_theta_is_overflow_safe():
    w = midas_weights("exp_almon", 12, (5.0, 0.0))  # large positive slope
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)


def test_builder_produces_aggregated_column():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(6, 3))
    frame = pd.DataFrame({f"x{k}": V[:, k] for k in range(3)})
    theta = (1.0, 2.0)
    block = build_midas_parametric(frame, "m", ("x0", "x1", "x2"), "beta", theta, 1e-6)
    w = midas_weights("beta", 3, theta)
    np.testing.assert_allclose(block.design.toarray().ravel(), V @ w)
    assert block.labels == ("m",)
    np.testing.assert_allclose(block.precision.toarray(), [[1e-6]])
    assert block.constraints.shape == (0, 1)


def test_builder_rejects_missing_and_nonfinite_columns():
    frame = pd.DataFrame({"x0": [1.0, 2.0], "x1": [np.nan, 1.0]})
    with pytest.raises(ValueError, match="missing"):
        build_midas_parametric(frame, "m", ("x0", "zzz"), "beta", (2.0, 2.0), 1e-6)
    with pytest.raises(ValueError, match="non-finite|finite"):
        build_midas_parametric(frame, "m", ("x0", "x1"), "beta", (2.0, 2.0), 1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_midas_parametric_kernels.py -v`
Expected: FAIL — `ImportError` on `midas_weights` / `build_midas_parametric`.

- [ ] **Step 3: Implement in `effects/midas.py`**

Append:

```python
def _softmax(log_w: np.ndarray) -> np.ndarray:
    shifted = log_w - log_w.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def midas_weights(kernel: str, n_lags: int, theta: tuple[float, float]) -> np.ndarray:
    """Normalized MIDAS lag weights (sum to 1) for a parametric kernel.

    ``kernel="exp_almon"``: log w_k = theta1*k + theta2*k**2 (theta real).
    ``kernel="beta"``: Beta density on an interior grid x_k=(k+1)/(K+1),
    log w_k = (a-1)*log x_k + (b-1)*log(1-x_k) with theta=(a, b), a,b>0.
    Both are softmax-normalized (overflow-safe).
    """
    k = np.arange(n_lags, dtype=float)
    if kernel == "exp_almon":
        theta1, theta2 = theta
        log_w = theta1 * k + theta2 * k * k
    elif kernel == "beta":
        a, b = theta
        x = (k + 1.0) / (n_lags + 1.0)  # strictly inside (0, 1): no log(0)
        log_w = (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)
    else:
        raise ValueError(f"unknown MIDAS kernel {kernel!r}; expected 'beta' or 'exp_almon'")
    return _softmax(log_w)


def build_midas_parametric(
    frame: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
    kernel: str,
    theta: tuple[float, float],
    prior_precision: float,
) -> LatentBlock:
    """Restricted-MIDAS block: one regressor = theta-weighted aggregate of the
    HF lag columns, with a fixed vague Gaussian prior on the loading."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"MIDAS lag columns missing from frame: {missing!r}")
    values = frame[list(columns)].to_numpy(dtype=float)  # (N, K)
    if not np.isfinite(values).all():
        raise ValueError(f"MIDAS lag columns for {name!r} contain non-finite values")
    weights = midas_weights(kernel, len(columns), theta)
    design = csr_matrix((values @ weights).reshape(-1, 1))
    precision = csr_matrix(np.array([[float(prior_precision)]]))
    constraints = np.empty((0, 1), dtype=float)
    return LatentBlock(name, (name,), design, precision, constraints)
```

(If `midas.py` does not already import `csr_matrix`/`np`/`pd`, add them alongside the existing S1 imports.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_midas_parametric_kernels.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects/midas.py tests/test_midas_parametric_kernels.py
git commit -m "feat: parametric MIDAS weight kernels + design builder"
```

---

### Task 4: `MIDASParametric` effect spec

**Files:**
- Modify: `src/pylgm/effects/spec.py` (add dataclass after `MIDAS` at `:230`-ish)
- Modify: `src/pylgm/__init__.py` and `src/pylgm/effects/__init__.py` (export `MIDASParametric`)
- Test: `tests/test_midas_parametric_spec.py`

**Interfaces:**
- Produces: `MIDASParametric(name, columns, kernel="beta", shape1=None, shape2=None, prior_precision=1e-6)` — a frozen `_ComposableEffect`. `.name` is the field. Kernel defaults for shapes (filled when `None`): Beta → `a=b=2.0` (`transform="log"`); exp-Almon → `θ1=0.0`, `θ2=-0.1` (`transform="identity"`). A `Hyperparameter` shape is estimated; a float is fixed. Auto-created shape hyperparameters are named `f"{name}.shape1"` / `f"{name}.shape2"`.
- Consumes: `Hyperparameter` (from `pylgm.parameters`), the `_ComposableEffect` base, `_non_empty_string`, `_positive_real` (already in `spec.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_midas_parametric_spec.py
import pytest

from pylgm import Fixed, MIDASParametric
from pylgm.parameters import Hyperparameter


def test_beta_defaults_are_positive_log_hyperparameters():
    e = MIDASParametric("m", ("x0", "x1", "x2"), kernel="beta")
    assert e.name == "m" and e.kernel == "beta"
    for shape in (e.shape1, e.shape2):
        assert isinstance(shape, Hyperparameter)
        assert shape.transform == "log" and shape.initial == 2.0
    assert e.shape1.name == "m.shape1" and e.shape2.name == "m.shape2"
    assert e.prior_precision == 1e-6


def test_exp_almon_defaults_are_identity_hyperparameters():
    e = MIDASParametric("m", ("x0", "x1"), kernel="exp_almon")
    assert e.shape1.transform == "identity" and e.shape1.initial == 0.0
    assert e.shape2.transform == "identity" and e.shape2.initial == -0.1


def test_fixed_float_shapes_are_kept():
    e = MIDASParametric("m", ("x0", "x1"), kernel="exp_almon", shape1=0.2, shape2=-0.4)
    assert e.shape1 == 0.2 and e.shape2 == -0.4


def test_validation_rejects_bad_kernel_and_short_columns_and_precision():
    with pytest.raises(ValueError, match="kernel"):
        MIDASParametric("m", ("x0", "x1"), kernel="nope")
    with pytest.raises(ValueError, match="at least 2|columns"):
        MIDASParametric("m", ("x0",))
    with pytest.raises(ValueError):
        MIDASParametric("m", ("x0", "x1"), prior_precision=0.0)


def test_composes_with_other_effects():
    predictor = Fixed("1") + MIDASParametric("m", ("x0", "x1"))
    names = [e.name for e in predictor.effects]
    assert "m" in names and "fixed" in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_midas_parametric_spec.py -v`
Expected: FAIL — `ImportError: cannot import name 'MIDASParametric'`.

- [ ] **Step 3: Add the dataclass** to `effects/spec.py` (after `MIDAS`)

```python
@dataclass(frozen=True)
class MIDASParametric(_ComposableEffect):
    """Restricted MIDAS: HF lag ``columns`` collapsed into one regressor
    ``beta * sum_k w(k; theta) * x_{t,k}`` under a parametric lag-weight kernel
    (``"beta"`` or ``"exp_almon"``). The two shape parameters are estimated
    (EB) / integrated (INLA) when given as ``Hyperparameter``s, or fixed when
    given as floats; the loading ``beta`` carries a fixed vague Gaussian prior
    (``prior_precision``). See the S2 design spec."""

    name: str
    columns: tuple[str, ...]
    kernel: str = "beta"
    shape1: "float | Hyperparameter | None" = None
    shape2: "float | Hyperparameter | None" = None
    prior_precision: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        columns = tuple(self.columns)
        for column in columns:
            _non_empty_string(column, "column")
        if len(columns) < 2:
            raise ValueError("MIDASParametric requires at least 2 lag columns")
        object.__setattr__(self, "columns", columns)
        if self.kernel not in ("beta", "exp_almon"):
            raise ValueError("kernel must be 'beta' or 'exp_almon'")
        object.__setattr__(self, "prior_precision", _positive_real(self.prior_precision, "prior_precision"))
        object.__setattr__(self, "shape1", self._resolve_shape(self.shape1, 1))
        object.__setattr__(self, "shape2", self._resolve_shape(self.shape2, 2))

    def _resolve_shape(self, shape, index):
        if shape is None:
            if self.kernel == "beta":
                return Hyperparameter(f"{self.name}.shape{index}", initial=2.0, transform="log")
            initial = 0.0 if index == 1 else -0.1
            return Hyperparameter(f"{self.name}.shape{index}", initial=initial, transform="identity")
        if isinstance(shape, Hyperparameter):
            return shape
        if type(shape) in (int, float):
            return float(shape)
        raise ValueError("shape must be a float, a Hyperparameter, or None")
```

Ensure `spec.py` imports `Hyperparameter` (it already imports it for `MIDAS.precision`; if not, add `from pylgm.parameters import Hyperparameter`).

- [ ] **Step 4: Export it**

- In `src/pylgm/effects/__init__.py`: add `MIDASParametric` to the `from .spec import (...)` list and `__all__`.
- In `src/pylgm/__init__.py`: add `MIDASParametric` next to the existing `MIDAS` import and `__all__` entry.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_midas_parametric_spec.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/effects/spec.py src/pylgm/effects/__init__.py src/pylgm/__init__.py tests/test_midas_parametric_spec.py
git commit -m "feat: MIDASParametric effect spec + exports"
```

---

### Task 5: Compiler wiring (fixed-θ + estimated-θ)

**Files:**
- Modify: `src/pylgm/compiler.py` — import `MIDASParametric`, `build_midas_parametric`, `midas_weights`, `ParametricDesignBlock`, `IdentityTransform`; add `_real_bounds`; `compile_lgm` branch (`~:302`); `compile_family` branch (`~:700`, after MIDAS); `build_prediction_context` entry (`~:783`)
- Test: `tests/test_midas_parametric_effect.py`

**Interfaces:**
- Consumes: `MIDASParametric` spec (Task 4), `build_midas_parametric`/`midas_weights` (Task 3), `ParametricDesignBlock` (Task 2), `IdentityTransform` (Task 1).
- Produces: `_real_bounds(hp: Hyperparameter) -> OptimizationBounds` using `IdentityTransform`. A `("midas_parametric", (name, columns, kernel, theta_spec))` prediction entry, where `theta_spec = (s1, s2)` and each `s` is a `float` (fixed) or the shape `Hyperparameter`'s `.name` (estimated) — consumed in Task 6.
- Helper `_resolved_shape(shape)`: `shape.initial if isinstance(shape, Hyperparameter) else shape` — reuse the existing `_resolved_precision` (it already does exactly this).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_midas_parametric_effect.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, MIDASParametric
from pylgm.effects.midas import midas_weights


def _frame(kernel="beta", theta=(2.0, 4.0), beta=2.0, n=160, K=6, seed=1):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, K))
    w = midas_weights(kernel, K, theta)
    y = 0.5 + beta * (V @ w) + rng.normal(scale=0.1, size=n)
    cols = {f"x{k}": V[:, k] for k in range(K)}
    cols["y"] = y
    return pd.DataFrame(cols), tuple(f"x{k}" for k in range(K))


def test_fixed_theta_fit_runs():
    frame, cols = _frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta", shape1=2.0, shape2=4.0),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert np.isfinite(result.latent_marginals("m").mean).all()


def test_estimated_beta_shapes_are_reported_and_in_domain():
    frame, cols = _frame(kernel="beta", theta=(2.0, 5.0))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.hyperparameters["m.shape1"] > 0.0
    assert result.hyperparameters["m.shape2"] > 0.0


def test_estimated_exp_almon_shapes_are_reported():
    frame, cols = _frame(kernel="exp_almon", theta=(0.0, -0.4))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert np.isfinite(result.hyperparameters["m.shape1"])
    assert np.isfinite(result.hyperparameters["m.shape2"])


def test_proper_block_accepted_under_full_laplace():
    # beta has a proper vague prior and no constraints -> full Laplace is fine
    frame, cols = _frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, hyperparameters="integrate", latent_strategy="laplace")
    assert np.isfinite(result.latent_marginals("m").mean).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_midas_parametric_effect.py -v`
Expected: FAIL — compiler raises `unsupported effect type: MIDASParametric` (no branch yet).

- [ ] **Step 3: Add imports + `_real_bounds` to `compiler.py`**

Add to the effect imports: `MIDASParametric`; to the builder imports: `build_midas_parametric, midas_weights`; to the IR imports: `ParametricDesignBlock`; to the transforms import: `IdentityTransform`.

Add next to `_log_bounds`:

```python
def _real_bounds(hp: Hyperparameter) -> OptimizationBounds:
    """Bounds for a real-line (identity-transform) hyperparameter, e.g. an
    exp-Almon MIDAS weight shape. The Hyperparameter already carries finite
    lower/upper (defaulted symmetrically) under transform='identity'."""
    if hp.transform != "identity":
        raise CompilationError(
            f"exp-Almon MIDAS shape {hp.name!r} must be declared transform='identity'; "
            f"got transform={hp.transform!r}"
        )
    return OptimizationBounds(hp.initial, hp.lower, hp.upper, transform=IdentityTransform())
```

- [ ] **Step 4: Add the `compile_lgm` fixed/plug-in branch** (in the `compile_lgm` effect loop, alongside the `MIDAS` branch at `~:302`)

```python
            elif isinstance(effect, MIDASParametric):
                theta = (_resolved_precision(effect.shape1), _resolved_precision(effect.shape2))
                block = build_midas_parametric(
                    frame, effect.name, effect.columns, effect.kernel, theta, effect.prior_precision
                )
```

(`_resolved_precision` returns `.initial` for a `Hyperparameter`, else the float — exactly the plug-in θ this path needs. This branch is also what runs post-EB to build the prediction context; Task 6 substitutes θ̂.)

- [ ] **Step 5: Add the `compile_family` estimated branch** (after the `MIDAS` branch, `~:700`)

```python
        if isinstance(effect, MIDASParametric):
            theta_init = (_resolved_precision(effect.shape1), _resolved_precision(effect.shape2))
            template = _compiled_block(
                effect.name, build_midas_parametric,
                frame, effect.name, effect.columns, effect.kernel, theta_init, effect.prior_precision,
            )
            shapes = (effect.shape1, effect.shape2)
            estimated = [s for s in shapes if isinstance(s, Hyperparameter)]
            if not estimated:
                # both shapes fixed: design bakes in, no per-theta rebuild
                scalable.append(ScalableBlock(template, None, 1.0))
                continue
            name1 = effect.shape1.name if isinstance(effect.shape1, Hyperparameter) else None
            name2 = effect.shape2.name if isinstance(effect.shape2, Hyperparameter) else None
            fixed1 = None if name1 else float(effect.shape1)
            fixed2 = None if name2 else float(effect.shape2)
            columns, kernel, prior_precision = effect.columns, effect.kernel, effect.prior_precision
            frame_ref = frame

            def build(values, columns=columns, kernel=kernel, prior_precision=prior_precision,
                      frame_ref=frame_ref, name1=name1, name2=name2, fixed1=fixed1, fixed2=fixed2):
                theta = (
                    values[name1] if name1 else fixed1,
                    values[name2] if name2 else fixed2,
                )
                V = frame_ref[list(columns)].to_numpy(dtype=float)
                w = midas_weights(kernel, len(columns), theta)
                return csr_matrix((V @ w).reshape(-1, 1))

            param_names = tuple(s.name for s in estimated)
            scalable.append(ParametricDesignBlock(template, param_names, build))
            for shape in estimated:
                parameter_names.append(shape.name)
                parameter_bounds[shape.name] = (
                    _log_bounds(shape) if shape.transform == "log" else _real_bounds(shape)
                )
                if shape.prior is not None:
                    parameter_priors[shape.name] = shape.prior
            continue
```

(`csr_matrix` is already imported in `compiler.py`. The β loading is not a hyperparameter — its precision is baked into `template`.)

- [ ] **Step 6: Emit the prediction entry** in `build_prediction_context` (after the `MIDAS` branch, `~:785`)

```python
        elif isinstance(effect, MIDASParametric):
            theta_spec = tuple(
                s.name if isinstance(s, Hyperparameter) else float(s)
                for s in (effect.shape1, effect.shape2)
            )
            entries.append(("midas_parametric", (effect.name, effect.columns, effect.kernel, theta_spec)))
```

(Place it before the generic `else:` structured branch. `Hyperparameter` is already importable in `compiler.py`.)

- [ ] **Step 7: Run to verify pass**

Run: `python -m pytest tests/test_midas_parametric_effect.py -v`
Expected: PASS (4 tests). If an estimated-shape test fails to converge, first confirm `sigma` matches the simulated noise and `n` is large enough; keep the assertions as in-domain/finite sanity floors (do not tighten to exact θ recovery here — full recovery is Task 6).

- [ ] **Step 8: Commit**

```bash
git add src/pylgm/compiler.py tests/test_midas_parametric_effect.py
git commit -m "feat: compile restricted MIDAS (fixed + estimated theta)"
```

---

### Task 6: Prediction at θ̂ + statistical recovery

**Files:**
- Modify: `src/pylgm/inference/prediction.py` (`_midas_parametric_block` + dispatch at `:173`)
- Modify: `src/pylgm/model.py` (add `_context_with_fitted_weights`; call it after `_context_with_fitted_likelihood` at `:373` and `:407`)
- Test: `tests/test_midas_parametric_predict.py`

**Interfaces:**
- Consumes: the `("midas_parametric", (name, columns, kernel, theta_spec))` entry from Task 5 and `midas_weights` from Task 3. After the fixup, `theta_spec` is a tuple of two floats.
- Produces: `_midas_parametric_block(entry, new_data) -> np.ndarray` of shape `(len(new_data), 1)`; `_context_with_fitted_weights(context, result)` returning a context whose `midas_parametric` entries have `theta_spec` resolved to floats via `result.hyperparameters[name]` (names) / kept (floats).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_midas_parametric_predict.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, LGM, MIDASParametric
from pylgm.effects.midas import midas_weights


def _data(kernel="beta", theta=(2.0, 5.0), beta=2.0, n=200, K=6, seed=3):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, K))
    w = midas_weights(kernel, K, theta)
    y = 0.3 + beta * (V @ w) + rng.normal(scale=0.08, size=n)
    cols = {f"x{k}": V[:, k] for k in range(K)}
    cols["y"] = y
    return pd.DataFrame(cols), tuple(f"x{k}" for k in range(K))


def test_predict_recovers_the_field():
    frame, cols = _data()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta"),
        likelihood=Gaussian(sigma=0.08),
    )
    result = model.fit(frame)
    eta = result.predict(frame).predictive_mean
    c_truth = frame["y"].to_numpy() - frame["y"].mean()
    c_eta = eta - eta.mean()
    assert np.corrcoef(c_eta, c_truth)[0, 1] > 0.9


def test_predict_uses_fitted_weights_not_initial():
    # exp-Almon truth decays strongly; the initial (0, -0.1) is nearly flat.
    frame, cols = _data(kernel="exp_almon", theta=(0.0, -0.8))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon"),
        likelihood=Gaussian(sigma=0.08),
    )
    result = model.fit(frame)
    # Score a fresh block of rows; the prediction context must aggregate them
    # with the ESTIMATED weights, so the fit should track the true signal.
    hold, _ = _data(kernel="exp_almon", theta=(0.0, -0.8), seed=99)
    eta = result.predict(hold).predictive_mean
    c_truth = hold["y"].to_numpy() - hold["y"].mean()
    assert np.corrcoef(eta - eta.mean(), c_truth)[0, 1] > 0.85
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_midas_parametric_predict.py -v`
Expected: FAIL — `predict() context has an unknown block kind 'midas_parametric'`.

- [ ] **Step 3: Add the prediction block builder + dispatch** (`prediction.py`)

After `_midas_block` (`:132`):

```python
def _midas_parametric_block(entry, new_data: pd.DataFrame) -> np.ndarray:
    from pylgm.effects.midas import midas_weights

    name, columns, kernel, theta = entry
    missing = [column for column in columns if column not in new_data.columns]
    if missing:
        raise ValueError(
            f"predict() new_data is missing lag columns {missing!r} required by the "
            f"{name!r} block"
        )
    values = new_data[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"predict() new_data has non-finite values in the {name!r} lag columns")
    weights = midas_weights(kernel, len(columns), theta)
    return (values @ weights).reshape(-1, 1)
```

In `_design_for` (`:173`), add before the `else:`:

```python
        elif kind == "midas_parametric":
            blocks.append(_midas_parametric_block(payload, new_data))
```

- [ ] **Step 4: Add the fitted-weights fixup** (`model.py`)

Add near `_context_with_fitted_likelihood`:

```python
def _context_with_fitted_weights(context, result):
    """Substitute fitted shape estimates into parametric-MIDAS design entries.

    ``build_prediction_context`` runs on a ``compile_lgm`` result, which resolves
    an estimated shape Hyperparameter to its ``.initial`` (the starting guess). A
    parametric-MIDAS design is a function of that shape, so predict() would
    otherwise aggregate new rows with the initial weights, not the fitted ones.
    Mirror ``_context_with_fitted_likelihood`` and swap in the estimates.
    """
    from dataclasses import replace

    fitted = getattr(result, "hyperparameters", {}) or {}
    new_entries = []
    changed = False
    for kind, payload in context.entries:
        if kind == "midas_parametric":
            name, columns, kernel, theta_spec = payload
            theta = tuple(fitted[s] if isinstance(s, str) else s for s in theta_spec)
            new_entries.append((kind, (name, columns, kernel, theta)))
            changed = True
        else:
            new_entries.append((kind, payload))
    if not changed:
        return context
    return replace(context, entries=tuple(new_entries))
```

Then at both call sites (`:373` and `:407`), add the line after the likelihood fixup:

```python
        context = _context_with_fitted_likelihood(context, result, self)
        context = _context_with_fitted_weights(context, result)
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_midas_parametric_predict.py -v`
Expected: PASS (2 tests). If `test_predict_recovers_the_field` is marginally under 0.9 on the small grid, confirm `sigma=0.08` matches the simulated noise before loosening to `> 0.88` (sanity floor, not a tight bound).

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/inference/prediction.py src/pylgm/model.py tests/test_midas_parametric_predict.py
git commit -m "feat: predict() rebuilds restricted-MIDAS design at fitted theta"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/effects.md` (add a `## Restricted MIDAS` section before `## Linear constraints`)
- Modify: `docs/roadmap.md` (extend the shipped Effects line)

- [ ] **Step 1: Add the effects-guide section**

Insert into `docs/effects.md` immediately before the `## Linear constraints (\`extraconstr\`)` heading:

````markdown
## Restricted MIDAS effect (parametric lag weights)

`MIDASParametric(name, columns, kernel="beta", shape1=None, shape2=None, prior_precision=1e-6)`
collapses the high-frequency lag `columns` into a **single** regressor
`β · Σ_k w(k; θ) · x_{t,k}`, where `w(·; θ)` is a parametric lag-weight kernel.
This is the *restricted* counterpart to the U-MIDAS
[`MIDAS`](#midas-smooth-lag-effect) effect, which keeps every lag as its own
smoothed coefficient.

| `kernel` | shape params | weight `log w_k` (normalized to Σ=1) |
|------|------|------|
| `"beta"` | `a, b > 0` | `(a−1)·log x_k + (b−1)·log(1−x_k)`, `x_k=(k+1)/(K+1)` |
| `"exp_almon"` | `θ1, θ2` real | `θ1·k + θ2·k²` (θ2<0 ⇒ decay) |

Lags are indexed `columns[0]` (shallowest) … `columns[K-1]` (deepest); weights
are softmax-normalized. Each shape is a float (fixed θ) or a `Hyperparameter`
(estimated by EB, integrated by INLA); omit them for kernel-appropriate
defaults (Beta `a=b=2`; exp-Almon `θ1=0, θ2=−0.1`). The loading `β` is a single
coefficient with a fixed vague Gaussian prior (`prior_precision`), so the block
is proper and unconstrained — it runs under every latent strategy, including
`latent_strategy="laplace"`. `predict()` rebuilds the aggregate for new rows at
the fitted weights.

```python
from pylgm import Fixed, Gaussian, LGM, MIDASParametric

model = LGM(
    response="y",
    predictor=Fixed("1") + MIDASParametric("m", ("x0", "x1", "x2", "x3"), kernel="beta"),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.hyperparameters["m.shape1"]  # estimated Beta a
```
````

- [ ] **Step 2: Update the roadmap**

In `docs/roadmap.md`, extend the shipped `**Effects**` line to mention restricted MIDAS alongside the existing MIDAS/SpaceTime entries (keep every already-shipped item; add "restricted (parametric) MIDAS"). Remove any now-shipped parametric-MIDAS bullet from `## Next` if one exists; otherwise leave `## Next` unchanged.

- [ ] **Step 3: Commit**

```bash
git add docs/effects.md docs/roadmap.md
git commit -m "docs: restricted MIDAS effect guide + roadmap"
```

---

## Self-Review

**Spec coverage:** IdentityTransform (T1) ✓; ParametricDesignBlock (T2) ✓; kernels+builder (T3) ✓; MIDASParametric spec+exports (T4) ✓; compiler fixed+estimated+bounds+prediction-entry (T5) ✓; prediction block + fitted-weights fixup (T6) ✓; docs (T7) ✓. The spec's "no new inference code" holds — T5 reuses `ParametricDesignBlock.materialize` through the existing optimize/integrate path.

**Placeholder scan:** every code step carries runnable code; test steps carry assertions; commands have expected output. No TBD/TODO.

**Type consistency:** `midas_weights(kernel, n_lags, theta)` and `build_midas_parametric(frame, name, columns, kernel, theta, prior_precision)` are used identically in T3/T5/T6. The prediction `theta_spec` is names-or-floats out of T5 and resolved to floats by `_context_with_fitted_weights` (T6) before `_midas_parametric_block` (T6) consumes it. `_resolved_precision` reused for shape plug-in resolution. `ParametricDesignBlock.parameters` is validated by the family checks extended in T2.

**Cross-task ordering:** T6's recovery test uses `predict()`, which needs both the prediction block (T6) and the fitted-weights fixup (T6) — self-contained within T6. T5's tests avoid `predict()` (assert on `result.hyperparameters` / `latent_marginals`), so T5 is independently testable before T6 lands.

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints.

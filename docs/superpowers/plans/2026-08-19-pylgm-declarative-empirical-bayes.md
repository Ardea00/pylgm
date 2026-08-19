# pyLGM Declarative Empirical-Bayes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make declared `Hyperparameter`s get estimated by type-II ML (empirical Bayes) in `LGM.fit`, for both the exact-Gaussian and Laplace engines, reusing the existing optimizer.

**Architecture:** A likelihood-agnostic `CompiledFamily` (built by the declarative compiler when the model declares any `Hyperparameter`) exposes `parameter_names` + `materialize(params) -> CompiledLGM` via a likelihood factory. `optimize_empirical_bayes` is relaxed to accept any family with that structural shape and drives it with `fit=fit_gaussian` or `fit=fit_laplace`. `LGM.fit` runs EB automatically when hyperparameters are declared and attaches the optimized estimates to the returned posterior result; otherwise the current fixed-plug-in fast path runs unchanged.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`optimize.minimize` L-BFGS-B), Pandas, pytest, Ruff. PySpark optional (existing adapter, unchanged).

## Global Constraints

- No new runtime dependencies.
- Objective is **type-II ML** (pure EB): the existing `optimize_empirical_bayes` minimizes `-log_marginal_likelihood`. Do NOT add prior penalties (penalized MAP-II is deferred).
- Do not change existing behavior: a model with no `Hyperparameter` must produce a result identical to today; `CompiledGaussianFamily` and the legacy schema-v2 EB path must be untouched in behavior (their tests are the guard).
- Reuse `ScalableBlock`, `_assemble_compiled_model`, `_validate_parameter_mapping`, and `optimize_empirical_bayes`; do not duplicate their logic.
- EB is triggered automatically by any `Hyperparameter` on `sigma` or an effect `precision`; plain floats stay fixed. `LGM.fit`'s return type stays `GaussianResult`/`LaplaceResult`.
- Optimized estimates are exposed via a new read-only `result.hyperparameters: Mapping[str, float] | None` (default `None`).
- Preserve the Pandas caller-order and Spark canonical-key contracts for posterior arrays.
- YAML EB is out of scope (YAML stays numeric plug-in).
- Use `PYTHONPATH=src` for all test and lint commands.
- Spec: `docs/superpowers/specs/2026-08-19-pylgm-declarative-empirical-bayes-design.md`.

---

## File Map

- `src/pylgm/parameters.py`: optional `lower`/`upper` on `Hyperparameter`.
- `src/pylgm/ir/family.py`: `_materialize_blocks` helper; `CompiledFamily`.
- `src/pylgm/ir/__init__.py`: export `CompiledFamily`.
- `src/pylgm/optimization/empirical_bayes.py`: structural family check.
- `src/pylgm/optimization/result.py`: `EmpiricalBayesResult.fit` annotation.
- `src/pylgm/inference/result.py`: `hyperparameters` field on both result types.
- `src/pylgm/compiler.py`: `compile_family` + declared-hyperparameter collection.
- `src/pylgm/model.py`: EB dispatch in `_fit_pandas`/`_fit_spark`; result rebuild carrying `hyperparameters`.
- `examples/empirical_bayes/`: runnable example.
- Tests: `tests/test_modeling_vocabulary.py`, `tests/ir/test_family.py` (or existing `test_gaussian_family.py`), `tests/optimization/test_empirical_bayes.py`, `tests/inference/test_result.py`, `tests/test_compiler.py`, `tests/test_model.py`, README + arch-spec docs.

---

### Task 1: Hyperparameter Bounds

**Files:**
- Modify: `src/pylgm/parameters.py`
- Modify: `tests/test_modeling_vocabulary.py`

**Interfaces:**
- Produces: `Hyperparameter(name, initial, prior=None, transform="log", lower=None, upper=None)` where resolved `lower`/`upper` are read-only float attributes satisfying `0 < lower < initial < upper`; defaults `lower = initial*1e-3`, `upper = initial*1e3`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_modeling_vocabulary.py
import pytest
from pylgm import Hyperparameter


def test_hyperparameter_default_bounds_bracket_initial():
    hp = Hyperparameter("p", initial=2.0)
    assert hp.lower == pytest.approx(2.0e-3)
    assert hp.upper == pytest.approx(2.0e3)


def test_hyperparameter_explicit_bounds():
    hp = Hyperparameter("p", initial=1.0, lower=0.1, upper=10.0)
    assert (hp.lower, hp.upper) == (0.1, 10.0)


def test_hyperparameter_rejects_bounds_not_bracketing_initial():
    with pytest.raises(ValueError, match="lower"):
        Hyperparameter("p", initial=1.0, lower=2.0)
    with pytest.raises(ValueError, match="upper"):
        Hyperparameter("p", initial=1.0, upper=0.5)
    with pytest.raises(ValueError):
        Hyperparameter("p", initial=1.0, lower=-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q`
Expected: FAIL (`Hyperparameter` has no `lower`).

- [ ] **Step 3: Implement bounds**

Add fields to the `Hyperparameter` dataclass after `transform`:

```python
    lower: float | None = None
    upper: float | None = None
```

In `__post_init__`, after the existing validation, add:

```python
        lower = self.initial * 1e-3 if self.lower is None else _positive_real(self.lower, "lower")
        upper = self.initial * 1e3 if self.upper is None else _positive_real(self.upper, "upper")
        if not lower < self.initial:
            raise ValueError("lower bound must be below initial")
        if not self.initial < upper:
            raise ValueError("upper bound must be above initial")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/parameters.py tests/test_modeling_vocabulary.py
git commit -m "feat: add optional bounds to Hyperparameter"
```

---

### Task 2: Likelihood-Agnostic CompiledFamily

**Files:**
- Modify: `src/pylgm/ir/family.py`
- Modify: `src/pylgm/ir/__init__.py`
- Create: `tests/ir/test_compiled_family.py`

**Interfaces:**
- Consumes: `ScalableBlock`, `LatentBlock`, `_assemble_compiled_model`, `_validate_parameter_mapping`.
- Produces: module-level `_materialize_blocks(blocks: tuple[ScalableBlock, ...], resolved: Mapping[str, float]) -> tuple[LatentBlock, ...]`.
- Produces: `CompiledFamily(y, observed, offset, blocks: tuple[ScalableBlock, ...], parameter_names: tuple[str, ...], likelihood_factory: Callable[[Mapping[str, float]], object])` with `.materialize(values) -> CompiledLGM` and `.parameter_names`.
- Preserves: `CompiledGaussianFamily` public behavior (now delegating its scaling loop to `_materialize_blocks`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ir/test_compiled_family.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson


def _block(name, precision_value):
    return LatentBlock(
        name=name,
        labels=("a", "b"),
        design=csr_matrix(np.eye(2)),
        precision=eye(2, format="csr") * precision_value,
        constraints=np.empty((0, 2), dtype=float),
    )


def _family(factory, parameter_names, blocks):
    return CompiledFamily(
        y=np.array([1.0, 2.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        blocks=blocks,
        parameter_names=parameter_names,
        likelihood_factory=factory,
    )


def test_compiled_family_scales_bound_block_and_builds_likelihood():
    blocks = (ScalableBlock(_block("g", 1.0), "g_prec", 1.0),)
    family = _family(lambda r: CompiledGaussian(0.5), ("g_prec",), blocks)
    compiled = family.materialize({"g_prec": 4.0})
    # unit-base precision (eye) scaled by 4.0
    np.testing.assert_allclose(compiled.precision.toarray(), np.eye(2) * 4.0)
    assert isinstance(compiled.likelihood, CompiledGaussian)
    assert compiled.likelihood.sigma == 0.5


def test_compiled_family_non_gaussian_factory():
    blocks = (ScalableBlock(_block("g", 2.0), None, 1.0),)
    family = _family(lambda r: CompiledPoisson(), (), blocks)
    compiled = family.materialize({})
    assert isinstance(compiled.likelihood, CompiledPoisson)
    np.testing.assert_allclose(compiled.precision.toarray(), np.eye(2) * 2.0)


def test_compiled_family_rejects_wrong_parameters():
    blocks = (ScalableBlock(_block("g", 1.0), "g_prec", 1.0),)
    family = _family(lambda r: CompiledPoisson(), ("g_prec",), blocks)
    with pytest.raises(Exception):
        family.materialize({"wrong": 1.0})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/ir/test_compiled_family.py -q`
Expected: FAIL (`CompiledFamily` absent).

- [ ] **Step 3: Extract the shared scaling helper**

In `src/pylgm/ir/family.py`, add a module-level helper and refactor `CompiledGaussianFamily.materialize` to use it. The helper is the existing per-block scaling loop:

```python
def _materialize_blocks(
    blocks: tuple[ScalableBlock, ...], resolved: Mapping[str, float]
) -> tuple[LatentBlock, ...]:
    materialized: list[LatentBlock] = []
    for item in blocks:
        multiplier = resolved[item.parameter] if item.parameter else item.scale
        with np.errstate(over="ignore", invalid="ignore"):
            precision = item.block.precision * multiplier
        if not np.isfinite(precision.data).all():
            raise NumericalError(
                f"precision scaling for block {item.block.name!r} must remain finite"
            )
        materialized.append(
            LatentBlock(
                item.block.name,
                item.block.labels,
                item.block.design,
                precision,
                item.block.constraints,
            )
        )
    return tuple(materialized)
```

Replace the body of `CompiledGaussianFamily.materialize` after `resolved = _validate_parameter_mapping(...)` with:

```python
        blocks = _materialize_blocks(self.blocks, resolved)
        likelihood = CompiledGaussian(resolved.get("sigma", self.initial.sigma))
        return _assemble_compiled_model(
            self._y, self._observed, self._offset, blocks, likelihood
        )
```

- [ ] **Step 4: Add `CompiledFamily`**

```python
from collections.abc import Callable  # add to imports


@dataclass(frozen=True, init=False)
class CompiledFamily:
    """A likelihood-agnostic optimisable family for the declarative path."""

    _y: np.ndarray = field(repr=False)
    _observed: np.ndarray = field(repr=False)
    _offset: np.ndarray = field(repr=False)
    blocks: tuple[ScalableBlock, ...]
    parameter_names: tuple[str, ...]
    likelihood_factory: Callable[[Mapping[str, float]], object] = field(repr=False)

    def __init__(
        self,
        y: np.ndarray,
        observed: np.ndarray,
        offset: np.ndarray,
        blocks: tuple[ScalableBlock, ...],
        parameter_names: tuple[str, ...],
        likelihood_factory: Callable[[Mapping[str, float]], object],
    ) -> None:
        y_value = np.asarray(y)
        observed_value = np.asarray(observed)
        offset_value = np.asarray(offset)
        if y_value.ndim != 1 or not np.issubdtype(y_value.dtype, np.number) or not np.isrealobj(y_value):
            raise ModelValidationError("family y must be a one-dimensional numeric array")
        if observed_value.ndim != 1 or not np.issubdtype(observed_value.dtype, np.bool_):
            raise ModelValidationError("family observed must be a one-dimensional boolean array")
        if (
            offset_value.ndim != 1
            or not np.issubdtype(offset_value.dtype, np.number)
            or not np.isrealobj(offset_value)
            or not np.isfinite(offset_value).all()
        ):
            raise ModelValidationError("family offset must be a finite one-dimensional numeric array")
        if not (y_value.size == observed_value.size == offset_value.size):
            raise ModelValidationError("family arrays must have equal row counts")
        block_values = tuple(blocks)
        names = tuple(parameter_names)
        if any(not isinstance(item, ScalableBlock) for item in block_values):
            raise ModelValidationError("family blocks must be scalable blocks")
        if any(item.block.design.shape[0] != y_value.size for item in block_values):
            raise ModelValidationError("family block design rows must match the response")
        block_names = [item.block.name for item in block_values]
        if len(block_names) != len(set(block_names)):
            raise ModelValidationError("family block names must be unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise ModelValidationError("parameter names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ModelValidationError("parameter names must be unique")
        bound = tuple(item.parameter for item in block_values if item.parameter is not None)
        if len(bound) != len(set(bound)):
            raise ModelValidationError("scalable block parameters must be unique")
        for parameter in bound:
            if parameter not in names:
                raise ModelValidationError(
                    f"bound block parameter {parameter!r} must appear in parameter_names"
                )
        if not callable(likelihood_factory):
            raise ModelValidationError("likelihood_factory must be callable")
        object.__setattr__(self, "_y", _readonly_array(y_value))
        object.__setattr__(self, "_observed", _readonly_array(observed_value))
        object.__setattr__(self, "_offset", _readonly_array(offset_value))
        object.__setattr__(self, "blocks", block_values)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "likelihood_factory", likelihood_factory)

    def materialize(self, values: Mapping[str, float]) -> CompiledLGM:
        resolved = _validate_parameter_mapping(values, self.parameter_names)
        blocks = _materialize_blocks(self.blocks, resolved)
        likelihood = self.likelihood_factory(resolved)
        return _assemble_compiled_model(
            self._y, self._observed, self._offset, blocks, likelihood
        )
```

Export `CompiledFamily` from `src/pylgm/ir/__init__.py` and its `__all__`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/ir/test_compiled_family.py tests/ir/test_gaussian_family.py -q`
Expected: PASS (CompiledGaussianFamily behavior unchanged by the extraction).

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/ir/family.py src/pylgm/ir/__init__.py tests/ir/test_compiled_family.py
git commit -m "feat: add likelihood-agnostic CompiledFamily"
```

---

### Task 3: Generalize the Empirical-Bayes Optimizer

**Files:**
- Modify: `src/pylgm/optimization/empirical_bayes.py`
- Modify: `src/pylgm/optimization/result.py`
- Modify: `tests/optimization/test_empirical_bayes.py`

**Interfaces:**
- Consumes: `CompiledFamily`, `fit_laplace`.
- Produces: `optimize_empirical_bayes` accepting any family exposing a `parameter_names` tuple and a callable `materialize` (structural), unchanged for `CompiledGaussianFamily`.
- Produces: `EmpiricalBayesResult.fit` typed `GaussianResult | LaplaceResult`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/optimization/test_empirical_bayes.py
import numpy as np
from scipy.sparse import csr_matrix, eye

from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes
from pylgm.inference.laplace import fit_laplace


def _unit_block(name):
    return LatentBlock(name, ("a", "b"), csr_matrix(np.eye(2)), eye(2, format="csr"),
                       np.empty((0, 2), dtype=float))


def test_optimize_over_compiled_family_gaussian():
    blocks = (ScalableBlock(_unit_block("g"), "g_prec", 1.0),)
    family = CompiledFamily(
        y=np.array([1.0, -1.0]), observed=np.array([True, True]), offset=np.zeros(2),
        blocks=blocks, parameter_names=("g_prec",),
        likelihood_factory=lambda r: CompiledGaussian(1.0),
    )
    result = optimize_empirical_bayes(family, {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)})
    assert result.diagnostics.converged
    assert "g_prec" in result.parameters


def test_optimize_over_laplace_family_converges():
    blocks = (ScalableBlock(_unit_block("g"), "g_prec", 1.0),)
    family = CompiledFamily(
        y=np.array([2.0, 4.0]), observed=np.array([True, True]), offset=np.zeros(2),
        blocks=blocks, parameter_names=("g_prec",),
        likelihood_factory=lambda r: CompiledPoisson(),
    )
    result = optimize_empirical_bayes(
        family, {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)}, fit=fit_laplace
    )
    assert result.diagnostics.converged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/optimization/test_empirical_bayes.py -q`
Expected: FAIL (`_validate_problem` rejects a non-`CompiledGaussianFamily`).

- [ ] **Step 3: Relax the family check**

In `src/pylgm/optimization/empirical_bayes.py`, replace the `_validate_problem` type guard

```python
    if not isinstance(family, CompiledGaussianFamily):
        raise TypeError("family must be a compiled Gaussian family")
```

with a structural check:

```python
    parameter_names = getattr(family, "parameter_names", None)
    if not isinstance(parameter_names, tuple) or not callable(getattr(family, "materialize", None)):
        raise TypeError("family must expose parameter_names and a materialize method")
```

Then use `parameter_names` (the local) where the function currently reads `family.parameter_names`. Keep the `CompiledGaussianFamily` import if still referenced elsewhere; otherwise remove it to satisfy ruff.

In `src/pylgm/optimization/result.py`, change the `EmpiricalBayesResult.fit` annotation to `GaussianResult | LaplaceResult` and import `LaplaceResult` from `pylgm.inference`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/optimization/test_empirical_bayes.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/optimization/empirical_bayes.py src/pylgm/optimization/result.py tests/optimization/test_empirical_bayes.py
git commit -m "feat: drive empirical-Bayes over any optimisable family"
```

---

### Task 4: `hyperparameters` on Result Types

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Modify: `tests/inference/test_result.py`

**Interfaces:**
- Produces: `GaussianResult(..., hyperparameters: Mapping[str, float] | None = None)` and `LaplaceResult(..., hyperparameters: Mapping[str, float] | None = None)`, exposed as a read-only-copy `.hyperparameters` property (a fresh `dict`/`MappingProxyType`), default `None`. Validated as a string-keyed finite-float mapping when supplied.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/inference/test_result.py
def test_gaussian_result_hyperparameters_defaults_none_and_is_readonly():
    result = GaussianResult(
        labels=("x",), mean=np.array([0.0]), covariance=np.eye(1),
        log_marginal_likelihood=0.0, predictive_mean=np.array([1.0]),
        predictive_variance=np.array([1.0]),
    )
    assert result.hyperparameters is None
    result2 = GaussianResult(
        labels=("x",), mean=np.array([0.0]), covariance=np.eye(1),
        log_marginal_likelihood=0.0, predictive_mean=np.array([1.0]),
        predictive_variance=np.array([1.0]), hyperparameters={"p": 2.0},
    )
    exposed = result2.hyperparameters
    assert exposed == {"p": 2.0}
    with pytest.raises(Exception):
        exposed["p"] = 9.0  # read-only mapping
    assert result2.hyperparameters["p"] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`
Expected: FAIL (`hyperparameters` not accepted).

- [ ] **Step 3: Implement the field on both result types**

Add a shared module-level helper in `result.py`:

```python
def _readonly_hyperparameters(values):
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError("hyperparameters must be a mapping")
    resolved = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError("hyperparameter names must be non-empty strings")
        resolved[name] = float(value)
        if not np.isfinite(resolved[name]):
            raise ValueError("hyperparameter values must be finite")
    return MappingProxyType(resolved)
```

In both `GaussianResult.__init__` and `LaplaceResult.__init__`, add a keyword-only parameter `hyperparameters: Mapping[str, float] | None = None` (last parameter), store `object.__setattr__(self, "_hyperparameters", _readonly_hyperparameters(hyperparameters))`, and add a property:

```python
    @property
    def hyperparameters(self):
        return self._hyperparameters
```

(`MappingProxyType` is already read-only, so returning it directly is safe.)

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS (existing constructor calls omit `hyperparameters`, defaulting `None`).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference/result.py tests/inference/test_result.py
git commit -m "feat: expose optimized hyperparameters on results"
```

---

### Task 5: Declarative EB Compilation and `LGM.fit` Dispatch

**Files:**
- Modify: `src/pylgm/compiler.py`
- Modify: `src/pylgm/model.py`
- Modify: `tests/test_compiler.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `CompiledFamily`, `ScalableBlock`, `optimize_empirical_bayes`, `OptimizationBounds`, `Hyperparameter`, `build_fixed`/`build_iid`/`build_random_walk`, `fit_gaussian`/`fit_laplace`.
- Produces: `compile_family(model: LGM, panel: CanonicalPanel) -> CompiledFamily | None` (returns `None` when no `Hyperparameter` is declared).
- Produces: `LGM.fit` running EB automatically and attaching `result.hyperparameters` when hyperparameters are declared.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter


def test_gaussian_empirical_bayes_estimates_precision():
    rng = np.random.default_rng(0)
    regions = ["a", "b", "c", "d"]
    rows = []
    for t in range(1, 21):
        for r in regions:
            rows.append({"region": r, "t": t, "y": rng.normal()})
    frame = pd.DataFrame(rows)
    model = LGM(
        "y", Gaussian(0.5),
        Fixed("1") + IID("region", index="region",
                         precision=Hyperparameter("region_prec", initial=1.0)),
        panel=("region",), time="t",
    )
    result = model.fit(frame, engine="exact_gaussian")
    assert result.hyperparameters is not None
    assert "region_prec" in result.hyperparameters
    assert result.hyperparameters["region_prec"] > 0
    assert result.diagnostics["empirical_bayes_converged"] is True


def test_poisson_empirical_bayes_runs_via_laplace():
    rows = [{"region": r, "t": t, "y": float((t + (r == "b")) % 5)}
            for t in range(1, 16) for r in ["a", "b"]]
    frame = pd.DataFrame(rows)
    model = LGM(
        "y", Poisson(),
        Fixed("1") + IID("region", index="region",
                         precision=Hyperparameter("region_prec", initial=1.0)),
        panel=("region",), time="t",
    )
    result = model.fit(frame, engine="laplace")
    assert result.hyperparameters is not None and result.hyperparameters["region_prec"] > 0


def test_no_hyperparameter_model_has_none_hyperparameters():
    frame = pd.DataFrame({"t": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    result = LGM("y", Gaussian(1.0), Fixed("1"), time="t").fit(frame)
    assert result.hyperparameters is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: FAIL (no EB path; `hyperparameters` never populated).

- [ ] **Step 3: Implement `compile_family` in `compiler.py`**

Add helpers. `declared_hyperparameters(model)` returns the `(target_name, Hyperparameter)` pairs where `target_name` is the hyperparameter's own `.name`:

```python
def _model_hyperparameters(model) -> list:
    found = []
    if isinstance(model.likelihood, Gaussian) and isinstance(model.likelihood.sigma, Hyperparameter):
        found.append(("sigma", model.likelihood.sigma))
    for effect in model.predictor.effects:
        precision = getattr(effect, "precision", None)
        if isinstance(precision, Hyperparameter):
            found.append((effect.name, precision))
    return found


def compile_family(model, panel):
    """Build an optimisable family, or None when no Hyperparameter is declared."""
    hyper = _model_hyperparameters(model)
    if not hyper:
        return None
    frame = panel.frame
    offset = (
        frame[model.offset].to_numpy(dtype=float)
        if model.offset is not None else np.zeros(len(frame))
    )
    y = frame[panel.response].fillna(0.0).to_numpy(dtype=float)
    from pylgm.ir.family import ScalableBlock, CompiledFamily

    scalable = []
    parameter_names = []
    for effect in model.predictor.effects:
        precision = getattr(effect, "precision", None)
        optimised = isinstance(precision, Hyperparameter)
        value = 1.0 if optimised else (precision if not isinstance(precision, Hyperparameter) else precision.initial)
        if isinstance(effect, Fixed):
            block = build_fixed(frame, effect.formula, effect.prior_precision)
            scalable.append(ScalableBlock(block, None, 1.0))
        elif isinstance(effect, IID):
            block = build_iid(frame, effect.name, effect.index, value)
            scalable.append(ScalableBlock(block, precision.name if optimised else None, 1.0))
            if optimised:
                parameter_names.append(precision.name)
        else:
            order = 1 if isinstance(effect, RW1) else 2
            block = build_random_walk(frame, effect.name, effect.index, value, order)
            scalable.append(ScalableBlock(block, precision.name if optimised else None, 1.0))
            if optimised:
                parameter_names.append(precision.name)

    if isinstance(model.likelihood, Gaussian):
        sigma = model.likelihood.sigma
        if isinstance(sigma, Hyperparameter):
            parameter_names.append(sigma.name)
            sigma_name = sigma.name
            factory = lambda r: CompiledGaussian(r[sigma_name])
        else:
            fixed_sigma = sigma
            factory = lambda r: CompiledGaussian(fixed_sigma)
    else:
        likelihood = model.likelihood
        factory = lambda r: likelihood.materialize({})

    return CompiledFamily(
        y=y, observed=panel.observed, offset=offset,
        blocks=tuple(scalable), parameter_names=tuple(parameter_names),
        likelihood_factory=factory,
    )
```

Import `CompiledGaussian` and `Hyperparameter` at the top of `compiler.py` if not present. Note: `build_iid`/`build_random_walk` called with `value=1.0` build the unit-base precision that the bound `ScalableBlock` parameter then scales.

- [ ] **Step 4: Implement EB dispatch in `model.py`**

Add imports: `from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes`. Add a helper that reconstructs a result with `hyperparameters` set (extend `_align_predictions_with_source_rows` to also carry `hyperparameters`, and add a plain attach used before Spark keys):

Extend `_align_predictions_with_source_rows` to pass `hyperparameters=result.hyperparameters` in both the `LaplaceResult` and `GaussianResult` reconstructions. In `_fit_spark`, likewise pass `hyperparameters=result.hyperparameters` in both reconstructions.

Add:

```python
    def _run_empirical_bayes(self, family, engine):
        from pylgm.compiler import _model_hyperparameters

        fit = self._engine(engine)
        bounds = {}
        initial = {}
        for _, hp in _model_hyperparameters(self):
            bounds[hp.name] = OptimizationBounds(hp.initial, hp.lower, hp.upper)
            initial[hp.name] = hp.initial
        eb = optimize_empirical_bayes(family, bounds, initial=initial, fit=fit)
        base = eb.fit
        diagnostics = dict(base.diagnostics)
        diagnostics["empirical_bayes_converged"] = eb.diagnostics.converged
        diagnostics["empirical_bayes_evaluations"] = eb.diagnostics.evaluations
        # rebuild base with hyperparameters + EB diagnostics
        return _attach_estimates(base, dict(eb.parameters), diagnostics)
```

where `_attach_estimates(result, hyperparameters, diagnostics)` is a module-level helper that rebuilds the result of the correct type with `hyperparameters=hyperparameters` and `diagnostics=diagnostics`, arrays unchanged (mirror the branch structure already in `_align_predictions_with_source_rows`, but without reordering).

Then in `_fit_pandas`, replace the compile/fit block with:

```python
        from pylgm.compiler import compile_lgm, compile_family

        family = compile_family(self, panel)
        if family is None:
            compiled = compile_lgm(self, panel)
            result = self._engine(engine)(compiled)
        else:
            self._engine(engine)  # validates engine/likelihood pairing, raises on mismatch
            result = self._run_empirical_bayes(family, engine)
        return _align_predictions_with_source_rows(result, panel.source_positions)
```

And in `_fit_spark`, symmetric: build `family = compile_family(self, canonical.panel)`; if `None` use `compile_lgm` + engine fit; else `result = self._run_empirical_bayes(family, engine)`; then rebuild with `prediction_keys=canonical.prediction_keys` (carrying `hyperparameters`).

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/test_model.py tests/test_compiler.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py src/pylgm/model.py tests/test_compiler.py tests/test_model.py
git commit -m "feat: run declarative empirical-Bayes in LGM.fit"
```

---

### Task 6: Example, Docs, and Roadmap

**Files:**
- Create: `examples/empirical_bayes/config.yaml` is NOT added (YAML EB out of scope); create `examples/empirical_bayes/run.py`, `examples/empirical_bayes/data.csv`, `examples/empirical_bayes/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`
- Modify: `tests/test_package.py`

**Interfaces:**
- Produces: a runnable Python example declaring a `Hyperparameter` precision and printing `result.hyperparameters`.

- [ ] **Step 1: Write the example smoke test**

```python
# append to tests/test_package.py
def test_empirical_bayes_example_reports_estimates():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/empirical_bayes/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hyperparameters" in completed.stdout
```

(Reuse the `os`/`sys`/`subprocess`/`Path` imports already at the top of `tests/test_package.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_package.py -q`
Expected: FAIL (example missing).

- [ ] **Step 3: Create the example**

`examples/empirical_bayes/data.csv` — a small panel (`region,time,y`) with enough rows for a stable estimate. `run.py` — read the CSV, build an `LGM` with an `IID` precision declared as `Hyperparameter("region_precision", initial=1.0)`, `model.fit(frame, engine="exact_gaussian")`, and print `result.hyperparameters` and `result.diagnostics["empirical_bayes_converged"]`. `README.md` — how to run with `PYTHONPATH=src`, and the note that estimation is type-II ML at fixed priors.

- [ ] **Step 4: Update docs**

README: add an "Empirical Bayes" subsection — declaring a `Hyperparameter` (with optional `lower`/`upper`) makes `LGM.fit` estimate it by type-II ML for both engines; the estimate appears on `result.hyperparameters`; priors are declaration metadata (used later by INLA); YAML EB is not yet supported.

Arch spec (`2026-08-08-...`): mark declarative empirical Bayes shipped for the Gaussian and Laplace engines, and **add penalized MAP-II (prior-penalized hyperparameter estimation) as an explicit roadmap item** alongside INLA-style integration, preserving all other deferred scope.

- [ ] **Step 5: Run tests and lint**

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

Run: `PYTHONPATH=src ruff check src tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/empirical_bayes README.md docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md tests/test_package.py
git commit -m "docs: publish declarative empirical-Bayes"
```

---

## Self-Review Checklist

- Task 1 (bounds) → Task 2 (family) → Task 3 (optimizer) → Task 4 (result field) → Task 5 (compile + dispatch) → Task 6 (docs). Dependencies are linear.
- Every spec section maps to a task: bounds (T1), CompiledFamily + shared `_materialize_blocks` (T2), optimizer generalization + result annotation (T3), `result.hyperparameters` (T4), declarative EB compilation + `LGM.fit` dispatch + Pandas/Spark (T5), docs/example/MAP-II roadmap (T6).
- Correctness anchors: CompiledFamily-vs-CompiledGaussianFamily parity and `_materialize_blocks` extraction (T2, guarded by existing 14 family tests); optimizer parity + Laplace convergence (T3); closed-form/estimation and no-Hyperparameter invariance (T5).
- Type-II ML only; priors untouched; `CompiledGaussianFamily`/legacy path behavior preserved; no new dependencies.
- Deferred (unimplemented, unblocked): penalized MAP-II (roadmap note added in T6), YAML EB, hyperparameter marginals, shared result base.

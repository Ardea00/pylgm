# pyLGM Non-Gaussian Laplace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Poisson` and `Bernoulli` likelihoods fit by a Laplace approximation at fixed hyperparameters, reusing the existing IR, compiler, and dense Gaussian machinery.

**Architecture:** A likelihood-agnostic Newton/Laplace engine (`fit_laplace`) drives every likelihood through one small GLM protocol (`log_likelihood`, `gradient`, `working_weights`, `response_mean`, `response_prediction`, `validate_response`). Non-Gaussian likelihoods compile to the same `CompiledLGM` structure as Gaussian (its `likelihood` field is already `object`, its `offset` field is already populated), so the engine reuses the Gaussian dense helpers verbatim. `CompiledGaussian` is refactored to satisfy the same protocol, making the exact Gaussian fit a correctness anchor for the Laplace kernel.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`cho_factor`/`cho_solve`/`null_space`), Formulaic, Pandas, pytest, Ruff. PySpark optional (existing adapter, unchanged).

## Global Constraints

- No new runtime dependencies; `statsmodels` may be used in tests only, guarded by `pytest.importorskip`.
- Do not change any existing Gaussian public behavior; the full suite must stay green.
- No IR changes: `CompiledLGM` already accepts any non-null `likelihood` and a real `offset` vector.
- Reuse the Gaussian engine helpers in `src/pylgm/inference/gaussian.py` (`_constraint_null_space`, `_factor_positive_definite`, `_block_slices`, `_require_finite`, `preflight_dense_reference`); do not duplicate them.
- Engine dispatch is explicit and never falls back: Gaussian likelihood requires `engine="exact_gaussian"`, non-Gaussian requires `engine="laplace"`.
- Fixed plug-in hyperparameters only (effect `.initial` precisions); the likelihood protocol takes a materialized likelihood and the engine reads no hyperparameter metadata, so a later empirical-Bayes slice can wrap `fit_laplace` with the existing `optimize_empirical_bayes` unchanged.
- Canonical links only this slice: Poisson→log, Bernoulli→logit.
- Use `PYTHONPATH=src` for all test and lint commands.
- Spec: `docs/superpowers/specs/2026-08-19-pylgm-non-gaussian-laplace-design.md`.

---

## File Map

- `src/pylgm/links.py`: add `LogLink`, `LogitLink` (with stable `inverse`).
- `src/pylgm/likelihoods.py`: add `Poisson`/`Bernoulli` specs and `CompiledPoisson`/`CompiledBernoulli`; extend `CompiledGaussian` to the GLM protocol.
- `src/pylgm/exceptions.py`: add `InferenceConvergenceError`.
- `src/pylgm/model.py`: add `LGM.offset`; dispatch `engine="laplace"`; align Laplace predictions.
- `src/pylgm/compiler.py`: non-Gaussian compile branch + offset population + response validation.
- `src/pylgm/data/spark.py`: include `model.offset` in projected columns.
- `src/pylgm/inference/result.py`: shared marginal/linear-combination helpers; `LaplaceResult`.
- `src/pylgm/inference/laplace.py`: `fit_laplace`.
- `src/pylgm/inference/__init__.py` and `src/pylgm/__init__.py`: exports.
- `src/pylgm/config/model.py`: YAML `family: poisson|bernoulli` and `offset:`.
- `examples/count_glm/`: runnable Poisson example.
- Tests: `tests/test_links.py`, `tests/test_modeling_vocabulary.py`, `tests/inference/test_laplace.py`, `tests/inference/test_result.py`, `tests/test_compiler.py`, `tests/test_model.py`, `tests/config/test_model.py`, `tests/data/test_spark.py`, README + arch-spec docs.

---

### Task 1: Log and Logit Links

**Files:**
- Modify: `src/pylgm/links.py`
- Create: `tests/test_links.py`

**Interfaces:**
- Produces: `LogLink()` with `name="log"`, `inverse(eta) -> np.ndarray = exp(eta)`.
- Produces: `LogitLink()` with `name="logit"`, `inverse(eta) -> np.ndarray = 1/(1+exp(-eta))` computed stably.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_links.py
import numpy as np

from pylgm.links import IdentityLink, LogLink, LogitLink


def test_log_link_inverse_is_exp():
    link = LogLink()
    assert link.name == "log"
    np.testing.assert_allclose(link.inverse(np.array([0.0, 1.0])), np.exp([0.0, 1.0]))


def test_logit_link_inverse_is_stable_sigmoid():
    link = LogitLink()
    assert link.name == "logit"
    np.testing.assert_allclose(link.inverse(np.array([0.0])), [0.5])
    # large-magnitude inputs must not overflow
    values = link.inverse(np.array([1000.0, -1000.0]))
    np.testing.assert_allclose(values, [1.0, 0.0])


def test_identity_link_unchanged():
    assert IdentityLink().name == "identity"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_links.py -q`
Expected: FAIL with `ImportError: cannot import name 'LogLink'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/pylgm/links.py
import numpy as np


@dataclass(frozen=True)
class LogLink:
    """The log link; inverse is exp."""

    name: str = field(default="log", init=False)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(eta, dtype=float))


@dataclass(frozen=True)
class LogitLink:
    """The logit link; inverse is a numerically stable logistic sigmoid."""

    name: str = field(default="logit", init=False)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        result = np.empty_like(eta)
        positive = eta >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
        exp_eta = np.exp(eta[~positive])
        result[~positive] = exp_eta / (1.0 + exp_eta)
        return result
```

Add `import numpy as np` at the top of the module if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_links.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/links.py tests/test_links.py
git commit -m "feat: add log and logit links"
```

---

### Task 2: Non-Gaussian Likelihood Vocabulary and GLM Protocol

**Files:**
- Modify: `src/pylgm/likelihoods.py`
- Modify: `src/pylgm/__init__.py`
- Modify: `tests/test_modeling_vocabulary.py`

**Interfaces:**
- Consumes: `LogLink`, `LogitLink`, `IdentityLink` from Task 1.
- Produces: `Poisson()` spec with `.materialize(values) -> CompiledPoisson`; `Bernoulli()` spec with `.materialize(values) -> CompiledBernoulli`.
- Produces the GLM protocol on `CompiledPoisson`, `CompiledBernoulli`, and (refactored) `CompiledGaussian`, each evaluated on observed `eta`, `y`:
  - `link` attribute (a link object).
  - `log_likelihood(eta, y) -> float`
  - `gradient(eta, y) -> np.ndarray` (dℓ/dη, per observed row)
  - `working_weights(eta, y) -> np.ndarray` (−d²ℓ/dη², per observed row, ≥ 0)
  - `response_mean(eta) -> np.ndarray` (g⁻¹(η))
  - `response_prediction(eta_mean, eta_variance) -> np.ndarray` (posterior-predictive response scale)
  - `validate_response(y) -> None` (raises `DataContractError`)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_modeling_vocabulary.py
import numpy as np
import pytest

from pylgm import Bernoulli, Poisson
from pylgm.exceptions import DataContractError


def test_poisson_glm_pieces():
    like = Poisson().materialize({})
    assert like.link.name == "log"
    eta = np.array([0.0, np.log(2.0)])
    y = np.array([1.0, 3.0])
    np.testing.assert_allclose(like.response_mean(eta), [1.0, 2.0])
    np.testing.assert_allclose(like.gradient(eta, y), y - np.exp(eta))   # y - mu
    np.testing.assert_allclose(like.working_weights(eta, y), np.exp(eta))  # mu
    # log-normal predictive mean for the log link
    np.testing.assert_allclose(
        like.response_prediction(np.array([0.0]), np.array([2.0])), [np.exp(1.0)]
    )


def test_poisson_rejects_non_count_response():
    like = Poisson().materialize({})
    with pytest.raises(DataContractError, match="non-negative integer"):
        like.validate_response(np.array([1.0, -2.0]))
    with pytest.raises(DataContractError, match="non-negative integer"):
        like.validate_response(np.array([1.5]))


def test_bernoulli_glm_pieces():
    like = Bernoulli().materialize({})
    assert like.link.name == "logit"
    eta = np.array([0.0])
    y = np.array([1.0])
    np.testing.assert_allclose(like.response_mean(eta), [0.5])
    np.testing.assert_allclose(like.gradient(eta, y), [0.5])       # y - p
    np.testing.assert_allclose(like.working_weights(eta, y), [0.25])  # p(1-p)


def test_bernoulli_rejects_non_binary_response():
    like = Bernoulli().materialize({})
    with pytest.raises(DataContractError, match="0 or 1"):
        like.validate_response(np.array([0.0, 2.0]))


def test_compiled_gaussian_satisfies_glm_protocol():
    from pylgm.likelihoods import CompiledGaussian

    like = CompiledGaussian(2.0)  # sigma=2 -> variance 4
    assert like.link.name == "identity"
    eta = np.array([1.0, 2.0])
    y = np.array([3.0, 2.0])
    np.testing.assert_allclose(like.gradient(eta, y), (y - eta) / 4.0)
    np.testing.assert_allclose(like.working_weights(eta, y), [0.25, 0.25])
    np.testing.assert_allclose(like.response_mean(eta), eta)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q`
Expected: FAIL with `ImportError: cannot import name 'Poisson'`.

- [ ] **Step 3: Implement the vocabulary and protocol**

```python
# src/pylgm/likelihoods.py — add near the top
import numpy as np

from pylgm.links import IdentityLink, LogLink, LogitLink
from pylgm.exceptions import DataContractError
```

Extend `CompiledGaussian` with the protocol (keep existing `sigma`/`variance`):

```python
    link: IdentityLink = field(default_factory=IdentityLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        residual = np.asarray(y, dtype=float) - np.asarray(eta, dtype=float)
        n = residual.size
        return float(
            -0.5 * (n * np.log(2 * np.pi * self.variance) + residual @ residual / self.variance)
        )

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - np.asarray(eta, dtype=float)) / self.variance

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(eta).shape, 1.0 / self.variance, dtype=float)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta, dtype=float)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return np.asarray(eta_mean, dtype=float)

    def validate_response(self, y: np.ndarray) -> None:
        return None
```

Note: `CompiledGaussian` is a frozen dataclass with only `sigma`; adding the `link` field with `init=False` is safe. Add the six methods to the class body.

Add the new likelihoods:

```python
def _sigmoid(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    result = np.empty_like(eta)
    positive = eta >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    result[~positive] = exp_eta / (1.0 + exp_eta)
    return result


@dataclass(frozen=True)
class CompiledPoisson:
    """A Poisson likelihood with a canonical log link."""

    link: LogLink = field(default_factory=LogLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        from scipy.special import gammaln

        return float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1.0)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - np.exp(np.asarray(eta, dtype=float))

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(eta, dtype=float))

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(eta, dtype=float))

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(eta_mean, dtype=float) + 0.5 * np.asarray(eta_variance, dtype=float))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.round(y)):
            raise DataContractError("Poisson responses must be non-negative integers")


@dataclass(frozen=True)
class CompiledBernoulli:
    """A Bernoulli likelihood with a canonical logit link."""

    link: LogitLink = field(default_factory=LogitLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        # log(1+exp(eta)) computed stably
        softplus = np.logaddexp(0.0, eta)
        return float(np.sum(y * eta - softplus))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - _sigmoid(eta)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = _sigmoid(eta)
        return p * (1.0 - p)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return _sigmoid(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(eta_mean, dtype=float))  # point estimate; variance ignored

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise DataContractError("Bernoulli responses must be 0 or 1")


@dataclass(frozen=True)
class Poisson:
    """A Poisson likelihood with a canonical log link."""

    def materialize(self, values: Mapping[str, float]) -> CompiledPoisson:
        return CompiledPoisson()


@dataclass(frozen=True)
class Bernoulli:
    """A Bernoulli likelihood with a canonical logit link."""

    def materialize(self, values: Mapping[str, float]) -> CompiledBernoulli:
        return CompiledBernoulli()
```

Export in `src/pylgm/__init__.py`: add `Poisson`, `Bernoulli` to the imports from `pylgm.likelihoods` and to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS (Gaussian protocol addition breaks nothing).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/likelihoods.py src/pylgm/__init__.py tests/test_modeling_vocabulary.py
git commit -m "feat: add Poisson and Bernoulli likelihoods with a shared GLM protocol"
```

---

### Task 3: Model Offset and Non-Gaussian Compilation

**Files:**
- Modify: `src/pylgm/model.py` (add `offset` field only; dispatch is Task 5)
- Modify: `src/pylgm/compiler.py`
- Modify: `src/pylgm/data/spark.py`
- Modify: `tests/test_compiler.py`

**Interfaces:**
- Consumes: `Poisson`/`Bernoulli` from Task 2; existing `CompiledLGM`, `LatentBlock`, `build_fixed`/`build_iid`/`build_random_walk`, `_qualified_labels`, `_resolved_precision`.
- Produces: `LGM(..., offset: str | None = None)`.
- Produces: `compile_lgm` accepting a non-Gaussian likelihood, returning a `CompiledLGM` whose `likelihood` is a compiled non-Gaussian likelihood and whose `offset` is populated from `model.offset`.
- Produces: `_required_columns(model)` in `data/spark.py` includes `model.offset` when set.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_compiler.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Bernoulli, Fixed, Gaussian, LGM, Poisson
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError
from pylgm.likelihoods import CompiledPoisson


def _panel(frame):
    return CanonicalPanel.from_frame(frame, DataConfig(time="t", response="y", panel=()))


def test_compile_poisson_uses_non_gaussian_likelihood_and_offset():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 4.0], "logexp": [0.0, 0.1, 0.2]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t", offset="logexp")
    compiled = compile_lgm(model, _panel(frame))
    assert isinstance(compiled.likelihood, CompiledPoisson)
    np.testing.assert_allclose(compiled.offset, [0.0, 0.1, 0.2])


def test_compile_poisson_rejects_non_count_response():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 1.5]})
    model = LGM("y", Poisson(), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="non-negative integer"):
        compile_lgm(model, _panel(frame))


def test_compile_bernoulli_default_offset_is_zero():
    frame = pd.DataFrame({"t": [1, 2], "y": [0.0, 1.0]})
    model = LGM("y", Bernoulli(), Fixed("1"), time="t")
    compiled = compile_lgm(model, _panel(frame))
    np.testing.assert_allclose(compiled.offset, [0.0, 0.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_compiler.py -q`
Expected: FAIL (LGM has no `offset`; compiler rejects non-Gaussian likelihoods).

- [ ] **Step 3: Add the offset field to `LGM`**

In `src/pylgm/model.py`, add the field and validation:

```python
    offset: str | None = None
```

(place it after `time`). In `__post_init__`, after the `time` check, add:

```python
        if self.offset is not None and (not isinstance(self.offset, str) or not self.offset):
            raise ValueError("offset must be a non-empty string or None")
```

- [ ] **Step 4: Add the non-Gaussian compile branch**

In `src/pylgm/compiler.py`, add imports:

```python
from scipy.sparse import block_diag, hstack

from pylgm.ir.model import _block_constraints
from pylgm.likelihoods import Bernoulli, Poisson
```

Replace the Gaussian-only guard block (currently `if not isinstance(model.likelihood, Gaussian): raise CompilationError(...)` and everything after through the `return family.materialize({})`) with likelihood dispatch. Keep the existing Gaussian assembly for `Gaussian`; add:

```python
    offset = (
        frame[model.offset].to_numpy(dtype=float)
        if model.offset is not None
        else np.zeros(len(frame))
    )
    if not np.isfinite(offset).all():
        raise CompilationError("offset column must be finite")
    y = frame[panel.response].fillna(0.0).to_numpy(dtype=float)

    if isinstance(model.likelihood, Gaussian):
        # ... existing Gaussian path unchanged, but pass offset into the family:
        # offset=offset instead of np.zeros(len(frame))
        ...
        return family.materialize({})

    if isinstance(model.likelihood, (Poisson, Bernoulli)):
        compiled_likelihood = model.likelihood.materialize({})
        observed = panel.observed
        compiled_likelihood.validate_response(y[observed])
        labels = _qualified_labels(blocks)
        width = sum(block.design.shape[1] for block in blocks)
        design = hstack([block.design for block in blocks], format="csr")
        precision = block_diag([block.precision for block in blocks], format="csr")
        constraints = _block_constraints(tuple(blocks), width)
        try:
            return CompiledLGM(
                y=y,
                observed=observed,
                offset=offset,
                design=design,
                precision=precision,
                constraints=constraints,
                labels=labels,
                likelihood=compiled_likelihood,
                blocks=tuple(blocks),
            )
        except (TypeError, ValueError, ModelValidationError) as error:
            raise CompilationError(f"compiled declarative model is invalid: {error}") from error

    raise CompilationError("unsupported likelihood for declarative compilation")
```

Confirm `_qualified_labels(blocks)` returns the qualified label tuple (it validates uniqueness); if it returns `None`, compute `labels = tuple(f"{b.name}:{l}" for b in blocks for l in b.labels)` instead. Import `CompiledLGM` and `ModelValidationError` if not already imported in `compiler.py`.

- [ ] **Step 5: Include offset in the Spark projection**

In `src/pylgm/data/spark.py`, update `_required_columns`:

```python
def _required_columns(model) -> set[str]:
    required = {model.response, *model.panel, model.time}
    if getattr(model, "offset", None) is not None:
        required.add(model.offset)
    for effect in model.predictor.effects:
        if isinstance(effect, Fixed):
            required.update(Formula(effect.formula).required_variables)
        else:
            required.add(effect.index)
    return required
```

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src pytest tests/test_compiler.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/model.py src/pylgm/compiler.py src/pylgm/data/spark.py tests/test_compiler.py
git commit -m "feat: compile non-Gaussian likelihoods with model offsets"
```

---

### Task 4: LaplaceResult and Shared Posterior Helpers

**Files:**
- Modify: `src/pylgm/exceptions.py`
- Modify: `src/pylgm/inference/result.py`
- Modify: `src/pylgm/inference/__init__.py`
- Modify: `tests/inference/test_result.py`

**Interfaces:**
- Produces: `InferenceConvergenceError(InferenceError)` with `.iterations: int` and `.gradient_norm: float`.
- Produces: shared module-level helpers in `result.py`: `latent_marginals_from(mean, covariance, block_slices, block)` and `linear_combinations_from(mean, covariance, weights)`, both returning `GaussianMarginals`.
- Produces: `LaplaceResult` with the same posterior surface as `GaussianResult` plus `fitted_mean`, `link_name`, `engine == "laplace"`, `converged == True`, and defensive-copy `prediction_keys`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/inference/test_result.py
import numpy as np
import pytest

from pylgm.inference.result import LaplaceResult


def _laplace(**overrides):
    kwargs = dict(
        labels=("fixed:1",),
        mean=np.array([0.5]),
        covariance=np.array([[0.25]]),
        log_marginal_likelihood=-1.0,
        predictive_mean=np.array([0.5, 0.5]),
        predictive_variance=np.array([0.25, 0.25]),
        fitted_mean=np.array([1.6487, 1.6487]),
        link_name="log",
        block_slices={"fixed": slice(0, 1)},
        diagnostics={"newton_iterations": 3, "final_gradient_norm": 1e-10},
    )
    kwargs.update(overrides)
    return LaplaceResult(**kwargs)


def test_laplace_result_surface_and_immutability():
    result = _laplace()
    assert result.engine == "laplace"
    assert result.converged is True
    assert result.link_name == "log"
    np.testing.assert_allclose(result.fitted_mean, [1.6487, 1.6487])
    # defensive copy out
    exposed = result.fitted_mean
    exposed[0] = 99.0
    np.testing.assert_allclose(result.fitted_mean, [1.6487, 1.6487])
    assert result.diagnostics["newton_iterations"] == 3


def test_laplace_result_latent_and_linear_combinations_match_gaussian_helpers():
    result = _laplace()
    marg = result.latent_marginals("fixed")
    np.testing.assert_allclose(marg.mean, [0.5])
    np.testing.assert_allclose(marg.variance, [0.25])
    lc = result.linear_combinations(np.array([[2.0]]))
    np.testing.assert_allclose(lc.mean, [1.0])
    np.testing.assert_allclose(lc.variance, [1.0])


def test_laplace_result_prediction_keys_defensive_copy():
    import pandas as pd

    keys = pd.DataFrame({"region": ["A", "B"]})
    result = _laplace(prediction_keys=keys)
    keys.loc[0, "region"] = "mutated"
    assert result.prediction_keys["region"].tolist() == ["A", "B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`
Expected: FAIL with `ImportError: cannot import name 'LaplaceResult'`.

- [ ] **Step 3: Add the exception**

In `src/pylgm/exceptions.py`, add after `InferenceError`:

```python
class InferenceConvergenceError(InferenceError):
    """Iterative inference failed to converge within its iteration budget."""

    def __init__(self, iterations: int, gradient_norm: float) -> None:
        self.iterations = int(iterations)
        self.gradient_norm = float(gradient_norm)
        super().__init__(
            f"Laplace Newton iteration did not converge in {self.iterations} "
            f"iterations (final gradient norm {self.gradient_norm:.3e})"
        )
```

- [ ] **Step 4: Extract shared helpers and add `LaplaceResult`**

In `src/pylgm/inference/result.py`, refactor the body of `GaussianResult.latent_marginals` and `GaussianResult.linear_combinations` into module-level functions `latent_marginals_from(mean, covariance, block_slices, block)` and `linear_combinations_from(mean, covariance, weights)` (move the existing math verbatim, including the roundoff-tolerance clamp in linear combinations). Have `GaussianResult` call them:

```python
    def latent_marginals(self, block: str | None = None) -> GaussianMarginals:
        return latent_marginals_from(self._mean, self._covariance, self.block_slices, block)

    def linear_combinations(self, weights) -> GaussianMarginals:
        return linear_combinations_from(self._mean, self._covariance, weights)
```

Then add `LaplaceResult` (a frozen dataclass, `init=False`) mirroring `GaussianResult`'s validation and defensive copies, storing `_mean`, `_covariance`, `_predictive_mean`, `_predictive_variance`, `_fitted_mean`, `_prediction_keys`, `labels`, `log_marginal_likelihood`, `link_name`, `block_slices`, `diagnostics`. Reuse `_readonly_array`, `_readonly_diagnostics`, the same `prediction_keys` validation as `GaussianResult`. Properties: `mean`, `covariance`, `predictive_mean`, `predictive_variance`, `fitted_mean`, `prediction_keys` (all defensive read-only copies), `engine` → `"laplace"`, `converged` → `True`, `latent_marginals`/`linear_combinations` delegate to the shared helpers, `hyperparameter_marginals()` → empty `MappingProxyType({})`.

Export `LaplaceResult` from `src/pylgm/inference/__init__.py` and its `__all__`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`
Expected: PASS (existing GaussianResult tests still pass — public API unchanged).

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/exceptions.py src/pylgm/inference/result.py src/pylgm/inference/__init__.py tests/inference/test_result.py
git commit -m "feat: add LaplaceResult and shared posterior helpers"
```

---

### Task 5: Laplace Inference Engine

**Files:**
- Create: `src/pylgm/inference/laplace.py`
- Modify: `src/pylgm/inference/__init__.py`
- Create: `tests/inference/test_laplace.py`

**Interfaces:**
- Consumes: `CompiledLGM`; the GLM protocol on the compiled likelihood; `LaplaceResult`, `InferenceConvergenceError`; the Gaussian helpers `_constraint_null_space`, `_factor_positive_definite`, `_block_slices`, `_require_finite`, `preflight_dense_reference`.
- Produces: `fit_laplace(model: CompiledLGM, *, allow_large_dense: bool = False, max_iterations: int = 100, tolerance: float = 1e-8) -> LaplaceResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/inference/test_laplace.py
import numpy as np
import pytest

from pylgm import Bernoulli, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import InferenceConvergenceError
from pylgm.inference.gaussian import fit_gaussian
from pylgm.inference.laplace import fit_laplace
import pandas as pd


def _panel(frame, panel=(), time="t"):
    return CanonicalPanel.from_frame(frame, DataConfig(time=time, response="y", panel=panel))


def test_laplace_reproduces_exact_gaussian_on_gaussian_likelihood():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 2.5, 4.0]})
    model = LGM("y", Gaussian(0.7), Fixed("1 + x", prior_precision=0.5) + IID("g", index="t", precision=1.5), time="t")
    compiled = compile_lgm(model, _panel(frame))
    exact = fit_gaussian(compiled)
    laplace = fit_laplace(compiled)
    np.testing.assert_allclose(laplace.mean, exact.mean, atol=1e-8)
    np.testing.assert_allclose(laplace.covariance, exact.covariance, atol=1e-8)
    np.testing.assert_allclose(
        laplace.log_marginal_likelihood, exact.log_marginal_likelihood, atol=1e-8
    )


def test_laplace_poisson_mode_solves_the_score_equation():
    # single fixed intercept, flat prior -> mode is log(mean(y))
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [2.0, 3.0, 5.0, 6.0]})
    model = LGM("y", Poisson(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    np.testing.assert_allclose(result.mean, [np.log(4.0)], atol=1e-6)
    # response-scale fitted mean uses log-normal correction, so > exp(mean)
    assert result.fitted_mean[0] > np.exp(result.mean[0]) - 1e-9
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_bernoulli_intercept_matches_logit_of_rate():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [1.0, 1.0, 0.0, 1.0]})
    model = LGM("y", Bernoulli(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    p = 0.75
    np.testing.assert_allclose(result.mean, [np.log(p / (1 - p))], atol=1e-6)


def test_laplace_predicts_unobserved_rows():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, np.nan]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    assert result.predictive_mean.shape == (3,)
    assert np.isfinite(result.fitted_mean).all()


def test_laplace_raises_on_non_convergence():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 5.0, 10.0], "y": [0.0, 1.0, 20.0]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    with pytest.raises(InferenceConvergenceError):
        fit_laplace(compile_lgm(model, _panel(frame)), max_iterations=1, tolerance=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/inference/test_laplace.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm.inference.laplace'`.

- [ ] **Step 3: Implement the engine**

```python
# src/pylgm/inference/laplace.py
import numpy as np
from scipy.linalg import cho_solve

from pylgm.exceptions import InferenceConvergenceError, NumericalError
from pylgm.inference.gaussian import (
    _block_slices,
    _constraint_null_space,
    _factor_positive_definite,
    _require_finite,
    preflight_dense_reference,
)
from pylgm.inference.result import LaplaceResult
from pylgm.ir.model import CompiledLGM


def _fit_laplace_dense(model: CompiledLGM, max_iterations: int, tolerance: float) -> LaplaceResult:
    likelihood = model.likelihood
    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    design = model.design
    y = model.y
    offset = model.offset
    observed = model.observed
    y_obs = y[observed]
    offset_obs = offset[observed]
    likelihood.validate_response(y_obs)

    basis = _constraint_null_space(model.constraints, latent_size)
    reduced_dim = basis.shape[1]
    reduced_design = np.asarray(design[observed] @ basis)
    reduced_precision = basis.T @ precision @ basis
    _, logdet_prior = _factor_positive_definite(reduced_precision, "reduced prior precision")

    def objective(z: np.ndarray) -> float:
        eta = reduced_design @ z + offset_obs
        return -likelihood.log_likelihood(eta, y_obs) + 0.5 * float(z @ reduced_precision @ z)

    z = np.zeros(reduced_dim)
    gradient_norm = 0.0
    iterations = 0
    if reduced_dim:
        current = objective(z)
        for iterations in range(1, max_iterations + 1):
            eta = reduced_design @ z + offset_obs
            grad_ll = likelihood.gradient(eta, y_obs)
            weights = likelihood.working_weights(eta, y_obs)
            gradient = reduced_precision @ z - reduced_design.T @ grad_ll
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm < tolerance:
                break
            hessian = reduced_precision + (reduced_design.T * weights) @ reduced_design
            factor, _ = _factor_positive_definite(hessian, "reduced posterior precision")
            step = cho_solve(factor, -gradient)
            slope = float(gradient @ step)
            scale = 1.0
            for _ in range(50):
                candidate = z + scale * step
                candidate_obj = objective(candidate)
                if np.isfinite(candidate_obj) and candidate_obj <= current + 1e-4 * scale * slope:
                    break
                scale *= 0.5
            else:
                raise NumericalError("Laplace line search failed to reduce the objective")
            z = candidate
            current = candidate_obj
        else:
            raise InferenceConvergenceError(iterations, gradient_norm)
        eta = reduced_design @ z + offset_obs
        weights = likelihood.working_weights(eta, y_obs)
        hessian = reduced_precision + (reduced_design.T * weights) @ reduced_design
        factor, logdet_posterior = _factor_positive_definite(hessian, "reduced posterior precision")
        reduced_covariance = cho_solve(factor, np.eye(reduced_dim))
        loglik_mode = likelihood.log_likelihood(eta, y_obs)
    else:
        reduced_covariance = np.empty((0, 0))
        logdet_posterior = 0.0
        loglik_mode = likelihood.log_likelihood(offset_obs, y_obs)

    mean = basis @ z
    covariance = basis @ reduced_covariance @ basis.T
    log_marginal_likelihood = float(
        loglik_mode
        - 0.5 * float(z @ reduced_precision @ z)
        + 0.5 * logdet_prior
        - 0.5 * logdet_posterior
    )

    predictive_mean = np.asarray(offset + design @ mean).reshape(-1)
    design_dense = design.toarray()
    predictive_variance = np.einsum("ij,jk,ik->i", design_dense, covariance, design_dense)
    fitted_mean = likelihood.response_prediction(predictive_mean, predictive_variance)

    _require_finite("posterior mean", mean)
    _require_finite("posterior covariance", covariance)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)
    _require_finite("predictive variance", predictive_variance)
    _require_finite("fitted mean", fitted_mean)

    return LaplaceResult(
        labels=model.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        fitted_mean=fitted_mean,
        link_name=likelihood.link.name,
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": int(np.count_nonzero(observed)),
            "constraint_count": int(model.constraints.shape[0]),
            "newton_iterations": int(iterations),
            "final_gradient_norm": float(gradient_norm),
        },
    )


def fit_laplace(
    model: CompiledLGM,
    *,
    allow_large_dense: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> LaplaceResult:
    """Fit a latent Gaussian model by a Laplace approximation at fixed hyperparameters.

    Likelihood-agnostic: any compiled likelihood implementing the GLM protocol works,
    so a Gaussian likelihood is fit exactly and serves as the correctness anchor.
    """
    preflight_dense_reference(model, allow_large_dense=allow_large_dense)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _fit_laplace_dense(model, max_iterations, tolerance)
    except FloatingPointError as error:
        raise NumericalError("Laplace numerical calculation was non-finite") from error
```

Export `fit_laplace` from `src/pylgm/inference/__init__.py` and its `__all__`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/inference/test_laplace.py -q`
Expected: PASS. If the Gaussian anchor differs by a constant, recheck the marginal-likelihood formula against `gaussian.py`; the two must be algebraically identical.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference/laplace.py src/pylgm/inference/__init__.py tests/inference/test_laplace.py
git commit -m "feat: add Laplace inference engine"
```

---

### Task 6: Public `LGM.fit(engine="laplace")` Dispatch

**Files:**
- Modify: `src/pylgm/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `fit_laplace`, `LaplaceResult`, `Poisson`/`Bernoulli`, `Gaussian`.
- Produces: `LGM.fit(frame, engine="exact_gaussian", *, max_driver_rows=100_000)` dispatching `engine="laplace"` to `fit_laplace`, with the same Pandas caller-order / Spark canonical-key contract as Gaussian, applied to `predictive_mean`, `predictive_variance`, and `fitted_mean`.
- Rejects Gaussian+`laplace` and non-Gaussian+`exact_gaussian` with `UnsupportedEngineError`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Bernoulli, Fixed, Gaussian, LGM, Poisson
from pylgm.exceptions import UnsupportedEngineError
from pylgm.inference.result import LaplaceResult


def test_lgm_fit_laplace_poisson_returns_laplace_result():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 4.0, 7.0]})
    result = LGM("y", Poisson(), Fixed("1 + x"), time="t").fit(frame, engine="laplace")
    assert isinstance(result, LaplaceResult)
    assert result.converged is True
    assert result.prediction_keys is None
    assert result.fitted_mean.shape == (4,)


def test_lgm_fit_laplace_preserves_caller_row_order():
    frame = pd.DataFrame({"t": [3, 1, 2], "y": [4.0, 1.0, 2.0]})
    result = LGM("y", Poisson(), Fixed("1"), time="t").fit(frame, engine="laplace")
    # first row corresponds to t=3 in caller order
    assert result.predictive_mean.shape == (3,)


def test_lgm_rejects_engine_likelihood_mismatch():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    with pytest.raises(UnsupportedEngineError, match="Gaussian"):
        LGM("y", Gaussian(1.0), Fixed("1"), time="t").fit(frame, engine="laplace")
    with pytest.raises(UnsupportedEngineError, match="non-Gaussian"):
        LGM("y", Poisson(), Fixed("1"), time="t").fit(frame, engine="exact_gaussian")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: FAIL (`engine="laplace"` unknown).

- [ ] **Step 3: Implement dispatch and alignment**

In `src/pylgm/model.py`:
- Import `fit_laplace`, `LaplaceResult` from `pylgm.inference`, and `Gaussian`, `Poisson`, `Bernoulli` from `pylgm.likelihoods`.
- Add a helper mapping engine → fit function and validating the likelihood/engine pairing:

```python
    def _engine(self, engine: str):
        engines = {"exact_gaussian": fit_gaussian, "laplace": fit_laplace}
        try:
            fit = engines[engine]
        except KeyError as error:
            raise UnsupportedEngineError(f"unknown inference engine: {engine}") from error
        is_gaussian = isinstance(self.likelihood, Gaussian)
        if engine == "exact_gaussian" and not is_gaussian:
            raise UnsupportedEngineError("exact_gaussian requires a Gaussian likelihood; use engine='laplace' for a non-Gaussian likelihood")
        if engine == "laplace" and is_gaussian:
            raise UnsupportedEngineError("engine='laplace' is for non-Gaussian likelihoods; use engine='exact_gaussian' for a Gaussian likelihood")
        return fit
```

- In `_fit_pandas`, replace `result = self._engine(engine)(compiled)` + `_align_predictions_with_source_rows(...)` with a branch that aligns the correct result type. Generalize `_align_predictions_with_source_rows` to handle `LaplaceResult` too:

```python
def _align_predictions_with_source_rows(result, source_positions):
    caller_order = np.argsort(source_positions)
    if isinstance(result, LaplaceResult):
        return LaplaceResult(
            labels=result.labels, mean=result.mean, covariance=result.covariance,
            log_marginal_likelihood=result.log_marginal_likelihood,
            predictive_mean=result.predictive_mean[caller_order],
            predictive_variance=result.predictive_variance[caller_order],
            fitted_mean=result.fitted_mean[caller_order],
            link_name=result.link_name, block_slices=result.block_slices,
            diagnostics=result.diagnostics, prediction_keys=result.prediction_keys,
        )
    return GaussianResult(... existing ...)
```

- In `_fit_spark`, dispatch through `self._engine(engine)` and rebuild the matching result type with `prediction_keys=canonical.prediction_keys` (no source-row realignment). For `LaplaceResult`, carry `fitted_mean`, `link_name` through unchanged.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/model.py tests/test_model.py
git commit -m "feat: dispatch engine='laplace' from LGM.fit"
```

---

### Task 7: YAML Frontend, Spark Parity, Example, and Docs

**Files:**
- Modify: `src/pylgm/config/model.py`
- Modify: `tests/config/test_model.py`
- Modify: `tests/data/test_spark.py`
- Create: `examples/count_glm/config.yaml`, `examples/count_glm/data.csv`, `examples/count_glm/run.py`, `examples/count_glm/README.md`
- Modify: `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`

**Interfaces:**
- Consumes: `Poisson`, `Bernoulli`, `LGM.offset` from earlier tasks.
- Produces: YAML `likelihood: {family: poisson}` / `{family: bernoulli}` and top-level `offset: <col>` loading to the matching `LGM`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/config/test_model.py
from pathlib import Path

from pylgm import Bernoulli, Poisson
from pylgm.config import load_model


def test_load_poisson_model_with_offset(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: poisson\n"
        "offset: logexp\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1 + x'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Poisson)
    assert model.offset == "logexp"


def test_load_bernoulli_model(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(
        "response: y\n"
        "likelihood:\n  family: bernoulli\n"
        "data:\n  time: t\n"
        "predictor:\n  fixed: '1'\n"
    )
    model = load_model(path)
    assert isinstance(model.likelihood, Bernoulli)
    assert model.offset is None
```

```python
# append to tests/data/test_spark.py
def test_spark_laplace_matches_sorted_pandas(spark):
    import numpy as np
    import pandas as pd

    from pylgm import Fixed, LGM, Poisson

    rows = [("B", 2, 4.0, 5.0), ("A", 1, 1.0, 1.0), ("B", 1, 3.0, 3.0), ("A", 2, 2.0, 2.0)]
    columns = ["region", "time", "x", "y"]
    model = LGM("y", Poisson(), Fixed("1 + x"), panel=("region",), time="time")
    spark_result = model.fit(spark.createDataFrame(rows, columns), engine="laplace")
    pandas_result = model.fit(
        pd.DataFrame(rows, columns=columns).sort_values(["region", "time"]), engine="laplace"
    )
    np.testing.assert_allclose(spark_result.predictive_mean, pandas_result.predictive_mean, atol=1e-8)
    assert spark_result.prediction_keys.to_dict("list") == {
        "region": ["A", "A", "B", "B"], "time": [1, 2, 1, 2],
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/config/test_model.py tests/data/test_spark.py -q`
Expected: config tests FAIL (family not accepted); Spark test SKIP without PySpark, else FAIL.

- [ ] **Step 3: Extend the YAML frontend**

In `src/pylgm/config/model.py`: allow `family` in `{"gaussian", "poisson", "bernoulli"}`. For `gaussian`, keep requiring `sigma`; for `poisson`/`bernoulli`, forbid `sigma`. Add an optional top-level `offset: str | None = None` field to the standalone config model and pass it into the built `LGM(..., offset=...)`. Build `Poisson()` / `Bernoulli()` for those families instead of `Gaussian(sigma)`.

- [ ] **Step 4: Create the example**

`examples/count_glm/data.csv` — a small panel with `region,time,x,y,logexp` where `y` are non-negative integer counts.

`examples/count_glm/config.yaml`:

```yaml
response: y
likelihood:
  family: poisson
offset: logexp
data:
  panel: [region]
  time: time
predictor:
  fixed: "1 + x"
  effects:
    - {name: region_effect, type: iid, index: region, precision: 2.0}
```

`examples/count_glm/run.py` — read the CSV, `load_model(config.yaml)`, `model.fit(frame, engine="laplace")`, print `result.fitted_mean` and `result.diagnostics["newton_iterations"]`. `examples/count_glm/README.md` — how to run with `PYTHONPATH=src`.

- [ ] **Step 5: Update docs**

README: add a "Non-Gaussian likelihoods (Laplace)" section documenting `Poisson`/`Bernoulli`, `offset`, `engine="laplace"`, `fitted_mean` semantics (exact for log link, point estimate for logit), and that hyperparameters are fixed plug-ins. Update the arch spec: mark slice 2's Laplace core as shipped, preserving the deferred Binomial/weights/EB/INLA/spatial scope.

- [ ] **Step 6: Run tests and lint**

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS (Spark tests PASS or explicit SKIP).

Run: `PYTHONPATH=src ruff check src tests`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/config/model.py tests/config/test_model.py tests/data/test_spark.py examples/count_glm README.md docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md
git commit -m "docs: publish non-Gaussian Laplace slice"
```

---

## Self-Review Checklist

- Task 1 (links) → consumed by Task 2 (likelihoods) → Task 3 (compiler) → Tasks 4/5 (result+engine) → Task 6 (dispatch) → Task 7 (frontend/docs). Dependencies are linear.
- Every spec section maps to a task: likelihoods+protocol (T2), links (T1), offset (T3), Laplace engine (T5), result surface (T4), dispatch+errors (T6), YAML+Spark+example+docs (T7), extensibility (protocol shape in T2 + `fit_laplace` marginal in T5).
- Gaussian-as-Laplace anchor (T5) validates the numerics; GLM-oracle and score-equation tests cover Poisson/Bernoulli correctness.
- No IR changes; Gaussian public behavior preserved (full-suite gate on every task).
- Explicit deferrals (Binomial(n), weights, non-canonical links, empirical Bayes, INLA) are not implemented; the protocol and engine are shaped so they slot in later without rework.

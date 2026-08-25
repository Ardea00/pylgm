# pyLGM General LGM Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the existing exact-Gaussian implementation behind a general LGM API and IR without changing its numerical behavior or breaking schema-v1/v2 users.

**Architecture:** Add immutable likelihood, link, prior, and hyperparameter values; move Gaussian-specific state out of `CompiledLGM`; expose a common marginal/result API; and compile both the new declarative Python API and legacy configuration through the existing Gaussian compiler. This plan deliberately stops before PySpark, Laplace, INLA, spatial effects, and forecasting-module relocation so the core contract can stabilize first.

**Tech Stack:** Python 3.11+, NumPy 2, SciPy sparse, Pandas 2.2, Pydantic 2, Formulaic, PyYAML, pytest, Ruff.

## Global Constraints

- Preserve all existing schema-v1 and schema-v2 behavior and public imports.
- Keep the existing exact Gaussian numerical path as the reference backend.
- Keep compiled design and precision operators sparse; this plan does not replace the dense reference factorization.
- Do not add PySpark as a required dependency.
- Do not add non-Gaussian likelihood execution or inference-engine fallback.
- Use immutable public value objects and defensive copies for arrays and sparse matrices.
- Every task must leave the full test suite passing.

---

## File Map

- `src/pylgm/parameters.py`: immutable hyperparameter declaration and validation.
- `src/pylgm/links.py`: supported link declarations.
- `src/pylgm/priors.py`: prior declarations, including PC precision.
- `src/pylgm/likelihoods.py`: public and compiled Gaussian likelihood values.
- `src/pylgm/ir/model.py`: inference-independent compiled LGM container.
- `src/pylgm/ir/family.py`: materialization of parameterized Gaussian families.
- `src/pylgm/inference/result.py`: posterior marginal and common result surface.
- `src/pylgm/inference/gaussian.py`: compatibility-preserving exact Gaussian backend.
- `src/pylgm/effects/spec.py`: declarative fixed/IID/RW effect specifications.
- `src/pylgm/model.py`: public `LGM` model and engine dispatch.
- `src/pylgm/compiler.py`: compilation shared by declarative and legacy frontends.
- `src/pylgm/config/model.py`: new standalone model YAML schema and loader.
- `src/pylgm/__init__.py`: public modeling exports while retaining legacy exports.
- `README.md`: distinguish general LGM API from legacy forecasting orchestration.

### Task 1: Modeling Vocabulary

**Files:**
- Create: `src/pylgm/parameters.py`
- Create: `src/pylgm/links.py`
- Create: `src/pylgm/priors.py`
- Create: `src/pylgm/likelihoods.py`
- Test: `tests/test_modeling_vocabulary.py`

**Interfaces:**
- Produces: `Prior` protocol with `logpdf(value: float) -> float` and `Hyperparameter(name: str, initial: float, prior: Prior | None = None, transform: Literal["identity", "log"] = "log")`.
- Produces: `IdentityLink()` with `name == "identity"`.
- Produces: `GaussianPrior(mean: float, precision: float)` and `PCPrecision(upper_sd: float, alpha: float)` with `logpdf(value: float) -> float` on the precision scale.
- Produces: `Gaussian(sigma: float | Hyperparameter = 1.0)` and internal `CompiledGaussian(sigma: float)`.

- [ ] **Step 1: Write validation and density tests**

```python
import numpy as np
import pytest

from pylgm.likelihoods import CompiledGaussian, Gaussian
from pylgm.links import IdentityLink
from pylgm.parameters import Hyperparameter
from pylgm.priors import GaussianPrior, PCPrecision


def test_gaussian_vocabulary_is_immutable_and_validated():
    parameter = Hyperparameter("sigma", 2.0)
    assert Gaussian(parameter).link == IdentityLink()
    assert CompiledGaussian(2.0).variance == 4.0
    with pytest.raises(ValueError, match="positive"):
        CompiledGaussian(0.0)


def test_pc_precision_matches_change_of_variables_density():
    prior = PCPrecision(upper_sd=1.0, alpha=0.01)
    rate = -np.log(0.01)
    expected = np.log(rate / 2.0) - 1.5 * np.log(4.0) - rate / 2.0
    assert prior.logpdf(4.0) == pytest.approx(expected)


def test_prior_and_parameter_reject_invalid_values():
    with pytest.raises(ValueError, match="alpha"):
        PCPrecision(1.0, alpha=1.0)
    with pytest.raises(ValueError, match="precision"):
        GaussianPrior(0.0, precision=-1.0)
    with pytest.raises(ValueError, match="name"):
        Hyperparameter("", 1.0)
```

- [ ] **Step 2: Run the new tests and verify import failures**

Run: `pytest tests/test_modeling_vocabulary.py -q`

Expected: FAIL because the four modules do not exist.

- [ ] **Step 3: Implement the immutable values and exact validations**

Use frozen dataclasses. Accept only finite ordinary real values; require positive `initial`, `sigma`, and prior precisions; require `0 < alpha < 1`. Implement PC precision density as

```python
rate = -np.log(alpha) / upper_sd
return np.log(rate / 2.0) - 1.5 * np.log(value) - rate / np.sqrt(value)
```

`Gaussian.materialize(values)` must resolve a numeric sigma directly or resolve its named `Hyperparameter` from an exact-key mapping and return `CompiledGaussian`.

- [ ] **Step 4: Run focused and full tests**

Run: `pytest tests/test_modeling_vocabulary.py -q`

Expected: PASS.

Run: `pytest -q`

Expected: all existing tests and the new tests PASS.

- [ ] **Step 5: Commit the vocabulary**

```bash
git add src/pylgm/parameters.py src/pylgm/links.py src/pylgm/priors.py src/pylgm/likelihoods.py tests/test_modeling_vocabulary.py
git commit -m "feat: add general LGM modeling vocabulary"
```

### Task 2: Remove Gaussian State from the General IR

**Files:**
- Modify: `src/pylgm/ir/model.py`
- Modify: `src/pylgm/ir/family.py`
- Modify: `src/pylgm/compiler.py`
- Modify: `src/pylgm/inference/gaussian.py`
- Modify: `tests/ir/test_model_validation.py`
- Modify: `tests/ir/test_gaussian_family.py`
- Modify: `tests/inference/test_gaussian.py`

**Interfaces:**
- Consumes: `CompiledGaussian` from Task 1.
- Produces: `CompiledLGM(..., likelihood: object, ...)` with no stored `sigma` field.
- Preserves: read-only `CompiledLGM.sigma` compatibility property for compiled Gaussian models.
- Preserves: `fit_gaussian(model, allow_large_dense=False) -> GaussianResult` and all current numerical outputs.

- [ ] **Step 1: Add tests proving likelihood ownership and compatibility**

```python
from pylgm.exceptions import UnsupportedEngineError
from pylgm.likelihoods import CompiledGaussian


def _compiled(likelihood: object) -> CompiledLGM:
    return CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=likelihood,
        blocks=(),
    )


def test_compiled_lgm_owns_a_likelihood_not_gaussian_state():
    model = _compiled(CompiledGaussian(1.0))
    assert model.likelihood == CompiledGaussian(model.sigma)
    assert "sigma" not in model.__dataclass_fields__


def test_exact_gaussian_rejects_an_incompatible_likelihood():
    with pytest.raises(UnsupportedEngineError, match="Gaussian"):
        fit_gaussian(_compiled(object()))
```

Add `UnsupportedEngineError(PyLGMError)` to `src/pylgm/exceptions.py` if it is not already present, and import it in the test.

- [ ] **Step 2: Run the affected tests and verify failure**

Run: `pytest tests/ir/test_model_validation.py tests/ir/test_gaussian_family.py tests/inference/test_gaussian.py -q`

Expected: FAIL because `CompiledLGM` still stores `sigma` directly.

- [ ] **Step 3: Generalize `CompiledLGM` and materialization**

Replace constructor parameter `sigma: float` with `likelihood: object`. Validate that the likelihood is present, store it immutably, and implement the compatibility property:

```python
@property
def sigma(self) -> float:
    if not isinstance(self.likelihood, CompiledGaussian):
        raise AttributeError("sigma is only defined for a Gaussian likelihood")
    return self.likelihood.sigma
```

Update `_assemble_compiled_model` to accept a compiled likelihood. Update `CompiledGaussianFamily.materialize()` to construct `CompiledGaussian(sigma)`. At the beginning of `fit_gaussian`, reject any other likelihood with `UnsupportedEngineError`.

- [ ] **Step 4: Prove numerical behavior is unchanged**

Run: `pytest tests/ir tests/inference tests/test_compiler.py tests/test_pipeline.py tests/optimization -q`

Expected: PASS with the same posterior means, covariance, predictions, and marginal likelihood assertions as before.

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit the generalized IR**

```bash
git add src/pylgm/exceptions.py src/pylgm/ir/model.py src/pylgm/ir/family.py src/pylgm/compiler.py src/pylgm/inference/gaussian.py tests/ir tests/inference
git commit -m "refactor: separate likelihood from compiled LGM"
```

### Task 3: Common Posterior Marginals and Result Surface

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Modify: `src/pylgm/inference/gaussian.py`
- Modify: `src/pylgm/inference/__init__.py`
- Test: `tests/inference/test_result.py`
- Modify: `tests/inference/test_gaussian.py`

**Interfaces:**
- Produces: immutable `GaussianMarginals(mean, variance)` with `std`, `quantile(probability)`, and defensive arrays.
- Produces: `GaussianResult(..., block_slices: Mapping[str, slice], diagnostics: Mapping[str, object])` with `engine == "exact_gaussian"`, `converged is True`, `latent_marginals(block: str | None = None) -> GaussianMarginals`, `hyperparameter_marginals() -> Mapping[str, GaussianMarginals]`, and `linear_combinations(weights: csr_matrix | np.ndarray) -> GaussianMarginals`.
- Preserves: `mean`, `covariance`, `predictive_mean`, `predictive_variance`, and `log_marginal_likelihood`.

- [ ] **Step 1: Write result-contract tests**

```python
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.inference import fit_gaussian
from pylgm.ir import CompiledLGM
from pylgm.likelihoods import CompiledGaussian


def _fitted_result():
    model = CompiledLGM(
        y=np.array([2.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix(np.eye(2)),
        constraints=np.empty((0, 2)),
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )
    return fit_gaussian(model)


def test_gaussian_result_exposes_common_marginals():
    result = _fitted_result()
    marginals = result.latent_marginals()
    assert np.array_equal(marginals.mean, result.mean)
    assert np.array_equal(marginals.variance, np.diag(result.covariance))
    assert result.engine == "exact_gaussian"
    assert result.converged is True
    assert result.hyperparameter_marginals() == {}


def test_linear_combinations_propagate_covariance():
    result = _fitted_result()
    weights = csr_matrix([[1.0, -1.0]])
    combined = result.linear_combinations(weights)
    expected = weights.toarray() @ result.covariance @ weights.toarray().T
    assert combined.mean[0] == pytest.approx(weights.toarray()[0] @ result.mean)
    assert combined.variance[0] == pytest.approx(expected[0, 0])
```

- [ ] **Step 2: Run and verify missing API failures**

Run: `pytest tests/inference/test_result.py tests/inference/test_gaussian.py -q`

Expected: FAIL because the common marginal methods do not exist.

- [ ] **Step 3: Implement marginal and result methods**

Use `scipy.stats.norm.ppf` for `quantile`. Require `0 < probability < 1`, validate matrix width against the latent dimension, and require finite numeric weights. Add `_block_slices(model: CompiledLGM) -> Mapping[str, slice]` to the Gaussian backend and pass its result into `GaussianResult`. Store `block_slices` and `diagnostics` behind `MappingProxyType`; diagnostics contains the integer keys `latent_dimension`, `observed_count`, and `constraint_count`.

- [ ] **Step 4: Run focused and full tests**

Run: `pytest tests/inference -q`

Expected: PASS.

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit the result API**

```bash
git add src/pylgm/inference tests/inference
git commit -m "feat: expose common LGM posterior results"
```

### Task 4: Declarative Effect Specifications and `LGM`

**Files:**
- Create: `src/pylgm/effects/spec.py`
- Modify: `src/pylgm/effects/__init__.py`
- Create: `src/pylgm/model.py`
- Modify: `src/pylgm/compiler.py`
- Test: `tests/effects/test_spec.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `Gaussian`, `Hyperparameter`, existing effect builders, `CanonicalPanel`, `fit_gaussian`.
- Produces: `Fixed(formula: str, prior_precision: float = 1e-6)`, `IID(name: str, index: str, precision: float | Hyperparameter = 1.0)`, `RW1(...)`, and `RW2(...)` with the same arguments as `IID`.
- Produces: additive immutable `Predictor` preserving declaration order and rejecting duplicate names.
- Produces: `LGM(response, likelihood, predictor, panel=(), time=None)` and `fit(frame, engine="exact_gaussian")`.

- [ ] **Step 1: Test composition and one end-to-end declarative fit**

```python
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, RW1
from pylgm.exceptions import UnsupportedEngineError


def _example():
    frame = pd.DataFrame({
        "region": ["a", "a", "b", "b"],
        "time": [1, 2, 1, 2],
        "y": [1.0, 2.0, 1.5, None],
    })
    model = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed("1") + IID("region", "region", 2.0) + RW1("trend", "time", 3.0),
        panel=("region",),
        time="time",
    )
    return frame, model


def test_effects_compose_in_declaration_order():
    predictor = Fixed("1 + x") + IID("region", "region", 2.0) + RW1("trend", "time", 3.0)
    assert [effect.name for effect in predictor.effects] == ["fixed", "region", "trend"]


def test_declarative_model_fits_existing_gaussian_path():
    frame, model = _example()
    result = model.fit(frame)
    assert result.engine == "exact_gaussian"
    assert result.predictive_mean.shape == (4,)


def test_model_does_not_fallback_from_unknown_engine():
    frame, model = _example()
    with pytest.raises(UnsupportedEngineError, match="unknown"):
        model.fit(frame, engine="unknown")
```

- [ ] **Step 2: Run and verify import failures**

Run: `pytest tests/effects/test_spec.py tests/test_model.py -q`

Expected: FAIL because declarative specifications and `LGM` do not exist.

- [ ] **Step 3: Implement specifications and compile through existing builders**

Effect specifications contain declarations only; do not duplicate sparse assembly. Add `compile_lgm(model: LGM, panel: CanonicalPanel) -> CompiledLGM` that translates declarations to existing `build_fixed`, `build_iid`, and `build_random_walk` calls. Resolve a declared `Hyperparameter` to its `initial` value in the exact conditional Gaussian fit. When `LGM.time is None`, `LGM.fit` works on a defensive frame copy with a collision-checked `__pylgm_row__` integer key; an RW effect continues to use its own declared `index`. `LGM.fit` accepts only Pandas in this plan and raises `DataContractError("PySpark support requires the spark extra")` for Spark-shaped objects until the dedicated adapter plan lands.

Dispatch engines with an exact lookup:

```python
engines = {"exact_gaussian": fit_gaussian}
try:
    fit = engines[engine]
except KeyError as error:
    raise UnsupportedEngineError(f"unknown inference engine: {engine}") from error
return fit(compiled)
```

- [ ] **Step 4: Run new, compiler, and full tests**

Run: `pytest tests/effects/test_spec.py tests/test_model.py tests/test_compiler.py -q`

Expected: PASS.

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit the declarative API**

```bash
git add src/pylgm/effects src/pylgm/model.py src/pylgm/compiler.py tests/effects/test_spec.py tests/test_model.py
git commit -m "feat: add declarative LGM model API"
```

### Task 5: Standalone YAML Model Frontend

**Files:**
- Create: `src/pylgm/config/model.py`
- Modify: `src/pylgm/config/__init__.py`
- Create: `tests/config/test_model.py`
- Create: `examples/general_lgm/config.yaml`
- Create: `examples/general_lgm/README.md`

**Interfaces:**
- Consumes: public `LGM`, likelihood, and effect declarations from Tasks 1 and 4.
- Produces: `load_model(path: Path) -> LGM` for the approved standalone YAML shape.
- Preserves: `load_config` and `load_experiment_config` unchanged.

- [ ] **Step 1: Write strict parsing and equivalence tests**

```python
def test_load_model_builds_the_same_declaration_as_python(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """
response: y
likelihood: {family: gaussian, sigma: 0.5}
data: {panel: [region], time: time}
predictor:
  fixed: "1 + x"
  effects:
    - {name: region_effect, type: iid, index: region, precision: 2.0}
    - {name: trend, type: rw1, index: time, precision: 3.0}
""".strip()
    )
    model = load_model(path)
    assert model.response == "y"
    assert [effect.name for effect in model.predictor.effects] == [
        "fixed", "region_effect", "trend"
    ]


def test_load_model_forbids_unknown_keys(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("response: y\nlikelihood: {family: gaussian}\npredictor: {}\nunknown: true")
    with pytest.raises(ConfigurationError, match="unknown"):
        load_model(path)
```

- [ ] **Step 2: Run and verify missing loader failure**

Run: `pytest tests/config/test_model.py -q`

Expected: FAIL because `load_model` does not exist.

- [ ] **Step 3: Implement strict Pydantic schema and translation**

Use frozen `extra="forbid"` models. Support only Gaussian, fixed, IID, RW1, and RW2 in this milestone. Require explicit `sigma` and effect `precision` to be finite and positive. Translate config objects into the public declarations, then construct `LGM`. Do not route this file through legacy `RunConfig`; the two frontends share the compiler, not schemas.

- [ ] **Step 4: Add and run a config-driven example**

Document the YAML beside a short Python snippet:

```python
model = load_model(Path("config.yaml"))
result = model.fit(frame)
```

Run: `pytest tests/config/test_model.py tests/config/test_load.py tests/config/test_experiment.py -q`

Expected: PASS, including all legacy config tests.

Run: `pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit the YAML frontend**

```bash
git add src/pylgm/config tests/config/test_model.py examples/general_lgm
git commit -m "feat: add standalone LGM YAML frontend"
```

### Task 6: Public Exports, Documentation, and Regression Gate

**Files:**
- Modify: `src/pylgm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_package.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: all public objects produced by Tasks 1-5.
- Produces: top-level imports `LGM`, `Gaussian`, `Fixed`, `IID`, `RW1`, `RW2`, `Hyperparameter`, `GaussianPrior`, and `PCPrecision`.
- Preserves: `Pipeline`, `Experiment`, `ComparisonResult`, `CandidateFailure`, and `FailureCause` imports.

- [ ] **Step 1: Extend the package-contract test**

```python
def test_general_lgm_api_is_exported_without_removing_legacy_api():
    expected = {
        "LGM", "Gaussian", "Fixed", "IID", "RW1", "RW2",
        "Hyperparameter", "GaussianPrior", "PCPrecision",
        "Pipeline", "Experiment",
    }
    assert expected.issubset(set(pylgm.__all__))
    assert pylgm.__version__ == "0.3.0"
```

- [ ] **Step 2: Run and verify export/version failure**

Run: `pytest tests/test_package.py -q`

Expected: FAIL because the new API is not exported and the version is still 0.2.0.

- [ ] **Step 3: Export the API and rewrite the README lead**

Set project and package version to `0.3.0`. Lead the README with “General-purpose latent Gaussian models for Python, with Pandas and PySpark data adapters.” Clearly mark exact Gaussian as the only stable engine in 0.3. Keep legacy forecasting commands under a “Legacy forecasting utilities” section and link the approved architecture document.

- [ ] **Step 4: Run formatting, all tests, and installed CLI smoke test**

Run: `ruff check src tests`

Expected: PASS.

Run: `pytest -q`

Expected: all tests PASS, including the original 357 regression tests.

Run: `python -m pip install -e . --no-deps && pylgm --help`

Expected: editable installation succeeds and CLI help exits zero.

- [ ] **Step 5: Commit the completed foundation**

```bash
git add src/pylgm/__init__.py README.md tests/test_package.py pyproject.toml
git commit -m "docs: publish general LGM foundation API"
```

## Self-Review Checklist

- Every approved first-core requirement is covered without claiming PySpark, Laplace, INLA, spatial, or HMC support.
- Legacy configurations, orchestration imports, and Gaussian outputs remain covered by the full regression suite.
- `CompiledLGM`, `Gaussian`, and `GaussianResult` names and ownership are consistent across tasks.
- Every implementation step names the concrete behavior and files it changes.
- Follow-up plans are required for the PySpark adapter and `contrib.forecasting` relocation after this API is stable.

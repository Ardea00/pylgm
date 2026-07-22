# pyLGM Predictive Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded empirical-Bayes tuning, leakage-safe rolling evaluation, candidate selection, and persistence benchmarking to the exact Gaussian foundation.

**Architecture:** A model is compiled once into an immutable Gaussian family whose precision blocks can be rescaled cheaply. A deterministic optimizer fits selected hyperparameters per candidate and origin; a rolling evaluator scores forecasts at configured horizons and an `Experiment` orchestrates selection and transactional artifacts.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Pandas, Formulaic, Pydantic 2, PyYAML, Typer, PyArrow-compatible Parquet support, Pytest, Ruff.

## Global Constraints

- Core code is domain-neutral; NIC/Italian inflation appears only in examples and external-data verification.
- Existing schema-version-1 `Pipeline` and `pylgm fit` behavior remains compatible.
- Schema version 2 is explicit; unknown keys, duplicate YAML keys, coercive numerics, unsupported strategies, and invalid horizons are errors.
- Only `empirical_bayes` hyperparameter optimization is implemented now.
- Inference code never imports YAML, Pandas, Spark, evaluation, or artifact modules.
- Optimizer code consumes numerical Gaussian-family interfaces, not configuration or DataFrames.
- Every response after a backtest origin is masked; covariate availability is checked before fitting.
- Candidate failure is isolated and persisted; there is no silent fallback to initial hyperparameters or another strategy.
- Candidate selection is eligibility first, mean log predictive density second, and RMSE tie-break third.
- Existing exact-Gaussian dense safety limits apply to every materialized fit.
- No plugin loader, automatic candidate generation, NUTS, Laplace, spatial effect, or Spark integration is added in this plan.

---

### Task 1: Version-2 experiment configuration and candidate resolution

**Files:**
- Create: `src/pylgm/config/experiment.py`
- Modify: `src/pylgm/config/__init__.py`
- Modify: `src/pylgm/config/load.py`
- Test: `tests/config/test_experiment.py`

**Interfaces:**
- Consumes: existing `DataConfig`, `ModelConfig`, and unique-key YAML loader.
- Produces: `ExperimentConfig`, `ResolvedCandidate`, `load_experiment_config(path)`, and `resolve_candidates(config)`.

- [ ] **Step 1: Write strict schema and resolution tests**

```python
def test_experiment_config_resolves_complete_candidates(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text("""
schema_version: 2
data:
  time: month
  response: y
  panel: [region]
  covariates:
    lag1: {availability: observed_with_lag, lag: 1}
model:
  fixed: "1 + lag1"
  sigma: 1.0
  effects: [{name: trend, type: rw1, index: month, precision: 2.0}]
candidates:
  - {name: base}
  - name: iid_only
    overrides:
      effects: [{name: region, type: iid, index: region, precision: 1.0}]
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      region.precision: {initial: 1.0, lower: 0.01, upper: 100.0}
inference:
  engine: exact_gaussian
  hyperparameters:
    strategy: empirical_bayes
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      trend.precision: {initial: 2.0, lower: 0.01, upper: 100.0}
evaluation:
  horizons: [1, 3, 6]
  origins: {last: 8}
  window: {type: expanding}
""")
    config = load_experiment_config(path)
    candidates = resolve_candidates(config)
    assert [candidate.name for candidate in candidates] == ["base", "iid_only"]
    assert candidates[0].model.effects[0].name == "trend"
    assert candidates[1].model.effects[0].name == "region"
    assert "region.precision" in candidates[1].optimize
```

Also test: schema version must be integer `2`; duplicate candidate names; empty candidates; duplicate/zero horizons; rolling window without positive length; vintage mode without a vintage column; unsupported strategy; unknown optimized effect; non-finite or unordered bounds; list replacement rather than merging; missing availability declaration for formula covariates.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/config/test_experiment.py -v`  
Expected: collection fails because `load_experiment_config` does not exist.

- [ ] **Step 3: Implement the focused v2 schema**

```python
# src/pylgm/config/experiment.py
class CovariateConfig(StrictModel):
    availability: Literal["known_future", "observed_with_lag", "unavailable_future"]
    lag: int | None = Field(default=None, ge=1)


class ExperimentDataConfig(DataConfig):
    vintage: str | None = None
    covariates: dict[str, CovariateConfig] = Field(default_factory=dict)


class ParameterBounds(StrictModel):
    initial: FinitePositiveFloat
    lower: FinitePositiveFloat
    upper: FinitePositiveFloat

    @model_validator(mode="after")
    def ordered(self) -> "ParameterBounds":
        if not self.lower <= self.initial <= self.upper or self.lower == self.upper:
            raise ValueError("parameter bounds must satisfy lower <= initial <= upper")
        return self


class ModelOverride(StrictModel):
    fixed: str | None = None
    fixed_prior_precision: FinitePositiveFloat | None = None
    sigma: FinitePositiveFloat | None = None
    effects: tuple[EffectConfig, ...] | None = None


class CandidateConfig(StrictModel):
    name: str
    overrides: ModelOverride = Field(default_factory=ModelOverride)
    optimize: dict[str, ParameterBounds] | None = None


class HyperparameterConfig(StrictModel):
    strategy: Literal["empirical_bayes"]
    optimize: dict[str, ParameterBounds]


class InferenceConfig(StrictModel):
    engine: Literal["exact_gaussian"]
    hyperparameters: HyperparameterConfig


class OriginConfig(StrictModel):
    last: int | None = Field(default=None, ge=1)
    values: tuple[str | int | date | datetime, ...] | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "OriginConfig":
        if (self.last is None) == (self.values is None):
            raise ValueError("origins require exactly one of last or values")
        return self


class WindowConfig(StrictModel):
    type: Literal["expanding", "rolling"] = "expanding"
    length: int | None = Field(default=None, ge=2)


class EvaluationConfig(StrictModel):
    mode: Literal["latest", "vintage"] = "latest"
    horizons: tuple[int, ...]
    origins: OriginConfig
    window: WindowConfig = WindowConfig()
    interval_levels: tuple[float, ...] = (0.5, 0.8, 0.95)
    max_abs_coverage_error: float = Field(default=0.15, ge=0, le=1)
    rmse_tie_tolerance: float = Field(default=1e-9, ge=0)


class ExperimentConfig(StrictModel):
    schema_version: Literal[2]
    data: ExperimentDataConfig
    model: ModelConfig
    candidates: tuple[CandidateConfig, ...]
    inference: InferenceConfig
    evaluation: EvaluationConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version_two(cls, value: object) -> int:
        if type(value) is not int or value != 2:
            raise ValueError("schema_version must be the integer 2")
        return value


@dataclass(frozen=True)
class ResolvedCandidate:
    name: str
    model: ModelConfig
    optimize: Mapping[str, ParameterBounds]
```

Use Formulaic's `Formula(config.model.fixed).required_variables` to require availability declarations for non-role formula variables. Resolve an override with `model.model_copy(update=override.model_dump(exclude_none=True))`, then validate the resulting `ModelConfig`; list values replace completely.

`CandidateConfig.optimize`, when present, replaces the global optimization mapping;
otherwise it inherits it. Validate every resolved mapping against that candidate's
`sigma` and effect names. Add model validators requiring unique positive horizons,
levels strictly inside `(0, 1)`, a rolling length only for rolling windows, and a
vintage column exactly when `mode: vintage` is selected.

`load_experiment_config` must reuse `_UniqueKeySafeLoader` and translate YAML/Pydantic failures to `ConfigurationError`. Do not change `load_config`, `RunConfig`, or v1 parsing.

- [ ] **Step 4: Run focused and existing config tests**

Run: `python -m pytest tests/config/test_experiment.py tests/config/test_load.py -q && ruff check src/pylgm/config tests/config`  
Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/config tests/config/test_experiment.py
git commit -m "feat: add predictive experiment configuration"
```

---

### Task 2: Immutable compiled Gaussian family

**Files:**
- Create: `src/pylgm/ir/family.py`
- Modify: `src/pylgm/ir/__init__.py`
- Modify: `src/pylgm/compiler.py`
- Test: `tests/ir/test_gaussian_family.py`
- Modify: `tests/test_compiler.py`

**Interfaces:**
- Consumes: `DataConfig`, `ModelConfig`, `CanonicalPanel`, effect builders, and `CompiledLGM`.
- Produces: `Hyperparameters`, `CompiledGaussianFamily.materialize(values)`, and `compile_gaussian_family(data, model, panel, optimized)`.

- [ ] **Step 1: Write analytical materialization and isolation tests**

```python
def test_family_rescales_precision_without_rebuilding_design(panel, model_config) -> None:
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma", "trend.precision")
    )
    low = family.materialize({"sigma": 0.5, "trend.precision": 2.0})
    high = family.materialize({"sigma": 1.0, "trend.precision": 8.0})
    np.testing.assert_allclose(low.design.toarray(), high.design.toarray())
    trend = family.block_slice("trend")
    np.testing.assert_allclose(
        high.precision[trend, trend].toarray(),
        4.0 * low.precision[trend, trend].toarray(),
    )
```

Also test sigma-only and multiple-effect materialization; unknown/missing/non-finite parameters; fixed blocks unchanged; source/accessor mutation isolation; labels/constraints identical to `compile_model`; and rematerialization at configured values produces the same exact Gaussian result.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/ir/test_gaussian_family.py -v`  
Expected: collection fails because `CompiledGaussianFamily` is absent.

- [ ] **Step 3: Implement family types and compiler refactor**

```python
@dataclass(frozen=True)
class Hyperparameters:
    sigma: float
    precisions: Mapping[str, float]


@dataclass(frozen=True)
class ScalableBlock:
    block: LatentBlock
    parameter: str | None
    scale: float


@dataclass(frozen=True, init=False)
class CompiledGaussianFamily:
    _y: np.ndarray
    _observed: np.ndarray
    _offset: np.ndarray
    blocks: tuple[ScalableBlock, ...]
    parameter_names: tuple[str, ...]
    initial: Hyperparameters

    @property
    def y(self) -> np.ndarray:
        """Return an isolated response snapshot for numerical problems/tests."""
        result = self._y.copy()
        result.setflags(write=False)
        return result

    def materialize(self, values: Mapping[str, float]) -> CompiledLGM:
        """Validate a complete optimized mapping and return an immutable model."""
        resolved = _validate_parameter_mapping(values, self.parameter_names)
        blocks = tuple(
            LatentBlock(
                item.block.name,
                item.block.labels,
                item.block.design,
                item.block.precision
                * (resolved[item.parameter] if item.parameter else item.scale),
                item.block.constraints,
            )
            for item in self.blocks
        )
        sigma = resolved.get("sigma", self.initial.sigma)
        return _assemble_compiled_model(
            self._y, self._observed, self._offset, blocks, sigma
        )

    def block_slice(self, name: str) -> slice:
        """Return the global latent slice for a unique block."""
        start = 0
        for item in self.blocks:
            stop = start + item.block.design.shape[1]
            if item.block.name == name:
                return slice(start, stop)
            start = stop
        raise KeyError(name)
```

`compile_gaussian_family` builds fixed effects once at configured precision. Each optimized structured effect is built at unit precision and receives parameter `<effect>.precision`; each non-optimized effect is built at configured precision with no binding. `sigma` is either bound or fixed.

Extract `_assemble_compiled_model` from the current final half of `compile_model`; it
constructs global design, precision, constraints, labels, and `CompiledLGM` from a
tuple of blocks. `_validate_parameter_mapping` requires exactly the declared names
and ordinary finite positive real values. Store every array, sparse matrix, mapping,
and accessor result defensively as in the existing IR.

Refactor `compile_model(config, panel)` to call the family compiler with no optimized values and materialize the configured values. Existing tests must remain unchanged.

- [ ] **Step 4: Run IR/compiler and full regression tests**

Run: `python -m pytest tests/ir/test_gaussian_family.py tests/test_compiler.py tests/inference/test_gaussian.py -q && python -m pytest -q && ruff check .`  
Expected: all tests pass; existing exact results are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/ir src/pylgm/compiler.py tests/ir/test_gaussian_family.py tests/test_compiler.py
git commit -m "feat: compile reusable Gaussian families"
```

---

### Task 3: Bounded empirical-Bayes optimizer

**Files:**
- Create: `src/pylgm/optimization/__init__.py`
- Create: `src/pylgm/optimization/empirical_bayes.py`
- Create: `src/pylgm/optimization/result.py`
- Modify: `src/pylgm/exceptions.py`
- Test: `tests/optimization/test_empirical_bayes.py`

**Interfaces:**
- Consumes: `CompiledGaussianFamily`, numerical `OptimizationBounds`, and `fit_gaussian`.
- Produces: `EmpiricalBayesResult`, `OptimizationDiagnostics`, and `optimize_empirical_bayes(family, bounds, initial=None)`.

- [ ] **Step 1: Write optimizer tests against an analytical scale optimum**

```python
def test_sigma_optimization_matches_zero_latent_gaussian_mle() -> None:
    family = zero_latent_family(y=np.array([-2.0, 1.0, 3.0]))
    expected = np.sqrt(np.mean(family.y**2))
    result = optimize_empirical_bayes(
        family,
        {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
    )
    assert result.parameters["sigma"] == pytest.approx(expected, rel=1e-5)
    assert result.diagnostics.converged
```

Also test log transformation; a precision optimum on a scalar conjugate model; deterministic repeated fits; warm-start override; cache hit for repeated vectors; active-bound reporting; invalid numerical evaluations recorded; unsuccessful/no-valid optimum raises `OptimizationError`; and rematerialized optimum equals the returned Gaussian result.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/optimization/test_empirical_bayes.py -v`  
Expected: collection fails because `pylgm.optimization` does not exist.

- [ ] **Step 3: Implement the numerical optimizer**

```python
@dataclass(frozen=True)
class OptimizationDiagnostics:
    converged: bool
    objective: float
    evaluations: int
    cache_hits: int
    elapsed_seconds: float
    initial: Mapping[str, float]
    optimum: Mapping[str, float]
    active_bounds: tuple[str, ...]
    numerical_failures: tuple[str, ...]


@dataclass(frozen=True)
class EmpiricalBayesResult:
    parameters: Mapping[str, float]
    fit: GaussianResult
    diagnostics: OptimizationDiagnostics


@dataclass(frozen=True)
class OptimizationBounds:
    initial: float
    lower: float
    upper: float


def optimize_empirical_bayes(
    family: CompiledGaussianFamily,
    bounds: Mapping[str, OptimizationBounds],
    *,
    initial: Mapping[str, float] | None = None,
) -> EmpiricalBayesResult:
    names = tuple(bounds)
    lower = np.log([bounds[name].lower for name in names])
    upper = np.log([bounds[name].upper for name in names])
    start = np.log([
        initial[name] if initial is not None else bounds[name].initial
        for name in names
    ])
    solution = scipy.optimize.minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper, strict=True)),
    )
```

Cache by the exact tuple of log parameters within the optimization. The objective is negative conditional log marginal likelihood. Catch only typed inference/numerical failures, record them, and return a finite sentinel objective; programming errors propagate. Require SciPy success and a valid cached fit at the final point.

Add `OptimizationError(PyLGMError)`.

- [ ] **Step 4: Run focused and full quality gates**

Run: `python -m pytest tests/optimization/test_empirical_bayes.py -q && python -m pytest -q && ruff check .`  
Expected: all tests pass and Ruff is clean.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/optimization src/pylgm/exceptions.py tests/optimization
git commit -m "feat: add empirical Bayes optimization"
```

---

### Task 4: Leakage-safe rolling folds and persistence predictions

**Files:**
- Create: `src/pylgm/evaluation/__init__.py`
- Create: `src/pylgm/evaluation/folds.py`
- Create: `src/pylgm/evaluation/availability.py`
- Create: `src/pylgm/evaluation/persistence.py`
- Modify: `src/pylgm/exceptions.py`
- Test: `tests/evaluation/test_folds.py`
- Test: `tests/evaluation/test_persistence.py`

**Interfaces:**
- Consumes: experiment data/evaluation configuration and a raw Pandas frame.
- Produces: `FoldDefinition`, `FoldData`, `build_fold_definitions`, `materialize_fold`, and `persistence_predictions`.

- [ ] **Step 1: Write fold tests that expose leakage**

```python
def test_horizon_three_masks_every_response_after_origin(panel_frame) -> None:
    definition = FoldDefinition(origin=3, target=6, horizon=3)
    fold = materialize_fold(panel_frame, data_config, evaluation_config, definition)
    assert fold.model_frame.loc[fold.model_frame.month > 3, "y"].isna().all()
    assert fold.target_frame["month"].eq(6).all()


def test_vintage_fold_uses_latest_release_available_at_origin(vintage_frame) -> None:
    fold = materialize_fold(vintage_frame, vintage_data_config, evaluation, definition)
    assert fold.training_frame.loc[fold.training_frame.month.eq(2), "y"].item() == 2.1
    assert 2.9 not in fold.training_frame["y"].tolist()  # later revision
```

Also test arbitrary horizons, explicit and last-n origins, insufficient future levels, expanding/rolling windows, irregular ordered time values, exact target coordinates, no rows after target, `known_future`, `observed_with_lag` allowed only when `lag >= horizon`, `unavailable_future`, missing covariates, duplicate vintage releases, and typed errors.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/evaluation/test_folds.py tests/evaluation/test_persistence.py -v`  
Expected: collection fails because evaluation modules are absent.

- [ ] **Step 3: Implement compact fold materialization**

```python
@dataclass(frozen=True)
class FoldDefinition:
    origin: object
    target: object
    horizon: int


@dataclass(frozen=True)
class FoldData:
    definition: FoldDefinition
    training_frame: pd.DataFrame
    model_frame: pd.DataFrame
    target_frame: pd.DataFrame


def build_fold_definitions(frame, data, evaluation) -> tuple[FoldDefinition, ...]:
    levels = tuple(sorted(frame[data.time].drop_duplicates()))
    positions = {value: index for index, value in enumerate(levels)}
    maximum_horizon = max(evaluation.horizons)
    eligible = levels[:-maximum_horizon]
    origins = (
        eligible[-evaluation.origins.last :]
        if evaluation.origins.last is not None
        else tuple(evaluation.origins.values)
    )
    return tuple(
        FoldDefinition(origin, levels[positions[origin] + horizon], horizon)
        for origin in origins
        for horizon in evaluation.horizons
    )


def materialize_fold(frame, data, evaluation, definition) -> FoldData:
    source = _latest_available_vintage(frame, data, definition.origin)
    truth = _latest_vintage(frame, data)
    source = _apply_training_window(source, data.time, evaluation.window, definition)
    model_frame = source.loc[source[data.time].le(definition.target)].copy()
    target_frame = truth.loc[
        truth[data.time].eq(definition.target) & truth[data.response].notna()
    ].copy()
    _require_target_coordinates(model_frame, target_frame, data)
    model_frame.loc[model_frame[data.time].gt(definition.origin), data.response] = np.nan
    _validate_target_covariates(target_frame, data, definition.horizon)
    training_frame = model_frame.loc[model_frame[data.time].le(definition.origin)].copy()
    return FoldData(definition, training_frame, model_frame, target_frame)
```

`_latest_available_vintage` returns the input unchanged in latest mode. In vintage
mode it filters `vintage <= origin`, sorts by panel/time/vintage, and keeps the last
row per panel/time key. `_latest_vintage` keeps the final release solely as scoring
truth. `_require_target_coordinates` ensures every truth key has an as-of-origin
model/scenario row; it never copies later-release covariates into the model frame.
`_apply_training_window` uses all prior levels for expanding windows and the
configured number for rolling windows. Helpers raise `FoldConstructionError` for
missing or incomparable levels.

`persistence_predictions(fold)` groups by panel keys, takes the last observed response at or before the origin, and joins it one-to-one to target rows. Its horizon-specific predictive variance is the pre-origin mean squared persistence error for that horizon, with a documented positive numerical floor. Missing history or insufficient residual history raises `FoldConstructionError`; neither the mean nor variance uses target/future values.

Add `FoldConstructionError` and `CovariateAvailabilityError`.

- [ ] **Step 4: Run evaluation and full tests**

Run: `python -m pytest tests/evaluation -q && python -m pytest -q && ruff check .`  
Expected: all tests pass and Ruff is clean.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/evaluation src/pylgm/exceptions.py tests/evaluation
git commit -m "feat: add leakage-safe rolling folds"
```

---

### Task 5: Probabilistic metrics and hierarchical selection

**Files:**
- Create: `src/pylgm/evaluation/metrics.py`
- Create: `src/pylgm/evaluation/selection.py`
- Modify: `src/pylgm/evaluation/__init__.py`
- Modify: `src/pylgm/exceptions.py`
- Test: `tests/evaluation/test_metrics.py`
- Test: `tests/evaluation/test_selection.py`

**Interfaces:**
- Consumes: coordinate-preserving prediction rows and evaluation thresholds.
- Produces: `score_predictions`, `aggregate_metrics`, `CandidateDecision`, and `select_candidate`.

- [ ] **Step 1: Write analytical scoring and ranking tests**

```python
def test_gaussian_log_score_matches_scipy() -> None:
    predictions = pd.DataFrame({
        "actual": [1.0], "mean": [0.5], "variance": [2.0],
        "candidate": ["a"], "origin": [1], "horizon": [1],
    })
    scored = score_predictions(predictions, interval_levels=(0.95,))
    assert scored.loc[0, "log_predictive_density"] == pytest.approx(
        scipy.stats.norm.logpdf(1.0, loc=0.5, scale=np.sqrt(2.0))
    )


def test_selection_uses_log_score_then_rmse_tie_break() -> None:
    result = select_candidate(metrics, failures={}, config=selection_config)
    assert result.selected == "better_log_score"
```

Also test positive finite variance enforcement; RMSE/MAE/bias; interval bounds and coverage; grouping by candidate/horizon; weighted aggregation by prediction rows rather than averaging fold means; numerical/calibration ineligibility; RMSE tie tolerance; deterministic name tie-break; all-ineligible raises `SelectionError`; and benchmark rows are reported but never selected.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/evaluation/test_metrics.py tests/evaluation/test_selection.py -v`  
Expected: collection fails because metrics/selection functions are absent.

- [ ] **Step 3: Implement vectorized scoring and explicit decisions**

```python
@dataclass(frozen=True)
class CandidateDecision:
    selected: str
    ranking: tuple[str, ...]
    eligible: Mapping[str, bool]
    reasons: Mapping[str, tuple[str, ...]]


def score_predictions(predictions: pd.DataFrame, interval_levels: tuple[float, ...]) -> pd.DataFrame:
    standard_deviation = np.sqrt(predictions["variance"].to_numpy())
    result = predictions.copy()
    result["log_predictive_density"] = norm.logpdf(
        result["actual"], loc=result["mean"], scale=standard_deviation
    )
    result["squared_error"] = (result["actual"] - result["mean"]) ** 2
    result["absolute_error"] = np.abs(result["actual"] - result["mean"])
    for level in interval_levels:
        quantile = norm.ppf((1.0 + level) / 2.0)
        suffix = str(level).replace(".", "_")
        lower = result["mean"] - quantile * standard_deviation
        upper = result["mean"] + quantile * standard_deviation
        result[f"lower_{suffix}"] = lower
        result[f"upper_{suffix}"] = upper
        result[f"covered_{suffix}"] = result["actual"].between(lower, upper)
    return result
```

Aggregate sums/counts first, then derive mean log density, RMSE, MAE, bias, coverage, and average width. `select_candidate` constructs reasons before sorting; it never hides excluded candidates.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/evaluation/test_metrics.py tests/evaluation/test_selection.py -q && python -m pytest -q && ruff check .`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/evaluation src/pylgm/exceptions.py tests/evaluation
git commit -m "feat: score and select predictive candidates"
```

---

### Task 6: Experiment orchestration with failure isolation

**Files:**
- Create: `src/pylgm/experiment.py`
- Modify: `src/pylgm/__init__.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Consumes: resolved candidates, folds, Gaussian families, empirical-Bayes optimizer, persistence predictions, metrics, and selection.
- Produces: `Experiment.from_yaml`, `Experiment.compare`, and immutable `ComparisonResult`.

- [ ] **Step 1: Write an end-to-end in-memory comparison test**

```python
def test_experiment_compares_candidates_and_persistence(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml(EXPERIMENT_CONFIG)
    result = experiment.compare(synthetic_panel(), output=None)
    assert set(result.candidates) == {"base", "trend"}
    assert set(result.predictions["candidate"]) == {"base", "trend", "persistence"}
    assert result.selected in {"base", "trend"}
    assert result.predictions["origin"].notna().all()
    assert result.predictions["target_time"].notna().all()
```

Also test: optimizer runs once per candidate/origin rather than per horizon; previous-origin warm start; target responses never enter optimization; one failing candidate does not stop others; all candidates failing raises `SelectionError`; dense guard failures are recorded; repeated runs are identical; result DataFrames are isolated from mutation.

- [ ] **Step 2: Run focused test and verify RED**

Run: `python -m pytest tests/test_experiment.py -v`  
Expected: collection fails because `Experiment` is absent.

- [ ] **Step 3: Implement the minimal orchestrator**

```python
@dataclass(frozen=True, init=False)
class ComparisonResult:
    candidates: tuple[str, ...]
    selected: str
    _predictions: pd.DataFrame
    _metrics: pd.DataFrame
    _diagnostics: pd.DataFrame
    failures: Mapping[str, tuple[str, ...]]
    decision: CandidateDecision


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Experiment":
        return cls(load_experiment_config(Path(path)))

    def compare(self, frame: pd.DataFrame, output: str | Path | None = None) -> ComparisonResult:
        definitions = build_fold_definitions(frame, self.config.data, self.config.evaluation)
        runs, failures = _run_candidates(self.config, frame, definitions)
        predictions = pd.concat(
            [run.predictions for run in runs] + [_run_persistence(self.config, frame, definitions)],
            ignore_index=True,
        )
        scored = score_predictions(predictions, self.config.evaluation.interval_levels)
        metrics = aggregate_metrics(scored)
        decision = select_candidate(metrics, failures, self.config.evaluation)
        result = ComparisonResult.from_tables(runs, scored, metrics, failures, decision)
        if output is not None:
            write_experiment(Path(output), self.config, result, frame)
        return result
```

Keep orchestration explicit; do not add a task graph or callback framework.
`_run_candidates` groups definitions by origin, optimizes each candidate once on the
training frame, passes that optimum to `_predict_horizon` for every horizon, and
catches only typed pyLGM errors into the candidate failure table. `_run_persistence`
uses the same definitions and target rows.

- [ ] **Step 4: Run experiment and full tests**

Run: `python -m pytest tests/test_experiment.py -q && python -m pytest -q && ruff check .`  
Expected: all tests pass and public v1 `Pipeline` tests remain green.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/experiment.py src/pylgm/__init__.py tests/test_experiment.py
git commit -m "feat: orchestrate predictive experiments"
```

---

### Task 7: Transactional experiment artifacts and CLI

**Files:**
- Create: `src/pylgm/artifacts/experiment.py`
- Create: `src/pylgm/data/fingerprint.py`
- Modify: `src/pylgm/artifacts/__init__.py`
- Modify: `src/pylgm/artifacts/run.py`
- Modify: `src/pylgm/pipeline.py`
- Modify: `src/pylgm/experiment.py`
- Modify: `src/pylgm/cli.py`
- Modify: `src/pylgm/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/artifacts/test_experiment_artifacts.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `ComparisonResult`, and existing atomic publication helpers.
- Produces: `write_experiment`, artifact schema version 2, `pylgm compare`, and CSV/Parquet input detection.

- [ ] **Step 1: Write artifact and CLI tests**

```python
def test_experiment_artifact_is_complete_and_versioned(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    Experiment.from_yaml(CONFIG).compare(synthetic_panel(), output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["artifact_schema_version"] == 2
    assert summary["selected"] in {"base", "trend"}
    assert (output / "predictions.parquet").exists()
    assert (output / "metrics.parquet").exists()
    assert (output / "optimization.parquet").exists()
    assert (output / "failures.json").exists()
    assert (output / "resolved_candidates.json").exists()
```

Also test fingerprints change with fold/candidate configuration; output is atomic on injected Parquet failure; existing output is not replaced; environment and direct dependencies are recorded; `pylgm compare CONFIG DATA --output DIR` works with CSV and Parquet; unsupported suffix fails clearly; existing `pylgm fit` remains compatible.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/artifacts/test_experiment_artifacts.py tests/test_cli.py -v`  
Expected: comparison artifact/CLI tests fail because they are absent.

- [ ] **Step 3: Reuse one transactional publication primitive**

Move `_panel_fingerprint` and its encoding helpers from `pipeline.py` to
`data/fingerprint.py`; keep the algorithm byte-identical and import it from both v1
and experiment pipelines. Refactor the existing lock/temp/no-replace sequence into:

```python
def write_transactionally(output: Path, writer: Callable[[Path], None]) -> None:
    """Call writer in a sibling temporary directory and publish it once."""
```

Keep `write_run` output byte-compatible except for internal refactoring. `write_experiment` writes canonical JSON plus Parquet tables, hashes every payload, and places those hashes in `summary.json`.

Add `pyarrow>=17` as a runtime dependency because version-2 experiment artifacts
always use Parquet, record its version in `environment.json`, and bump the package
and `__version__` to `0.2.0`.

Add CLI input loading:

```python
def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise typer.BadParameter("data must be CSV or Parquet")


@app.command()
def compare(config: Path, data: Path, output: Path = typer.Option(...)) -> None:
    result = Experiment.from_yaml(config).compare(_read_frame(data), output)
    typer.echo(f"selected={result.selected} candidates={len(result.candidates)}")
```

- [ ] **Step 4: Run artifact, CLI, and full tests**

Run: `python -m pytest tests/artifacts/test_experiment_artifacts.py tests/test_cli.py -q && python -m pytest -q && ruff check .`  
Expected: all tests pass; transactional failure leaves no final directory.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/pylgm/artifacts src/pylgm/data/fingerprint.py src/pylgm/pipeline.py src/pylgm/experiment.py src/pylgm/cli.py src/pylgm/__init__.py tests/artifacts tests/test_cli.py
git commit -m "feat: persist and run predictive comparisons"
```

---

### Task 8: Reproducible example, NIC verification, and documentation

**Files:**
- Create: `examples/predictive_selection/config.yaml`
- Create: `examples/predictive_selection/data.csv`
- Create: `examples/predictive_selection/README.md`
- Create: `examples/nic_backtest/config.yaml`
- Create: `examples/nic_backtest/README.md`
- Modify: `README.md`
- Create: `tests/integration/test_predictive_selection.py`
- Create: `tests/integration/test_nic_shaped_backtest.py`

**Interfaces:**
- Consumes: public `Experiment` and `pylgm compare` interfaces only.
- Produces: a small committed general panel example and a NIC-shaped regression fixture/config that does not redistribute the external source dataset.

- [ ] **Step 1: Write public-interface integration tests**

```python
def test_predictive_selection_example_runs(tmp_path: Path) -> None:
    result = Experiment.from_yaml(
        Path("examples/predictive_selection/config.yaml")
    ).compare(
        pd.read_csv("examples/predictive_selection/data.csv"),
        tmp_path / "comparison",
    )
    assert result.selected in result.candidates
    assert set(result.predictions["horizon"]) == {1, 2}


def test_nic_shaped_backtest_has_no_leakage_and_beats_global_mean() -> None:
    result = Experiment.from_yaml(Path("examples/nic_backtest/config.yaml")).compare(
        synthetic_nic_frame(), output=None
    )
    assert result.metrics["log_predictive_density"].notna().all()
    assert "persistence" in set(result.predictions["candidate"])
```

- [ ] **Step 2: Run tests and verify fixtures are absent**

Run: `python -m pytest tests/integration/test_predictive_selection.py tests/integration/test_nic_shaped_backtest.py -v`  
Expected: failure because example configs/data do not exist.

- [ ] **Step 3: Add concise examples and limitations**

The general example must contain at least two model candidates, horizons `[1, 2]`, empirical-Bayes bounds, and persistence comparison. The NIC config uses only the generic roles `date`, `region_code`, `coicop`, and `yoy_pct`, so it runs directly on the source Parquet file. Its candidates vary region/COICOP/time effects; persistence supplies the autoregressive benchmark.

The NIC README documents:

```bash
pylgm compare examples/nic_backtest/config.yaml \
  /path/to/nic_regionale_mensile.parquet \
  --output nic-comparison
```

State that the source data is not redistributed, hyperparameter uncertainty is not integrated, and the exact engine is for small/medium latent dimensions.

- [ ] **Step 4: Run the final completion gate**

Run:

```bash
export PYLGM_VERIFY_DIR=$(mktemp -d)
python -m pytest -q
ruff check .
git diff --check
pylgm compare examples/predictive_selection/config.yaml examples/predictive_selection/data.csv --output "$PYLGM_VERIFY_DIR/comparison"
python -c 'import json, os, pathlib; p=pathlib.Path(os.environ["PYLGM_VERIFY_DIR"])/"comparison"/"summary.json"; s=json.loads(p.read_text()); assert s["artifact_schema_version"] == 2; assert s["selected"]'
git status --short
```

Expected: every test passes, Ruff and diff checks are clean, the CLI prints a selected candidate, artifact assertions pass, and Git status is clean.

- [ ] **Step 5: Commit**

```bash
git add examples README.md tests/integration
git commit -m "docs: add predictive selection examples"
```

## Completion gate

Before declaring the milestone complete:

1. run the full verification in Task 8 from a fresh temporary output directory;
2. review the complete branch range against this plan and the approved design;
3. fix all correctness or contract findings and re-run the gate;
4. retain exact-Gaussian fixtures for future NUTS/Laplace comparison;
5. do not start Spark, spatial, or alternative optimizer work in this milestone.

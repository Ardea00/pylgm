# pyLGM PySpark Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional PySpark adapter that feeds the existing canonical-panel compiler and exact-Gaussian engine without adding distributed inference.

**Architecture:** A lazy-imported Spark adapter validates and projects a Spark DataFrame, establishes canonical model-key order, applies a driver-row preflight, and collects only the selected model table. `LGM.fit()` dispatches to that adapter, while `CanonicalPanel`, `compile_lgm`, and `fit_gaussian` remain the sole model path. Spark results carry immutable canonical prediction keys; Pandas results retain caller-row alignment.

**Tech Stack:** Python 3.11+, PySpark >=3.5 (optional), Java 17 for Spark CI, Formulaic, Pandas, NumPy, SciPy sparse, pytest, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-17-pylgm-pyspark-adapter-design.md`

## Global Constraints

- PySpark is optional; importing or using Pandas-only pyLGM must not import PySpark.
- Spark is a data boundary only; inference remains on the driver through the existing exact-Gaussian engine.
- Do not create a separate Spark compiler or Spark-specific model language.
- Spark input requires explicit `LGM.time`; no synthetic Spark row key is permitted.
- Project only response, canonical keys, Formulaic fixed-effect variables, and structured-effect index columns before collecting.
- Never call `toPandas()` on the original Spark DataFrame.
- Default `max_driver_rows` is 100,000; `None` disables only this adapter preflight.
- Preserve all existing Pandas, legacy configuration, and pipeline behavior.
- Use `PYTHONPATH=src` for local tests. Spark integration tests skip when PySpark or Java is unavailable; CI must run them with PySpark and Java 17.

---

## File Map

- `src/pylgm/inference/result.py`: immutable optional prediction-key metadata.
- `src/pylgm/data/spark.py`: lazy PySpark detection, validation, projection, row-limit preflight, collection, and canonical-key extraction.
- `src/pylgm/data/__init__.py`: export backend-neutral adapter types only if callers need them.
- `src/pylgm/model.py`: Pandas/Spark dispatch and result construction.
- `pyproject.toml`: optional `spark` extra.
- `tests/inference/test_result.py`: prediction-key immutability/shape validation.
- `tests/data/test_spark.py`: local Spark adapter and Spark/Pandas compilation parity.
- `tests/test_model_spark.py`: public `LGM.fit` Spark contract.
- `README.md`: Spark installation, driver boundary, canonical-key behavior, and limitations.
- `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`: mark the adapter slice as shipped while retaining later scope.
- `.github/workflows/test.yml`: core and provisioned Spark test jobs.

### Task 1: Immutable Prediction Keys on Gaussian Results

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Modify: `src/pylgm/model.py`
- Modify: `tests/inference/test_result.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Produces: `GaussianResult(..., prediction_keys: pd.DataFrame | None = None)`.
- Produces: read-only-by-copy `GaussianResult.prediction_keys -> pd.DataFrame | None`.
- Preserves: all existing constructor arguments, result arrays, `block_slices`, and diagnostics.
- Preserves: `_align_predictions_with_source_rows()` while retaining any key metadata it receives.

- [ ] **Step 1: Write failing prediction-key contract tests**

```python
import pandas as pd
import pytest


def test_result_preserves_prediction_keys_by_defensive_copy():
    keys = pd.DataFrame({"region": ["A", "B"], "time": [1, 2]})
    result = GaussianResult(
        labels=("x",), mean=np.array([0.0]), covariance=np.eye(1),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([1.0, 2.0]),
        predictive_variance=np.array([1.0, 1.0]), prediction_keys=keys,
    )
    keys.loc[0, "region"] = "mutated"
    exposed = result.prediction_keys
    assert exposed is not None
    assert exposed["region"].tolist() == ["A", "B"]
    exposed.loc[1, "time"] = 99
    assert result.prediction_keys["time"].tolist() == [1, 2]


def test_result_rejects_prediction_keys_with_wrong_row_count():
    with pytest.raises(ValueError, match="prediction keys.*row count"):
        GaussianResult(
            labels=("x",), mean=np.array([0.0]), covariance=np.eye(1),
            log_marginal_likelihood=0.0,
            predictive_mean=np.array([1.0, 2.0]),
            predictive_variance=np.array([1.0, 1.0]),
            prediction_keys=pd.DataFrame({"time": [1]}),
        )
```

- [ ] **Step 2: Run focused tests and verify the API is absent**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`

Expected: FAIL because `GaussianResult` does not accept `prediction_keys`.

- [ ] **Step 3: Implement immutable DataFrame metadata**

Add a private `_prediction_keys: pd.DataFrame | None` field. Require a Pandas DataFrame when supplied, string and unique column labels, no null key cells, and exactly one key row per predictive result row. Store a deep copy and return a new deep copy from the property. Update `_align_predictions_with_source_rows()` to pass `result.prediction_keys` through unchanged; Pandas fits continue to pass `None`.

- [ ] **Step 4: Run result and regression tests**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py tests/test_model.py -q`

Expected: PASS.

Run: `PYTHONPATH=src pytest -q`

Expected: all existing tests PASS.

- [ ] **Step 5: Commit prediction-key support**

```bash
git add src/pylgm/inference/result.py src/pylgm/model.py tests/inference/test_result.py tests/test_model.py
git commit -m "feat: add immutable prediction keys to Gaussian results"
```

### Task 2: Spark Canonicalization Adapter

**Files:**
- Create: `src/pylgm/data/spark.py`
- Modify: `src/pylgm/data/__init__.py`
- Create: `tests/data/test_spark.py`

**Interfaces:**
- Consumes: declarative `LGM`, `CanonicalPanel`, `DataConfig`, Formulaic `Formula`, and `DataContractError`/`CompilationError`.
- Produces: `is_spark_dataframe(frame: object) -> bool` without importing PySpark when it is unavailable.
- Produces: `SparkCanonicalized(panel: CanonicalPanel, prediction_keys: pd.DataFrame)`.
- Produces: `canonicalize_spark_frame(frame: object, model: LGM, *, max_driver_rows: int | None) -> SparkCanonicalized`.

- [ ] **Step 1: Write Spark adapter tests with an explicit local fixture**

```python
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    session = SparkSession.builder.master("local[2]").appName("pylgm-tests").getOrCreate()
    yield session
    session.stop()


def test_spark_adapter_projects_only_model_columns_and_canonicalizes(spark):
    frame = spark.createDataFrame([
        ("B", 2, 4.0, 8.0, "discard"),
        ("A", 1, 1.0, 2.0, "discard"),
    ], ["region", "time", "x", "y", "raw_only"])
    model = LGM("y", Gaussian(1.0), Fixed("0 + x"), panel=("region",), time="time")

    canonical = canonicalize_spark_frame(frame, model, max_driver_rows=10)

    assert canonical.panel.frame.columns.tolist() == ["region", "time", "x", "y"]
    assert canonical.prediction_keys.to_dict("list") == {"region": ["A", "B"], "time": [1, 2]}


def test_spark_adapter_rejects_missing_time_and_duplicate_keys(spark):
    without_time = spark.createDataFrame([(1.0,)], ["y"])
    model = LGM("y", Gaussian(1.0), Fixed("1"))
    with pytest.raises(DataContractError, match="explicit time"):
        canonicalize_spark_frame(without_time, model, max_driver_rows=10)

    duplicate = spark.createDataFrame([("A", 1, 1.0), ("A", 1, 2.0)], ["region", "time", "y"])
    keyed = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    with pytest.raises(DataContractError, match="duplicate"):
        canonicalize_spark_frame(duplicate, keyed, max_driver_rows=10)
```

Add tests for missing projected columns, null keys, `max_driver_rows=1`, `None`, and Formulaic projection (`Fixed("1 + x + C(region)")`). Add a test that monkeypatches `pyspark.sql.DataFrame.toPandas` and asserts it is invoked only on the projected ordered DataFrame whose columns exclude `raw_only`.

- [ ] **Step 2: Run tests and verify the module is absent**

Run: `PYTHONPATH=src pytest tests/data/test_spark.py -q`

Expected: SKIPPED when PySpark is not installed; in a provisioned environment, FAIL because `pylgm.data.spark` does not exist.

- [ ] **Step 3: Implement lazy detection, validation, projection, and collection**

Use lazy imports inside `src/pylgm/data/spark.py`; module import must work without PySpark. Implement required columns with:

```python
required = {model.response, *model.panel, model.time}
for effect in model.predictor.effects:
    if isinstance(effect, Fixed):
        required.update(Formula(effect.formula).required_variables)
    else:
        required.add(effect.index)
columns = tuple(column for column in frame.columns if column in required)
```

Reject `model.time is None` before Spark actions. Use `filter` plus `isNull`, `groupBy(*keys).count().filter("count > 1")`, and `limit(1).count()` for typed key failures. Select `columns`, apply `limit(max_driver_rows + 1).count()` when the limit is not `None`, then call `orderBy(*keys).toPandas()` only on the selected frame. Construct `CanonicalPanel.from_frame()` using `DataConfig(time=model.time, response=model.response, panel=model.panel)` and set `prediction_keys` from its canonical frame.

Only normalize Spark `AnalysisException` instances raised by the known projection/order validation operations; let unrelated Spark/JVM failures propagate.

- [ ] **Step 4: Run adapter tests**

Run: `PYTHONPATH=src pytest tests/data/test_spark.py -q`

Expected: PASS with PySpark/Java; otherwise explicit SKIP.

Run: `PYTHONPATH=src pytest tests/data/test_panel.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the adapter**

```bash
git add src/pylgm/data/spark.py src/pylgm/data/__init__.py tests/data/test_spark.py
git commit -m "feat: add Spark canonicalization adapter"
```

### Task 3: Integrate Spark Dispatch with `LGM.fit`

**Files:**
- Modify: `src/pylgm/model.py`
- Create: `tests/test_model_spark.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `canonicalize_spark_frame`, `SparkCanonicalized`, `GaussianResult`, `compile_lgm`, and `fit_gaussian`.
- Produces: `LGM.fit(frame: object, engine: str = "exact_gaussian", *, max_driver_rows: int | None = 100_000) -> GaussianResult`.
- Preserves: Pandas caller-order predictions, synthetic Pandas row key, legacy behavior, and explicit unknown-engine errors.
- Produces: Spark `GaussianResult.prediction_keys` in canonical key order.

- [ ] **Step 1: Write public Spark-fit parity tests**

```python
import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")


def test_spark_fit_matches_sorted_pandas_and_exposes_join_keys(spark):
    rows = [("B", 2, 4.0, 8.0), ("A", 1, 1.0, 2.0), ("B", 1, 3.0, 6.0), ("A", 2, 2.0, 4.0)]
    columns = ["region", "time", "x", "y"]
    model = LGM("y", Gaussian(1.0), Fixed("0 + x", prior_precision=1.0), panel=("region",), time="time")

    spark_result = model.fit(spark.createDataFrame(rows, columns))
    pandas_result = model.fit(pd.DataFrame(rows, columns=columns).sort_values(["region", "time"]))

    np.testing.assert_allclose(spark_result.predictive_mean, pandas_result.predictive_mean)
    np.testing.assert_allclose(spark_result.predictive_variance, pandas_result.predictive_variance)
    assert spark_result.prediction_keys.to_dict("list") == {"region": ["A", "A", "B", "B"], "time": [1, 2, 1, 2]}


def test_pandas_fit_keeps_no_prediction_keys():
    result = LGM("y", Gaussian(1.0), Fixed("1")).fit(pd.DataFrame({"y": [1.0]}))
    assert result.prediction_keys is None
```

Add a test that Spark `max_driver_rows=1` raises before fitting and a Pandas test that retains caller-order output when `max_driver_rows` is left at its default.

- [ ] **Step 2: Run tests and verify Spark still follows the old rejection path**

Run: `PYTHONPATH=src pytest tests/test_model_spark.py tests/test_model.py -q`

Expected: Spark file SKIPPED without PySpark; provisioned Spark tests initially FAIL because `LGM.fit` rejects Spark input.

- [ ] **Step 3: Implement explicit Pandas/Spark dispatch**

Split current `fit` logic into private helpers `_fit_pandas()` and `_fit_spark()`. `_fit_pandas()` must retain the current synthetic-key and caller-order code exactly. `_fit_spark()` calls `canonicalize_spark_frame`, compiles `canonical.panel`, dispatches `exact_gaussian`, and rebuilds `GaussianResult` with unchanged posterior arrays and `prediction_keys=canonical.prediction_keys`.

Validate `max_driver_rows` in the Spark adapter only: permit `None` or a positive ordinary `int`, reject `bool`, zero, negative, float, and string values with `DataContractError`. For non-Pandas/non-Spark input retain the existing `frame must be a Pandas DataFrame` error; update it to mention Spark only when the adapter identifies a genuine Spark frame.

- [ ] **Step 4: Run public and regression tests**

Run: `PYTHONPATH=src pytest tests/test_model.py tests/test_model_spark.py tests/data/test_spark.py -q`

Expected: PASS with Spark or explicit SKIP only for Spark tests.

Run: `PYTHONPATH=src pytest -q`

Expected: all existing tests PASS.

- [ ] **Step 5: Commit public Spark support**

```bash
git add src/pylgm/model.py tests/test_model.py tests/test_model_spark.py
git commit -m "feat: accept Spark DataFrames in declarative fits"
```

### Task 4: Package, Document, and Provision Spark Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`
- Create: `.github/workflows/test.yml`
- Modify: `examples/general_lgm/README.md`
- Create: `examples/general_lgm/run_spark.py`
- Modify: `tests/test_package.py`

**Interfaces:**
- Produces: `pylgm[spark]` extra containing `pyspark>=3.5`.
- Produces: a runnable Spark example that starts a local session, reads the committed example CSV, fits the existing YAML model, and prints prediction keys and mean.
- Produces: CI core and Spark jobs; Spark job installs Java 17 and `.[dev,spark]`, then runs Spark tests unskipped.

- [ ] **Step 1: Write package and example contract tests**

```python
import tomllib


def test_spark_extra_is_optional_and_declared():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["optional-dependencies"]["spark"] == ["pyspark>=3.5"]
    assert "pyspark" not in metadata["project"]["dependencies"]
```

Add a subprocess smoke test guarded by `pytest.importorskip("pyspark")` that runs `examples/general_lgm/run_spark.py` with `PYTHONPATH=src` and asserts output contains `prediction_keys`.

- [ ] **Step 2: Run package tests and verify the Spark extra is absent**

Run: `PYTHONPATH=src pytest tests/test_package.py -q`

Expected: FAIL because the `spark` optional dependency is not declared.

- [ ] **Step 3: Add optional dependency, docs, example, and CI workflow**

Add exactly:

```toml
[project.optional-dependencies]
spark = ["pyspark>=3.5"]
```

Keep the existing `dev` extra. Document that Spark performs data preparation and collection only, that exact-Gaussian inference remains driver-bound, that Spark results are joined using `result.prediction_keys`, and that `model.time` is required. Update the Latte/PySpark architecture document to mark this adapter as shipped while preserving deferred non-Gaussian, spatial, AR1, new-data prediction, and distributed-inference scope.

Create the workflow with a core job (`python -m pip install -e ".[dev]"`, `ruff check src tests`, `pytest -q`) and a Spark job using `actions/setup-java` with Temurin 17 and `python -m pip install -e ".[dev,spark]"`, followed by `pytest tests/data/test_spark.py tests/test_model_spark.py -q`. The Spark job is provisioned with both PySpark and Java 17; therefore either test file skipping is a CI failure and must be diagnosed rather than accepted.

- [ ] **Step 4: Run local verification**

Run: `PYTHONPATH=src pytest tests/test_package.py -q`

Expected: PASS.

Run: `PYTHONPATH=src ruff check src tests`

Expected: PASS.

Run: `PYTHONPATH=src pytest -q`

Expected: full suite PASS; Spark tests are either PASS or explicit SKIP locally.

- [ ] **Step 5: Commit packaging and docs**

```bash
git add pyproject.toml README.md docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md .github/workflows/test.yml examples/general_lgm tests/test_package.py
git commit -m "docs: publish optional Spark adapter"
```

## Self-Review Checklist

- Task 1 supplies the immutable result metadata consumed by Task 3.
- Task 2 is the only Spark-specific implementation and has no import-time PySpark dependency.
- Task 3 is the only public dispatch change and preserves Pandas behavior.
- Task 4 supplies package installation, example, and provisioned Spark verification.
- The plan implements every included spec item and excludes CLI work, distributed inference, non-Gaussian engines, spatial effects, AR1, and forecasting relocation.
- The only optional local outcome is an explicit Spark-test skip when PySpark/Java is absent; CI must run Spark tests unskipped.

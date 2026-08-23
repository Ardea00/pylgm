# Out-of-Sample Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `result.predict(new_data)` — score rows not supplied to `LGM.fit`, using the fitted latent posterior.

**Architecture:** A `PredictionContext` (built once at fit time from the model, panel and compiled model) carries what the result lacks in order to rebuild a design for new rows: each `Fixed` effect's formulaic `ModelSpec`, each structured effect's `(name, index column, fitted labels)`, the compiled likelihood, the offset column, and the latent width. `model.py` attaches it to the result via the existing `_rebuild_result`, so **no engine, `CompiledLGM`, or family changes are needed**. `predict()` rebuilds `X_new`, then returns `η = X μ + offset` and `Var(η) = diag(X Σ Xᵀ)` as an immutable `Prediction`.

**Tech Stack:** Python, NumPy, SciPy sparse, pandas, formulaic. No new runtime dependency.

## Global Constraints

- `predict` reuses the fitted latent posterior and **cannot create latent columns**. An index level absent from the fit is an **error** naming the block, the offending levels, and directing to the NaN-response workflow (include such rows at `fit` time with a `NaN` response — this already works and is the statistically correct joint fit).
- The fitted categorical encoding MUST be reproduced with the retained formulaic `ModelSpec` (`model_spec.get_model_matrix(new_data)`). Re-running `model_matrix(formula, new_data)` can silently yield different columns when the new frame's categorical levels differ — never do that.
- `predictive_variance` follows **each engine's existing convention** (exact-Gaussian adds `σ²`; Laplace does not), and `fitted_mean` uses **exactly** the existing response-scale semantics. Reuse the helpers the engines already use rather than reimplementing them.
- `Var(η)` is computed as `np.einsum("ij,jk,ik->i", X, Σ, X)` — never form the `n_new × n_new` matrix.
- `new_data` is a pandas `DataFrame` and must NOT be required to carry the response column.
- Everything is additive: existing results, engines, predictions and `_rebuild_result` behaviour unchanged; full suite green; `ruff check src tests` clean.

---

### Task 1: `Prediction`, `PredictionContext`, and `predict_from`

**Files:**
- Create: `src/pylgm/inference/prediction.py`
- Test: `tests/inference/test_prediction.py`

**Interfaces:**
- Produces:
  - `PredictionContext(fixed_specs, structured, likelihood, offset, width)` — frozen; `fixed_specs` is a tuple of formulaic `ModelSpec`; `structured` is a tuple of `(block_name, index_column, labels_tuple)` in fitted block order (fixed blocks first, matching the compiled order).
  - `Prediction(predictive_mean, predictive_variance, fitted_mean, keys)` — frozen, read-only arrays, with `.to_frame()`.
  - `predict_from(context, mean, covariance, new_data) -> Prediction`.

> **Block order matters.** The rebuilt design must concatenate blocks in exactly the fitted order. Represent that by keeping `fixed_specs` and `structured` as ordered tuples plus an explicit `order` tuple of `("fixed", i)` / `("structured", j)` entries, OR — simpler and preferred — make `structured` carry every block including fixed ones by using a single ordered tuple of entries tagged with their kind. Pick one and keep it consistent; Task 2 must build it in the same order the compiler builds blocks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/inference/test_prediction.py
import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix

from pylgm.inference.prediction import Prediction, PredictionContext, predict_from
from pylgm.likelihoods import CompiledGaussian


def _context(frame, formula="1 + x"):
    spec = model_matrix(formula, frame).model_spec
    return PredictionContext(
        entries=(("fixed", spec), ("structured", ("region", "region", ("a", "b")))),
        likelihood=CompiledGaussian(0.5),
        offset=None,
        width=len(spec.column_names) + 2,
    )


def test_predicts_linear_predictor_mean_and_variance():
    frame = pd.DataFrame({"x": [1.0, 2.0], "region": ["a", "b"]})
    context = _context(frame)
    width = context.width
    mean = np.arange(1.0, width + 1.0)
    covariance = np.eye(width) * 0.25
    prediction = predict_from(context, mean, covariance, frame)

    # design: [1, x, region==a, region==b]
    design = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0]])
    expected_mean = design @ mean
    expected_var = np.einsum("ij,jk,ik->i", design, covariance, design) + 0.5**2
    assert np.allclose(prediction.predictive_mean, expected_mean)
    assert np.allclose(prediction.predictive_variance, expected_var)


def test_to_frame_is_aligned_and_has_sd():
    frame = pd.DataFrame({"x": [1.0, 2.0], "region": ["a", "b"]}, index=[10, 20])
    context = _context(frame)
    prediction = predict_from(context, np.ones(context.width), np.eye(context.width), frame)
    table = prediction.to_frame()
    assert list(table.index) == [10, 20]
    assert list(table.columns) == ["predictive_mean", "predictive_sd", "fitted_mean"]
    assert np.allclose(table["predictive_sd"], np.sqrt(prediction.predictive_variance))


def test_unseen_level_names_the_block_and_the_workflow():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    unseen = pd.DataFrame({"x": [1.0], "region": ["zzz"]})
    with pytest.raises(ValueError, match="region"):
        predict_from(context, np.ones(context.width), np.eye(context.width), unseen)
    with pytest.raises(ValueError, match="NaN"):
        predict_from(context, np.ones(context.width), np.eye(context.width), unseen)


def test_missing_index_column_raises():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    with pytest.raises(ValueError, match="region"):
        predict_from(context, np.ones(context.width), np.eye(context.width),
                     pd.DataFrame({"x": [1.0]}))


def test_empty_frame_raises():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    with pytest.raises(ValueError, match="empty"):
        predict_from(context, np.ones(context.width), np.eye(context.width),
                     frame.iloc[:0])


def test_prediction_arrays_are_read_only():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    prediction = predict_from(context, np.ones(context.width), np.eye(context.width), frame)
    with pytest.raises(ValueError):
        prediction.predictive_mean[0] = 5.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_prediction.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `prediction.py`**

Write the module with:

1. `PredictionContext` — frozen dataclass with `entries: tuple[tuple[str, object], ...]` (each `("fixed", ModelSpec)` or `("structured", (name, index, labels))`, in fitted block order), `likelihood`, `offset: str | None`, `width: int`.
2. `_design_for(context, new_data) -> np.ndarray` — validates `new_data` is a non-empty DataFrame; for each entry builds its block and horizontally stacks:
   - fixed: `np.asarray(spec.get_model_matrix(new_data), dtype=float)`, wrapping formulaic's error in a `ValueError` that names the missing variable;
   - structured: check the index column exists (else `ValueError` naming it); map `str(value)` onto `labels`; collect values with no match and raise
     ```python
     raise ValueError(
         f"predict() cannot score rows whose {name!r} level was not in the fitted "
         f"model: {missing!r}. predict reuses the fitted latent posterior, so it "
         "cannot create a new latent component. To forecast new levels, include "
         "those rows at fit time with a NaN response instead."
     )
     ```
     then build the one-hot block.
   Finally assert the stacked width equals `context.width` (else `ValueError`).
3. `predict_from(context, mean, covariance, new_data)`:
   - `design = _design_for(...)`; `offset` from `new_data[context.offset]` when set (missing → `ValueError` naming it), else zeros;
   - `eta = design @ mean + offset`;
   - `variance = np.einsum("ij,jk,ik->i", design, covariance, design)`;
   - **reuse the engines' response-scale logic** — read `src/pylgm/inference/gaussian.py` and `src/pylgm/inference/laplace.py` and call the same helper/likelihood methods they use. For a `CompiledGaussian` likelihood add `sigma**2` to the variance and set `fitted_mean = eta` (matching `GaussianResult`); otherwise leave the variance as the `η` variance and compute `fitted_mean` exactly the way `fit_laplace` does. Do NOT re-derive the formulas by hand.
   - raise `NumericalError` if any output is non-finite;
   - return `Prediction(..., keys=new_data.index)`.
4. `Prediction` — frozen dataclass with read-only arrays (match the `_readonly_array` idiom already used in `inference/result.py`) and:
   ```python
   def to_frame(self) -> pd.DataFrame:
       return pd.DataFrame(
           {
               "predictive_mean": self.predictive_mean,
               "predictive_sd": np.sqrt(self.predictive_variance),
               "fitted_mean": self.fitted_mean,
           },
           index=self.keys,
       )
   ```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_prediction.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference/prediction.py tests/inference/test_prediction.py
git commit -m "feat: add out-of-sample prediction context and helper"
```

---

### Task 2: Build the prediction context at compile time

**Files:**
- Modify: `src/pylgm/compiler.py`
- Test: `tests/test_compiler.py` (append)

**Interfaces:**
- Consumes: `PredictionContext` (Task 1).
- Produces: `build_prediction_context(model: "LGM", panel: CanonicalPanel, compiled: CompiledLGM) -> PredictionContext` — entries in the **same order** the compiler builds blocks, fixed specs recomputed via `model_matrix(effect.formula, panel.frame).model_spec`, structured labels taken from `compiled.blocks` (never rebuilt — spatial builders are expensive), likelihood/offset/width from the compiled model and the `LGM`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compiler.py  (append)
def test_build_prediction_context_matches_the_compiled_blocks():
    import pandas as pd

    from pylgm import Fixed, Gaussian, IID, LGM, RW1
    from pylgm.compiler import build_prediction_context, compile_lgm
    from pylgm.data import CanonicalPanel

    frame = pd.DataFrame({
        "y": [0.1, 0.2, 0.3, 0.4],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
        "t": [0, 1, 2, 3],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region") + RW1("trend", index="t"),
        likelihood=Gaussian(sigma=0.5),
    )
    panel = CanonicalPanel(frame, np.array([True] * 4), ("region",), "y")
    compiled = compile_lgm(model, panel)
    context = build_prediction_context(model, panel, compiled)

    assert context.width == compiled.design.shape[1]
    kinds = [kind for kind, _ in context.entries]
    assert kinds == ["fixed", "structured", "structured"]
    # structured entries carry the compiled labels, in compiled block order
    structured = [payload for kind, payload in context.entries if kind == "structured"]
    assert structured[0][:2] == ("region", "region")
    assert structured[0][2] == compiled.blocks[1].labels
    assert structured[1][:2] == ("trend", "t")
    assert structured[1][2] == compiled.blocks[2].labels
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_compiler.py -q -k prediction_context`
Expected: FAIL (function missing).

- [ ] **Step 3: Implement `build_prediction_context`**

Add to `compiler.py`:

```python
def build_prediction_context(
    model: "LGM", panel: CanonicalPanel, compiled: CompiledLGM
) -> PredictionContext:
    """Capture what ``result.predict(new_data)`` needs to rebuild a design.

    Structured labels come from the already-compiled blocks so the (sometimes
    expensive) effect builders are not run twice; only each fixed effect's
    formulaic spec is recomputed, which is what reproduces the fitted
    categorical encoding on new rows.
    """
    blocks = {block.name: block for block in compiled.blocks}
    entries: list[tuple[str, object]] = []
    for effect in model.predictor.effects:
        if isinstance(effect, Fixed):
            spec = model_matrix(effect.formula, panel.frame).model_spec
            entries.append(("fixed", spec))
            continue
        block = blocks[effect.name]
        entries.append(("structured", (effect.name, effect.index, block.labels)))
    return PredictionContext(
        entries=tuple(entries),
        likelihood=compiled.likelihood,
        offset=model.offset,
        width=compiled.design.shape[1],
    )
```

Add the imports: `from formulaic import model_matrix` and `from pylgm.inference.prediction import PredictionContext`. **Check for an import cycle** (`inference` importing `compiler`); if one exists, import `PredictionContext` inside the function.

> The effect order in `model.predictor.effects` is exactly the block order the
> compiler builds, so `entries` lines up with `compiled.blocks`. The test above
> asserts that alignment — do not assume it silently.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_compiler.py -q -k prediction_context`
Expected: PASS.

- [ ] **Step 5: Full suite + ruff + commit**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`

```bash
git add src/pylgm/compiler.py tests/test_compiler.py
git commit -m "feat: capture the prediction context when compiling a model"
```

---

### Task 3: `predict()` on the result types, wired through `LGM.fit`

**Files:**
- Modify: `src/pylgm/inference/result.py`, `src/pylgm/model.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: an optional `prediction_context` field on `GaussianResult`, `LaplaceResult`, `INLAResult` (default `None`, carried through `_rebuild_result`), and `result.predict(new_data) -> Prediction` on each. `LGM.fit` attaches the context.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_predict.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson
from pylgm.priors import PCPrecision


def _frame():
    return pd.DataFrame({
        "y": [0.5, 1.5, 2.5, 3.5],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })


def _gaussian_model():
    return LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Gaussian(sigma=0.5),
    )


def test_predicting_the_fit_rows_reproduces_the_fit_predictions():
    # The strongest oracle: predict() on the training frame must reproduce
    # exactly what fit already reported for those rows.
    frame = _frame()
    result = _gaussian_model().fit(frame)
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)
    assert np.allclose(prediction.predictive_variance, result.predictive_variance)


def test_predicts_new_covariate_values_on_known_levels():
    frame = _frame()
    result = _gaussian_model().fit(frame)
    new = pd.DataFrame({"x": [10.0, -3.0], "region": ["b", "a"]})
    prediction = result.predict(new)
    assert prediction.predictive_mean.shape == (2,)
    assert np.all(np.isfinite(prediction.predictive_mean))
    assert np.all(prediction.predictive_variance > 0)


def test_poisson_fit_row_round_trip():
    frame = pd.DataFrame({
        "y": [2.0, 3.0, 5.0, 4.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)
    assert np.allclose(prediction.fitted_mean, result.fitted_mean)


def test_categorical_subset_reproduces_the_fitted_encoding():
    # New data containing only ONE of the fitted categorical levels must still
    # produce the fitted columns. Re-running model_matrix here would not.
    frame = pd.DataFrame({
        "y": [0.5, 1.5, 2.5, 3.5],
        "g": ["a", "b", "c", "a"],
        "region": ["a", "b", "a", "b"],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + g") + IID("region", index="region", precision=2.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame)
    only_b = pd.DataFrame({"g": ["b"], "region": ["a"]})
    prediction = result.predict(only_b)
    assert prediction.predictive_mean.shape == (1,)
    assert np.all(np.isfinite(prediction.predictive_mean))


def test_unseen_level_raises_and_points_at_the_nan_workflow():
    result = _gaussian_model().fit(_frame())
    with pytest.raises(ValueError, match="NaN response"):
        result.predict(pd.DataFrame({"x": [1.0], "region": ["zzz"]}))


def test_spatial_effect_predicts_a_graph_node_absent_from_the_fit_data():
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    frame = pd.DataFrame({"y": [0.5, 1.5], "region": ["a", "b"]})
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", precision=1.0, graph=graph),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame)
    # "c" is a graph node, so it has a fitted latent component even though no
    # row referenced it.
    prediction = result.predict(pd.DataFrame({"region": ["c"]}))
    assert np.all(np.isfinite(prediction.predictive_mean))


def test_inla_result_predicts_with_the_integrated_posterior():
    frame = _frame()
    model = LGM(
        response="y",
        predictor=Fixed("1 + x")
        + IID("region", index="region",
              precision=Hyperparameter("region.precision", initial=1.0,
                                       prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame, hyperparameters="integrate")
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)


def test_offset_is_read_from_the_new_rows():
    frame = pd.DataFrame({
        "y": [2.0, 3.0, 5.0, 4.0],
        "region": ["a", "b", "a", "b"],
        "logexp": [0.0, 0.1, 0.2, 0.3],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1") + IID("region", index="region", precision=2.0),
        likelihood=Poisson(),
        offset="logexp",
    )
    result = model.fit(frame, engine="laplace")
    low = result.predict(pd.DataFrame({"region": ["a"], "logexp": [0.0]}))
    high = result.predict(pd.DataFrame({"region": ["a"], "logexp": [1.0]}))
    assert high.predictive_mean[0] == pytest.approx(low.predictive_mean[0] + 1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_predict.py -q`
Expected: FAIL (`predict` missing).

- [ ] **Step 3: Add the field and method to the three result types**

In `inference/result.py`, for `GaussianResult`, `LaplaceResult` and `INLAResult`:
- accept `prediction_context: object | None = None` in `__init__`, store it with `object.__setattr__` (duck-typed; do not import `PredictionContext` at module load if that cycles);
- add:
  ```python
  def predict(self, new_data):
      """Score new rows with the fitted latent posterior."""
      from pylgm.inference.prediction import predict_from

      if self.prediction_context is None:
          raise ValueError(
              "this result carries no prediction context; predict() is available "
              "on results produced by LGM.fit"
          )
      return predict_from(self.prediction_context, self.mean, self.covariance, new_data)
  ```

In `model.py`'s `_rebuild_result`, add `prediction_context` to the `common` dict, carrying `result.prediction_context` forward unless overridden (mirror the existing `hyperparameters`/`diagnostics` override idiom, adding a `prediction_context=None` keyword).

- [ ] **Step 4: Attach the context in `LGM.fit`**

In `model.py`, after the engine produces a result and the compiled model is in scope, build the context and attach it via `_rebuild_result(result, prediction_context=context)`. Read `_fit_pandas` first: there are separate branches for `hyperparameters="integrate"` (family/INLA) and the plug-in path. Both must attach it. For the integrate path, materialize the compiled model the context needs from the family (or call `compile_lgm` — prefer whatever the branch already has in scope; do NOT compile twice if a compiled model is already available).

> `build_prediction_context` needs a `CompiledLGM` only for block labels, the
> likelihood, and the design width — any materialization of the same model gives
> identical values for those, since they do not depend on hyperparameter values.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_predict.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite + ruff + commit**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`

```bash
git add src/pylgm/inference/result.py src/pylgm/model.py tests/test_predict.py
git commit -m "feat: predict new rows from a fitted result"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Document `predict` in `README.md`**

Add a "Predicting new rows" section covering: `result.predict(new_data)` returning a `Prediction` with `predictive_mean`, `predictive_variance`, `fitted_mean` and `.to_frame()`; the columns `new_data` must carry (each effect's index, the formula variables, the offset — but **not** the response); that `predictive_variance`/`fitted_mean` follow the same conventions as the fit-row outputs; and the **table distinguishing `predict` from the NaN-response workflow**, with a short runnable example of each. State plainly that `predict` cannot create latent columns and that unseen levels raise.

Also correct the 0.3-scope paragraph, which currently lists `result.predict(new_data)` as deferred and says predictions "cover the rows supplied to `LGM.fit`".

- [ ] **Step 2: Update the roadmap**

Mark out-of-sample prediction shipped. Record the deferred pieces: Spark `new_data`, prior fallback for unseen levels, predictive quantiles/simulation, and a future-frame construction helper.

- [ ] **Step 3: Verify**

Run BOTH README examples verbatim under `PYTHONPATH=src` (paste into a temp file and execute); fix them if they error. Then `PYTHONPATH=src python -m pytest -q`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: document out-of-sample prediction"
```

---

## Self-Review Notes

- **Spec coverage:** `Prediction`/`PredictionContext`/`predict_from` incl. errors and `to_frame` (T1) ✓; context construction in compiler order (T2) ✓; result wiring, `_rebuild_result`, `LGM.fit` attachment, and the end-to-end matrix incl. the fit-row oracle, categorical trap, spatial node, INLA, offset (T3) ✓; docs incl. the predict-vs-NaN-rows distinction (T4) ✓.
- **Minimal blast radius:** attaching the context in `model.py` avoids touching the engines, `CompiledLGM`, and the families entirely — the earlier slices showed those are the risky shared surfaces.
- **The load-bearing detail** is the retained formulaic `ModelSpec`; T3's categorical-subset test is what proves it, and it fails if the implementation re-runs `model_matrix` on the new frame.
- **Type consistency:** `PredictionContext(entries, likelihood, offset, width)` is constructed identically in T1's tests and T2's builder; `predict_from(context, mean, covariance, new_data)` is called identically in T1 and T3.
- **Reuse over re-derivation:** T1 Step 3 explicitly instructs reading `gaussian.py`/`laplace.py` and reusing their response-scale logic rather than reimplementing `exp(μ+σ²/2)` etc., so the fit-row round-trip tests in T3 can actually hold.
- **Known asymmetry (documented, not fixed):** `GaussianResult.predictive_variance` includes `σ²` while `LaplaceResult.predictive_variance` does not. `predict` mirrors it deliberately so round-trips match; the docs say so.

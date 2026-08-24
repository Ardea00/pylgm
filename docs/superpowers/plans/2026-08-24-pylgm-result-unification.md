# Result-Type Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplication across `pylgm`'s three result types and three marginal types, then resolve the `predictive_variance` convention inconsistency.

**Architecture:** Two sequenced parts in separate commits. **Part A** (Tasks 2–3) is a pure refactor whose acceptance criterion is that *nothing observable changes*; **Part B** (Task 4) is a deliberate behaviour change. Task 1 captures a golden baseline first, so Part A can be proven equivalent and Part B's blast radius is visible as an explicit diff to that baseline.

**Tech Stack:** Python, NumPy, pandas. No new runtime dependency.

## Global Constraints

- **The three result types must remain siblings.** They may inherit `_BaseResult`; **none may inherit another**. `model.py:_rebuild_result` tests `isinstance(result, LaplaceResult)` *before* `isinstance(result, INLAResult)`, and `optimization/inla.py:310` tests `isinstance(reference, LaplaceResult)`. Making `INLAResult` a subclass of `LaplaceResult` would silently misroute every INLA rebuild. This is the highest-risk property in the slice.
- **Part A changes no observable behaviour** — not values, not `repr`, not `isinstance` outcomes, not immutability, not error messages. Proven by the golden baseline, not by "the suite passes".
- Part B changes `predictive_variance` for Gaussian models only, and must keep `predict()` consistent with the fit rows.
- Criteria (DIC/WAIC/CPO/PIT) must not move in either part — `_model_criteria` never reads `predictive_variance`; assert it.
- Public API names are preserved. No new runtime dependency. Full suite green; `ruff check src tests` clean.

---

### Task 1: Golden baseline of the public result surface

**Files:**
- Create: `tests/inference/test_result_surface.py`
- Create: `tests/inference/result_surface_baseline.json`

**Interfaces:**
- Produces: `_surface(result) -> dict` capturing every public attribute and method output of a result, and a test comparing a matrix of fits against a committed baseline.

> Run this task **before any refactoring**, on unmodified code, so the baseline records current behaviour.

- [ ] **Step 1: Write the harness and generate the baseline**

Create `tests/inference/test_result_surface.py` with:

- `MATRIX`: a list of named model builders covering Gaussian/Poisson/Bernoulli × plug-in/optimise/integrate × a few effects (`IID`, `RW1`, `AR1` with fixed rho, `Besag`). Use small fixed frames and fixed seeds so results are deterministic. Keep it to roughly 8–12 fits so the file stays reviewable.
- `_surface(result)` returning a JSON-able dict with, for each fit:
  - `type` (class name), and `isinstance` against all three result classes;
  - every public scalar/array attribute: `labels`, `mean`, `covariance`, `log_marginal_likelihood`, `predictive_mean`, `predictive_variance`, `engine`, `converged`, `hyperparameters`, and — when present — `fitted_mean`, `link_name`;
  - method outputs: `latent_marginals()` (`mean`/`variance`/`std`), `hyperparameter_marginals()` (per name), `linear_combinations(W)` for a fixed small `W`, and `criteria` fields when present;
  - `repr(result)`;
  - immutability: whether writing to each returned array raises.
  Round floats to 12 significant digits when serialising so the file is stable across platforms.
- `test_result_surface_matches_baseline()` comparing the freshly computed surface to the committed JSON, reporting the first differing key path.
- `test_constructor_error_messages_are_stable()` capturing the message from a handful of invalid constructions (wrong ndim, non-finite, mismatched lengths) for each result type.
- `test_result_types_are_siblings()`: for each pair of the three result types, an instance of one is **not** an instance of the other.

Generate the baseline with a small `__main__` block or a one-off script, and commit the JSON.

- [ ] **Step 2: Verify the harness is meaningful**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_result_surface.py -q` → PASS.
Then temporarily perturb something observable (e.g. add `+1e-9` to `predictive_variance` in `gaussian.py`), re-run, confirm the test FAILS naming the key, and revert.
Report both outcomes — a harness that cannot fail is worthless.

- [ ] **Step 3: Commit**

```bash
git add tests/inference/test_result_surface.py tests/inference/result_surface_baseline.json
git commit -m "test: pin the public result surface with a golden baseline"
```

---

### Task 2: Consolidate the marginal validators and add the protocol

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Test: `tests/inference/test_marginals.py` (or the existing marginal tests; append)

**Interfaces:**
- Produces: `_marginal_grid_array(value, name)` replacing both `_skew_normal_marginal_array` and `_tabulated_marginal_array`; a `LatentMarginals` `typing.Protocol` documenting `mean`/`variance`/`std`/`quantile`.

- [ ] **Step 1: Replace the duplicated validators**

`_skew_normal_marginal_array` and `_tabulated_marginal_array` are byte-identical. Define once:

```python
def _marginal_grid_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional (n_components, n_grid)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result
```

and update both call sites. Delete the two originals. **The error messages must be unchanged** — Task 1's error-message test enforces this.

- [ ] **Step 2: Add the protocol**

```python
@runtime_checkable
class LatentMarginals(Protocol):
    """The read surface every latent-marginal representation provides.

    The three implementations differ irreducibly -- a Gaussian mean/variance
    pair, a grid-weighted mixture of skew-normals, and a tabulated density --
    so this documents the contract rather than sharing an implementation.
    """

    @property
    def mean(self) -> np.ndarray: ...
    @property
    def variance(self) -> np.ndarray: ...
    @property
    def std(self) -> np.ndarray: ...
    def quantile(self, probability: float) -> np.ndarray: ...
```

Export it from `pylgm.inference` if the package exports the marginal types; add to `__all__` where relevant (a `tests/test_public_exports.py` test asserts `__all__` entries resolve).

- [ ] **Step 3: Assert the protocol is satisfied**

Add a test that each of `GaussianMarginals`, `SkewNormalMarginals`, `TabulatedMarginals` instances `isinstance(..., LatentMarginals)`.

- [ ] **Step 4: Verify and commit**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
The golden-surface test from Task 1 must still pass untouched.

```bash
git add src/pylgm/inference/result.py tests/
git commit -m "refactor: share the marginal grid validator and document the contract"
```

---

### Task 3: `_BaseResult`

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Test: `tests/inference/test_result_surface.py` (unchanged — it is the oracle)

**Interfaces:**
- Produces: `_BaseResult` carrying the nine shared read-only properties, the four delegating methods, and a `_init_common(...)` performing the shared constructor validation and storage. `GaussianResult`, `LaplaceResult`, `INLAResult` inherit it as **siblings**.

- [ ] **Step 1: Read all three constructors carefully first**

Before editing, list for each of the three types: its full field set, the exact validation performed, the order of validation (error messages depend on it), and which fields are optional. The shared subset goes in `_init_common`; anything type-specific stays in the subclass. Do not change the order in which validation errors are raised.

- [ ] **Step 2: Introduce `_BaseResult`**

```python
class _BaseResult:
    """Shared read surface for the fitted-result types.

    The three results are SIBLINGS under this base: none inherits another.
    ``model._rebuild_result`` dispatches with isinstance and tests
    ``LaplaceResult`` before ``INLAResult``, so making one a subclass of
    another would silently misroute every rebuild of the more specific type.
    """

    def _init_common(self, *, labels, mean, covariance, log_marginal_likelihood,
                     predictive_mean, predictive_variance, block_slices,
                     diagnostics, prediction_keys, hyperparameters,
                     prediction_context) -> None:
        ...  # the validation and object.__setattr__ storage shared today
```

Then the shared properties (`mean`, `covariance`, `predictive_mean`,
`predictive_variance`, `prediction_keys`, `hyperparameters`,
`prediction_context`, `engine`, `converged`) and the four delegating methods
(`latent_marginals`, `hyperparameter_marginals`, `linear_combinations`,
`predict`).

`engine` currently returns a per-type constant — keep that by having each
subclass define a class attribute (e.g. `_ENGINE = "exact_gaussian"`) that the
base property reads, so the value is still per-type but the property is
defined once. Do the same for anything else that is "same shape, different
constant".

> `INLAResult.latent_marginals` has a richer signature than the other two (it
> can return the skew/tabulated table). Keep its override in the subclass; the
> base provides the common two.

- [ ] **Step 3: Make each result inherit and delegate**

Each subclass keeps its explicit `__init__` signature, calls
`self._init_common(...)` with the shared fields, then validates and stores its
own extras (`fitted_mean`, `link_name`, `latent_marginal_table`, `criteria`,
`hyperparameter_marginals` …). Keep the `@dataclass(frozen=True, init=False)`
decoration and `object.__setattr__` storage exactly as now.

- [ ] **Step 4: Verify equivalence — this is the acceptance gate**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_result_surface.py -q`
Expected: PASS with the baseline **unmodified**. If any key differs, the
refactor changed behaviour — fix the refactor, do **not** regenerate the
baseline.

Then: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Report the line count of `result.py` before and after.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference/result.py
git commit -m "refactor: share the fitted-result read surface and validation"
```

---

### Task 4: `predictive_variance` becomes the linear-predictor variance

**Files:**
- Modify: `src/pylgm/inference/gaussian.py`, `src/pylgm/inference/result.py`, `src/pylgm/inference/prediction.py`
- Modify: `tests/inference/result_surface_baseline.json` (deliberately — see Step 4)
- Test: `tests/inference/test_predictive_variance_convention.py`

**Interfaces:**
- Produces: `predictive_variance` = `Var(η)` for every result type and for `predict()`; `GaussianResult.observation_variance` exposing the σ² previously folded in.

- [ ] **Step 1: Write the failing tests**

```python
# tests/inference/test_predictive_variance_convention.py
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson


def _frame():
    return pd.DataFrame({
        "y": [0.5, 1.5, 2.5, 3.5],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })


def _gaussian_result(sigma=0.5):
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Gaussian(sigma=sigma),
    )
    return model.fit(_frame())


def test_gaussian_predictive_variance_is_the_linear_predictor_variance():
    sigma = 0.5
    result = _gaussian_result(sigma)
    design = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0],
                       [1.0, 3.0, 1.0, 0.0], [1.0, 4.0, 0.0, 1.0]])
    expected = np.einsum("ij,jk,ik->i", design, result.covariance, design)
    assert np.allclose(result.predictive_variance, expected)
    # ...and specifically NOT the old value
    assert not np.allclose(result.predictive_variance, expected + sigma**2)


def test_observation_variance_reconstructs_the_previous_value():
    sigma = 0.5
    result = _gaussian_result(sigma)
    assert result.observation_variance == pytest.approx(sigma**2)
    old = result.predictive_variance + result.observation_variance
    assert np.all(old > result.predictive_variance)


def test_predict_matches_the_fit_rows_under_the_new_convention():
    result = _gaussian_result()
    prediction = result.predict(_frame())
    assert np.allclose(prediction.predictive_variance, result.predictive_variance)


def test_laplace_predictive_variance_is_unchanged():
    frame = _frame().assign(y=[2.0, 3.0, 5.0, 4.0])
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")
    design = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0],
                       [1.0, 3.0, 1.0, 0.0], [1.0, 4.0, 0.0, 1.0]])
    expected = np.einsum("ij,jk,ik->i", design, result.covariance, design)
    assert np.allclose(result.predictive_variance, expected)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_predictive_variance_convention.py -q`
Expected: the Gaussian and `observation_variance` tests FAIL.

- [ ] **Step 3: Make the change**

- `inference/gaussian.py`: `predictive_variance = np.einsum(...)` — drop `+ variance`. Pass `observation_variance=variance` to `GaussianResult`.
- `inference/result.py`: `GaussianResult` accepts and exposes `observation_variance` (validated finite and non-negative, matching the existing validation style). Document on the base property that `predictive_variance` is the linear-predictor variance for every result type.
- `inference/prediction.py`: in `predict_from`, remove the `+ likelihood.variance` branch so `predictive_variance` is the η variance for every likelihood. Keep `fitted_mean` computed exactly as now — for Gaussian the link is the identity, so nothing else moves.

- [ ] **Step 4: Regenerate the baseline as a reviewable diff**

Part A proved the baseline unchanged; Part B deliberately changes it. Regenerate
`result_surface_baseline.json` and **inspect the diff before committing**:
only `predictive_variance` entries of *Gaussian-likelihood* fits (and any
`observation_variance` addition) may change. If a Poisson/Bernoulli entry, a
`criteria` field, a `mean`, or a `log_marginal_likelihood` moves, that is a bug
— stop and investigate. Record in the report exactly which keys changed.

- [ ] **Step 5: Assert criteria did not move**

Add to the new test file a check that DIC/WAIC/log-CPO for an integrated
Gaussian fit match the values recorded in the pre-change baseline (copy the
numbers into the test as literals so the assertion is independent of the
regenerated file).

- [ ] **Step 6: Verify and commit**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`

```bash
git add src/pylgm/inference/ tests/inference/
git commit -m "fix: report the linear-predictor variance consistently"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Document the convention change prominently**

The README currently states the asymmetry as intended behaviour — find every
place (search `predictive_variance`, `observation`, `excludes`, `adds it`) and
correct it. State plainly:

- `predictive_variance` is the **linear-predictor** variance `Var(η)` for every
  result type, engine, and for `predict()`;
- this **changed** in this version for Gaussian models, whose reported values
  previously included σ²;
- the reconstruction is `predictive_variance + observation_variance`;
- why: it is the only convention definable for Poisson/Bernoulli, and it
  matches R-INLA's `summary.linear.predictor`.

- [ ] **Step 2: Note the unification**

Briefly record that the result types now share a base and that the marginal
types satisfy a documented `LatentMarginals` protocol — a structural note, not
a user-facing API change.

- [ ] **Step 3: Update the roadmap**

Mark result-type unification done. Keep recorded: the bounded-parameter
bounds-block extraction, the in-loop Newton decrement, predictive quantiles,
Spark `new_data`, config-file spatial/AR1 types, irregular AR1 spacing.

- [ ] **Step 4: Verify and commit**

Run any README example touching `predictive_variance` verbatim; then
`PYTHONPATH=src python -m pytest -q`.

```bash
git add README.md docs/
git commit -m "docs: document the linear-predictor variance convention"
```

---

## Self-Review Notes

- **Baseline first is the whole safety design.** Task 1 runs on unmodified code so Parts A and B can be judged against recorded behaviour rather than intent. Task 3's acceptance gate is "the baseline passes unmodified"; Task 4's is "the baseline diff contains only the keys we meant to change".
- **The sibling invariant** is called out in the Global Constraints, restated in Task 3's docstring, and pinned by a test in Task 1 — three layers, because breaking it is silent and would misroute every INLA rebuild.
- **Sequencing matters:** Part A (Tasks 2–3) and Part B (Task 4) are separate commits precisely so a regression is attributable. Do not merge them.
- **Harness credibility:** Task 1 Step 2 requires demonstrating the harness *fails* on an injected perturbation. A golden test that cannot fail is worse than none, because it looks like coverage.
- **Type consistency:** `_marginal_grid_array(value, name)` (Task 2) and `_init_common(...)` (Task 3) are referenced identically wherever they appear; `observation_variance` (Task 4) is named the same in `gaussian.py`, `result.py`, and the tests.
- **Known non-goal:** `result.py` may remain large after unification. Splitting it is permitted but explicitly not required — correctness and equivalence come first, and a file split would obscure the equivalence diff.

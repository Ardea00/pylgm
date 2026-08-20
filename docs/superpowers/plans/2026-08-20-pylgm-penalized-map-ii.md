# pyLGM Penalized MAP-II Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate declared `Hyperparameter`s that carry a `prior` by MAP-II (marginal likelihood + log prior) instead of pure type-II ML, for both engines.

**Architecture:** `optimize_empirical_bayes` gains an optional `penalty` callable subtracted from `-log_marginal_likelihood` in its objective. `LGM.fit` builds that penalty from the declared hyperparameters' priors (native-scale `logpdf`, no Jacobian) and passes it; a diagnostics flag records penalized estimation. Everything else — family building, dispatch, result attachment, Pandas/Spark contracts — is reused unchanged.

**Tech Stack:** Python 3.11+, NumPy, SciPy (L-BFGS-B), Pandas, pytest, Ruff.

## Global Constraints

- No new runtime dependencies.
- Penalty is the native-scale log prior summed over prior'd hyperparameters: `Σ prior.logpdf(θ_i)`; the objective becomes `-log_marginal_likelihood - penalty(θ)`. No Jacobian term.
- Automatic trigger: a `Hyperparameter` with a `prior` → MAP; without a prior → pure ML; a model may mix both.
- `penalty=None` and prior-free / no-`Hyperparameter` models must reproduce the current empirical-Bayes (pure ML) behavior exactly.
- Works for `engine="exact_gaussian"` and `engine="laplace"`; Pandas caller-order and Spark canonical-key contracts preserved.
- Do not modify the legacy schema-v2 `Experiment` path.
- Use `PYTHONPATH=src` for all test/lint commands.
- Spec: `docs/superpowers/specs/2026-08-20-pylgm-penalized-map-ii-design.md`.

---

## File Map

- `src/pylgm/optimization/empirical_bayes.py`: `penalty` parameter + objective term.
- `src/pylgm/model.py`: build the penalty from priors in `_run_empirical_bayes`; `hyperparameter_penalized` diagnostic.
- `examples/map_ii/`: runnable example.
- `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`: docs + roadmap.
- Tests: `tests/optimization/test_empirical_bayes.py`, `tests/test_model.py`, `tests/test_package.py`.

---

### Task 1: Penalty Term in the Optimizer

**Files:**
- Modify: `src/pylgm/optimization/empirical_bayes.py`
- Modify: `tests/optimization/test_empirical_bayes.py`

**Interfaces:**
- Produces: `optimize_empirical_bayes(family, bounds, *, initial=None, allow_large_dense=False, fit=None, penalty: Callable[[Mapping[str, float]], float] | None = None)`. When `penalty` is given, the per-evaluation objective is `-log_marginal_likelihood - penalty(parameters)`; `penalty=None` is behavior-identical to today.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/optimization/test_empirical_bayes.py
import numpy as np
from scipy.sparse import csr_matrix, eye

from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian
from pylgm.optimization.empirical_bayes import OptimizationBounds, optimize_empirical_bayes


def _unit_family():
    block = LatentBlock("g", ("a", "b"), csr_matrix(np.eye(2)), eye(2, format="csr"),
                        np.empty((0, 2), dtype=float))
    return CompiledFamily(
        y=np.array([1.0, -1.0]), observed=np.array([True, True]), offset=np.zeros(2),
        blocks=(ScalableBlock(block, "g_prec", 1.0),), parameter_names=("g_prec",),
        likelihood_factory=lambda r: CompiledGaussian(1.0),
    )


def test_penalty_none_matches_unpenalized():
    family = _unit_family()
    bounds = {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)}
    a = optimize_empirical_bayes(family, bounds)
    b = optimize_empirical_bayes(family, bounds, penalty=None)
    assert a.parameters["g_prec"] == b.parameters["g_prec"]


def test_penalty_shifts_the_optimum():
    family = _unit_family()
    bounds = {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)}
    plain = optimize_empirical_bayes(family, bounds)
    # a penalty that strongly rewards larger g_prec must move the optimum up
    penalized = optimize_empirical_bayes(
        family, bounds, penalty=lambda values: 5.0 * np.log(values["g_prec"]),
    )
    assert penalized.parameters["g_prec"] > plain.parameters["g_prec"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/optimization/test_empirical_bayes.py -q`
Expected: FAIL (`optimize_empirical_bayes` has no `penalty` parameter).

- [ ] **Step 3: Implement the penalty**

Add `Callable` to the imports if not present (it is: `from collections.abc import Callable, Mapping`). Add the parameter to the signature (after `fit`):

```python
    penalty: Callable[[Mapping[str, float]], float] | None = None,
```

In the `objective` closure, where it currently reads:

```python
            raw_objective = -float(result.log_marginal_likelihood)
            if not np.isfinite(raw_objective):
                raise NumericalError("optimization objective must be finite")
```

change it to subtract the penalty before the finiteness check:

```python
            raw_objective = -float(result.log_marginal_likelihood)
            if penalty is not None:
                raw_objective -= float(penalty(parameters))
            if not np.isfinite(raw_objective):
                raise NumericalError("optimization objective must be finite")
```

`parameters` here is the natural-scale mapping already computed by `_parameter_values` in the same closure. A `-inf` log prior makes `raw_objective` `+inf`, which the existing guard turns into an invalid evaluation the optimizer routes away from.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/optimization/test_empirical_bayes.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/optimization/empirical_bayes.py tests/optimization/test_empirical_bayes.py
git commit -m "feat: add optional prior penalty to the empirical-Bayes objective"
```

---

### Task 2: MAP-II Dispatch in `LGM.fit`

**Files:**
- Modify: `src/pylgm/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `optimize_empirical_bayes(..., penalty=...)`, `_model_hyperparameters`.
- Produces: `LGM.fit` estimating prior'd hyperparameters by MAP-II, exposing the estimate on `result.hyperparameters` and setting `result.diagnostics["hyperparameter_penalized"]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter, PCPrecision


def _panel_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"region": r, "t": t, "y": rng.normal()} for t in range(1, 21) for r in "abcd"]
    return pd.DataFrame(rows)


def test_map_ii_sets_penalized_flag_and_estimate_both_engines():
    frame = _panel_frame()
    gauss = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0,
                                                          prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
                panel=("region",), time="t")
    result = gauss.fit(frame, engine="exact_gaussian")
    assert result.diagnostics["hyperparameter_penalized"] is True
    assert result.hyperparameters["p"] > 0

    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    pois = LGM("y", Poisson(),
               Fixed("1") + IID("region", index="region",
                                precision=Hyperparameter("p", initial=1.0,
                                                         prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
               panel=("region",), time="t")
    pres = pois.fit(counts, engine="laplace")
    assert pres.diagnostics["hyperparameter_penalized"] is True
    assert pres.hyperparameters["p"] > 0


def test_ml_when_no_prior_and_penalty_changes_estimate():
    frame = _panel_frame()
    def build(prior):
        return LGM("y", Gaussian(0.5),
                   Fixed("1") + IID("region", index="region",
                                    precision=Hyperparameter("p", initial=1.0, prior=prior)),
                   panel=("region",), time="t")
    ml = build(None).fit(frame, engine="exact_gaussian")
    assert ml.diagnostics["hyperparameter_penalized"] is False
    map_ii = build(PCPrecision(upper_sd=0.2, alpha=0.01)).fit(frame, engine="exact_gaussian")
    # the prior must actually move the estimate
    assert not np.isclose(ml.hyperparameters["p"], map_ii.hyperparameters["p"])
```

Note on the last assertion: `PCPrecision` shrinks the effect standard deviation toward zero, so a strong prior (`upper_sd=0.2`) pulls the precision estimate away from the ML value. If the two happen to coincide, adjust the prior strength (smaller `upper_sd`) until they demonstrably differ — do not weaken the assertion to a tautology.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: FAIL (`hyperparameter_penalized` not set; prior not applied).

- [ ] **Step 3: Build the penalty in `_run_empirical_bayes`**

In `src/pylgm/model.py`, `_run_empirical_bayes` currently builds `bounds`/`initial`, calls `optimize_empirical_bayes(family, bounds, initial=initial, fit=fit)`, and sets two diagnostics. Change it to build a penalty from the prior'd hyperparameters and pass it:

```python
        priored = [hp for _, hp in _model_hyperparameters(self) if hp.prior is not None]
        penalty = None
        if priored:
            def penalty(values, priored=priored):
                return sum(hp.prior.logpdf(values[hp.name]) for hp in priored)
        eb = optimize_empirical_bayes(family, bounds, initial=initial, fit=fit, penalty=penalty)
        diagnostics = dict(eb.fit.diagnostics)
        diagnostics["empirical_bayes_converged"] = eb.diagnostics.converged
        diagnostics["empirical_bayes_evaluations"] = eb.diagnostics.evaluations
        diagnostics["hyperparameter_penalized"] = penalty is not None
        return _attach_estimates(eb.fit, dict(eb.parameters), diagnostics)
```

Keep the existing `bounds`/`initial` construction above it unchanged.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/model.py tests/test_model.py
git commit -m "feat: estimate prior'd hyperparameters by MAP-II"
```

---

### Task 3: Example, Docs, and Roadmap

**Files:**
- Create: `examples/map_ii/data.csv`, `examples/map_ii/run.py`, `examples/map_ii/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`
- Modify: `tests/test_package.py`

**Interfaces:**
- Produces: a runnable Python example declaring a `PCPrecision`-prior'd `Hyperparameter` and printing the MAP estimate and `hyperparameter_penalized` flag.

- [ ] **Step 1: Write the example smoke test**

```python
# append to tests/test_package.py
def test_map_ii_example_reports_penalized_estimate():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/map_ii/run.py")],
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

`examples/map_ii/data.csv` — a small region panel over time (`region,t,y`) with enough rows for a stable Gaussian fit. `run.py` — read the CSV, build an `LGM` with an `IID` precision declared as `Hyperparameter("region_precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))`, `model.fit(frame, engine="exact_gaussian")`, and print `result.hyperparameters` and `result.diagnostics["hyperparameter_penalized"]`. `README.md` — how to run with `PYTHONPATH=src`, and that this is MAP-II (marginal likelihood + PC-prior penalty on the native precision scale).

- [ ] **Step 4: Update docs**

README: extend the "Empirical Bayes" subsection — when a declared `Hyperparameter` also carries a `prior`, `LGM.fit` estimates it by MAP-II (marginal likelihood + native-scale log prior) automatically, for both engines; `result.diagnostics["hyperparameter_penalized"]` records it; a prior-free `Hyperparameter` stays pure type-II ML. Note MAP is on the prior's native scale (no Jacobian), and full hyperparameter *integration* remains the later INLA slice.

Arch spec (`2026-08-08-...`): mark penalized MAP-II as shipped (it was the roadmap item added in the empirical-Bayes slice), and keep INLA-style hyperparameter integration, spatial effects, and HMC-Laplace as the remaining deferred slices.

- [ ] **Step 5: Run tests and lint**

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

Run: `PYTHONPATH=src ruff check src tests`
Expected: PASS.

Also run the example directly and confirm exit 0 with sensible output:

Run: `PYTHONPATH=src python examples/map_ii/run.py`
Expected: prints `hyperparameters` and `hyperparameter_penalized: True`.

- [ ] **Step 6: Commit**

```bash
git add examples/map_ii README.md docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md tests/test_package.py
git commit -m "docs: publish penalized MAP-II"
```

---

## Self-Review Checklist

- Task 1 (optimizer penalty) → Task 2 (LGM.fit builds it) → Task 3 (docs/example). Linear.
- Every spec section maps to a task: penalty seam (T1), automatic native-scale MAP dispatch + diagnostic (T2), docs/example/roadmap (T3).
- Correctness anchors: `penalty=None` parity (T1), penalty shifts the optimum in the expected direction (T1), MAP flag + estimate for both engines (T2), prior changes the estimate vs ML (T2).
- Native-scale log prior, no Jacobian; pure-ML and no-Hyperparameter behavior preserved; no new dependencies; legacy path untouched.
- Deferred (unimplemented): log-scale MAP with Jacobians, hyperparameter integration/marginals (INLA), new priors, YAML priors.

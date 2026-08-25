# pyLGM INLA Integration Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add INLA-style grid integration over hyperparameters to `LGM.fit(..., hyperparameters="integrate")`, producing integrated latent marginals, populated hyperparameter marginals, and an integrated marginal likelihood, for both engines.

**Architecture:** A driver `integrate_inla` reuses `optimize_empirical_bayes` for the posterior mode + conditional fit, finite-differences the log-posterior Hessian, builds a whitened grid, evaluates the conditional fit at each grid point, and combines them into an `INLAResult` (grid-weighted Gaussian mixture). `LGM.fit` gains a `hyperparameters` mode selector that dispatches optimize (current) vs integrate (new). Pure numeric helpers are unit-tested independently.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.special.logsumexp`, `numpy.linalg`), Pandas, pytest, Ruff.

## Global Constraints

- No new runtime dependencies.
- INLA is an integration wrapper: reuse the family (`materialize`), the conditional engines (`fit_gaussian`/`fit_laplace`), `optimize_empirical_bayes`, priors, and the result helpers. Do not add a new conditional engine.
- Objective/posterior semantics match the MAP-II slice: `s(u) = log π(y|θ(u)) + log π(θ(u))`, evaluated as the conditional fit's `log_marginal_likelihood` plus the declared priors' `logpdf`; integration in `u = log θ` space carries the `Σ_j u_j` log-Jacobian in the weights.
- `hyperparameters="optimize"` (default) must be byte-identical to today; `"integrate"` requires at least one declared `Hyperparameter`.
- Works for `engine="exact_gaussian"` and `engine="laplace"`; preserve Pandas caller-order and Spark canonical-key contracts for integrated predictive arrays.
- A `max_grid_points` guard rejects too many hyperparameters before doing work.
- No result-type unification refactor in this slice; `INLAResult` reuses the existing module-level helpers.
- Use `PYTHONPATH=src` for all test/lint commands.
- Spec: `docs/superpowers/specs/2026-08-20-pylgm-inla-integration-design.md`.

---

## File Map

- `src/pylgm/inference/result.py`: `INLAResult`.
- `src/pylgm/inference/__init__.py`: export `INLAResult`.
- `src/pylgm/optimization/inla.py`: `_finite_difference_hessian`, `_build_grid`, `_MixtureAccumulator`, `integrate_inla`.
- `src/pylgm/model.py`: `hyperparameters` mode on `fit`; shared `_family_optimization_inputs`; `_run_inla`; `_rebuild_result` handles `INLAResult`.
- `examples/inla/`: runnable example.
- `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`: docs + roadmap.
- Tests: `tests/inference/test_result.py`, `tests/optimization/test_inla.py`, `tests/test_model.py`, `tests/test_package.py`.

---

### Task 1: INLAResult

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Modify: `src/pylgm/inference/__init__.py`
- Modify: `tests/inference/test_result.py`

**Interfaces:**
- Consumes: `latent_marginals_from`, `linear_combinations_from`, `GaussianMarginals`, `_readonly_array`, `_readonly_diagnostics`, `_validate_prediction_keys`, `_readonly_hyperparameters` (all already in `result.py`).
- Produces: `INLAResult(labels, mean, covariance, log_marginal_likelihood, predictive_mean, predictive_variance, hyperparameter_marginals, *, fitted_mean=None, link_name=None, block_slices=None, diagnostics=None, prediction_keys=None, hyperparameters=None)` with properties `mean`, `covariance`, `predictive_mean`, `predictive_variance`, `fitted_mean` (or `None`), `prediction_keys`, `hyperparameters`, `engine == "inla"`, `converged == True`, `latent_marginals`/`linear_combinations` (shared helpers), and `hyperparameter_marginals()` returning the stored `Mapping[str, GaussianMarginals]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/inference/test_result.py
from pylgm.inference.result import INLAResult, GaussianMarginals


def _inla(**overrides):
    kwargs = dict(
        labels=("g:a",),
        mean=np.array([0.5]),
        covariance=np.array([[0.25]]),
        log_marginal_likelihood=-2.0,
        predictive_mean=np.array([0.5, 0.5]),
        predictive_variance=np.array([0.3, 0.3]),
        hyperparameter_marginals={"p": GaussianMarginals(np.array([1.5]), np.array([0.4]))},
        block_slices={"g": slice(0, 1)},
        diagnostics={"inla_grid_points": 7},
    )
    kwargs.update(overrides)
    return INLAResult(**kwargs)


def test_inla_result_surface():
    result = _inla()
    assert result.engine == "inla"
    assert result.converged is True
    assert result.fitted_mean is None
    hm = result.hyperparameter_marginals()
    np.testing.assert_allclose(hm["p"].mean, [1.5])
    np.testing.assert_allclose(hm["p"].variance, [0.4])
    marg = result.latent_marginals("g")
    np.testing.assert_allclose(marg.mean, [0.5])
    np.testing.assert_allclose(marg.variance, [0.25])
    lc = result.linear_combinations(np.array([[2.0]]))
    np.testing.assert_allclose(lc.mean, [1.0])


def test_inla_result_carries_fitted_mean_and_is_immutable():
    result = _inla(fitted_mean=np.array([1.6, 1.6]), link_name="log")
    assert result.link_name == "log"
    exposed = result.fitted_mean
    with pytest.raises(ValueError):
        exposed[0] = 9.0
    np.testing.assert_allclose(result.fitted_mean, [1.6, 1.6])
    assert result.hyperparameters is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`
Expected: FAIL (`INLAResult` absent).

- [ ] **Step 3: Implement `INLAResult`**

Add to `src/pylgm/inference/result.py`, mirroring `LaplaceResult`'s defensive-copy pattern. Validate `hyperparameter_marginals` is a `Mapping` whose values are `GaussianMarginals`, store as `MappingProxyType`. `fitted_mean` optional (store read-only or `None`), `link_name` optional. Properties reuse the module helpers exactly as `LaplaceResult` does; `hyperparameter_marginals()` returns the stored proxy; `engine` returns `"inla"`; `converged` returns `True`. Delegate `latent_marginals`/`linear_combinations` to `latent_marginals_from`/`linear_combinations_from` on `self._mean`/`self._covariance`.

Export `INLAResult` from `src/pylgm/inference/__init__.py` and its `__all__`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/inference/test_result.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference/result.py src/pylgm/inference/__init__.py tests/inference/test_result.py
git commit -m "feat: add INLAResult with integrated marginals surface"
```

---

### Task 2: Grid and Hessian Numeric Helpers

**Files:**
- Create: `src/pylgm/optimization/inla.py`
- Create: `tests/optimization/test_inla.py`

**Interfaces:**
- Produces: `_finite_difference_hessian(func: Callable[[np.ndarray], float], center: np.ndarray, step: float = 1e-3) -> np.ndarray` (symmetric `d×d` central-difference Hessian).
- Produces: `_build_grid(center: np.ndarray, hessian: np.ndarray, *, grid_step: float = 1.0, radius: int = 3, max_grid_points: int = 4096) -> np.ndarray` returning candidate `u`-points of shape `(K, d)`; whitens `-hessian` (ridge-regularized to PD), lays an integer lattice `z ∈ {-radius..radius}^d`, and returns `center + grid_step * (V Λ^{-1/2}) z` for each `z`. Raises `OptimizationError` when `(2*radius+1)**d > max_grid_points`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_inla.py
import numpy as np
import pytest

from pylgm.exceptions import OptimizationError
from pylgm.optimization.inla import _build_grid, _finite_difference_hessian


def test_finite_difference_hessian_recovers_quadratic():
    A = np.array([[2.0, 0.3], [0.3, 1.5]])
    center = np.array([0.4, -0.2])
    # s(u) = -0.5 (u-c)^T A (u-c); Hessian is -A everywhere
    def s(u):
        d = u - center
        return -0.5 * d @ A @ d
    H = _finite_difference_hessian(s, center, step=1e-3)
    np.testing.assert_allclose(H, -A, atol=1e-4)


def test_build_grid_centers_and_bounds_count():
    center = np.array([0.0])
    hessian = np.array([[-4.0]])  # -H = 4 -> sigma = 0.5 in u-space
    grid = _build_grid(center, hessian, grid_step=1.0, radius=3)
    assert grid.shape == (7, 1)
    # the mode (z=0) is present
    assert np.any(np.all(np.isclose(grid, center), axis=1))
    # step of one lattice unit is 1/sqrt(4) = 0.5 in u-space
    offsets = np.sort(np.unique(np.round(grid[:, 0] - center[0], 6)))
    np.testing.assert_allclose(offsets, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])


def test_build_grid_guards_dimensionality():
    center = np.zeros(6)
    hessian = -np.eye(6)
    with pytest.raises(OptimizationError, match="grid"):
        _build_grid(center, hessian, radius=3, max_grid_points=4096)  # 7**6 >> 4096
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/optimization/test_inla.py -q`
Expected: FAIL (`pylgm.optimization.inla` absent).

- [ ] **Step 3: Implement the helpers**

```python
# src/pylgm/optimization/inla.py
from collections.abc import Callable
from itertools import product

import numpy as np

from pylgm.exceptions import OptimizationError


def _finite_difference_hessian(
    func: Callable[[np.ndarray], float], center: np.ndarray, step: float = 1e-3
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    d = center.size
    hessian = np.zeros((d, d))
    f0 = func(center)
    for i in range(d):
        ei = np.zeros(d); ei[i] = step
        f_plus = func(center + ei); f_minus = func(center - ei)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step * step)
        for j in range(i + 1, d):
            ej = np.zeros(d); ej[j] = step
            f_pp = func(center + ei + ej); f_pm = func(center + ei - ej)
            f_mp = func(center - ei + ej); f_mm = func(center - ei - ej)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _whitening_directions(hessian: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    negative = -np.asarray(hessian, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(negative)
    clamped = np.clip(eigenvalues, ridge, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(clamped))


def _build_grid(
    center: np.ndarray, hessian: np.ndarray, *,
    grid_step: float = 1.0, radius: int = 3, max_grid_points: int = 4096,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    d = center.size
    candidate_count = (2 * radius + 1) ** d
    if candidate_count > max_grid_points:
        raise OptimizationError(
            f"INLA grid would need {candidate_count} points for {d} hyperparameters; "
            f"exceeds max_grid_points={max_grid_points}"
        )
    directions = _whitening_directions(hessian)
    offsets = range(-radius, radius + 1)
    points = [
        center + grid_step * directions @ np.asarray(z, dtype=float)
        for z in product(offsets, repeat=d)
    ]
    return np.asarray(points)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/optimization/test_inla.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/optimization/inla.py tests/optimization/test_inla.py
git commit -m "feat: add INLA grid and finite-difference Hessian helpers"
```

---

### Task 3: `integrate_inla` Driver

**Files:**
- Modify: `src/pylgm/optimization/inla.py`
- Modify: `tests/optimization/test_inla.py`

**Interfaces:**
- Consumes: `optimize_empirical_bayes`, `OptimizationBounds`, a family with `parameter_names` + `materialize`, a conditional `fit` callable, an optional `penalty`, `INLAResult`, `GaussianMarginals`, and the Task 2 helpers.
- Produces: `integrate_inla(family, bounds, *, initial=None, fit=None, penalty=None, allow_large_dense=False, grid_step=1.0, radius=3, log_density_drop=2.5, max_grid_points=4096) -> INLAResult`.

- [ ] **Step 1: Write the failing tests (1-D quadrature anchor + widening)**

```python
# append to tests/optimization/test_inla.py
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp

from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian
from pylgm.inference import fit_gaussian
from pylgm.optimization.empirical_bayes import OptimizationBounds
from pylgm.optimization.inla import integrate_inla


def _one_hyperparameter_family():
    # 4 observed rows, single IID block with an optimisable precision
    design = csr_matrix(np.eye(4))
    block = LatentBlock("g", tuple("abcd"), design, eye(4, format="csr"),
                        np.empty((0, 4), dtype=float))
    return CompiledFamily(
        y=np.array([1.0, -0.5, 0.8, 0.2]), observed=np.array([True] * 4),
        offset=np.zeros(4), blocks=(ScalableBlock(block, "p", 1.0),),
        parameter_names=("p",), likelihood_factory=lambda r: CompiledGaussian(1.0),
    )


def test_integrate_inla_matches_fine_1d_quadrature():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    result = integrate_inla(family, bounds, fit=fit_gaussian, grid_step=0.75, radius=6)

    # independent fine 1-D quadrature over u = log p
    us = np.linspace(np.log(1e-2), np.log(1e2), 4001)
    logw, means, variances = [], [], []
    for u in us:
        fit = fit_gaussian(family.materialize({"p": float(np.exp(u))}))
        logw.append(float(fit.log_marginal_likelihood) + u)  # + Jacobian
        means.append(fit.mean[0]); variances.append(fit.covariance[0, 0])
    logw = np.asarray(logw); w = np.exp(logw - logsumexp(logw))
    ref_mean = float(np.sum(w * np.asarray(means)))
    ref_var = float(np.sum(w * (np.asarray(variances) + np.asarray(means) ** 2)) - ref_mean ** 2)

    np.testing.assert_allclose(result.mean[0], ref_mean, atol=5e-3)
    np.testing.assert_allclose(result.covariance[0, 0], ref_var, rtol=5e-2)
    assert result.hyperparameter_marginals()["p"].mean[0] > 0
    assert result.diagnostics["inla_grid_points"] >= 3


def test_integrate_inla_widens_variance_vs_plug_in():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    inla = integrate_inla(family, bounds, fit=fit_gaussian, radius=6)
    from pylgm.optimization.empirical_bayes import optimize_empirical_bayes
    eb = optimize_empirical_bayes(family, bounds, fit=fit_gaussian)
    # integrated latent variance >= plug-in latent variance (adds between-theta spread)
    assert inla.covariance[0, 0] >= eb.fit.covariance[0, 0] - 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/optimization/test_inla.py -q`
Expected: FAIL (`integrate_inla` absent).

- [ ] **Step 3: Implement the driver**

Append to `src/pylgm/optimization/inla.py`:

```python
import numpy as np
from scipy.special import logsumexp

from pylgm.exceptions import NumericalError
from pylgm.inference import GaussianResult, LaplaceResult, fit_gaussian
from pylgm.inference.result import GaussianMarginals, INLAResult
from pylgm.optimization.empirical_bayes import optimize_empirical_bayes


def integrate_inla(
    family, bounds, *, initial=None, fit=None, penalty=None, allow_large_dense=False,
    grid_step=1.0, radius=3, log_density_drop=2.5, max_grid_points=4096,
) -> INLAResult:
    names = tuple(family.parameter_names)
    conditional_fit = fit if fit is not None else fit_gaussian
    lower = np.array([bounds[name].lower for name in names])
    upper = np.array([bounds[name].upper for name in names])

    def evaluate(u):
        theta = {
            name: float(np.clip(np.exp(value), lower[i], upper[i]))
            for i, (name, value) in enumerate(zip(names, u, strict=True))
        }
        compiled = family.materialize(theta)
        conditional = (
            conditional_fit(compiled, allow_large_dense=True)
            if allow_large_dense else conditional_fit(compiled)
        )
        s_value = float(conditional.log_marginal_likelihood)
        if penalty is not None:
            s_value += float(penalty(theta))
        return s_value, conditional, theta

    eb = optimize_empirical_bayes(
        family, bounds, initial=initial, fit=conditional_fit, penalty=penalty,
        allow_large_dense=allow_large_dense,
    )
    u_star = np.array([np.log(eb.parameters[name]) for name in names])
    hessian = _finite_difference_hessian(lambda u: evaluate(u)[0], u_star)
    grid = _build_grid(
        u_star, hessian, grid_step=grid_step, radius=radius, max_grid_points=max_grid_points,
    )

    points = [evaluate(u) for u in grid]
    s_values = np.array([s for s, _, _ in points])
    s_max = float(s_values.max())
    kept = [(u, s, cond, theta) for u, (s, cond, theta) in zip(grid, points, strict=True)
            if s >= s_max - log_density_drop]
    if not kept:
        raise NumericalError("INLA grid retained no points above the density threshold")

    log_weights = np.array([s + float(np.sum(u)) for (u, s, _, _) in kept])
    weights = np.exp(log_weights - logsumexp(log_weights))

    reference = kept[0][2]
    is_laplace = isinstance(reference, LaplaceResult)
    link_name = reference.link_name if is_laplace else None

    mean_acc = np.zeros_like(reference.mean)
    cov_acc = np.zeros_like(reference.covariance)
    pm_acc = np.zeros_like(reference.predictive_mean)
    pv_acc = np.zeros_like(reference.predictive_variance)
    fitted_acc = np.zeros_like(reference.fitted_mean) if is_laplace else None
    theta_mean = {name: 0.0 for name in names}
    theta_sq = {name: 0.0 for name in names}

    for (_, _, cond, theta), w in zip(kept, weights, strict=True):
        m = cond.mean
        pm = cond.predictive_mean
        mean_acc += w * m
        cov_acc += w * (cond.covariance + np.outer(m, m))
        pm_acc += w * pm
        pv_acc += w * (cond.predictive_variance + pm * pm)
        if is_laplace:
            fitted_acc += w * cond.fitted_mean
        for name in names:
            theta_mean[name] += w * theta[name]
            theta_sq[name] += w * theta[name] * theta[name]

    mean = mean_acc
    covariance = cov_acc - np.outer(mean, mean)
    predictive_mean = pm_acc
    predictive_variance = pv_acc - pm_acc * pm_acc

    hyper_marginals = {
        name: GaussianMarginals(
            np.array([theta_mean[name]]),
            np.array([max(theta_sq[name] - theta_mean[name] ** 2, 0.0)]),
        )
        for name in names
    }

    eigenvalues = np.clip(np.linalg.eigvalsh(-hessian), 1e-6, None)
    integrated_lml = float(
        logsumexp(log_weights) + len(names) * np.log(grid_step)
        - 0.5 * float(np.sum(np.log(eigenvalues)))
    )

    diagnostics = {
        "inla_grid_points": int(len(kept)),
        "inla_effective_weight": float(1.0 / np.sum(weights**2)),
        "inla_conditional_engine": "laplace" if is_laplace else "exact_gaussian",
    }
    for name in names:
        diagnostics[f"inla_mode_{name}"] = float(eb.parameters[name])

    for label, value in (("mean", mean), ("covariance", covariance),
                         ("predictive mean", predictive_mean),
                         ("predictive variance", predictive_variance)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA produced non-finite {label}")

    return INLAResult(
        labels=reference.labels,
        mean=mean, covariance=covariance, log_marginal_likelihood=integrated_lml,
        predictive_mean=predictive_mean, predictive_variance=predictive_variance,
        hyperparameter_marginals=hyper_marginals,
        fitted_mean=fitted_acc, link_name=link_name,
        block_slices=dict(reference.block_slices), diagnostics=diagnostics,
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/optimization/test_inla.py -q`
Expected: PASS (the 1-D anchor matches within tolerance; variance widens).

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/optimization/inla.py tests/optimization/test_inla.py
git commit -m "feat: add INLA grid-integration driver"
```

---

### Task 4: `LGM.fit(hyperparameters="integrate")` Dispatch

**Files:**
- Modify: `src/pylgm/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Consumes: `integrate_inla`, `INLAResult`, `_model_hyperparameters`, `OptimizationBounds`.
- Produces: `LGM.fit(frame, engine="exact_gaussian", *, max_driver_rows=100_000, hyperparameters="optimize")`; `"integrate"` runs INLA (requires a declared `Hyperparameter`), attaches the result through the existing realignment/keys machinery.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter
from pylgm.inference.result import INLAResult


def _region_panel(seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"region": r, "t": t, "y": rng.normal()}
                         for t in range(1, 21) for r in "abcd"])


def test_fit_integrate_returns_inla_result_both_engines():
    frame = _region_panel()
    gauss = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    result = gauss.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    assert isinstance(result, INLAResult)
    assert result.engine == "inla"
    assert "p" in result.hyperparameter_marginals()
    assert result.diagnostics["inla_grid_points"] >= 3

    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    pois = LGM("y", Poisson(),
               Fixed("1") + IID("region", index="region",
                                precision=Hyperparameter("p", initial=1.0)),
               panel=("region",), time="t")
    pres = pois.fit(counts, engine="laplace", hyperparameters="integrate")
    assert isinstance(pres, INLAResult)
    assert pres.fitted_mean is not None


def test_integrate_requires_a_hyperparameter():
    frame = _region_panel()
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="t")
    with pytest.raises(ValueError, match="integrate"):
        model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")


def test_fit_optimize_default_unchanged():
    frame = _region_panel()
    model = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    default = model.fit(frame, engine="exact_gaussian")
    explicit = model.fit(frame, engine="exact_gaussian", hyperparameters="optimize")
    np.testing.assert_allclose(default.hyperparameters["p"], explicit.hyperparameters["p"])


def test_invalid_hyperparameters_mode_rejected():
    frame = _region_panel()
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="t")
    with pytest.raises(ValueError, match="hyperparameters"):
        model.fit(frame, engine="exact_gaussian", hyperparameters="nonsense")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: FAIL (`fit` has no `hyperparameters` parameter).

- [ ] **Step 3: Implement the dispatch**

In `src/pylgm/model.py`:
- Import `from pylgm.optimization.inla import integrate_inla` and `from pylgm.inference import INLAResult` (add to the existing `pylgm.inference` import).
- Extract the shared bounds/penalty construction from `_run_empirical_bayes` into a helper so INLA reuses it:

```python
    def _family_optimization_inputs(self):
        from pylgm.compiler import _model_hyperparameters

        hyperparameters = list(_model_hyperparameters(self))
        bounds, initial = {}, {}
        for _, hp in hyperparameters:
            bounds[hp.name] = OptimizationBounds(hp.initial, hp.lower, hp.upper)
            initial[hp.name] = hp.initial
        priored = [hp for _, hp in hyperparameters if hp.prior is not None]
        penalty = None
        if priored:
            def penalty(values, priored=priored):
                return sum(hp.prior.logpdf(values[hp.name]) for hp in priored)
        return bounds, initial, penalty
```

Refactor `_run_empirical_bayes` to call `_family_optimization_inputs` (behavior unchanged), and add `_run_inla`:

```python
    def _run_inla(self, family, engine: str) -> INLAResult:
        fit = self._engine(engine)
        bounds, initial, penalty = self._family_optimization_inputs()
        return integrate_inla(family, bounds, initial=initial, fit=fit, penalty=penalty)
```

- Add `hyperparameters: str = "optimize"` (keyword-only) to `fit`, validate it is `"optimize"` or `"integrate"` (`ValueError` otherwise), and thread it to `_fit_pandas`/`_fit_spark`.
- In `_fit_pandas`/`_fit_spark`, after building `family`: if `hyperparameters == "integrate"`: require `family is not None` (else `ValueError("hyperparameters='integrate' requires a declared Hyperparameter")`) and `result = self._run_inla(family, engine)`; otherwise keep the existing optimize/plug-in branch. Then realign (Pandas) / set `prediction_keys` (Spark) as today.
- Extend `_rebuild_result` to handle `INLAResult`: reorder `predictive_mean`/`predictive_variance` (and `fitted_mean` when present), carry `hyperparameter_marginals`, `link_name`, `hyperparameters`, `diagnostics`, `prediction_keys` through. Model it on the existing `LaplaceResult` branch, adding `hyperparameter_marginals=result.hyperparameter_marginals()`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/model.py tests/test_model.py
git commit -m "feat: integrate hyperparameters via LGM.fit(hyperparameters='integrate')"
```

---

### Task 5: Example, Docs, and Roadmap

**Files:**
- Create: `examples/inla/data.csv`, `examples/inla/run.py`, `examples/inla/README.md`
- Modify: `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`
- Modify: `tests/test_package.py`

**Interfaces:**
- Produces: a runnable example fitting a `Hyperparameter`-precision model with `hyperparameters="integrate"` and printing the integrated hyperparameter marginal and a latent marginal.

- [ ] **Step 1: Write the example smoke test**

```python
# append to tests/test_package.py
def test_inla_example_reports_integrated_marginals():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/inla/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hyperparameter_marginals" in completed.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_package.py -q`
Expected: FAIL (example missing).

- [ ] **Step 3: Create the example**

`examples/inla/data.csv` — a region panel over time (`region,t,y`). `run.py` — build an `LGM` with an `IID` precision `Hyperparameter("region_precision", initial=1.0)`, fit with `hyperparameters="integrate"`, and print `result.hyperparameter_marginals()` (mean/sd of `region_precision`), a `latent_marginals("region")` summary, and `result.diagnostics["inla_grid_points"]`. `README.md` — how to run with `PYTHONPATH=src`, and that integration accounts for hyperparameter uncertainty (marginals wider than the plug-in fit).

- [ ] **Step 4: Update docs**

README: add an "INLA integration" subsection — `hyperparameters="integrate"` integrates over declared hyperparameters (grid quadrature) instead of plugging in the optimum, giving integrated latent marginals, populated `hyperparameter_marginals()`, and an integrated marginal likelihood, for both engines; note it's the Gaussian latent strategy and is practical for a handful of hyperparameters (grid guard). State that DIC/WAIC/CPO/PIT and richer latent strategies are later INLA sub-slices.

Arch spec: mark the INLA integration core (3a) shipped; keep model-assessment criteria (3b: DIC/WAIC/CPO/PIT), richer latent strategies (3c), spatial effects, and HMC-Laplace as remaining deferred slices.

- [ ] **Step 5: Run tests, lint, and the example**

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

Run: `PYTHONPATH=src ruff check src tests`
Expected: PASS.

Run: `PYTHONPATH=src python examples/inla/run.py`
Expected: exit 0, prints `hyperparameter_marginals` and a grid-point count.

- [ ] **Step 6: Commit**

```bash
git add examples/inla README.md docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md tests/test_package.py
git commit -m "docs: publish INLA integration core"
```

---

## Self-Review Checklist

- Task 1 (INLAResult) → Task 2 (numeric helpers) → Task 3 (driver) → Task 4 (dispatch) → Task 5 (docs). Linear.
- Every spec section maps to a task: result surface (T1), grid/Hessian (T2), integration + integrated LML + marginals (T3), `hyperparameters="integrate"` dispatch + both engines + Pandas/Spark + errors (T4), docs/example/roadmap (T5).
- Correctness anchors: Hessian-of-quadratic and grid geometry/guard (T2); integrated quantities vs an independent fine 1-D quadrature and variance-widening (T3); end-to-end both engines, optimize-unchanged, and error paths (T4).
- Reuses the family, conditional engines, `optimize_empirical_bayes`, priors, and result helpers; no new conditional engine; no result-unification refactor; no new dependencies.
- Deferred (unimplemented, unblocked): DIC/WAIC/CPO/PIT (3b), simplified/full-Laplace latent strategies (3c), CCD, result unification.

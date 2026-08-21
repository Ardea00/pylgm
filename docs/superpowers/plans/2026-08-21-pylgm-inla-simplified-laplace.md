# pyLGM INLA Simplified-Laplace Latent Marginals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the simplified Laplace approximation (SLA) for latent marginals — location- and skewness-corrected skew-normal marginals — to the integrated fit, faithful to Rue–Martino–Chopin (2009) §3.2.3 + Appendix B.

**Architecture:** A `third_derivative` likelihood method feeds a `_simplified_laplace_marginals` helper that computes per-coefficient RMC corrections (γ¹, γ³) per grid point and fits a skew-normal (Appendix B). The integrated marginal is a grid-weighted mixture of skew-normals, exposed via a new `SkewNormalMarginals` type when `latent_strategy="simplified_laplace"`. The Gaussian strategy remains the default and unchanged.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`special.owens_t`, `stats.norm`), pytest, Ruff.

## Global Constraints

- No new runtime dependencies (SciPy already used; `scipy.special.owens_t` for the skew-normal CDF).
- Equations are exactly as transcribed in the spec (`docs/superpowers/specs/2026-08-21-pylgm-inla-simplified-laplace-design.md`), cited by RMC paper numbers. Do not improvise the math.
- `latent_strategy="gaussian"` (default) leaves integrated fits byte-identical; `latent_marginals` still returns `GaussianMarginals`.
- SLA corrects only the latent-coefficient marginals returned by `latent_marginals`; the integrated `mean`/`covariance`, predictions, and criteria are unchanged.
- Gaussian likelihood ⇒ third derivative 0 ⇒ SLA reduces EXACTLY to the Gaussian marginal (correctness anchor).
- No R-INLA available: validate via (a) Gaussian exact reduction and (b) a brute-force nested-quadrature true-marginal oracle for a tiny non-Gaussian model.
- Use `PYTHONPATH=src` for all test/lint commands.

## File Map

- `src/pylgm/likelihoods.py`: `third_derivative`.
- `src/pylgm/inference/result.py`: `SkewNormalMarginals`; `INLAResult` stores SLA marginals + `latent_marginals` returns them under SLA.
- `src/pylgm/inference/__init__.py`: export `SkewNormalMarginals`.
- `src/pylgm/optimization/inla.py`: `_fit_skew_normal`, `_solve_omega`, `_simplified_laplace_marginals`; wire into `integrate_inla`.
- `src/pylgm/model.py`: `latent_strategy` through `fit`/`_run_inla`; `_rebuild_result` carries SLA marginals.
- `examples/inla_sla/`, `README.md`, arch spec, tests.

---

### Task 1: Likelihood Third Derivative

**Files:** Modify `src/pylgm/likelihoods.py`, `tests/test_modeling_vocabulary.py`

**Interfaces:** `third_derivative(eta, y) -> np.ndarray` on each compiled likelihood: `∂³/∂η³ log p(y|η)`. Gaussian `0`; Poisson `−exp(η)`; Bernoulli `−p(1−p)(1−2p)` with `p=sigmoid(η)`. (`y` is accepted for protocol uniformity but the third derivative does not depend on it.)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_modeling_vocabulary.py
import numpy as np
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson, CompiledBernoulli


def test_third_derivative_values_and_finite_difference():
    eta = np.array([0.2, -0.6, 1.1]); y = np.zeros(3)
    np.testing.assert_allclose(CompiledGaussian(1.7).third_derivative(eta, y), np.zeros(3))
    np.testing.assert_allclose(CompiledPoisson().third_derivative(eta, y), -np.exp(eta))
    p = 1.0 / (1.0 + np.exp(-eta))
    np.testing.assert_allclose(CompiledBernoulli().third_derivative(eta, y),
                               -p * (1 - p) * (1 - 2 * p))
    # cross-check: d3 log p = -d/deta working_weights, via central finite difference
    for like in (CompiledPoisson(), CompiledBernoulli()):
        h = 1e-4
        fd = -(like.working_weights(eta + h, y) - like.working_weights(eta - h, y)) / (2 * h)
        np.testing.assert_allclose(like.third_derivative(eta, y), fd, atol=1e-5)
```

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q` (FAIL: no `third_derivative`).

- [ ] **Step 3: Implement**

```python
# CompiledGaussian
    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(eta, dtype=float).shape)
# CompiledPoisson
    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -np.exp(np.asarray(eta, dtype=float))
# CompiledBernoulli
    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(np.asarray(eta, dtype=float))
        return -p * (1.0 - p) * (1.0 - 2.0 * p)
```

- [ ] **Step 4: Run** — focused + `PYTHONPATH=src pytest -q` (all pass).

- [ ] **Step 5: Commit** — `git commit -m "feat: add likelihood third derivative"`

---

### Task 2: Skew-Normal Fit and `SkewNormalMarginals`

**Files:** Modify `src/pylgm/optimization/inla.py` (solver), `src/pylgm/inference/result.py` (type), `src/pylgm/inference/__init__.py`; create `tests/optimization/test_skew_normal.py`; modify `tests/inference/test_result.py`.

**Interfaces:**
- `_fit_skew_normal(gamma1, gamma3) -> (xi, omega, a, clamped)` (vectorized): standardized skew-normal with mean `γ¹`, variance `1`, and leading-order cubic match `(4−π)√2/π^{3/2}·(a/ω)³ = γ³` (RMC eq 32); `clamped` bool array where `|a/ω|` hit the robustness cap.
- `SkewNormalMarginals(weights, location, scale, shape)` — grid-mixture skew-normal per latent component; arrays shape `(n_components, n_grid)` (weights broadcast per grid). Properties `mean`, `variance`, `std`, `skewness` (mixture moments), `quantile(p)` (mixture-CDF root-find), `pdf(x)`, `cdf(x)`. Reduces to Gaussian when `shape==0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_skew_normal.py
import numpy as np
from pylgm.optimization.inla import _fit_skew_normal


def _sn_stats(xi, omega, a):
    d = a / np.sqrt(1 + a * a)
    mean = xi + omega * d * np.sqrt(2 / np.pi)
    var = omega ** 2 * (1 - 2 * d * d / np.pi)
    return mean, var


def test_fit_skew_normal_zero_reduces_to_gaussian():
    xi, omega, a, clamped = _fit_skew_normal(np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(xi, [0.0]); np.testing.assert_allclose(omega, [1.0])
    np.testing.assert_allclose(a, [0.0]); assert not clamped.any()


def test_fit_skew_normal_matches_mean_variance_and_cubic():
    g1 = np.array([0.15]); g3 = np.array([0.4])
    xi, omega, a, _ = _fit_skew_normal(g1, g3)
    mean, var = _sn_stats(xi[0], omega[0], a[0])
    np.testing.assert_allclose(mean, g1[0], atol=1e-6)
    np.testing.assert_allclose(var, 1.0, atol=1e-6)
    C = (4 - np.pi) * np.sqrt(2) / np.pi ** 1.5
    np.testing.assert_allclose(C * (a[0] / omega[0]) ** 3, g3[0], rtol=1e-6)


def test_fit_skew_normal_clamps_extreme_skew():
    _, _, _, clamped = _fit_skew_normal(np.array([0.0]), np.array([1e6]))
    assert clamped.all()
```

```python
# append to tests/inference/test_result.py
import numpy as np
from pylgm.inference.result import SkewNormalMarginals


def test_skew_normal_marginals_gaussian_reduction():
    # single grid point, shape 0 -> Normal(loc, scale)
    m = SkewNormalMarginals(weights=np.array([[1.0]]), location=np.array([[0.5]]),
                            scale=np.array([[2.0]]), shape=np.array([[0.0]]))
    np.testing.assert_allclose(m.mean, [0.5]); np.testing.assert_allclose(m.variance, [4.0])
    from scipy.stats import norm
    np.testing.assert_allclose(m.quantile(0.975), [0.5 + norm.ppf(0.975) * 2.0], atol=1e-6)


def test_skew_normal_marginals_quantile_cdf_roundtrip_and_skew():
    m = SkewNormalMarginals(weights=np.array([[1.0]]), location=np.array([[0.0]]),
                            scale=np.array([[1.0]]), shape=np.array([[4.0]]))
    q = m.quantile(0.3)
    np.testing.assert_allclose(m.cdf(q)[0], 0.3, atol=1e-4)
    assert m.skewness[0] > 0  # positive shape -> right skew
```

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=src pytest tests/optimization/test_skew_normal.py tests/inference/test_result.py -q`.

- [ ] **Step 3: Implement the solver** (`src/pylgm/optimization/inla.py`)

```python
import math
from scipy.special import owens_t

_SN_C = (4.0 - math.pi) * math.sqrt(2.0) / math.pi ** 1.5
_SN_R_MAX = 5.0  # cap on |a/omega| for robustness (|skew| ~ 0.99)


def _solve_omega(r: np.ndarray) -> np.ndarray:
    # solve omega^2 (1 - 2 delta(omega)^2 / pi) = 1, delta = r*omega/sqrt(1+r^2 omega^2)
    # monotone increasing in omega on (0, inf); bisection (vectorized).
    lo = np.zeros_like(r); hi = np.full_like(r, 1e6)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        a = r * mid
        delta = a / np.sqrt(1.0 + a * a)
        val = mid * mid * (1.0 - 2.0 * delta * delta / math.pi) - 1.0
        hi = np.where(val > 0.0, mid, hi)
        lo = np.where(val > 0.0, lo, mid)
    return 0.5 * (lo + hi)


def _fit_skew_normal(gamma1, gamma3):
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma3 = np.asarray(gamma3, dtype=float)
    r = np.sign(gamma3) * (np.abs(gamma3) / _SN_C) ** (1.0 / 3.0)
    clamped = np.abs(r) > _SN_R_MAX
    r = np.clip(r, -_SN_R_MAX, _SN_R_MAX)
    omega = np.where(r == 0.0, 1.0, _solve_omega(r))
    a = r * omega
    delta = a / np.sqrt(1.0 + a * a)
    xi = gamma1 - omega * delta * math.sqrt(2.0 / math.pi)
    return xi, omega, a, clamped
```

- [ ] **Step 4: Implement `SkewNormalMarginals`** (`src/pylgm/inference/result.py`)

A frozen dataclass storing read-only `_weights`, `_location`, `_scale`, `_shape` (each `(n_components, n_grid)`; weights sum to 1 across grid per component). Helpers per component/grid: `delta = shape/sqrt(1+shape²)`, component mean `mu = loc + scale·delta·√(2/π)`, component variance `v = scale²(1−2δ²/π)`, component third central moment `mu3 = scale³·(4−π)/2·(δ√(2/π))³`.
- `mean = Σ_k w·mu`.
- `variance = Σ_k w·(v + mu²) − mean²`.
- `std = √variance`.
- `skewness = [Σ_k w·(mu3 + 3(mu−mean)v + (mu−mean)³)] / std³`.
- `pdf(x)`: `Σ_k w·(2/scale)·φ((x−loc)/scale)·Φ(shape·(x−loc)/scale)`.
- `cdf(x)`: `Σ_k w·[Φ(z) − 2·owens_t(z, shape)]`, `z=(x−loc)/scale` (skew-normal CDF).
- `quantile(p)`: per component, bracketed bisection on the scalar mixture `cdf` (monotone) to solve `cdf(q)=p`; validate `0<p<1`. Return read-only array.
Reduces to `GaussianMarginals` behavior when all `shape==0` (owens_t(z,0)=0 ⇒ SN CDF = Φ). Export from `inference/__init__.py`.

- [ ] **Step 5: Run** — focused + `PYTHONPATH=src pytest -q`.

- [ ] **Step 6: Commit** — `git commit -m "feat: add skew-normal fit and SkewNormalMarginals"`

---

### Task 3: `_simplified_laplace_marginals` Helper

**Files:** Modify `src/pylgm/optimization/inla.py`; create `tests/optimization/test_simplified_laplace.py`.

**Interfaces:** `_simplified_laplace_marginals(design, offset, y, grid) -> (location, scale, shape, weights, clamped_count)`, where `design` is the observed-row design, `grid` is `[(weight, conditional_fit, likelihood)]`. Returns arrays `(p, n_grid)` of the mixture skew-normal params (in `x_i` coordinates) for the `p` latent coefficients, plus a `clamped_count`.

- [ ] **Step 1: Write the failing tests (Gaussian anchor + brute-force oracle)**

```python
# tests/optimization/test_simplified_laplace.py
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp
from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.inference import fit_gaussian, fit_laplace
from pylgm.optimization.inla import _simplified_laplace_marginals


class _Fit:
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, float); self.covariance = np.asarray(cov, float)


def test_sla_gaussian_likelihood_is_exact_gaussian():
    design = csr_matrix(np.eye(2)); offset = np.zeros(2); y = np.array([0.3, -0.4])
    fit = _Fit([0.2, -0.1], np.diag([0.3, 0.5]))
    loc, scale, shape, w, clamped = _simplified_laplace_marginals(
        design, offset, y, [(1.0, fit, CompiledGaussian(1.0))])
    np.testing.assert_allclose(shape, 0.0, atol=1e-12)   # d3=0 -> no skew
    np.testing.assert_allclose(loc[:, 0], fit.mean)
    np.testing.assert_allclose(scale[:, 0], np.sqrt(np.diag(fit.covariance)))


def test_sla_poisson_matches_true_marginal_better_than_gaussian():
    # p=1 intercept, Poisson counts; the true latent posterior is 1-D -> exact quadrature.
    n = 12
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, size=n).astype(float)
    design = csr_matrix(np.ones((n, 1)))
    prior_prec = 1.0
    block = LatentBlock("b", ("i",), design, eye(1, format="csr") * prior_prec,
                        np.empty((0, 1), dtype=float))
    family = CompiledFamily(y=y, observed=np.array([True] * n), offset=np.zeros(n),
                            blocks=(ScalableBlock(block, None, 1.0),), parameter_names=(),
                            likelihood_factory=lambda r: CompiledPoisson())
    compiled = family.materialize({})
    fit = fit_laplace(compiled)  # single conditional fit at fixed theta
    like = CompiledPoisson()
    loc, scale, shape, w, _ = _simplified_laplace_marginals(
        design, np.zeros(n), y, [(1.0, fit, like)])
    # SLA skew-normal for the single latent:
    from pylgm.inference.result import SkewNormalMarginals
    sla = SkewNormalMarginals(weights=w, location=loc, scale=scale, shape=shape)

    # TRUE 1-D posterior of x: pi(x) ∝ exp(-0.5 prior_prec x^2) * prod Poisson(y_j; exp(x))
    xs = np.linspace(-2.0, 4.0, 8001)
    logp = -0.5 * prior_prec * xs ** 2 + np.array(
        [np.sum(y * x - np.exp(x)) for x in xs])
    wq = np.exp(logp - logsumexp(logp)); dx = xs[1] - xs[0]; wq = wq / (wq.sum())
    true_mean = np.sum(wq * xs)
    true_var = np.sum(wq * (xs - true_mean) ** 2)
    true_skew = np.sum(wq * (xs - true_mean) ** 3) / true_var ** 1.5

    gauss_mean = fit.mean[0]; gauss_var = fit.covariance[0, 0]
    # SLA closer in mean & skewness than the Gaussian strategy; correct skew sign
    assert abs(sla.mean[0] - true_mean) <= abs(gauss_mean - true_mean) + 1e-9
    assert np.sign(sla.skewness[0]) == np.sign(true_skew)
    assert abs(sla.skewness[0] - true_skew) < abs(0.0 - true_skew)  # Gaussian skew is 0
```

(If the SLA mean tie is too tight numerically on some seeds, assert the skewness improvements — the load-bearing check — and that `sla.variance[0]` is finite/positive. Do not weaken the skew-sign / skew-closer-than-Gaussian assertions.)

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=src pytest tests/optimization/test_simplified_laplace.py -q`.

- [ ] **Step 3: Implement**

```python
def _simplified_laplace_marginals(design, offset, y, grid):
    dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
    offset = np.asarray(offset, float); y = np.asarray(y, float)
    points = list(grid)
    p = points[0][1].mean.shape[0]
    n_grid = len(points)
    location = np.zeros((p, n_grid)); scale = np.zeros((p, n_grid))
    shape = np.zeros((p, n_grid)); weights = np.zeros((p, n_grid))
    clamped_count = 0
    for k, (w, fit, likelihood) in enumerate(points):
        m = np.asarray(fit.mean, float); cov = np.asarray(fit.covariance, float)
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        eta_mean = offset + dense @ m
        d3 = np.asarray(likelihood.third_derivative(eta_mean, y), float)   # (n,)
        cx_eta = cov @ dense.T                                             # (p, n) cov(x_i, eta_j)
        eta_var = np.clip(np.einsum("ij,jk,ik->i", dense, cov, dense), 0.0, None)  # (n,)
        sigma_eta = np.sqrt(eta_var)
        safe_si = np.where(sigma > 0, sigma, 1.0)
        # gamma3_i = (1/sigma_i^3) sum_j d3_j cov(x_i,eta_j)^3
        gamma3 = (cx_eta ** 3 @ d3) / safe_si ** 3
        # a_ij = cov / (sigma_i sigma_eta_j); gamma1_i = (1/(2 sigma_i)) sum_j sigma_eta_j^2 (1-a_ij^2) d3_j cov
        safe_se = np.where(sigma_eta > 0, sigma_eta, 1.0)
        a_ij = cx_eta / (safe_si[:, None] * safe_se[None, :])
        term = (sigma_eta ** 2)[None, :] * (1.0 - a_ij ** 2) * d3[None, :] * cx_eta
        gamma1 = term.sum(axis=1) / (2.0 * safe_si)
        # degenerate sigma_i -> no correction
        gamma1 = np.where(sigma > 0, gamma1, 0.0)
        gamma3 = np.where(sigma > 0, gamma3, 0.0)
        xi, omega, sn_a, clamped = _fit_skew_normal(gamma1, gamma3)
        clamped_count += int(clamped.sum())
        location[:, k] = m + sigma * xi
        scale[:, k] = sigma * omega
        shape[:, k] = sn_a
        weights[:, k] = w
    return location, scale, shape, weights, clamped_count
```

- [ ] **Step 4: Run** — focused (anchor + oracle pass) + `PYTHONPATH=src pytest -q`.

- [ ] **Step 5: Commit** — `git commit -m "feat: compute simplified-Laplace skew-normal latent marginals"`

---

### Task 4: `latent_strategy` Dispatch and `INLAResult` Integration

**Files:** Modify `src/pylgm/optimization/inla.py`, `src/pylgm/inference/result.py`, `src/pylgm/model.py`, `tests/test_model.py`, `tests/optimization/test_inla.py`.

**Interfaces:**
- `integrate_inla(..., latent_strategy="gaussian")`; when `"simplified_laplace"`, compute `_simplified_laplace_marginals` from the kept grid (observed-row design/offset/y from `kept[0]`'s compiled) and attach to the result.
- `INLAResult(..., latent_marginal_table=None)`: an internal per-latent-index `SkewNormalMarginals` (full latent vector); `latent_marginals(block)` returns the block's slice as `SkewNormalMarginals` when present, else the current `GaussianMarginals`.
- `LGM.fit(..., latent_strategy="gaussian")` validated; `"simplified_laplace"` requires `hyperparameters="integrate"`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
def test_sla_latent_strategy_returns_skew_marginals_both_engines():
    import numpy as np, pandas as pd
    from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter
    from pylgm.inference.result import SkewNormalMarginals, GaussianMarginals

    rng = np.random.default_rng(0)
    regions = [f"r{i}" for i in range(25)]
    eff = {r: rng.normal() for r in regions}
    rows = [{"region": r, "t": t, "y": eff[r] + rng.normal()} for t in range(6) for r in regions]
    frame = pd.DataFrame(rows)
    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    model = LGM("y", Poisson(),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    sla = model.fit(counts, engine="laplace", hyperparameters="integrate",
                    latent_strategy="simplified_laplace")
    assert isinstance(sla.latent_marginals("region"), SkewNormalMarginals)
    # default stays Gaussian
    gauss = model.fit(counts, engine="laplace", hyperparameters="integrate")
    assert isinstance(gauss.latent_marginals("region"), GaussianMarginals)


def test_sla_requires_integration_and_valid_strategy():
    import numpy as np, pandas as pd
    from pylgm import Fixed, Gaussian, IID, LGM, Hyperparameter
    frame = pd.DataFrame({"region": ["a", "b"] * 10, "t": list(range(20)),
                          "y": np.random.default_rng(0).normal(size=20)})
    model = LGM("y", Gaussian(1.0),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    with pytest.raises(ValueError, match="integrate"):
        model.fit(frame, hyperparameters="optimize", latent_strategy="simplified_laplace")
    with pytest.raises(ValueError, match="latent_strategy"):
        model.fit(frame, hyperparameters="integrate", latent_strategy="nonsense")
```

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=src pytest tests/test_model.py -q`.

- [ ] **Step 3: Implement**

- `integrate_inla`: add `latent_strategy="gaussian"` param. When `"simplified_laplace"`, after the mixture accumulation, build `criteria_grid`-style `sla_grid=[(w, cond, compiled.likelihood)]` and the observed-row `design`/`offset`/`y` from `kept[0][4]` (as the criteria code already does), call `_simplified_laplace_marginals`, and construct a full-latent `SkewNormalMarginals` (`location`/`scale`/`shape`/`weights` shape `(p, n_grid)`), passed to `INLAResult(latent_marginal_table=...)`. When `"gaussian"`, pass `None`.
- `INLAResult`: add `latent_marginal_table: SkewNormalMarginals | None = None`, stored; `latent_marginals(block)` returns the block-sliced `SkewNormalMarginals` (slice its component axis by `block_slices[block]`) when the table is present, else the existing Gaussian path. Add a `SkewNormalMarginals.select(slice)` helper returning the component-subset marginals. `_rebuild_result` carries `latent_marginal_table` through (not row-reordered).
- `model.py`: `fit(..., latent_strategy="gaussian")` validated in `{"gaussian","simplified_laplace"}`; `"simplified_laplace"` with `hyperparameters!="integrate"` → `ValueError`; thread to `_run_inla(family, engine, latent_strategy)` → `integrate_inla(..., latent_strategy=...)`.

- [ ] **Step 4: Run** — `PYTHONPATH=src pytest tests/test_model.py tests/optimization/test_inla.py -q` + `PYTHONPATH=src pytest -q` + `ruff check src tests`.

- [ ] **Step 5: Commit** — `git commit -m "feat: select simplified-Laplace latent strategy in LGM.fit"`

---

### Task 5: Example, Docs, and Roadmap

**Files:** create `examples/inla_sla/` (data.csv, run.py, README.md); modify `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`, `tests/test_package.py`.

- [ ] **Step 1: Example smoke test** (append to `tests/test_package.py`): run `examples/inla_sla/run.py`, assert output contains `skewness`.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Create example** — a Poisson region panel (interior posterior); `run.py` fits with `latent_strategy="simplified_laplace"`, prints a region marginal's `mean`, `std`, `skewness`, and `quantile(0.025)/quantile(0.975)`, contrasting with the Gaussian strategy's symmetric interval. `README.md` explains SLA corrects skewness (per RMC 2009) and is exact for Gaussian likelihoods.
- [ ] **Step 4: Docs** — README: an "Simplified Laplace latent marginals" subsection (latent_strategy, skew-aware marginals, exact-for-Gaussian, faithful to RMC 2009 §3.2.3/App. B, validated against a brute-force true-marginal oracle not R-INLA). Arch spec: mark the SLA latent strategy (3c, simplified Laplace) SHIPPED; keep **full Laplace**, η-marginal SLA, region-of-interest pruning, **R-INLA parity fixtures**, and the other INLA follow-ups (integrand-mode recentering, randomized discrete PIT, robust discrete CPO, non-integrated-fit criteria) deferred/roadmap.
- [ ] **Step 5: Run suite + ruff + example.**
- [ ] **Step 6: Commit** — `git commit -m "docs: publish INLA simplified-Laplace latent marginals"`

---

## Self-Review Checklist

- Task 1 (third derivative) → Task 2 (skew-normal fit + type) → Task 3 (SLA helper) → Task 4 (dispatch + result) → Task 5 (docs). Linear.
- Every spec section maps to a task: third derivative (T1); Appendix-B skew-normal + skew marginal type (T2); RMC γ¹/γ³ corrections + skew-normal per grid (T3); latent_strategy dispatch + INLAResult + both engines (T4); docs/example/roadmap (T5).
- Correctness anchors: Gaussian exact reduction (T2 skew-normal zero case, T3 Gaussian-likelihood, T4 default unchanged); brute-force true-marginal oracle for tiny Poisson (T3); skew-normal moment/quantile self-consistency (T2).
- Math is exactly the transcribed RMC equations (eqs 14,16,21,22,23,31,32); no improvisation. No new dependency. Predictions/criteria/optimize paths unchanged.
- Deferred (unimplemented, recorded): full Laplace, η-marginal SLA, region-of-interest pruning, R-INLA parity, and the other INLA refinement follow-ups.

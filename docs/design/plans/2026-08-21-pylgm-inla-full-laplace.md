# pyLGM INLA Full-Laplace Latent Marginals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the full Laplace approximation for latent marginals — spline-corrected tabulated densities — for unconstrained integrated fits, faithful to RMC (2009) §3.2.2 (eqs 12/13/16/17).

**Architecture:** `_full_laplace_marginals` evaluates, per latent and Gauss–Hermite abscissa, `log π̃_LA` via the eq-13 conditional-mean configuration and the eq-12 `π̃_GG` determinant, fits a cubic spline (eq 17), and mixes the per-θ normalized densities on a common per-latent grid into a `TabulatedMarginals`. Exposed via `latent_strategy="laplace"`; unconstrained models only.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`interpolate.CubicSpline`, `integrate.cumulative_trapezoid`, `linalg.cho_factor`), pytest, Ruff.

## Global Constraints

- No new runtime dependencies (SciPy already used).
- Equations exactly as in the spec (`docs/superpowers/specs/2026-08-21-pylgm-inla-full-laplace-design.md`); do not improvise.
- **Unconstrained latent fields only.** `latent_strategy="laplace"` with any latent constraint (`compiled.constraints.shape[0] > 0`, i.e. RW1/RW2) raises `UnsupportedEngineError`; `gaussian`/`simplified_laplace` unchanged.
- `latent_strategy` default `"gaussian"` byte-identical; `"laplace"` requires `hyperparameters="integrate"`.
- Gaussian likelihood ⇒ spline ≡ 0 ⇒ per-grid-point full Laplace equals the Gaussian marginal (anchor).
- No R-INLA: validate via Gaussian exact reduction + a brute-force nested-quadrature true-marginal oracle (compare Gaussian vs simplified-Laplace vs full-Laplace against truth).
- Use `PYTHONPATH=src` for all test/lint commands.

## File Map
- `src/pylgm/inference/result.py`: `TabulatedMarginals`; `INLAResult` tabulated table + `latent_marginals` under `"laplace"`.
- `src/pylgm/inference/__init__.py`: export `TabulatedMarginals`.
- `src/pylgm/optimization/inla.py`: `_full_laplace_marginals` (+ spline eval); wire into `integrate_inla`; constrained guard.
- `src/pylgm/model.py`: widen `latent_strategy` to include `"laplace"`; thread + guards.
- `examples/inla_full_laplace/`, `README.md`, arch spec, tests.

---

### Task 1: `TabulatedMarginals`

**Files:** Modify `src/pylgm/inference/result.py`, `src/pylgm/inference/__init__.py`; create `tests/inference/test_tabulated_marginals.py`.

**Interfaces:** `TabulatedMarginals(x, density)` — `x`, `density` shape `(n_components, n_grid)`; each row's `x` strictly increasing, `density ≥ 0`, normalized to unit trapezoidal integral on construction. Properties `mean`, `variance`, `std`, `skewness` (trapezoidal moments per row); `pdf(x0)`, `cdf(x0)`, `quantile(p)` (interpolation; `0<p<1`); `select(sl)` (component subset). Read-only defensive copies.

- [ ] **Step 1: Write the failing tests**

```python
# tests/inference/test_tabulated_marginals.py
import numpy as np
from scipy.stats import norm, skewnorm
from pylgm.inference.result import TabulatedMarginals


def _grid_density(dist, lo, hi, n=4001):
    x = np.linspace(lo, hi, n)
    return x, dist.pdf(x)


def test_tabulated_matches_normal_moments_and_quantiles():
    x, d = _grid_density(norm(loc=0.5, scale=2.0), -12, 13)
    m = TabulatedMarginals(x[None, :], d[None, :])
    np.testing.assert_allclose(m.mean, [0.5], atol=1e-3)
    np.testing.assert_allclose(m.variance, [4.0], atol=1e-2)
    np.testing.assert_allclose(m.skewness, [0.0], atol=1e-3)
    np.testing.assert_allclose(m.quantile(0.975), [0.5 + norm.ppf(0.975) * 2.0], atol=2e-2)
    np.testing.assert_allclose(m.cdf(m.quantile(0.3))[0], 0.3, atol=1e-3)


def test_tabulated_matches_skewnormal_skewness():
    x, d = _grid_density(skewnorm(a=6.0, loc=0.0, scale=1.0), -6, 8)
    m = TabulatedMarginals(x[None, :], d[None, :])
    assert m.skewness[0] > 0.4   # skewnorm(a=6) skewness ~0.77
    np.testing.assert_allclose(m.mean, [skewnorm.mean(6.0)], atol=1e-2)


def test_tabulated_normalizes_and_is_immutable():
    x = np.linspace(-5, 5, 2001)
    m = TabulatedMarginals(x[None, :], norm.pdf(x)[None, :] * 7.0)  # unnormalized
    np.testing.assert_allclose(np.trapz(m.pdf(x)[0] if False else m.density[0], m.x[0]), 1.0, atol=1e-6)
    with __import__("pytest").raises(ValueError):
        m.density[0, 0] = 1.0
```

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=src pytest tests/inference/test_tabulated_marginals.py -q`.

- [ ] **Step 3: Implement** (`result.py`, reusing `_readonly_array`; `from scipy.integrate import cumulative_trapezoid`)

Frozen dataclass storing read-only `_x`, `_density` (each `(n_components, n_grid)`), validated (2-D, matching shapes, each `x` row strictly increasing, `density ≥ 0` and not all-zero), each density row divided by its `np.trapz(density_row, x_row)`. Properties `x`, `density` (read-only copies). `mean = trapz(x*d, x)` per row; `variance = trapz((x-mean)²*d, x)`; `std = √variance`; `skewness = trapz((x-mean)³*d, x)/std³`. `pdf(x0)`: per-row `np.interp(x0, x_row, d_row, left=0.0, right=0.0)`. `cdf(x0)`: per row `cdf_row = cumulative_trapezoid(d_row, x_row, initial=0.0)` (ends at ~1), `np.interp(x0, x_row, cdf_row, left=0.0, right=1.0)`. `quantile(p)` (`0<p<1`): per row `np.interp(p, cdf_row, x_row)`. `select(sl)`: `TabulatedMarginals(self._x[sl], self._density[sl])` with `sl` normalized to a 2-D slice. Export from `inference/__init__.py`.

- [ ] **Step 4: Run** focused + `PYTHONPATH=src pytest -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat: add TabulatedMarginals numerical density type"`

---

### Task 2: `_full_laplace_marginals` Helper

**Files:** Modify `src/pylgm/optimization/inla.py`; create `tests/optimization/test_full_laplace.py`.

**Interfaces:** `_full_laplace_marginals(design, offset, y, grid, precision, *, n_abscissae=7, grid_points=201, grid_radius=8.0) -> TabulatedMarginals`, where `design` is the observed-row design (n_obs×p), `grid` is `[(weight, conditional_fit, likelihood)]`, `precision` is the dense prior precision `Q` (p×p). Unconstrained models only (caller guards).

- [ ] **Step 1: Write the failing tests (Gaussian exactness + 3-way brute-force oracle)**

```python
# tests/optimization/test_full_laplace.py
import numpy as np
from scipy.sparse import csr_matrix, eye
from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.inference import fit_gaussian, fit_laplace
from pylgm.optimization.inla import _full_laplace_marginals, _simplified_laplace_marginals
from pylgm.inference.result import SkewNormalMarginals


class _Fit:
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, float); self.covariance = np.asarray(cov, float)


def test_full_laplace_gaussian_likelihood_is_exact_gaussian():
    n, p = 6, 2
    X = csr_matrix(np.column_stack([np.ones(n), np.linspace(-1, 1, n)]))
    y = np.array([0.4, -0.3, 0.8, 0.1, -0.6, 0.2])
    Q = np.eye(2) * 0.5
    fit = _Fit([0.2, -0.1], np.array([[0.3, 0.05], [0.05, 0.2]]))
    tab = _full_laplace_marginals(X, np.zeros(n), y, [(1.0, fit, CompiledGaussian(1.0))],
                                  Q, n_abscissae=7)
    # exact-Gaussian: tabulated mean/variance match the Gaussian marginal
    np.testing.assert_allclose(tab.mean, fit.mean, atol=5e-3)
    np.testing.assert_allclose(tab.variance, np.diag(fit.covariance), rtol=5e-2)
    np.testing.assert_allclose(tab.skewness, [0.0, 0.0], atol=2e-2)


def test_full_laplace_beats_or_matches_sla_vs_true_marginal():
    # tiny UNCONSTRAINED Poisson: Fixed("1 + x"), p=2 -> true marginal by 2-D quadrature
    n = 15
    rng = np.random.default_rng(0)
    xcov = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([np.ones(n), xcov])
    true_beta = np.array([0.5, 0.8])
    y = rng.poisson(np.exp(X @ true_beta)).astype(float)
    Xs = csr_matrix(X)
    prior_prec = 1.0
    block = LatentBlock("b", ("i", "s"), Xs, eye(2, format="csr") * prior_prec,
                        np.empty((0, 2), dtype=float))
    fam = CompiledFamily(y=y, observed=np.array([True] * n), offset=np.zeros(n),
                         blocks=(ScalableBlock(block, None, 1.0),), parameter_names=(),
                         likelihood_factory=lambda r: CompiledPoisson())
    compiled = fam.materialize({})
    fit = fit_laplace(compiled)
    Q = np.eye(2) * prior_prec
    like = CompiledPoisson()

    tab = _full_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, like)], Q, n_abscissae=9)
    loc, scale, shape, w, _ = _simplified_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, like)])
    sla = SkewNormalMarginals(weights=w, location=loc, scale=scale, shape=shape)

    # TRUE marginal of x_0 (intercept) via 2-D quadrature of the joint posterior
    g0 = np.linspace(fit.mean[0] - 6 * np.sqrt(fit.covariance[0, 0]),
                     fit.mean[0] + 6 * np.sqrt(fit.covariance[0, 0]), 241)
    g1 = np.linspace(fit.mean[1] - 6 * np.sqrt(fit.covariance[1, 1]),
                     fit.mean[1] + 6 * np.sqrt(fit.covariance[1, 1]), 241)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    eta = G0[..., None] * X[:, 0] + G1[..., None] * X[:, 1]      # (241,241,n)
    loglik = np.sum(y * eta - np.exp(eta), axis=-1)
    logprior = -0.5 * prior_prec * (G0 ** 2 + G1 ** 2)
    dens = np.exp((loglik + logprior) - (loglik + logprior).max())
    marg0 = dens.sum(axis=1); marg0 /= np.trapz(marg0, g0)
    tm = float(np.trapz(g0 * marg0, g0))
    tv = float(np.trapz((g0 - tm) ** 2 * marg0, g0))
    tskew = float(np.trapz((g0 - tm) ** 3 * marg0, g0) / tv ** 1.5)

    full_skew_err = abs(tab.skewness[0] - tskew)
    sla_skew_err = abs(sla.skewness[0] - tskew)
    gauss_skew_err = abs(0.0 - tskew)
    assert np.sign(tab.skewness[0]) == np.sign(tskew)
    assert full_skew_err <= sla_skew_err + 1e-3     # full LA at least as good as SLA on skew
    assert full_skew_err < gauss_skew_err            # and better than the Gaussian strategy
    np.testing.assert_allclose(tab.mean[0], tm, atol=2e-2)
```

(If `full_skew_err <= sla_skew_err` is a razor-thin tie on some seed, keep `<= sla_skew_err + 1e-3` and the skew-sign + mean assertions — the load-bearing checks. Do not weaken the "better than Gaussian" assertion.)

- [ ] **Step 2: Run to verify fail** — `PYTHONPATH=src pytest tests/optimization/test_full_laplace.py -q`.

- [ ] **Step 3: Implement**

```python
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor

def _logdet_spd(matrix):
    factor = cho_factor(matrix, lower=True, check_finite=True)
    return 2.0 * np.sum(np.log(np.diag(factor[0])))


def _full_laplace_marginals(design, offset, y, grid, precision, *,
                            n_abscissae=7, grid_points=201, grid_radius=8.0):
    X = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
    offset = np.asarray(offset, float); y = np.asarray(y, float)
    Q = precision.toarray() if hasattr(precision, "toarray") else np.asarray(precision, float)
    points = list(grid)
    p = points[0][1].mean.shape[0]
    std_nodes = np.polynomial.hermite.hermgauss(n_abscissae)[0] * np.sqrt(2.0)  # ~N(0,1) abscissae

    x_out = np.zeros((p, grid_points)); dens_out = np.zeros((p, grid_points))
    for i in range(p):
        # common per-latent grid from the mixture spread
        mu_bar = sum(w * f.mean[i] for w, f, _ in points)
        sig_max = max(np.sqrt(max(f.covariance[i, i], 0.0)) for _, f, _ in points)
        gx = np.linspace(mu_bar - grid_radius * sig_max, mu_bar + grid_radius * sig_max, grid_points)
        mixed = np.zeros(grid_points)
        keep = [c for c in range(p) if c != i]
        for w, fit, likelihood in points:
            m = np.asarray(fit.mean, float); cov = np.asarray(fit.covariance, float)
            sigma_i = np.sqrt(max(cov[i, i], 0.0))
            if sigma_i <= 0.0:
                continue
            abscissae = m[i] + sigma_i * std_nodes
            delta = np.empty(n_abscissae)
            for a, xi in enumerate(abscissae):
                x = m.copy()
                x[keep] = m[keep] + cov[keep, i] / cov[i, i] * (xi - m[i])   # eq 13 conditional mean
                x[i] = xi
                eta = offset + X @ x
                loglik = float(likelihood.pointwise_log_density(eta, y).sum())
                joint = -0.5 * float(x @ Q @ x) + loglik
                wgt = likelihood.working_weights(eta, y)
                info = Q + (X.T * wgt) @ X
                info_mi = np.delete(np.delete(info, i, axis=0), i, axis=1)
                log_gg = 0.5 * _logdet_spd(info_mi) if p > 1 else 0.0
                log_la = joint - log_gg
                log_g = -0.5 * std_nodes[a] ** 2 - np.log(sigma_i) - 0.5 * np.log(2 * np.pi)
                delta[a] = log_la - log_g
            spline = CubicSpline(abscissae, delta, extrapolate=False)
            s = spline(gx)
            s[gx < abscissae[0]] = delta[0]      # constant tail extrapolation
            s[gx > abscissae[-1]] = delta[-1]
            log_dens = -0.5 * ((gx - m[i]) / sigma_i) ** 2 - np.log(sigma_i) - 0.5 * np.log(2 * np.pi) + s
            dens = np.exp(log_dens - log_dens.max())
            area = np.trapz(dens, gx)
            if not np.isfinite(area) or area <= 0:
                raise NumericalError("full-Laplace density normalization failed")
            mixed += w * (dens / area)
        total = np.trapz(mixed, gx)
        if not np.isfinite(total) or total <= 0:
            raise NumericalError("full-Laplace mixture normalization failed")
        x_out[i] = gx; dens_out[i] = mixed / total
    from pylgm.inference.result import TabulatedMarginals
    return TabulatedMarginals(x_out, dens_out)
```

Note: `p == 1` short-circuits the `π̃_GG` determinant (no `x_{-i}`), so the correction is purely the likelihood curvature vs the Gaussian — still a valid 1-D full Laplace. `NumericalError` is already imported in `inla.py`.

- [ ] **Step 4: Run** focused (Gaussian-exact + oracle pass) + `PYTHONPATH=src pytest -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat: compute full-Laplace tabulated latent marginals"`

---

### Task 3: `latent_strategy="laplace"` Dispatch + Constrained Guard

**Files:** Modify `src/pylgm/optimization/inla.py`, `src/pylgm/inference/result.py`, `src/pylgm/model.py`, `tests/test_model.py`, `tests/optimization/test_inla.py`.

**Interfaces:** `integrate_inla(..., latent_strategy)` handles `"laplace"` (build `TabulatedMarginals` via `_full_laplace_marginals` from `kept`'s observed-row design/offset/y, per-grid `(w,cond,likelihood)`, and `kept[0][4].precision`); raise `UnsupportedEngineError` if `kept[0][4].constraints.shape[0] > 0`. `INLAResult` returns the tabulated table from `latent_marginals` under `"laplace"`. `LGM.fit(..., latent_strategy)` accepts `"laplace"`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
def test_full_laplace_latent_strategy_returns_tabulated_both_engines():
    import numpy as np, pandas as pd
    from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter
    from pylgm.inference.result import TabulatedMarginals
    rng = np.random.default_rng(0)
    regions = [f"r{i}" for i in range(25)]
    eff = {r: rng.normal() for r in regions}
    rows = [{"region": r, "t": t, "x": rng.normal(), "y": eff[r] + rng.normal()}
            for t in range(6) for r in regions]
    frame = pd.DataFrame(rows)
    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    # UNCONSTRAINED: Fixed + IID (no RW)
    model = LGM("y", Poisson(),
                Fixed("1 + x") + IID("region", index="region",
                                     precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    res = model.fit(counts, engine="laplace", hyperparameters="integrate",
                    latent_strategy="laplace")
    assert isinstance(res.latent_marginals("region"), TabulatedMarginals)


def test_full_laplace_rejects_constrained_effects():
    import numpy as np, pandas as pd
    from pylgm import Fixed, Gaussian, LGM, RW1, Hyperparameter
    from pylgm.exceptions import UnsupportedEngineError
    rng = np.random.default_rng(0)
    rows = [{"t": t, "y": rng.normal()} for t in range(1, 40)]
    frame = pd.DataFrame(rows)
    model = LGM("y", Gaussian(1.0),
                Fixed("1") + RW1("trend", index="t",
                                 precision=Hyperparameter("p", initial=1.0)),
                time="t")
    with pytest.raises(UnsupportedEngineError, match="constrained|laplace"):
        model.fit(frame, engine="exact_gaussian", hyperparameters="integrate",
                  latent_strategy="laplace")
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — `integrate_inla`: add `"laplace"` branch (guard `constraints.shape[0] > 0` → `UnsupportedEngineError("full Laplace does not support constrained (RW) effects; use latent_strategy='gaussian' or 'simplified_laplace'")`; else `_full_laplace_marginals(design_obs, offset_obs, y_obs, sla_grid, kept[0][4].precision)`), store on `INLAResult(latent_marginal_table=...)` — reuse the same `latent_marginal_table` slot the SLA path uses, since both `SkewNormalMarginals` and `TabulatedMarginals` support `.select(sl)`; `latent_marginals(block)` already returns `table.select(block_slices[block])`. `model.py`: widen validation set to `{"gaussian","simplified_laplace","laplace"}`; keep the "requires integrate" guard.
- [ ] **Step 4: Run** `PYTHONPATH=src pytest tests/test_model.py tests/optimization/test_inla.py -q` + full suite + ruff.
- [ ] **Step 5: Commit** — `git commit -m "feat: select full-Laplace latent strategy in LGM.fit"`

---

### Task 4: Example, Docs, and Roadmap

**Files:** create `examples/inla_full_laplace/`; modify `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`, `tests/test_package.py`.

- [ ] **Step 1: Example smoke test** (append to `tests/test_package.py`): run `examples/inla_full_laplace/run.py`, assert output contains `quantile`.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Create example** — an UNCONSTRAINED Poisson model (Fixed + IID region), fit with `latent_strategy="laplace"`, print a region marginal's `mean/std/skewness/quantile(0.025)/quantile(0.975)` and contrast with `"gaussian"` and `"simplified_laplace"` on the same fit. `README.md`: how to run; that full Laplace is the most accurate (and most expensive) latent strategy, faithful to RMC 2009 §3.2.2, unconstrained models only.
- [ ] **Step 4: Docs** — README "Full-Laplace latent marginals" subsection (latent_strategy="laplace", TabulatedMarginals, unconstrained-only, exact-for-Gaussian, brute-force-oracle-validated not R-INLA, most-accurate/most-expensive tier). Arch spec: mark full Laplace (3d) SHIPPED (unconstrained); keep constrained-effect full Laplace, region-of-interest pruning, sparse determinant updates, η-marginals, R-INLA parity, and the other INLA follow-ups deferred/roadmap.
- [ ] **Step 5: Run suite + ruff + example.**
- [ ] **Step 6: Commit** — `git commit -m "docs: publish INLA full-Laplace latent marginals"`

---

## Self-Review Checklist
- Task 1 (TabulatedMarginals) → Task 2 (full-Laplace helper) → Task 3 (dispatch + guard) → Task 4 (docs). Linear.
- Every spec section maps to a task: tabulated type (T1); RMC eqs 12/13/16/17 + spline + mixture (T2); latent_strategy="laplace" + constrained guard + result (T3); docs/example/roadmap (T4).
- Correctness anchors: TabulatedMarginals vs analytic Normal/skew-normal (T1); Gaussian exact reduction + 3-way brute-force oracle (Gaussian vs SLA vs full-LA vs truth) (T2); constrained-guard + dispatch + both engines (T3).
- Faithful RMC math, no improvisation; unconstrained-only (documented); no new deps; other strategies + predictions/criteria/optimize unchanged.
- Deferred/recorded: constrained-effect full Laplace, region-of-interest pruning, sparse determinant updates, η-marginals, R-INLA parity, result-type unification.

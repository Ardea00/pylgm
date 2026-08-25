# pyLGM INLA Model-Assessment Criteria Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute DIC, WAIC, CPO, and PIT from the INLA-integrated fit and attach them to `result.criteria`, for both engines.

**Architecture:** Two new per-observation likelihood methods (`pointwise_log_density`, `cdf`) feed a `_model_criteria` helper that integrates each observation's linear-predictor posterior (a grid mixture of Gaussians) by Gauss–Hermite quadrature. The helper folds into `integrate_inla`'s grid loop; its `ModelCriteria` result is attached to `INLAResult`.

**Tech Stack:** Python 3.11+, NumPy (`numpy.polynomial.hermite.hermgauss`), SciPy (`special.erf`, `special.gammaincc`, `special.gammaln`), pytest, Ruff.

## Global Constraints

- No new runtime dependencies (SciPy already used).
- Definitions are exactly as in the spec (`docs/superpowers/specs/2026-08-20-pylgm-inla-criteria-design.md`): per-observation η-posterior `N(m_i, v_i)` per grid point, Gauss–Hermite over each, grid-weighted aggregation. WAIC `= −2Σ(lppd−Var[logp])`; DIC `= D̄ + p_D`, `p_D = D̄ − D(m̄)`; CPO `= 1/E[1/p]`; PIT `= E[cdf/p]/E[1/p]` (non-randomized CDF for discrete).
- Criteria are attached only to the INLA-integrated result; the `optimize`/plug-in paths and the other result types are unchanged.
- Always computed when INLA runs (no `LGM.fit` signature change).
- Preserve Pandas caller-order and Spark canonical-key contracts; `criteria` is not row-dependent (not reordered).
- Use `PYTHONPATH=src` for all test/lint commands.

## File Map

- `src/pylgm/likelihoods.py`: `pointwise_log_density` + `cdf` on the three compiled likelihoods.
- `src/pylgm/inference/result.py`: `ModelCriteria`; `INLAResult.criteria`.
- `src/pylgm/inference/__init__.py`: export `ModelCriteria`.
- `src/pylgm/optimization/inla.py`: `_model_criteria`; wire into `integrate_inla` (`evaluate` returns the compiled model; compute + attach criteria).
- `src/pylgm/model.py`: `_rebuild_result` carries `criteria` on the `INLAResult` branch.
- `examples/inla_criteria/`: runnable example. `README.md`, arch spec, `tests/*`.

---

### Task 1: Per-Observation Likelihood Methods

**Files:**
- Modify: `src/pylgm/likelihoods.py`
- Modify: `tests/test_modeling_vocabulary.py`

**Interfaces:**
- Produces on `CompiledGaussian`/`CompiledPoisson`/`CompiledBernoulli`:
  `pointwise_log_density(eta, y) -> np.ndarray` (per-observation, summing to the existing `log_likelihood`) and `cdf(eta, y) -> np.ndarray` (`P(Y_i ≤ y_i | η_i)`).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_modeling_vocabulary.py
import numpy as np
from scipy.stats import norm, poisson
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson, CompiledBernoulli


def test_pointwise_log_density_sums_to_log_likelihood():
    eta = np.array([0.3, -0.5, 1.2]); y = np.array([1.0, 0.0, 2.0])
    for like in (CompiledGaussian(1.3), CompiledPoisson(), CompiledBernoulli()):
        yy = y if not isinstance(like, CompiledBernoulli) else np.array([1.0, 0.0, 1.0])
        pw = like.pointwise_log_density(eta, yy)
        np.testing.assert_allclose(pw.sum(), like.log_likelihood(eta, yy))


def test_cdf_matches_scipy():
    eta = np.array([0.2, -1.0, 0.5])
    g = CompiledGaussian(2.0)
    np.testing.assert_allclose(g.cdf(eta, np.array([0.0, -1.0, 1.5])),
                               norm.cdf(np.array([0.0, -1.0, 1.5]), loc=eta, scale=2.0))
    p = CompiledPoisson()
    y = np.array([0.0, 2.0, 5.0])
    np.testing.assert_allclose(p.cdf(eta, y), poisson.cdf(y, np.exp(eta)), atol=1e-12)
    b = CompiledBernoulli()
    yb = np.array([1.0, 0.0, 1.0])
    expected = np.where(yb >= 1, 1.0, 1.0 - 1.0 / (1.0 + np.exp(-eta)))
    np.testing.assert_allclose(b.cdf(eta, yb), expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q`
Expected: FAIL (methods absent).

- [ ] **Step 3: Implement the methods**

Add to `likelihoods.py` (imports `from scipy.special import erf, gammaincc` — `gammaln` already imported; `np` already imported).

`CompiledGaussian`:
```python
    def pointwise_log_density(self, eta, y):
        eta = np.asarray(eta, dtype=float); y = np.asarray(y, dtype=float)
        return -0.5 * (np.log(2 * np.pi * self.variance) + (y - eta) ** 2 / self.variance)

    def cdf(self, eta, y):
        eta = np.asarray(eta, dtype=float); y = np.asarray(y, dtype=float)
        return 0.5 * (1.0 + erf((y - eta) / (np.sqrt(self.variance) * np.sqrt(2.0))))
```

`CompiledPoisson`:
```python
    def pointwise_log_density(self, eta, y):
        eta = np.asarray(eta, dtype=float); y = np.asarray(y, dtype=float)
        return y * eta - np.exp(eta) - gammaln(y + 1.0)

    def cdf(self, eta, y):
        eta = np.asarray(eta, dtype=float); y = np.asarray(y, dtype=float)
        return gammaincc(np.floor(y) + 1.0, np.exp(eta))
```

`CompiledBernoulli`:
```python
    def pointwise_log_density(self, eta, y):
        eta = np.asarray(eta, dtype=float); y = np.asarray(y, dtype=float)
        return y * eta - np.logaddexp(0.0, eta)

    def cdf(self, eta, y):
        eta = np.asarray(eta, dtype=float); y = np.asarray(y, dtype=float)
        return np.where(y >= 1.0, 1.0, 1.0 - self.link.inverse(eta))
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src pytest tests/test_modeling_vocabulary.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/likelihoods.py tests/test_modeling_vocabulary.py
git commit -m "feat: add pointwise log-density and CDF to likelihoods"
```

---

### Task 2: `ModelCriteria` and the `_model_criteria` Helper

**Files:**
- Modify: `src/pylgm/inference/result.py` (add `ModelCriteria`)
- Modify: `src/pylgm/inference/__init__.py`
- Modify: `src/pylgm/optimization/inla.py` (add `_model_criteria`)
- Create: `tests/optimization/test_inla_criteria.py`
- Modify: `tests/inference/test_result.py` (ModelCriteria immutability)

**Interfaces:**
- Produces: `ModelCriteria(dic, dic_effective_parameters, waic, waic_effective_parameters, cpo, pit, cpo_failures, log_cpo_sum)` — a frozen dataclass; `cpo`/`pit` are read-only per-observation arrays.
- Produces: `_model_criteria(design, offset, y, grid, *, n_nodes=21, cpo_failure_threshold=0.5) -> ModelCriteria`, where `design` is the observed-row design (dense or CSR), `offset`/`y` the observed-row offset/response, and `grid` an iterable of `(weight: float, conditional_fit, likelihood)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/optimization/test_inla_criteria.py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
from pylgm.likelihoods import CompiledGaussian
from pylgm.optimization.inla import _model_criteria


class _Fit:
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(cov, dtype=float)


def test_single_grid_gaussian_matches_closed_forms():
    # 2 observed rows, identity design (eta_i = x_i), latent posterior N(mean, cov diag)
    design = csr_matrix(np.eye(2))
    offset = np.zeros(2)
    y = np.array([0.4, -0.7])
    mean = np.array([0.5, -0.5]); v = np.array([0.1, 0.2])
    fit = _Fit(mean, np.diag(v))
    sigma = 1.3; var = sigma ** 2
    like = CompiledGaussian(sigma)
    crit = _model_criteria(design, offset, y, [(1.0, fit, like)], n_nodes=64)

    m = mean  # eta mean = design@mean = mean
    # WAIC ingredients (closed forms)
    elog = -0.5 * (np.log(2 * np.pi * var) + ((y - m) ** 2 + v) / var)
    lppd = norm.logpdf(y, loc=m, scale=np.sqrt(v + var))
    var_logp = (v ** 2 + 2 * v * (y - m) ** 2) / (2 * var ** 2)
    waic = -2.0 * np.sum(lppd - var_logp)
    np.testing.assert_allclose(crit.waic, waic, rtol=1e-6)
    np.testing.assert_allclose(crit.waic_effective_parameters, var_logp.sum(), rtol=1e-6)
    # DIC: D_bar = -2 sum elog ; D(mbar) = -2 sum logdensity(m, y) ; p_D = Dbar - Dmean
    d_bar = -2.0 * elog.sum()
    d_mean = -2.0 * like.pointwise_log_density(m, y).sum()
    np.testing.assert_allclose(crit.dic, d_bar + (d_bar - d_mean), rtol=1e-6)

    # CPO/PIT vs independent fine Gauss-Hermite (harmonic / reweighted definitions)
    from numpy.polynomial.hermite import hermgauss
    nodes, w = hermgauss(200); w = w / np.sqrt(np.pi)
    for i in range(2):
        en = m[i] + np.sqrt(2 * v[i]) * nodes
        lp = like.pointwise_log_density(en, np.full_like(en, y[i]))
        einv = np.sum(w / np.exp(lp))
        cdf = like.cdf(en, np.full_like(en, y[i]))
        f = np.sum(w * cdf / np.exp(lp))
        np.testing.assert_allclose(crit.cpo[i], 1.0 / einv, rtol=1e-4)
        np.testing.assert_allclose(crit.pit[i], f / einv, rtol=1e-4)


def test_cpo_failure_flag_on_diffuse_latent():
    design = csr_matrix(np.eye(1)); offset = np.zeros(1); y = np.array([0.0])
    fit = _Fit(np.array([0.0]), np.array([[4.0]]))  # v=4 >> sigma^2=0.25 -> 1/p heavy tail
    crit = _model_criteria(design, offset, y, [(1.0, fit, CompiledGaussian(0.5))], n_nodes=64)
    assert crit.cpo_failures >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/optimization/test_inla_criteria.py -q`
Expected: FAIL (`_model_criteria`/`ModelCriteria` absent).

- [ ] **Step 3: Implement `ModelCriteria`**

Add to `src/pylgm/inference/result.py` (near the other result types), reusing `_readonly_array`:

```python
@dataclass(frozen=True, init=False)
class ModelCriteria:
    dic: float
    dic_effective_parameters: float
    waic: float
    waic_effective_parameters: float
    _cpo: np.ndarray = field(repr=False)
    _pit: np.ndarray = field(repr=False)
    cpo_failures: int
    log_cpo_sum: float

    def __init__(self, dic, dic_effective_parameters, waic, waic_effective_parameters,
                 cpo, pit, cpo_failures, log_cpo_sum):
        object.__setattr__(self, "dic", float(dic))
        object.__setattr__(self, "dic_effective_parameters", float(dic_effective_parameters))
        object.__setattr__(self, "waic", float(waic))
        object.__setattr__(self, "waic_effective_parameters", float(waic_effective_parameters))
        object.__setattr__(self, "_cpo", _readonly_array(cpo))
        object.__setattr__(self, "_pit", _readonly_array(pit))
        object.__setattr__(self, "cpo_failures", int(cpo_failures))
        object.__setattr__(self, "log_cpo_sum", float(log_cpo_sum))

    @property
    def cpo(self) -> np.ndarray:
        return _readonly_array(self._cpo)

    @property
    def pit(self) -> np.ndarray:
        return _readonly_array(self._pit)
```

Export `ModelCriteria` from `src/pylgm/inference/__init__.py` and `__all__`. Add a small immutability test to `tests/inference/test_result.py` (constructing a `ModelCriteria` and asserting `cpo`/`pit` are read-only copies).

- [ ] **Step 4: Implement `_model_criteria`**

Add to `src/pylgm/optimization/inla.py`:

```python
from numpy.polynomial.hermite import hermgauss
from pylgm.inference.result import ModelCriteria


def _model_criteria(design, offset, y, grid, *, n_nodes=21, cpo_failure_threshold=0.5):
    dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, dtype=float)
    offset = np.asarray(offset, dtype=float)
    y = np.asarray(y, dtype=float)
    nodes, gh = hermgauss(n_nodes)
    gh = gh / np.sqrt(np.pi)
    n = y.size

    pd_sum = np.zeros(n)     # Σ_k w_k E[p]
    elog = np.zeros(n)       # Σ_k w_k E[log p]
    elog2 = np.zeros(n)      # Σ_k w_k E[(log p)^2]
    einv = np.zeros(n)       # Σ_k w_k E[1/p]
    ecdf_over_p = np.zeros(n)
    mbar = np.zeros(n)
    max_inv = np.zeros(n)    # largest single (k,j) contribution to einv
    points = list(grid)

    for weight, fit, likelihood in points:
        m = offset + np.asarray(dense @ fit.mean).reshape(-1)
        v = np.clip(np.einsum("ij,jk,ik->i", dense, np.asarray(fit.covariance), dense), 0.0, None)
        mbar += weight * m
        eta = m[:, None] + np.sqrt(2.0 * v)[:, None] * nodes[None, :]     # (n, n_nodes)
        logp = np.stack([likelihood.pointwise_log_density(eta[:, j], y) for j in range(n_nodes)], axis=1)
        cdf = np.stack([likelihood.cdf(eta[:, j], y) for j in range(n_nodes)], axis=1)
        p = np.exp(logp); inv = np.exp(-logp)
        pd_sum += weight * (gh * p).sum(axis=1)
        elog += weight * (gh * logp).sum(axis=1)
        elog2 += weight * (gh * logp ** 2).sum(axis=1)
        einv += weight * (gh * inv).sum(axis=1)
        ecdf_over_p += weight * (gh * cdf * inv).sum(axis=1)
        max_inv = np.maximum(max_inv, (weight * gh[None, :] * inv).max(axis=1))

    lppd = np.log(pd_sum)
    var_logp = np.clip(elog2 - elog ** 2, 0.0, None)
    waic = float(-2.0 * np.sum(lppd - var_logp))
    p_waic = float(var_logp.sum())

    d_bar = float(-2.0 * elog.sum())
    d_mean = 0.0
    for weight, _, likelihood in points:
        d_mean += weight * float(likelihood.pointwise_log_density(mbar, y).sum())
    d_mean = -2.0 * d_mean
    p_d = d_bar - d_mean
    dic = d_bar + p_d

    cpo = 1.0 / einv
    pit = ecdf_over_p / einv
    cpo_failures = int(np.sum(max_inv > cpo_failure_threshold * einv))
    log_cpo_sum = float(np.sum(np.log(cpo)))

    for name, value in (("waic", waic), ("dic", dic), ("cpo", cpo), ("pit", pit)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA criteria produced non-finite {name}")

    return ModelCriteria(dic, p_d, waic, p_waic, cpo, pit, cpo_failures, log_cpo_sum)
```

Note: `D(m̄)` uses the posterior-mean predictor evaluated under each grid point's likelihood, grid-weighted — for a single grid point it reduces to `−2 Σ_i pointwise_log_density(m̄_i, y_i)`, matching the anchor. `NumericalError` is already imported in `inla.py`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src pytest tests/optimization/test_inla_criteria.py tests/inference/test_result.py -q`
Expected: PASS (the single-grid closed forms and the fine-node CPO/PIT reference match).

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/inference/result.py src/pylgm/inference/__init__.py src/pylgm/optimization/inla.py tests/optimization/test_inla_criteria.py tests/inference/test_result.py
git commit -m "feat: compute DIC, WAIC, CPO, and PIT criteria"
```

---

### Task 3: Attach Criteria to the INLA Result

**Files:**
- Modify: `src/pylgm/optimization/inla.py` (wire `_model_criteria` into `integrate_inla`)
- Modify: `src/pylgm/inference/result.py` (`INLAResult.criteria`)
- Modify: `src/pylgm/model.py` (`_rebuild_result` carries `criteria`)
- Modify: `tests/optimization/test_inla.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `_model_criteria`, `ModelCriteria`.
- Produces: `INLAResult(..., criteria: ModelCriteria)` with a `criteria` property; `integrate_inla` computes and attaches it.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_model.py
def test_inla_result_exposes_model_criteria_both_engines():
    import numpy as np, pandas as pd
    from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter
    from pylgm.inference.result import ModelCriteria

    rng = np.random.default_rng(0)
    regions = [f"r{i}" for i in range(30)]
    effects = {r: rng.normal() for r in regions}
    rows = [{"region": r, "t": t, "y": effects[r] + rng.normal()}
            for t in range(6) for r in regions]
    frame = pd.DataFrame(rows)
    gauss = LGM("y", Gaussian(1.0),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    result = gauss.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    crit = result.criteria
    assert isinstance(crit, ModelCriteria)
    assert np.isfinite(crit.dic) and np.isfinite(crit.waic)
    n_obs = len(frame)
    assert crit.cpo.shape == (n_obs,) and crit.pit.shape == (n_obs,)
    assert np.all((crit.pit >= 0) & (crit.pit <= 1))

    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    pois = LGM("y", Poisson(),
               Fixed("1") + IID("region", index="region",
                                precision=Hyperparameter("p", initial=1.0)),
               panel=("region",), time="t")
    pres = pois.fit(counts, engine="laplace", hyperparameters="integrate")
    assert np.isfinite(pres.criteria.waic)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_model.py -q`
Expected: FAIL (`INLAResult` has no `criteria`).

- [ ] **Step 3: Add `criteria` to `INLAResult`**

Add a required keyword `criteria: ModelCriteria` to `INLAResult.__init__` (after `hyperparameter_marginals`), validate it is a `ModelCriteria`, store it, and add a `criteria` property returning it. Import `ModelCriteria` is same-module.

- [ ] **Step 4: Wire into `integrate_inla`**

In `src/pylgm/optimization/inla.py`, change `evaluate` to also return the compiled model:

```python
        return s_value, conditional, theta, compiled
```

Update the unpacking of `points`/`kept` to the 4-tuple `(u, s, cond, theta, compiled)` (adjust the comprehensions and the accumulation loop's unpacking accordingly — the accumulation loop ignores `compiled`). After the mixture accumulation, build the criteria grid and compute:

```python
    observed = kept[0][4].observed
    design_obs = kept[0][4].design[observed]
    offset_obs = kept[0][4].offset[observed]
    y_obs = kept[0][4].y[observed]
    criteria_grid = [(w, cond, compiled.likelihood)
                     for (_, _, cond, _, compiled), w in zip(kept, weights, strict=True)]
    criteria = _model_criteria(design_obs, offset_obs, y_obs, criteria_grid)
```

Pass `criteria=criteria` into the `INLAResult(...)` construction.

- [ ] **Step 5: Carry `criteria` in `_rebuild_result`**

In `src/pylgm/model.py`, the `INLAResult` branch of `_rebuild_result` must pass `criteria=result.criteria` (it is not row-dependent — do not reorder it).

- [ ] **Step 6: Run tests**

Run: `PYTHONPATH=src pytest tests/test_model.py tests/optimization/test_inla.py -q`
Expected: PASS.

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/optimization/inla.py src/pylgm/inference/result.py src/pylgm/model.py tests/optimization/test_inla.py tests/test_model.py
git commit -m "feat: attach model criteria to the INLA result"
```

---

### Task 4: Example, Docs, and Roadmap

**Files:**
- Create: `examples/inla_criteria/data.csv`, `examples/inla_criteria/run.py`, `examples/inla_criteria/README.md`
- Modify: `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`
- Modify: `tests/test_package.py`

- [ ] **Step 1: Write the example smoke test**

```python
# append to tests/test_package.py
def test_inla_criteria_example_reports_dic_waic():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/inla_criteria/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "waic" in completed.stdout.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_package.py -q`
Expected: FAIL (example missing).

- [ ] **Step 3: Create the example**

`examples/inla_criteria/data.csv` — a region panel with a genuine per-region effect and enough groups for an interior posterior (mirror the informative `examples/inla` data). `run.py` — fit with `hyperparameters="integrate"` and print `result.criteria.dic`, `.waic`, `.waic_effective_parameters`, `.cpo_failures`, `.log_cpo_sum`, and a couple of `result.criteria.pit` values. `README.md` — how to run; that DIC/WAIC are model-comparison criteria (lower is better), CPO/PIT are leave-one-out diagnostics (PIT ≈ Uniform(0,1) indicates good calibration), and CPO uses the harmonic identity with a reliability flag (`cpo_failures`).

- [ ] **Step 4: Update docs**

README: add a "Model-assessment criteria" subsection under INLA — `result.criteria` carries DIC, WAIC (with effective-parameter counts), per-observation CPO and PIT, `cpo_failures`, and `log_cpo_sum`, always computed for an integrated fit, for both engines; PIT for discrete responses is the non-randomized `P(Y ≤ y)`; CPO is the leave-one-out harmonic identity with a reliability flag.

Arch spec: mark model-assessment criteria (3b) SHIPPED; keep richer latent strategies (3c: simplified/full Laplace), spatial effects, and HMC-Laplace deferred; note randomized discrete PIT and criteria for non-integrated fits as follow-ups.

- [ ] **Step 5: Run tests, lint, and the example**

Run: `PYTHONPATH=src pytest -q`
Expected: full suite PASS.

Run: `PYTHONPATH=src ruff check src tests`
Expected: PASS.

Run: `PYTHONPATH=src python examples/inla_criteria/run.py`
Expected: exit 0, prints DIC/WAIC and a few PIT values.

- [ ] **Step 6: Commit**

```bash
git add examples/inla_criteria README.md docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md tests/test_package.py
git commit -m "docs: publish INLA model-assessment criteria"
```

---

## Self-Review Checklist

- Task 1 (likelihood methods) → Task 2 (ModelCriteria + helper) → Task 3 (wire into driver + result) → Task 4 (docs). Linear.
- Every spec section maps to a task: likelihood methods (T1), the four criteria + ModelCriteria + failure flag (T2), driver/result wiring both engines (T3), docs/example/roadmap (T4).
- Correctness anchors: pointwise-density-sums-to-log-likelihood and CDF-vs-scipy (T1); single-grid Gaussian DIC/WAIC closed forms + fine-node CPO/PIT reference + failure flag (T2); end-to-end both engines (T3).
- No new dependencies; `optimize`/plug-in paths and other result types unchanged; criteria not row-reordered; Pandas/Spark contracts preserved.
- Deferred (unimplemented): randomized discrete PIT, criteria for non-integrated fits, richer latent strategies (3c), spatial, HMC.

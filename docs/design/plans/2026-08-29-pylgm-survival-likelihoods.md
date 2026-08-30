# Survival Likelihoods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `WeibullSurv` and `ExponentialSurv` proportional-hazards survival likelihoods (right-censoring + optional left-truncation) on the existing Laplace engine.

**Architecture:** Survival PH has η-derivatives `δ−A`, `A`, `−A` (with `A=(tᵃ−vᵃ)·exp(η)`), structurally identical to Poisson, so the damped-Newton Laplace engine fits it unchanged. The only new plumbing is generalizing the per-row auxiliary-data hook (`for_observations`) from a single trials vector to a named mapping, so a family can bind two per-row vectors (event indicator + entry time). Shape `α` is a `phi`-style hyperparameter estimated over the existing marginal.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.special`), pytest. No new dependencies.

## Global Constraints

- No new runtime dependency (NumPy/SciPy only). Copied verbatim from spec.
- Git identity **must** be `Ardea00` on every commit (`git config user.name` = `Ardea00`). Never commit as iongroup/AndreaPanozzo.
- Parameterization is R-INLA `weibullsurv` variant 0 (proportional hazards): `h(t)=α·t^(α−1)·exp(η)`, `H(t)=t^α·exp(η)`, `S(t)=exp(−H(t))`.
- Unified log-likelihood per row: `log Lᵢ = δᵢ·(log α + (α−1)·log tᵢ + ηᵢ) − (tᵢ^α − vᵢ^α)·exp(ηᵢ)`.
- Contract violations raise `DataContractError` (bad values) or `CompilationError` (missing columns), matching the existing GLM families.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit. One behaviour per test.

---

### Task 1: Generalize `for_observations` to a named mapping (pure refactor)

Change the auxiliary-column hook from a single vector to a `Mapping[str, np.ndarray]` so a likelihood can bind more than one per-row vector. `Binomial` migrates onto it with **no behaviour change**. This is the enabling refactor; no survival code yet.

**Files:**
- Modify: `src/pylgm/likelihoods.py` (`_CompiledLikelihood.for_observations`, `CompiledBinomial.for_observations`)
- Modify: `src/pylgm/compiler.py` (`_binomial_trials` → `_likelihood_columns`; the two call sites at ~424-429 and ~894-897; add `_slice_aux` helper)
- Test: `tests/test_modeling_vocabulary.py` (update existing Binomial call sites), `tests/inference/test_laplace.py` (Binomial tests unaffected — verify green)

**Interfaces:**
- Produces: `_CompiledLikelihood.for_observations(self, aux: Mapping[str, np.ndarray] | None) -> _CompiledLikelihood` (default no-op). `CompiledBinomial.for_observations` reads `aux["trials"]`. Compiler helpers `_likelihood_columns(model, frame) -> dict | None` and `_slice_aux(aux, observed) -> dict | None`.

- [ ] **Step 1: Update the failing Binomial unit test to the new mapping signature**

In `tests/test_modeling_vocabulary.py`, change the three `for_observations` call sites in `test_binomial_glm_pieces_and_bernoulli_reduction`, `test_binomial_density_and_cdf_match_scipy`, and `test_binomial_rejects_out_of_support_response`:

```python
# was: .for_observations(n)  /  .for_observations(np.ones(4))  /  .for_observations(np.array([10.0, 8.0]))
lk = Binomial("n").materialize({}).for_observations({"trials": n})
# ...
lk1 = CompiledBinomial().for_observations({"trials": np.ones(4)})
# ...
lk = Binomial("n").materialize({}).for_observations({"trials": np.array([10.0, 8.0])})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_modeling_vocabulary.py -k binomial -v`
Expected: FAIL — `CompiledBinomial.for_observations` still expects a positional array, so `aux["trials"]` indexing or the `CompiledBinomial(trials)` construction breaks.

- [ ] **Step 3: Change the hook signature and Binomial to read the mapping**

In `src/pylgm/likelihoods.py`, replace the base hook:

```python
class _CompiledLikelihood:
    """Shared per-observation binding hook for compiled likelihoods.

    Default is a no-op: one compiled likelihood serves both the observed-row fit
    calls and the all-row prediction call unchanged. Families needing per-row
    data (Binomial trials; survival event/entry) override this to bind the
    vectors named in ``aux``.
    """

    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "_CompiledLikelihood":
        return self
```

And `CompiledBinomial.for_observations`:

```python
    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "CompiledBinomial":
        if aux is None:
            return self
        return CompiledBinomial(aux["trials"])
```

- [ ] **Step 4: Rename `_binomial_trials` → `_likelihood_columns` and add `_slice_aux` in the compiler**

In `src/pylgm/compiler.py`, replace `_binomial_trials` (lines ~271-285) with:

```python
def _likelihood_columns(model: "LGM", frame: object) -> "dict | None":
    """Extract and validate the per-row auxiliary data a likelihood binds.

    Returns ``{"trials": ...}`` for Binomial and ``None`` for every family that
    binds no per-row data (its ``for_observations`` hook is then a no-op).
    """
    like = model.likelihood
    if isinstance(like, Binomial):
        column = like.trials
        if column not in frame.columns:
            raise DataContractError(f"trials column not found: {column!r}")
        trials = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(trials)) or np.any(trials < 1.0) or np.any(trials != np.floor(trials)):
            raise CompilationError("binomial trials column must be positive integers")
        return {"trials": trials}
    return None


def _slice_aux(aux: "dict | None", observed: "np.ndarray") -> "dict | None":
    """Slice each bound vector to the observed rows (None entries stay None)."""
    if aux is None:
        return None
    return {k: (v[observed] if v is not None else None) for k, v in aux.items()}
```

- [ ] **Step 5: Update the two compiler call sites**

In `compile_lgm` (lines ~424-429), replace:

```python
        observed = panel.observed
        aux = _likelihood_columns(model, frame)
        compiled_likelihood.for_observations(_slice_aux(aux, observed)).validate_response(y[observed])
        compiled_likelihood = compiled_likelihood.for_observations(aux)
```

In `compile_gaussian_family` (lines ~894-897), replace:

```python
        aux = _likelihood_columns(model, frame)

        def factory(resolved: dict, likelihood: object = likelihood, aux=aux) -> object:
            return likelihood.materialize(resolved).for_observations(aux)
```

- [ ] **Step 6: Run the Binomial + Laplace suites to verify green**

Run: `python -m pytest tests/test_modeling_vocabulary.py -k binomial tests/inference/test_laplace.py -k binomial -v`
Expected: PASS (behaviour unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/pylgm/likelihoods.py src/pylgm/compiler.py tests/test_modeling_vocabulary.py
git commit -m "refactor(likelihoods): generalize for_observations to a named aux mapping"
```

---

### Task 2: Weibull/Exponential compiled likelihoods (likelihood math, unit-tested)

Add the spec classes and the single compiled class (exponential = Weibull with `α=1`). No compiler wiring yet — everything is tested by materializing and binding directly.

**Files:**
- Modify: `src/pylgm/likelihoods.py` (add `CompiledWeibullSurv`, `WeibullSurv`, `ExponentialSurv`)
- Test: `tests/test_modeling_vocabulary.py` (new survival tests)

**Interfaces:**
- Consumes: `_CompiledLikelihood.for_observations(aux)` from Task 1; `_resolve_phi`, `_positive_real`, `LogLink`, `Hyperparameter`, `DataContractError` (already imported in the module).
- Produces:
  - `CompiledWeibullSurv(alpha: float, event=None, entry=None)` with `.link` (LogLink), `.gradient/.working_weights/.third_derivative/.log_likelihood/.pointwise_log_density/.response_mean/.response_prediction/.cdf/.validate_response`, and `.for_observations(aux)` reading `aux["event"]`, `aux["entry"]`.
  - `WeibullSurv(event: str, shape: float | Hyperparameter = 1.0, entry: str | None = None)` with `.materialize(values) -> CompiledWeibullSurv`.
  - `ExponentialSurv(event: str, entry: str | None = None)` with `.materialize(values) -> CompiledWeibullSurv` (alpha fixed at 1.0).

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/test_modeling_vocabulary.py`:

```python
def test_weibull_surv_ph_pieces_match_poisson_trick():
    from pylgm import WeibullSurv

    alpha = 1.3
    t = np.array([0.5, 2.0, 1.0])          # follow-up time (the response)
    delta = np.array([1.0, 0.0, 1.0])      # event / right-censoring indicator
    eta = np.array([0.2, -0.4, 0.7])
    lk = WeibullSurv("d", shape=alpha).materialize({}).for_observations(
        {"event": delta, "entry": None}
    )
    A = (t ** alpha) * np.exp(eta)         # entry = 0
    np.testing.assert_allclose(lk.gradient(eta, t), delta - A)
    np.testing.assert_allclose(lk.working_weights(eta, t), A)
    assert np.all(lk.working_weights(eta, t) > 0)
    np.testing.assert_allclose(lk.third_derivative(eta, t), -A)


def test_weibull_surv_left_truncation_subtracts_entry_hazard():
    from pylgm import WeibullSurv

    alpha = 1.0                            # exponential for a clean hand value
    t = np.array([3.0, 4.0])
    v = np.array([1.0, 0.0])
    delta = np.array([1.0, 0.0])
    eta = np.array([0.0, 0.5])
    lk = WeibullSurv("d", shape=alpha).materialize({}).for_observations(
        {"event": delta, "entry": v}
    )
    A = (t - v) * np.exp(eta)             # (t^1 - v^1) exp(eta)
    expected = delta * (np.log(alpha) + (alpha - 1.0) * np.log(t) + eta) - A
    np.testing.assert_allclose(lk.pointwise_log_density(eta, t), expected)
    np.testing.assert_allclose(lk.gradient(eta, t), delta - A)


def test_exponential_surv_equals_weibull_shape_one():
    from pylgm import ExponentialSurv, WeibullSurv

    t = np.array([0.5, 2.0, 1.5])
    delta = np.array([1.0, 1.0, 0.0])
    eta = np.array([0.1, -0.2, 0.3])
    aux = {"event": delta, "entry": None}
    exp_lk = ExponentialSurv("d").materialize({}).for_observations(aux)
    wei_lk = WeibullSurv("d", shape=1.0).materialize({}).for_observations(aux)
    np.testing.assert_allclose(exp_lk.pointwise_log_density(eta, t),
                               wei_lk.pointwise_log_density(eta, t))
    np.testing.assert_allclose(exp_lk.gradient(eta, t), wei_lk.gradient(eta, t))


def test_weibull_surv_density_and_cdf_match_scipy():
    from scipy.stats import weibull_min
    from pylgm import WeibullSurv

    alpha = 1.7
    t = np.array([0.5, 1.2, 2.0])
    eta = np.array([0.2, -0.3, 0.6])
    delta = np.ones_like(t)               # all events -> full log density = log f(t)
    lk = WeibullSurv("d", shape=alpha).materialize({}).for_observations(
        {"event": delta, "entry": None}
    )
    scale = np.exp(-eta / alpha)          # S(t)=exp(-(t/scale)^alpha)
    np.testing.assert_allclose(lk.pointwise_log_density(eta, t),
                               weibull_min.logpdf(t, alpha, scale=scale), atol=1e-10)
    np.testing.assert_allclose(lk.cdf(eta, t),
                               weibull_min.cdf(t, alpha, scale=scale), atol=1e-10)
    # response_mean is E[T] = scale * Gamma(1 + 1/alpha)
    from scipy.special import gamma as gamma_fn
    np.testing.assert_allclose(lk.response_mean(eta), scale * gamma_fn(1.0 + 1.0 / alpha))


def test_weibull_surv_gradient_third_derivative_finite_difference():
    from pylgm import WeibullSurv

    alpha = 1.4
    t = np.array([0.7, 1.3, 2.1])
    delta = np.array([1.0, 0.0, 1.0])
    eta = np.array([0.2, -0.6, 0.4])
    lk = WeibullSurv("d", shape=alpha).materialize({}).for_observations(
        {"event": delta, "entry": None}
    )
    h = 1e-5
    num_g = (lk.log_likelihood(eta + h, t) - lk.log_likelihood(eta - h, t)) / (2 * h)
    np.testing.assert_allclose(lk.gradient(eta, t).sum(), num_g, atol=1e-5)

    def obs_second(e):
        return (lk.gradient(e + h, t) - lk.gradient(e - h, t)) / (2 * h)

    fd3 = (obs_second(eta + h) - obs_second(eta - h)) / (2 * h)
    np.testing.assert_allclose(lk.third_derivative(eta, t), fd3, atol=1e-3)


def test_survival_rejects_bad_response_and_construction():
    import pytest
    from pylgm import ExponentialSurv, WeibullSurv

    with pytest.raises(DataContractError, match="positive"):
        WeibullSurv("d").materialize({}).validate_response(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="event"):
        WeibullSurv("")                    # empty event column name
    with pytest.raises(ValueError, match="shape"):
        WeibullSurv("d", shape=0.0)        # non-positive fixed shape
    with pytest.raises(ValueError, match="entry"):
        ExponentialSurv("d", entry="")     # empty entry column name
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_modeling_vocabulary.py -k "surv or truncation or exponential" -v`
Expected: FAIL with `ImportError: cannot import name 'WeibullSurv'`.

- [ ] **Step 3: Implement the compiled class and spec classes**

Add to `src/pylgm/likelihoods.py`. First extend the scipy import at the top:

```python
from scipy.special import gammaln, erf, gammaincc, gammainc, betainc, digamma, polygamma, gamma as gamma_fn
```

Then add the compiled likelihood (after `CompiledBeta`):

```python
@dataclass(frozen=True)
class CompiledWeibullSurv(_CompiledLikelihood):
    """Weibull proportional-hazards survival with shape ``alpha`` and a log link.

    ``eta`` is the log relative risk: hazard ``h(t)=alpha*t**(alpha-1)*exp(eta)``,
    cumulative ``H(t)=t**alpha*exp(eta)``, survival ``S(t)=exp(-H(t))``. The event
    indicator ``delta`` and the left-truncation entry time ``v`` are bound per
    observation via ``for_observations``; ``entry=None`` means no truncation
    (``v=0``). Exponential survival is this class with ``alpha=1``.
    """

    alpha: float
    event: object = None          # delta, bound via for_observations
    entry: object = None          # v (left truncation), bound via for_observations
    link: LogLink = field(default_factory=LogLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _positive_real(self.alpha, "alpha"))

    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "CompiledWeibullSurv":
        if aux is None:
            return self
        return CompiledWeibullSurv(self.alpha, aux.get("event"), aux.get("entry"))

    def _delta(self) -> np.ndarray:
        return np.asarray(self.event, dtype=float)

    def _at_risk(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A = (t**alpha - v**alpha) * exp(eta), the at-risk cumulative hazard."""
        eta = np.asarray(eta, dtype=float)
        t = np.asarray(y, dtype=float)
        v = np.zeros_like(t) if self.entry is None else np.asarray(self.entry, dtype=float)
        return (t ** self.alpha - v ** self.alpha) * np.exp(eta)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(self.pointwise_log_density(eta, y)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._delta() - self._at_risk(eta, y)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._at_risk(eta, y)          # -d2/deta2 = A > 0 (posterior precision PD)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -self._at_risk(eta, y)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        # E[T] = scale * Gamma(1 + 1/alpha), scale = exp(-eta/alpha)
        return np.exp(-np.asarray(eta, dtype=float) / self.alpha) * gamma_fn(1.0 + 1.0 / self.alpha)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.response_mean(np.asarray(eta_mean, dtype=float))  # point estimate; variance ignored

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        t = np.asarray(y, dtype=float)
        d = self._delta()
        a = self.alpha
        return d * (math.log(a) + (a - 1.0) * np.log(t) + eta) - self._at_risk(eta, y)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        t = np.asarray(y, dtype=float)
        return 1.0 - np.exp(-(t ** self.alpha) * np.exp(eta))    # marginal event CDF (entry ignored)

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
            raise DataContractError("survival response (follow-up time) must be positive")
```

Then the spec classes (after `Beta`):

```python
def _survival_columns(event: object, entry: object) -> None:
    if not isinstance(event, str) or not event:
        raise ValueError("survival event must be a non-empty column name")
    if entry is not None and (not isinstance(entry, str) or not entry):
        raise ValueError("survival entry must be a non-empty column name or None")


@dataclass(frozen=True)
class WeibullSurv:
    """Weibull proportional-hazards survival; ``event`` names the 0/1 indicator column.

    ``shape`` is the Weibull shape alpha (fixed float or optimisable Hyperparameter).
    ``entry`` optionally names a left-truncation (delayed-entry) time column.
    """

    event: str
    shape: float | Hyperparameter = 1.0
    entry: str | None = None

    def __post_init__(self) -> None:
        _survival_columns(self.event, self.entry)
        if isinstance(self.shape, Hyperparameter):
            return
        object.__setattr__(self, "shape", _positive_real(self.shape, "shape"))

    def materialize(self, values: Mapping[str, float]) -> CompiledWeibullSurv:
        return CompiledWeibullSurv(_resolve_phi(self.shape, values))


@dataclass(frozen=True)
class ExponentialSurv:
    """Exponential proportional-hazards survival (Weibull with shape alpha = 1)."""

    event: str
    entry: str | None = None

    def __post_init__(self) -> None:
        _survival_columns(self.event, self.entry)

    def materialize(self, values: Mapping[str, float]) -> CompiledWeibullSurv:
        return CompiledWeibullSurv(1.0)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_modeling_vocabulary.py -k "surv or truncation or exponential" -v`
Expected: PASS (all survival unit tests green).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/likelihoods.py tests/test_modeling_vocabulary.py
git commit -m "feat(likelihoods): Weibull/exponential PH survival compiled likelihoods"
```

---

### Task 3: Compiler wiring — aux extraction, family registration, shape discovery, fixed-shape fit

Wire survival into `compile_lgm` / `compile_gaussian_family`: extract & validate `event`/`entry`, register the families, and discover an estimable `shape` alongside `phi`. Ends with a fixed-shape end-to-end Laplace fit.

**Files:**
- Modify: `src/pylgm/compiler.py` (`_likelihood_columns` survival branch; `_estimable_scalar` helper; the family `isinstance` tuple at ~418; the phi discovery at ~421-422 and ~886-889; imports of `WeibullSurv`, `ExponentialSurv`)
- Test: `tests/inference/test_laplace.py` (new survival fit tests)

**Interfaces:**
- Consumes: `WeibullSurv`, `ExponentialSurv`, `CompiledWeibullSurv` from Task 2; `_likelihood_columns`/`_slice_aux` from Task 1; `_log_bounds`, `Hyperparameter`, `_resolve_phi`.
- Produces: `_estimable_scalar(likelihood) -> Hyperparameter | None`. `compile_lgm` and `compile_gaussian_family` accept `WeibullSurv`/`ExponentialSurv` models.

- [ ] **Step 1: Write the failing fit tests**

Add to `tests/inference/test_laplace.py`:

```python
def test_laplace_exponential_surv_intercept_solves_score_equation():
    # Intercept-only exponential PH: the MLE of eta solves sum(delta) = sum(t*exp(eta)),
    # i.e. eta_hat = log( sum(delta) / sum(t) ). With a near-flat prior the mode matches.
    from pylgm import ExponentialSurv

    frame = pd.DataFrame({
        "t": np.arange(1, 6, dtype=float),
        "y": [1.0, 2.0, 1.5, 3.0, 2.5],          # follow-up times
        "d": [1.0, 0.0, 1.0, 1.0, 0.0],          # events
    })
    model = LGM("y", ExponentialSurv("d"), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    expected = np.log(frame["d"].sum() / frame["y"].sum())
    np.testing.assert_allclose(result.mean, [expected], atol=1e-6)
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_weibull_surv_fixed_shape_recovers_slope():
    # Fixed-shape Weibull PH: recover a planted slope from simulated event times.
    from pylgm import WeibullSurv

    rng = np.random.default_rng(0)
    n, alpha, b0, b1 = 6000, 1.5, 0.3, -0.8
    x = rng.normal(size=n)
    eta = b0 + b1 * x
    scale = np.exp(-eta / alpha)                  # S(t)=exp(-(t/scale)^alpha)
    t = rng.weibull(alpha, size=n) * scale        # all observed (no censoring)
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "x": x, "d": np.ones(n)})
    model = LGM("y", WeibullSurv("d", shape=alpha), Fixed("1 + x"), time="t")
    result = model.fit(frame, engine="laplace")
    idx = {lab: i for i, lab in enumerate(result.labels)}
    assert result.mean[idx["fixed:x"]] == pytest.approx(b1, abs=0.05)


def test_survival_missing_event_column_raises():
    from pylgm import ExponentialSurv
    from pylgm.exceptions import DataContractError

    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})   # no "d" column
    model = LGM("y", ExponentialSurv("d"), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="event column not found"):
        compile_lgm(model, _panel(frame))


def test_survival_entry_after_time_raises():
    from pylgm import WeibullSurv
    from pylgm.exceptions import DataContractError

    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0], "d": [1.0, 1.0], "v": [0.5, 3.0]})
    model = LGM("y", WeibullSurv("d", entry="v"), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="entry"):
        compile_lgm(model, _panel(frame))
```

Check the top of `tests/inference/test_laplace.py` imports `fit_laplace`, `compile_lgm`, `_panel`, `pytest`, `pd`, `np`, `LGM`, `Fixed`; add any missing to the existing import block.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/inference/test_laplace.py -k "surv" -v`
Expected: FAIL — `compile_lgm` raises `CompilationError: unsupported ... likelihood` (survival not in the registered family tuple).

- [ ] **Step 3: Add the survival branch to `_likelihood_columns`**

In `src/pylgm/compiler.py`, inside `_likelihood_columns` (from Task 1), before the final `return None`:

```python
    if isinstance(like, (WeibullSurv, ExponentialSurv)):
        if like.event not in frame.columns:
            raise DataContractError(f"event column not found: {like.event!r}")
        event = frame[like.event].to_numpy(dtype=float)
        if not np.all(np.isin(event, (0.0, 1.0))):
            raise DataContractError("survival event indicator must be 0 or 1")
        aux: dict = {"event": event, "entry": None}
        if like.entry is not None:
            if like.entry not in frame.columns:
                raise DataContractError(f"entry column not found: {like.entry!r}")
            entry = frame[like.entry].to_numpy(dtype=float)
            t = frame[model.response].to_numpy(dtype=float)
            if not np.all(np.isfinite(entry)) or np.any(entry < 0.0) or np.any(entry >= t):
                raise DataContractError("survival entry time must satisfy 0 <= entry < time")
            aux["entry"] = entry
        return aux
```

- [ ] **Step 4: Add `_estimable_scalar` and register the families**

In `src/pylgm/compiler.py`, add the helper near `_likelihood_columns`:

```python
def _estimable_scalar(likelihood: object) -> "Hyperparameter | None":
    """The likelihood's optimisable scalar (dispersion phi or Weibull shape), if any."""
    for attr in ("phi", "shape"):
        value = getattr(likelihood, attr, None)
        if isinstance(value, Hyperparameter):
            return value
    return None
```

Add the imports (top of file, with the other likelihood imports):

```python
from pylgm.likelihoods import (
    # ... existing names ...
    WeibullSurv,
    ExponentialSurv,
)
```

In `compile_lgm`, extend the family tuple (~line 418) and swap the phi discovery (~421-422):

```python
    if isinstance(model.likelihood, (Poisson, Bernoulli, Binomial, NegativeBinomial,
                                     Gamma, Beta, WeibullSurv, ExponentialSurv)):
        scalar = _estimable_scalar(model.likelihood)
        values = {scalar.name: scalar.initial} if scalar is not None else {}
```

In `compile_gaussian_family`'s non-Gaussian branch (~886-889), swap the phi discovery:

```python
        likelihood = model.likelihood
        scalar = _estimable_scalar(likelihood)
        if scalar is not None:
            parameter_names.append(scalar.name)
            parameter_bounds[scalar.name] = _log_bounds(scalar)
```

(Leave the `aux = _likelihood_columns(...)` + `factory` lines from Task 1 unchanged.)

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/inference/test_laplace.py -k "surv" -v`
Expected: PASS (fixed-shape fit, score equation, and both validation errors).

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/compiler.py tests/inference/test_laplace.py
git commit -m "feat(compiler): wire Weibull/exponential survival (aux columns + shape discovery)"
```

---

### Task 4: Shape estimation (EB/optimize) + left-truncation end-to-end

Verify the Weibull shape `α` flows through the factory and is estimated over the marginal, and that a left-truncated model fits end to end.

**Files:**
- Test: `tests/inference/test_laplace.py` (new tests only — no source change expected; if a test fails it reveals a Task 3 gap to fix)

**Interfaces:**
- Consumes: everything from Task 3; `Hyperparameter` from `pylgm`.

- [ ] **Step 1: Write the failing recovery tests**

Add to `tests/inference/test_laplace.py`:

```python
def test_empirical_bayes_recovers_planted_weibull_shape():
    from pylgm import Hyperparameter, WeibullSurv

    rng = np.random.default_rng(1)
    n, alpha_true, b0 = 6000, 1.8, 0.5
    scale = np.exp(-b0 / alpha_true)
    t = rng.weibull(alpha_true, size=n) * scale
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "d": np.ones(n)})
    model = LGM(
        "y",
        WeibullSurv("d", shape=Hyperparameter("alpha", initial=1.0, transform="log")),
        Fixed("1", prior_precision=1e-8),
        time="t",
    )
    result = model.fit(frame, engine="laplace", hyperparameters="optimize")
    assert result.hyperparameters["alpha"] == pytest.approx(alpha_true, rel=0.1)


def test_weibull_surv_left_truncation_fits_end_to_end():
    # Delayed entry: rows enter the risk set at v>0. Fit must run and the slope
    # estimate must be finite and close to the planted value.
    from pylgm import WeibullSurv

    rng = np.random.default_rng(2)
    n, alpha, b1 = 5000, 1.2, -0.6
    x = rng.normal(size=n)
    eta = b1 * x
    scale = np.exp(-eta / alpha)
    t = rng.weibull(alpha, size=n) * scale
    v = np.minimum(0.2 * rng.random(n), 0.99 * t)      # entry strictly before t
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "x": x, "d": np.ones(n), "v": v})
    model = LGM("y", WeibullSurv("d", shape=alpha, entry="v"), Fixed("1 + x"), time="t")
    result = model.fit(frame, engine="laplace")
    idx = {lab: i for i, lab in enumerate(result.labels)}
    assert np.isfinite(result.mean[idx["fixed:x"]])
    assert result.mean[idx["fixed:x"]] == pytest.approx(b1, abs=0.08)
```

- [ ] **Step 2: Run to verify (fail first if Task 3 has a gap, else pass)**

Run: `python -m pytest tests/inference/test_laplace.py -k "weibull_shape or left_truncation_fits" -v`
Expected: PASS. If `alpha` is not estimated (KeyError on `result.hyperparameters["alpha"]`), the `_estimable_scalar` discovery in Task 3 is not reaching the factory — fix there, re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/inference/test_laplace.py
git commit -m "test(laplace): Weibull shape EB recovery + left-truncation end-to-end"
```

---

### Task 5: Exports + YAML frontend

Export the new families from the package root and add `weibullsurv`/`exponentialsurv` to the declarative config.

**Files:**
- Modify: `src/pylgm/__init__.py` (import + `__all__`)
- Modify: `src/pylgm/config/model.py` (`_LikelihoodModelConfig` fields/validators; `_build_likelihood`)
- Test: `tests/config/test_model.py` (YAML round-trip), `tests/test_package.py` (export presence)

**Interfaces:**
- Consumes: `WeibullSurv`, `ExponentialSurv` from Task 2.
- Produces: `pylgm.WeibullSurv`, `pylgm.ExponentialSurv`; YAML `likelihood: {family: weibullsurv, event: ..., shape: ..., entry: ...}` and `{family: exponentialsurv, event: ..., entry: ...}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/config/test_model.py` (follow the file's existing `load_model`/YAML pattern — inspect a nearby test for the exact fixture shape):

```python
def test_weibullsurv_config_builds_model():
    from pylgm.config import load_model
    from pylgm.likelihoods import WeibullSurv

    text = """
    response: y
    time: t
    panel: [region]
    likelihood: {family: weibullsurv, event: d, shape: 1.5, entry: v}
    effects:
      - {name: b, type: iid, index: region, precision: 1.0}
    """
    model = load_model(text)
    assert isinstance(model.likelihood, WeibullSurv)
    assert model.likelihood.event == "d"
    assert model.likelihood.shape == 1.5
    assert model.likelihood.entry == "v"


def test_exponentialsurv_config_rejects_shape():
    import pytest
    from pylgm.config import load_model

    text = """
    response: y
    time: t
    panel: [region]
    likelihood: {family: exponentialsurv, event: d, shape: 2.0}
    effects:
      - {name: b, type: iid, index: region, precision: 1.0}
    """
    with pytest.raises(ValueError, match="shape"):
        load_model(text)
```

Add to `tests/test_package.py` inside `test_general_lgm_api_is_exported_without_removing_legacy_api`, extend the `expected` set with `"WeibullSurv"` and `"ExponentialSurv"`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/config/test_model.py -k surv tests/test_package.py -k exported -v`
Expected: FAIL — `weibullsurv` is not an allowed `family` literal / `WeibullSurv` not exported.

- [ ] **Step 3: Export from the package root**

In `src/pylgm/__init__.py`, add to the `from pylgm.likelihoods import (...)` block: `WeibullSurv,` and `ExponentialSurv,`; add `"WeibullSurv",` and `"ExponentialSurv",` to `__all__`.

- [ ] **Step 4: Extend the declarative config**

In `src/pylgm/config/model.py`, update `_LikelihoodModelConfig`:

```python
    family: Literal["gaussian", "poisson", "bernoulli", "binomial", "nbinomial",
                    "gamma", "beta", "weibullsurv", "exponentialsurv"]
    # ... existing sigma / phi / trials fields ...
    event: str | None = None          # survival event/censoring indicator column
    entry: str | None = None          # survival left-truncation entry-time column
    shape: FinitePositiveFloat | None = None   # weibullsurv shape (fixed in YAML)
```

Add to `_fields_match_family` (before `return self`):

```python
        survival = self.family in ("weibullsurv", "exponentialsurv")
        if survival and not self.event:
            raise ValueError(f"event is required for likelihood: {{family: {self.family}}}")
        if not survival and self.event is not None:
            raise ValueError(f"event is not a valid field for likelihood: {{family: {self.family}}}")
        if not survival and self.entry is not None:
            raise ValueError(f"entry is not a valid field for likelihood: {{family: {self.family}}}")
        if self.family != "weibullsurv" and self.shape is not None:
            raise ValueError(f"shape is not a valid field for likelihood: {{family: {self.family}}}")
```

Extend `_build_likelihood`:

```python
    from pylgm.likelihoods import (
        # ... existing imports ...
        WeibullSurv,
        ExponentialSurv,
    )
    # ... existing branches ...
    if config.family == "weibullsurv":
        shape = 1.0 if config.shape is None else config.shape
        return WeibullSurv(config.event, shape=shape, entry=config.entry)
    if config.family == "exponentialsurv":
        return ExponentialSurv(config.event, entry=config.entry)
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/config/test_model.py -k surv tests/test_package.py -k exported -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pylgm/__init__.py src/pylgm/config/model.py tests/config/test_model.py tests/test_package.py
git commit -m "feat(config): export survival families and add weibullsurv/exponentialsurv YAML types"
```

---

### Task 6: Prediction uses the fitted shape

`build_prediction_context` captures the likelihood materialized at the shape's `.initial`. On the optimise/integrate paths that is only the starting guess, and survival's `response_prediction` (E[T]) depends on `α`. Substitute the fitted shape, mirroring the Gaussian-sigma fixup.

**Files:**
- Modify: `src/pylgm/model.py` (`_context_with_fitted_likelihood`)
- Test: `tests/inference/test_laplace.py` (prediction test)

**Interfaces:**
- Consumes: `_context_with_fitted_likelihood(context, result, model)` (existing); `WeibullSurv`, `Hyperparameter`.

- [ ] **Step 1: Write the failing test**

Add to `tests/inference/test_laplace.py`:

```python
def test_weibull_surv_predict_uses_fitted_shape():
    # Fit with an estimated shape, predict on new rows: the predicted E[T] must use
    # the FITTED alpha, not the Hyperparameter's initial guess.
    from pylgm import Hyperparameter, WeibullSurv
    from scipy.special import gamma as gamma_fn

    rng = np.random.default_rng(3)
    n, alpha_true, b0 = 4000, 1.9, 0.4
    scale = np.exp(-b0 / alpha_true)
    t = rng.weibull(alpha_true, size=n) * scale
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "d": np.ones(n)})
    model = LGM(
        "y",
        WeibullSurv("d", shape=Hyperparameter("alpha", initial=1.0, transform="log")),
        Fixed("1", prior_precision=1e-8),
        time="t",
    )
    result = model.fit(frame, engine="laplace", hyperparameters="optimize")
    alpha_hat = result.hyperparameters["alpha"]
    new = pd.DataFrame({"t": [n], "y": [1.0], "d": [1.0]})
    pred = result.predict(new)
    eta_hat = float(result.mean[0])
    expected = np.exp(-eta_hat / alpha_hat) * gamma_fn(1.0 + 1.0 / alpha_hat)
    # would differ by >5% if the initial alpha=1.0 were used instead of alpha_hat
    np.testing.assert_allclose(pred["fitted_mean"].to_numpy(), [expected], rtol=1e-3)
```

Confirm the predict output column name against a nearby predict test in the file (e.g. the Binomial `predicts_counts` test) and adjust `pred["fitted_mean"]` if that test uses a different accessor.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/inference/test_laplace.py -k predict_uses_fitted_shape -v`
Expected: FAIL — predicted mean uses the initial `α=1.0`, not `α_hat`.

- [ ] **Step 3: Generalize the fitted-likelihood fixup**

In `src/pylgm/model.py`, `_context_with_fitted_likelihood`, replace the early `if not isinstance(model.likelihood, Gaussian): return context` structure so survival is handled too. After the existing Gaussian block (which ends `return context`), add before the final `return context`:

```python
    from pylgm.likelihoods import WeibullSurv

    if isinstance(model.likelihood, WeibullSurv) and isinstance(model.likelihood.shape, Hyperparameter):
        name = model.likelihood.shape.name
        marginals = getattr(result, "hyperparameter_marginals", None)
        if callable(marginals):
            table = marginals() or {}
            if name in table:
                fitted_shape = float(table[name].mean[0])
                return replace(context, likelihood=model.likelihood.materialize({name: fitted_shape}))
        fitted = getattr(result, "hyperparameters", None) or {}
        if name in fitted:
            return replace(context, likelihood=model.likelihood.materialize({name: float(fitted[name])}))
    return context
```

Restructure the leading guard so it only short-circuits the Gaussian-specific block: change `if not isinstance(model.likelihood, Gaussian): return context` to guard just the Gaussian branch (e.g. wrap the Gaussian-sigma logic in `if isinstance(model.likelihood, Gaussian) and isinstance(model.likelihood.sigma, Hyperparameter):`), then fall through to the survival block above. Keep the `CompiledGaussian`/`Gaussian`/`Hyperparameter` imports; the materialize path needs no new import beyond `WeibullSurv`.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/inference/test_laplace.py -k "predict_uses_fitted_shape or predict" -v`
Expected: PASS (survival prediction uses `α_hat`; existing predict tests still green).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/model.py tests/inference/test_laplace.py
git commit -m "feat(predict): survival prediction uses the fitted Weibull shape"
```

---

### Task 7: Economics duration example, docs, roadmap, full-suite regression

Ship a runnable econometric duration example, document the families, move survival from Next to Shipped, and confirm the whole suite is green.

**Files:**
- Create: `examples/survival_duration/` (a `run.py`-style script + `README.md`, matching the layout of an existing example such as `examples/count_regression/`)
- Modify: `docs/` (a survival section — extend `docs/likelihoods.md` if it exists, else add `docs/survival.md` and link it from `docs/index.md` and `mkdocs.yml`)
- Modify: `docs/roadmap.md` (move survival from **Next** to **Shipped in 0.5**; renumber remaining Next items)
- Test: `tests/test_package.py` (example smoke test, mirroring an existing example test)

**Interfaces:**
- Consumes: the full public API from Tasks 2–6.

- [ ] **Step 1: Inspect an existing example + its smoke test**

Read `examples/count_regression/README.md`, its script, and the matching `tests/test_package.py::test_..._example_...` to copy the structure exactly (script entry point, printed keys, how the smoke test imports and asserts).

- [ ] **Step 2: Write the failing example smoke test**

Add to `tests/test_package.py`, mirroring an existing example test:

```python
def test_survival_duration_example_reports_hazard_ratio():
    import runpy
    ns = runpy.run_path("examples/survival_duration/run.py")
    result = ns["main"]()
    assert "hazard_ratio" in result
    assert "shape" in result
    assert np.isfinite(result["hazard_ratio"])
```

Adjust the import mechanism (`runpy.run_path` vs module import) to match the sibling example tests in the file.

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_package.py -k survival_duration -v`
Expected: FAIL — the example does not exist yet.

- [ ] **Step 4: Write the example**

Create `examples/survival_duration/run.py`: simulate (or load) a small unemployment-duration or time-to-default panel with right-censoring, fit `WeibullSurv("event", shape=Hyperparameter("alpha", ...))` with a `Fixed` covariate and an `IID` frailty over individuals, print/return a dict with `hazard_ratio = exp(beta_x)`, `shape`, and a fitted-vs-observed summary. Keep it deterministic (seeded). Add `examples/survival_duration/README.md` explaining the PH reading, censoring column, and the frailty term.

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_package.py -k survival_duration -v`
Expected: PASS.

- [ ] **Step 6: Write the docs**

Add a survival section documenting: the PH parameterization, the `event`/`entry` data contract, `shape` estimation, the frailty-via-`IID` idiom, and the excluded-features note (interval-censoring via Bernoulli/person-period, time-varying covariates via episode-splitting). Link it from `docs/index.md` and `mkdocs.yml` if a new page.

- [ ] **Step 7: Update the roadmap**

In `docs/roadmap.md`, add a **Survival likelihoods** bullet under **Shipped in 0.5** (Weibull/exponential PH, right-censoring + left-truncation, shape estimation, frailty via IID), and remove item 1 from **Next**, renumbering the rest.

- [ ] **Step 8: Full-suite regression**

Run: `python -m pytest -q` (expect the documented pre-existing env failures only: Spark JVM errors + console-entrypoint PATH + any OneDrive `MoveFileW` temp-lock flake — no new failures attributable to this slice). Run `ruff check .` and confirm clean.

- [ ] **Step 9: Commit**

```bash
git add examples/survival_duration docs/ tests/test_package.py mkdocs.yml
git commit -m "docs(survival): econ duration example, guide, and roadmap; survival shipped in 0.5"
```

---

## Self-Review

**Spec coverage:**
- Weibull PH likelihood → Task 2 ✓; Exponential (α=1) → Task 2 ✓
- Right-censoring (event indicator) → Task 2 (math) + Task 3 (extraction/validation) ✓
- Left-truncation (entry) → Task 2 (math) + Task 3 (validation) + Task 4 (end-to-end) ✓
- `for_observations(aux)` generalization + Binomial migration → Task 1 ✓
- Shape estimation via EB/MAP-II/INLA → Task 3 (discovery) + Task 4 (recovery) ✓
- `response_mean`/`cdf`/`response_prediction`/`validate_response` → Task 2 ✓
- Fitted-shape prediction → Task 6 ✓
- YAML `weibullsurv`/`exponentialsurv` → Task 5 ✓
- Example + docs + roadmap → Task 7 ✓
- Excluded items (interval-censoring, time-varying covariates, competing risks) → not implemented, documented in Task 7 ✓

**Placeholder scan:** All code steps show real code; commands have expected output. The only deliberately open bits are Task 7's example content and docs prose (inherently authored, with explicit required contents), and "inspect a sibling test" steps that pin call-site details the plan cannot know without reading — each names the exact file to read.

**Type consistency:** `for_observations(aux: Mapping)` is defined in Task 1 and consumed identically in Tasks 2–3. `_estimable_scalar` returns `Hyperparameter | None`, consumed in both compiler branches (Task 3). `CompiledWeibullSurv(alpha, event, entry)` signature is consistent across Tasks 2/3/6. `materialize` uses `_resolve_phi(self.shape, values)` (Task 2), matching the `_estimable_scalar` name discovery (Task 3). Response accessor column name (`fitted_mean`) is flagged in Task 6 to verify against a sibling test.

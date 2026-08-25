# pyLGM

General-purpose latent Gaussian modeling foundations for Python.

Version 0.3 is a bounded foundation release: its stable inference engines are
exact Gaussian and Laplace (fixed plug-in hyperparameters), and its
declarative runtime accepts Pandas and Spark data. It provides `LGM`,
Gaussian/Poisson/Bernoulli likelihoods, fixed/IID/RW1/RW2 effects, fit-row
and out-of-sample (`result.predict(new_data)`) predictions, posterior
marginals, and linear combinations. Schema-v1 and schema-v2 forecasting
interfaces remain available for compatibility.

See the [approved Latte-parity and PySpark architecture](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md)
for the explicitly labeled target state and staged roadmap. It is not the 0.3
runtime contract.

## Runnable declarative 0.3 example

The [general LGM example](examples/general_lgm/README.md) fits the same Pandas
data through both a Python declaration and the standalone YAML frontend:

```bash
PYTHONPATH=src python examples/general_lgm/run.py
```

The committed YAML includes the required numeric `sigma` and every structured
effect `precision`.

## 0.3 scope and explicit deferrals

The 0.3 exact and Laplace engines condition on plain numeric hyperparameter
values. A declarative `Hyperparameter` contributes its validated `.initial`
value during compilation and, since the ["Empirical Bayes"](#empirical-bayes)
slice, is estimated by type-II ML rather than merely held fixed. Since the
same slice's MAP-II follow-up, a `Hyperparameter`'s `prior` — when declared —
is also consumed: its native-scale log density penalizes the marginal
likelihood, turning the estimate into a MAP-II point estimate. The prior is
still not copied into the materialized IR, and it is applied on the
hyperparameter's native scale with no Jacobian correction for `transform`.
`PCPrecision` and `GaussianPrior` drive both point estimation and, under
["INLA integration"](#inla-integration), the integrated (marginal) posterior.

Parameterized IR metadata, sparse production engines, and HMC are deferred to
later slices. The stationary `AR1` effect has shipped — see
["AR1 effect"](#ar1-effect) below. The spatial (CAR)
effect family is now complete for the dense reference regime: `Besag`
(intrinsic CAR/ICAR), `ProperCAR` (proper CAR, with its spatial-dependence
parameter `ρ` either fixed or estimated), and `BYM2` (the structured +
unstructured convolution, with its mixing parameter `φ` either fixed or
estimated) have all shipped (see ["Besag / intrinsic CAR (ICAR) spatial effect"](#besag--intrinsic-car-icar-spatial-effect),
["Proper CAR spatial effect"](#proper-car-spatial-effect),
["BYM2 spatial effect"](#bym2-spatial-effect), and
["Bounded hyperparameters"](#bounded-hyperparameters) below). Remaining
spatial follow-ups (the augmented 2n representation, config-file spatial
effect types, graceful isolated-node handling, and sparse/large-graph
scaling) are tracked in the
[spatial roadmap](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md#4-spatial-effects).
Optional PySpark input is now
supported as a data boundary (see below). Pandas fit-row predictions in 0.3
cover the rows supplied to `LGM.fit`, in the caller's original row order.
`result.predict(new_data)` extends this to out-of-sample rows by reusing the
fitted latent posterior (see ["Predicting new rows"](#predicting-new-rows)
below); `new_data` itself must always be a Pandas DataFrame, including for
results fitted from a Spark DataFrame.

## Non-Gaussian likelihoods (Laplace)

`Poisson()` (canonical log link) and `Bernoulli()` (canonical logit link) fit
through a dedicated `engine="laplace"` Newton-Raphson mode-finder; the Gaussian
likelihood keeps using `engine="exact_gaussian"` and `LGM.fit` rejects the
mismatched engine/likelihood combination:

```python
from pylgm import Fixed, IID, LGM, Poisson

model = LGM(
    response="claims",
    likelihood=Poisson(),
    predictor=Fixed("1 + x") + IID("region_effect", index="region", precision=2.0),
    panel=("region",),
    time="time",
    offset="log_exposure",
)
result = model.fit(frame, engine="laplace")
```

The mode-finder stops on an absolute gradient threshold, and falls back to the
**Newton decrement** `½·∇fᵀH⁻¹∇f` when that threshold is out of reach. The
gradient's scale follows the data — for a Poisson model its components are of
order the counts — so on many well-behaved problems the iteration reaches the
mode and then stalls just above the threshold, having already found it. The
decrement is scale-invariant and closely estimates the remaining suboptimality,
so such a point is accepted instead of raising. `result.diagnostics` records
`newton_decrement` when that fallback carried a fit.

**This changed non-Gaussian results in 0.3.x.** Before the fallback, a stalled
inner fit raised — which aborted an entire `hyperparameters="integrate"` run,
and, less visibly, made `hyperparameters="optimize"` treat that hyperparameter
as infeasible and search around it. Both now see the converged fit, so
previously-recorded Poisson and Bernoulli estimates can move, occasionally a
lot: in our test matrix roughly one non-Gaussian fit in ten changed, the
largest by 4.9 nats of log marginal likelihood, and in one case an AR1 `rho`
posterior mean corrected from −0.9998 to +0.91 on data simulated with
`rho = 0.8`. The new answers are the trustworthy ones; re-fit rather than
comparing against stored output from an earlier version.

`LGM.offset` names an optional column added directly to the linear predictor
`eta` before the link (e.g. a log-exposure column for a Poisson rate model);
it is not itself modeled. The YAML frontend exposes the same vocabulary:
`likelihood: {family: poisson}` / `{family: bernoulli}` and a top-level
`offset: <col>`. `family: gaussian` still requires `sigma`; `poisson` and
`bernoulli` forbid it, since they have no free dispersion parameter.

The Laplace engine conditions on its effect precisions by default; declaring
one as a `Hyperparameter` instead estimates it by type-II ML (see
["Empirical Bayes"](#empirical-bayes) below) — INLA-style numerical
integration over hyperparameters remains deferred to a later slice.
`result.fitted_mean` is the response-scale prediction, and its meaning
depends on the link: for the Poisson **log link** it is the exact lognormal
expectation `exp(mean + variance / 2)` of the linear predictor; for the
Bernoulli **logit link** there is no closed form, so it is a documented
**point estimate** `logit^-1(mean)` that ignores the linear-predictor
variance. `result.predictive_variance` is the linear-predictor (eta)
posterior variance and excludes response-scale observation noise, for every
result type including `GaussianResult` — see
["The `predictive_variance` convention"](#the-predictive_variance-convention)
below. A runnable example lives at
[`examples/count_glm/README.md`](examples/count_glm/README.md), including the
Spark data-boundary path (Spark only collects and canonicalizes data; the
Laplace fit itself still runs on the driver, same as `exact_gaussian`).

## The `predictive_variance` convention

`predictive_variance` is the **linear-predictor** posterior variance
`Var(eta)` for every result type (`GaussianResult`, `LaplaceResult`,
`INLAResult`) and for `result.predict(new_data)` — it never includes
response-scale observation noise.

**This changed for Gaussian models.** Before this version,
`GaussianResult.predictive_variance` (and the Gaussian path of `predict()`)
included the observation variance `sigma^2` — i.e. it reported
`Var(eta) + sigma^2` — while the Laplace path already reported `Var(eta)`
alone, so the same attribute name meant two different quantities depending on
engine. Reported Gaussian predictive variances (and predictive standard
deviations) recorded from before this version are too large by `sigma^2`;
**re-fit rather than reuse stored numbers**. Poisson and Bernoulli values are
unchanged — they never included an observation variance, since those
likelihoods have no such parameter.

`GaussianResult` gains **`observation_variance`** (the Gaussian likelihood's
`sigma^2`), and `INLAResult` carries it too when the conditional engine is
Gaussian — there it is `E[sigma^2]` over the hyperparameter grid, so an
integrated fit reconstructs the same way. It is `None` for a Laplace
conditional engine, which has no observation variance. The old, response-scale
value is recovered exactly:

```python
response_scale_variance = result.predictive_variance + result.observation_variance
```

`predict()`'s returned `Prediction` does not carry `observation_variance`
itself (it is a property of the fitted `GaussianResult`, constant across new
rows), so the same reconstruction reads `prediction.predictive_variance +
result.observation_variance`.

Why this convention: it is the only one definable for Poisson/Bernoulli
likelihoods, which have no observation-variance parameter at all; it matches
R-INLA's `summary.linear.predictor`; and it is what the Laplace path and
every non-Gaussian model already did. `fitted_mean`, `mean`,
`log_marginal_likelihood`, and the model-assessment criteria (DIC/WAIC/CPO/PIT)
are all unaffected by this change.

**Structural note.** The three result types (`GaussianResult`, `LaplaceResult`,
`INLAResult`) now share a private `_BaseResult` carrying their common
read-only properties, delegating methods, and constructor validation; they
remain siblings — none inherits another, and no public API was renamed,
removed, or moved. The three latent-marginal types (`GaussianMarginals`,
`SkewNormalMarginals`, `TabulatedMarginals`) satisfy a documented
`LatentMarginals` protocol, exported from `pylgm.inference`, formalizing the
`mean`/`variance`/`std`/`quantile` surface they already shared.

`_BaseResult` is now a dataclass declaring its eleven shared fields once
instead of once per result type. **This changed `repr()` output**: printed
fields now list the shared ones first, then each result type's own fields,
rather than interleaved in the previous per-type order. No value, attribute
name, or other behaviour changed — this affects only what `repr(result)`
prints.

## Empirical Bayes

Declaring an effect's precision (or a Gaussian likelihood's `sigma`) as a
`Hyperparameter` — instead of passing a plain number — makes `LGM.fit`
estimate it by **type-II maximum likelihood**: it optimizes the marginal
likelihood over the declared hyperparameter(s), starting from `.initial` and
respecting any declared `lower`/`upper` bounds, for both the
`exact_gaussian` and `laplace` engines:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM

model = LGM(
    response="y",
    likelihood=Gaussian(0.5),
    predictor=Fixed("1")
    + IID("region", index="region", precision=Hyperparameter("region_precision", initial=1.0)),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="exact_gaussian")
result.hyperparameters["region_precision"]  # the type-II ML estimate
result.diagnostics["empirical_bayes_converged"]
```

The fitted value is exposed on `result.hyperparameters`; a model with no
declared `Hyperparameter` leaves `result.hyperparameters` as `None`.

When a declared `Hyperparameter` also carries a `prior` (e.g. `PCPrecision`,
`GaussianPrior`), `LGM.fit` estimates it by **MAP-II** instead of pure
type-II ML: the same marginal-likelihood objective is penalized by the
prior's log density evaluated on the hyperparameter's native scale (no
Jacobian correction for `transform`), for both the `exact_gaussian` and
`laplace` engines. A prior-free `Hyperparameter` stays pure type-II ML.
`result.diagnostics["hyperparameter_penalized"]` records whether any
declared hyperparameter was penalized this way:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, PCPrecision

model = LGM(
    response="y",
    likelihood=Gaussian(0.5),
    predictor=Fixed("1")
    + IID(
        "region", index="region",
        precision=Hyperparameter(
            "region_precision", initial=1.0,
            prior=PCPrecision(upper_sd=1.0, alpha=0.01),
        ),
    ),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="exact_gaussian")
result.hyperparameters["region_precision"]  # the MAP-II estimate
result.diagnostics["hyperparameter_penalized"]  # True
```

This is still a point estimate, not a marginal over the hyperparameter —
full posterior *integration* over hyperparameters is the separate
`hyperparameters="integrate"` mode described below. YAML declaration of
`Hyperparameter`s is not yet supported — this is a Python-API-only
capability. Runnable examples live at
[`examples/empirical_bayes/README.md`](examples/empirical_bayes/README.md)
(prior-free, type-II ML) and
[`examples/map_ii/README.md`](examples/map_ii/README.md) (prior'd, MAP-II).

## INLA integration

Passing `hyperparameters="integrate"` to `LGM.fit` (instead of the default
`"optimize"`) resolves every declared `Hyperparameter` by **INLA-style grid
quadrature** rather than plugging in the type-II/MAP-II optimum: it builds a
grid of hyperparameter values in log space around the empirical-Bayes mode
(using a finite-difference Hessian to orient and scale it), weights each
grid point by its marginal likelihood and the log-space Jacobian, and
averages the conditional fit at each point. This is supported for both the
`exact_gaussian` and `laplace` engines, and for a declared `prior` (MAP-II
penalty) just as with `hyperparameters="optimize"`:

```python
result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
result.hyperparameter_marginals()["region_precision"]  # GaussianMarginals: mean, sd
result.latent_marginals("region")  # integrated latent marginals (wider than plug-in)
result.log_marginal_likelihood  # integrated marginal likelihood
result.diagnostics["inla_grid_points"]  # how many grid points were kept
```

Compared to `hyperparameters="optimize"`, this gives:

- a populated `result.hyperparameter_marginals()` (mean/sd per declared
  hyperparameter) instead of only a point estimate on `result.hyperparameters`;
- latent marginals that account for hyperparameter uncertainty, not just the
  latent-field uncertainty conditional on the mode;
- an *integrated* marginal likelihood rather than the conditional one at a
  single hyperparameter value.

This mode uses the **Gaussian latent strategy** at every grid point (the
exact-Gaussian posterior, or the Laplace approximation for non-Gaussian
likelihoods) — it does not yet implement INLA's simplified- or full-Laplace
latent correction. It is practical for a handful of declared
hyperparameters: the grid grows as `(2 * radius + 1) ** d`, and a
`max_grid_points` guard raises `OptimizationError` before building a grid
that would be too large. Model-assessment criteria (DIC, WAIC, CPO, PIT) are
computed for every integrated fit — see
["Model-assessment criteria"](#model-assessment-criteria) below.

**Limitation:** grid integration assumes the hyperparameter posterior is
reasonably interior and near-Gaussian in log space. When it is instead
boundary-dominated — an under-informative model, or the empirical-Bayes mode
pinned to a declared `lower`/`upper` bound — the grid degenerates toward a
single point and the integration quietly reduces to the plug-in fit.
`result.diagnostics["inla_active_bounds"]` surfaces which hyperparameters (if
any) hit a bound at the mode, so this degradation is visible rather than
silent. A runnable example lives at
[`examples/inla/README.md`](examples/inla/README.md).

### Simplified-Laplace latent marginals

By default (`latent_strategy="gaussian"`), latent marginals — from either
`hyperparameters="optimize"` or `hyperparameters="integrate"` — are Gaussian
summaries (mean/sd) of the conditional latent posterior, which is exact for
a Gaussian likelihood but only a local approximation for non-Gaussian ones
(Poisson, Bernoulli): it discards any skewness the true conditional marginal
has.

Passing `latent_strategy="simplified_laplace"` (which requires
`hyperparameters="integrate"`) instead applies **INLA's simplified-Laplace
correction** at every hyperparameter grid point: a third-order (skewness)
correction to the Gaussian latent approximation, fit as a skew-normal per
grid point and mixed across the grid the same way the Gaussian marginals
are. The result is a `SkewNormalMarginals` — `mean`, `std`, `skewness`, and
`quantile(p)` — instead of a `GaussianMarginals`:

```python
result = model.fit(
    frame, engine="laplace", hyperparameters="integrate",
    latent_strategy="simplified_laplace",
)
region = result.latent_marginals("region")  # SkewNormalMarginals
region.mean       # per-component mean
region.std        # per-component sd
region.skewness   # per-component skewness (0 per grid point for a Gaussian
                  # likelihood, but the grid-mixed value is generally nonzero)
region.quantile(0.025)  # per-component quantile (asymmetric interval when skewed)
```

This is faithful to the simplified-Laplace approximation of Rue, Martino &
Chopin (2009), *Approximate Bayesian Inference for Latent Gaussian Models by
Using Integrated Nested Laplace Approximations*, §3.2.3 and Appendix B — the
skew-normal fit uses the same location/scale/skewness matching described
there. For a Gaussian likelihood the per-grid-point SLA correction is zero
(each conditional marginal is exactly Gaussian, since the Gaussian
conditional posterior has no third-derivative correction to apply); the
integrated marginal still carries hyperparameter-mixture skewness, as any
INLA grid mixture does, so it is not identical to the symmetric
Gaussian-strategy summary. `latent_strategy` defaults to `"gaussian"`, so
existing fits are unaffected unless it is passed explicitly.

**Scope:** this is the *simplified*-Laplace correction to the latent
marginals, not INLA's full-Laplace strategy (which additionally corrects the
denominator via a Laplace approximation re-fit per latent component) —
see the next section for that. Validation is against a brute-force
true-marginal oracle for small, tractable models, not against R-INLA output;
R-INLA parity fixtures are future work. A runnable example lives at
[`examples/inla_sla/README.md`](examples/inla_sla/README.md).

### Full-Laplace latent marginals

Passing `latent_strategy="laplace"` (which, like `"simplified_laplace"`,
requires `hyperparameters="integrate"`) applies INLA's **full-Laplace**
correction: in addition to the numerator (skewness) correction the
simplified strategy applies, it re-fits a Laplace approximation to the
conditional posterior's denominator at each point of each latent
component's own grid, following Rue, Martino & Chopin (2009), §3.2.2 (the
numerator/denominator ratio, RMC eqs 12-13) and eq. 16-17 (the cubic-spline
correction, with constant tail extrapolation beyond the outer abscissae, used
when the raw ratio is not itself a well-behaved density). The result is a `TabulatedMarginals` — a
numerically tabulated density per component, mixed across the hyperparameter
grid the same way the other two strategies are — instead of a closed-form
`GaussianMarginals`/`SkewNormalMarginals`:

```python
result = model.fit(
    frame, engine="laplace", hyperparameters="integrate",
    latent_strategy="laplace",
)
region = result.latent_marginals("region")  # TabulatedMarginals
region.mean       # per-component mean (numerical integral over the tabulated grid)
region.std        # per-component sd
region.skewness   # per-component skewness (third numerical moment)
region.quantile(0.025)  # per-component quantile (asymmetric interval when skewed)
region.pdf(x)     # tabulated density evaluated (interpolated) at x
region.cdf(x)     # tabulated CDF evaluated at x
```

This is the **most accurate, and most expensive**, of the three latent
strategies: it is exact per hyperparameter grid point for a Gaussian
likelihood (the Laplace approximation to a Gaussian conditional posterior is
exact, so the denominator correction reduces to the identity), and for
non-Gaussian likelihoods it captures the full third- and higher-order shape
the simplified strategy only partially corrects for. It does not re-optimize
per component: the conditional-mean configuration at each Gauss-Hermite
abscissa is the RMC eq. 13 closed-form (a rank-one update of the modal
configuration), so the added cost is one dense (p-1)x(p-1) determinant per
abscissa per latent component per hyperparameter grid point, on top of what
the simplified strategy already does.

**Scope:** `latent_strategy="laplace"` supports **unconstrained models
only** — a model with any `RW`/intrinsic (constrained) effect raises
`UnsupportedEngineError` at fit time; use `latent_strategy="gaussian"` or
`"simplified_laplace"` for models with constrained effects. Validation is
against a brute-force true-marginal oracle (Gaussian vs. simplified-Laplace
vs. full-Laplace vs. numerically integrated truth) for small, tractable
models, not against R-INLA output; R-INLA parity fixtures remain future
work, as for the simplified strategy. `latent_strategy` still defaults to
`"gaussian"`, so existing fits are unaffected unless `"laplace"` is passed
explicitly. A runnable example, including a side-by-side contrast of all
three strategies' skewness and 95% intervals on the same fit, lives at
[`examples/inla_full_laplace/README.md`](examples/inla_full_laplace/README.md).

**The three-tier latent-strategy ladder**, in increasing accuracy and cost:
`"gaussian"` (default; exact for a Gaussian likelihood, a local
approximation otherwise) -> `"simplified_laplace"` (adds a numerator
skewness correction) -> `"laplace"` (adds the denominator correction too,
unconstrained models only).

## Model-assessment criteria

Every integrated fit (`hyperparameters="integrate"`, either the
`exact_gaussian` or `laplace` engine) carries `result.criteria`, a
`ModelCriteria` populated with DIC, WAIC, per-observation CPO and PIT, and
two summary fields:

```python
result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")

criteria = result.criteria
criteria.dic                        # deviance information criterion
criteria.dic_effective_parameters   # DIC's effective-parameter count
criteria.waic                       # widely applicable information criterion
criteria.waic_effective_parameters  # WAIC's effective-parameter count
criteria.cpo                        # per-observation CPO array
criteria.pit                        # per-observation PIT array
criteria.cpo_failures               # count of unreliable per-obs CPO estimates
criteria.log_cpo_sum                # sum of log CPO, a leave-one-out log score
```

- **DIC** and **WAIC** are model-*comparison* criteria — lower is better
  when comparing fits of the same data — each paired with an
  effective-parameter count as a complexity penalty.
- **CPO** (conditional predictive ordinate) is the leave-one-out predictive
  density for each observation, computed via the harmonic-mean identity
  rather than by refitting once per held-out point. That identity can be
  numerically unreliable for individual observations, so `cpo_failures`
  counts how many per-observation CPO values hit the reliability guard;
  `log_cpo_sum` sums the per-observation log CPO into a single leave-one-out
  log predictive score.
- **PIT** (probability integral transform) is the predictive CDF evaluated
  at each observed response. PIT values that resemble a Uniform(0, 1) sample
  indicate good calibration; clustering near 0 or 1 indicates miscalibration.
  For discrete-response likelihoods, `pit` is the non-randomized `P(Y <= y)`
  rather than the randomized PIT, so it is conservative even for a correctly
  specified model.
- For discrete-response likelihoods (Poisson, Bernoulli), the harmonic-mean
  CPO/PIT estimator integrates a heavy/divergent-tailed `E[1/p]` against the
  Gaussian eta-approximation, so the CPO/PIT reliability flag (`cpo_failures`)
  is dependent on the quadrature node count and is only a *lower bound* on
  unreliable observations, not a guarantee that the rest are trustworthy;
  better tail handling (randomized PIT, robust leave-one-out) is deferred.

Criteria are not yet computed for plug-in fits (`hyperparameters="optimize"`
or the default). A runnable example lives at
[`examples/inla_criteria/README.md`](examples/inla_criteria/README.md).

## Predicting new rows

`result.predict(new_data)` is available on `GaussianResult`, `LaplaceResult`,
and `INLAResult` (i.e. every result produced by `LGM.fit`). It scores rows
that were **not** passed to `fit` by rebuilding their design and reusing the
already-fitted latent posterior — no refit. It returns an immutable
`Prediction` with `predictive_mean`, `predictive_variance`, `fitted_mean`,
and `keys` (`new_data`'s own index), plus `.to_frame()`, a `DataFrame`
indexed like `new_data` with columns `predictive_mean`, `predictive_sd`, and
`fitted_mean`.

`new_data` must carry every column the fitted model reads — each effect's
`index` column, every variable used in a `Fixed` formula, and the `offset`
column when the model declares one — but **not** the response column.

```python
import pandas as pd
from pylgm import Fixed, Gaussian, IID, LGM

frame = pd.DataFrame({
    "claims": [0.5, 1.5, 2.5, 3.5],
    "x": [1.0, 2.0, 3.0, 4.0],
    "region": ["a", "b", "a", "b"],
})
model = LGM(
    response="claims",
    predictor=Fixed("1 + x") + IID("region_effect", index="region", precision=2.0),
    likelihood=Gaussian(sigma=0.5),
)
result = model.fit(frame)

# new_data carries "x" (the Fixed formula variable) and "region" (the IID
# effect's index), but not "claims" (the response).
new_scenarios = pd.DataFrame({"x": [10.0, -3.0], "region": ["b", "a"]})
prediction = result.predict(new_scenarios)
print(prediction.to_frame())
#    predictive_mean  predictive_sd  fitted_mean
# 0         9.499999       1.764396     9.499999
# 1        -3.499999       1.288687    -3.499999
```

`predict` **reuses** the fitted posterior; it cannot create a new latent
column. An index level absent from the fitted model — a region `predict`
never saw at `fit` time — raises `ValueError` naming both the offending block
and the unrecognized level(s), rather than silently falling back to a prior
or a zero contribution. The one exception is a spatial effect: it is keyed on
graph **nodes**, so a node present in the effect's `graph` but not referenced
by any fitted row already has a posterior and predicts without error.

Forecasting genuinely new levels — a future time point, a region never seen
at all — is a different operation, and it already worked before `predict`
existed: include those rows at `fit` time with a `NaN` response. They
contribute no likelihood but still receive a latent column (and, for a
structured effect such as `RW1`, an extrapolated posterior) plus a full
prediction, right alongside the observed rows.

```python
import numpy as np
import pandas as pd
from pylgm import Fixed, Gaussian, IID, LGM, RW1

n_hist = 30
history = pd.DataFrame({
    "y": 1.0 + 0.2 * np.sin(np.arange(n_hist) / 3.0),
    "region": (["a", "b"] * (n_hist // 2))[:n_hist],
    "t": range(n_hist),
})
# These 4 future periods never appeared at fit time. Give them a NaN
# response instead of calling predict() on them.
future = pd.DataFrame({
    "y": [np.nan, np.nan, np.nan, np.nan],
    "region": ["a", "b", "a", "b"],
    "t": [n_hist, n_hist + 1, n_hist + 2, n_hist + 3],
})
frame = pd.concat([history, future], ignore_index=True)

model = LGM(
    response="y",
    predictor=Fixed("1") + IID("region", index="region", precision=2.0)
    + RW1("trend", index="t", precision=2.0),
    likelihood=Gaussian(sigma=0.2),
)
result = model.fit(frame)

# The last 4 rows are the future periods; predictive_mean/_variance already
# cover them, in the caller's row order (same arrays fit() always returns).
print(np.round(result.predictive_mean[-4:], 3))    # [0.957 0.957 0.957 0.957]
print(np.round(result.predictive_variance[-4:], 3))  # [0.557 1.037 1.557 2.037]
```

An `RW1`/`RW2` forecast extrapolates **flat** from the last fitted level —
these examples show it exactly, because the region effect nets out to ~0 by
symmetry — and its variance strictly increases with each additional step
ahead, converging toward growth of `1/τ` per step as the fitted history
grows long relative to the forecast horizon (the finite-sample deviation
comes from the effect's sum-to-zero identifiability constraint, which
couples the whole chain including the unobserved tail).

| need | use |
| --- | --- |
| new covariate values / scenarios on **known** index levels | `result.predict(new_data)` |
| **new** time points, regions, or groups | `NaN`-response rows at `fit` time |

`predictive_variance` and `fitted_mean` follow exactly the same conventions
`predict` mirrors from the fit-row outputs: `predictive_variance` is the
linear-predictor variance `Var(eta)`, for every engine — see
["The `predictive_variance` convention"](#the-predictive_variance-convention)
above for the Gaussian reconstruction of the response-scale value.
`fitted_mean` is identity for Gaussian, the exact
lognormal expectation `exp(μ + σ²_η/2)` for the Poisson log link, and the
documented **point estimate** `logit⁻¹(μ)` (ignoring linear-predictor
variance) for the Bernoulli logit link — see
["Non-Gaussian likelihoods (Laplace)"](#non-gaussian-likelihoods-laplace)
above.

One exception is worth knowing about. Under `hyperparameters="integrate"`
with a **non-linear link**, the integrated `result.fitted_mean` mixes the
*transformed* per-hyperparameter values (`Σₖ wₖ·g(μₖ, σ²ₖ)`), while `predict`
transforms the *integrated* moments (`g(Σₖ wₖμₖ, Var)`). Jensen's inequality
separates the two by an amount that grows with the hyperparameter
uncertainty, so `predict().fitted_mean` is a moment-matched approximation of
`result.fitted_mean` there rather than an exact match (reproducing it exactly
would mean retaining every grid point's latent covariance). `predictive_mean`
and `predictive_variance` are exact, as is `fitted_mean` for the identity link
and for all plug-in and empirical-Bayes fits.

`predict` works on a result fitted from either Pandas or a Spark DataFrame,
but `new_data` itself must always be a Pandas DataFrame — Spark `new_data` is
not supported. **Not shipped**: a prior-based fallback for unseen levels,
predictive quantiles or simulation, response-scale predictive variance for
non-Gaussian links, and an automatic future-frame construction helper (the
`NaN`-response rows above are built by hand).

## AR1 effect

`AR1(name, index, precision=1.0, rho=0.5)` declares a stationary first-order
autoregressive latent effect on the ordered, observed levels of `index` (the
same level-ordering rule as `RW1`/`RW2`; at least two levels are required).
Its precision is `Q = τ/(1−ρ²)·T`, with `T` tridiagonal (unit corners,
`1+ρ²` on the interior diagonal, `−ρ` off it), which inverts to
`Cov[i, j] = (1/τ)·ρ^|i−j|`. **`τ` is the marginal precision** — INLA's
convention — so the marginal variance of every level is exactly `1/τ`,
directly comparable to an `IID` effect's precision or `BYM2`'s `τ`, *not* an
innovation precision. `ρ = 0` collapses `Q` to exactly `τ·I` (independent
levels); as `ρ → 1` the effect approaches the intrinsic `RW1` limit.

```python
import numpy as np
import pandas as pd
from pylgm import AR1, Fixed, Gaussian, LGM

frame = pd.DataFrame({
    "t": range(10),
    "y": [1.0, 1.4, 1.1, 1.6, 1.3, 1.8, 1.5, 2.0, 1.7, 2.2],
})
model = LGM(
    response="y",
    predictor=Fixed("1") + AR1("trend", index="t", precision=2.0, rho=0.7),
    likelihood=Gaussian(sigma=0.3),
)
result = model.fit(frame)
print(np.round(result.latent_marginals("trend").mean, 3))
# [-0.466 -0.244 -0.358 -0.065 -0.166  0.129  0.027  0.322  0.22   0.506]
```

`rho` accepts either a fixed float strictly inside `(-1, 1)` (plug-in, shown
above) or a declared `Hyperparameter` with `transform="logit"`, in which case
ρ is **estimated** by empirical Bayes and, under
`hyperparameters="integrate"`, integrated over jointly with τ by INLA — a
non-logit `Hyperparameter` for `rho` raises `CompilationError`. The interval
is the fixed `(-1, 1)` inset by `1e-6` (no graph-derived bound, unlike
`ProperCAR`'s ρ); a `Hyperparameter`'s own `lower`/`upper` narrow it further if
you set them.

One caveat on that interval: **conditioning degrades as ρ approaches the bound
and as the series grows** — `cond(Q)` runs roughly `2e6·n` at the inset edge, so
a 1000-level series fitted at ρ ≈ 1 retains only about seven significant digits.
Nothing fails, but treat an ρ̂ pinned at the bound on a long series as "the data
want a random walk" and consider `RW1` instead.

```python
from pylgm import AR1, Fixed, Gaussian, Hyperparameter, LGM
from pylgm.priors import PCPrecision

rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
tau = Hyperparameter("trend.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(
    response="y",
    predictor=Fixed("1") + AR1("trend", index="t", precision=tau, rho=rho),
    likelihood=Gaussian(sigma=0.3),
)
eb = model.fit(frame)                                 # empirical Bayes
eb.hyperparameters["trend.rho"]                        # point estimate of rho

post = model.fit(frame, hyperparameters="integrate")   # INLA over (tau, rho)
post.hyperparameter_marginals()["trend.rho"].mean      # rho marginal
```

Unlike `RW1`/`RW2`, `AR1` is proper/full-rank, so it carries **no
sum-to-zero constraint**: it works under all three latent strategies,
including `latent_strategy="laplace"` (full Laplace), where `RW1`/`RW2` are
rejected because they are intrinsic/constrained. `result.predict(new_data)`
works on `AR1` exactly as described in
["Predicting new rows"](#predicting-new-rows) above; an unseen time level
raises the same `ValueError` pointing at the `NaN`-response workflow.

**Regular spacing is assumed.** `AR1` relates *consecutive* ordered levels —
it has no notion of the gap between them — so a missing period (a skipped
month, say) is silently treated as a single step, understating the true
elapsed time and its correlation decay. If a period is absent from the data
but should count as a step, include it as a row with a `NaN` response at
`fit` time: it contributes no likelihood but still creates its latent column
and restores regular spacing, exactly like the future-period rows in
["Predicting new rows"](#predicting-new-rows) above. **Not shipped**:
irregular-spacing support (`ρ^Δt`), a config-file `ar1` effect type,
AR(p)/seasonal effects, and group-wise AR1 (a separate series per panel
unit).

## Besag / intrinsic CAR (ICAR) spatial effect

`Besag(name, index, graph, precision=1.0, scale=True)` declares an intrinsic
conditional-autoregressive (ICAR) spatial latent effect: precision
`τ·(D−W)`, the graph Laplacian of the adjacency, with density proportional to
`exp(−τ/2 · Σ_{i~j}(xᵢ−xⱼ)²)`. This structure is rank-deficient (it is
invariant to adding a constant within a connected component), so the effect
carries one sum-to-zero constraint per connected component of the graph.

`graph` is a neighbour dict `{region: [neighbours, ...]}` keyed by the same
labels as the data's `index` column, or the result of `load_graph_file(path)`,
which parses the R-INLA/latte `.graph` text format into that same dict shape.
The graph's nodes are the effect's domain, so it may include regions absent
from the observed data (e.g. to predict at unobserved regions); every
observed region must be a node in the graph, or `fit` raises a clear error.
A node with zero neighbours is rejected at declaration time — an isolated
region has no ICAR structure to borrow strength from and should be modeled as
an `IID` effect instead — while a multi-node disconnected component (an
"island" group) is supported and gets its own sum-to-zero constraint.

`scale=True` (the default) applies the Sørbye–Rue (2014) scaling
convention: each connected component's structure matrix is rescaled to a
unit geometric-mean marginal variance, so `τ` is comparable across different
graphs and to an `IID` effect's precision on the same scale. Pass
`scale=False` to use the raw, unscaled `D−W` structure instead.

```python
import numpy as np
import pandas as pd

from pylgm import Besag, Fixed, Gaussian, LGM, load_graph_file

# A small connected chain graph over regions "0".."5": i <-> i-1, i <-> i+1.
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5] for i in range(6)}

frame = pd.DataFrame({"region": [str(i) for i in range(6)], "y": np.sin(np.arange(6) / 2.0)})
model = LGM(
    response="y",
    predictor=Fixed("1") + Besag("region", index="region", graph=graph, precision=1.0),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.latent_marginals("region")  # GaussianMarginals over the 6 regions

# Loading a graph from an R-INLA / latte .graph file instead of an inline dict:
graph_from_file = load_graph_file("regions.graph")  # -> {"1": ["2"], "2": ["1", "3"], ...}
```

`precision` accepts a plain number or a declared `Hyperparameter`, and
participates in `hyperparameters="optimize"`/`"integrate"` exactly like
`IID`/`RW1`/`RW2` precisions. `Besag` composes with other effects the same
way (`Fixed(...) + Besag(...) + IID(...)`) and works under both the default
`latent_strategy="gaussian"` and `latent_strategy="simplified_laplace"`.
Because it is a constrained effect (like `RW1`/`RW2`), `latent_strategy="laplace"`
(full Laplace) rejects it with `UnsupportedEngineError`, the same restriction
already documented above for `RW`/intrinsic effects.

**Not yet built:** spatial autocorrelation diagnostics and shapefile/GeoJSON
graph construction. The YAML `ModelConfig` frontend also does not yet have a
`besag` effect type — see the
[spatial roadmap](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md#4-spatial-effects)
for what's next.

## Proper CAR spatial effect

`ProperCAR(name, index, graph, rho, precision=1.0)` declares a proper
conditional-autoregressive spatial latent effect: precision
`Q = τ·(D − ρW)`, where `D` is the degree matrix and `W` the adjacency of
`graph`. Unlike `Besag`/ICAR, this precision is full-rank (proper), so the
effect carries **no sum-to-zero constraint**.

`graph` uses the same neighbour-dict or `load_graph_file(path)` input as
`Besag`. `rho` (ρ) must lie in the open interval `(1/μ_min, 1/μ_max)`, where
`μ` are the eigenvalues of the normalized adjacency, for `D − ρW` to be
positive definite; for a graph with edges the upper bound is `1`. `ρ = 0`
recovers a spatially-independent, degree-weighted precision `τD`. An
out-of-range `rho` raises a `ValueError` naming the valid interval.

`rho` accepts either a **fixed float** (plug-in, shown below) or a declared
`Hyperparameter`, in which case ρ is **estimated** — see
["Estimating ρ"](#estimating-ρ).

```python
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, LGM, ProperCAR

# The same small connected chain graph over regions "0".."5" used above.
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5] for i in range(6)}

frame = pd.DataFrame({"region": [str(i) for i in range(6)], "y": np.sin(np.arange(6) / 2.0)})
model = LGM(
    response="y",
    predictor=Fixed("1") + ProperCAR("region", index="region", graph=graph, rho=0.9),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.latent_marginals("region").mean  # posterior mean over the 6 regions
```

`precision` (τ) accepts a plain number or a declared `Hyperparameter`, and
participates in `hyperparameters="optimize"`/`"integrate"` exactly like the
other structured effects' precisions. Because it is unconstrained,
`ProperCAR` is the **first spatial effect to work under
`latent_strategy="laplace"`** (full Laplace) — `Besag` is rejected there
since it carries a sum-to-zero constraint.

### Estimating ρ

Passing a `Hyperparameter` for `rho` estimates the spatial dependence instead
of fixing it. Declare it with `transform="logit"`: ρ lives on a bounded
interval, so it is inferred on a scaled-logit scale rather than the log scale
used for positive parameters. The interval itself is resolved **from the
graph** at compile time — leave `lower`/`upper` as `None` and the compiler
supplies `(1/μ_min, 1/μ_max)` — and `initial` may be any finite value inside
it, including `0.0`.

```python
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, ProperCAR
from pylgm.priors import PCPrecision

rho = Hyperparameter("region.rho", initial=0.0, transform="logit")
tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(
    response="y",
    predictor=Fixed("1")
    + ProperCAR("region", index="region", graph=graph, rho=rho, precision=tau),
    likelihood=Gaussian(sigma=0.1),
)

eb = model.fit(frame)                                   # empirical Bayes
eb.hyperparameters["region.rho"]                        # point estimate of rho

post = model.fit(frame, hyperparameters="integrate")    # INLA over (tau, rho)
post.hyperparameter_marginals()["region.rho"].mean      # rho marginal
```

ρ and τ are estimated (or integrated over) **jointly**. Under the hood the
effect's precision is no longer a scalar multiple of a fixed matrix, so it is
carried as a parametric block that rebuilds `τ(D − ρW)` at each hyperparameter
value; see ["Bounded hyperparameters"](#bounded-hyperparameters).

**Not yet built:** Sørbye–Rue scaling of the proper-CAR structure itself, and
shapefile/GeoJSON graph construction — see the
[spatial roadmap](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md#4-spatial-effects)
for what's next.

## BYM2 spatial effect

`BYM2(name, index, graph, precision=1.0, phi=0.5)` declares Riebler et al.'s
(2016) BYM2 convolution model: a marginal-precision reparameterization of the
classic BYM structured-plus-unstructured mixture,

```
x = τ^(-1/2) · ( √(1-φ)·v + √φ·u* )
```

with `v ~ N(0, I)` unstructured noise and `u*` the **Sørbye–Rue-scaled**
ICAR component (`Besag`'s scaling, always applied — there is no `scale` flag
on `BYM2`). `τ` is the **marginal** precision of `x` (not the ICAR
precision), and `φ ∈ (0, 1)` is the fraction of the marginal variance
attributed to the spatially structured component: `φ → 0` behaves like plain
IID noise, `φ → 1` like `Besag`/ICAR.

Rather than the classical augmented `2n`-dimensional representation
(structured and unstructured components stacked and reported separately),
this is implemented in the **marginal, `n`-dimensional** parameterization:
`Q = τ·[(1-φ)·I + φ·R*⁻]⁻¹`, built once per graph via an eigendecomposition
of the scaled structure's generalized inverse and then reassembled cheaply at
any `(τ, φ)`. Because this precision is full-rank, `BYM2` carries **no
constraint at all** — unlike `Besag` (one sum-to-zero constraint per
connected component). `graph` takes the same neighbour-dict or
`load_graph_file(path)` input as `Besag`/`ProperCAR`, with the same isolated-
node rejection and disconnected-graph support.

```python
import numpy as np
import pandas as pd

from pylgm import BYM2, Fixed, Gaussian, LGM

# The same small connected chain graph over regions "0".."5" used above.
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5] for i in range(6)}

frame = pd.DataFrame({"region": [str(i) for i in range(6)], "y": np.sin(np.arange(6) / 2.0)})
model = LGM(
    response="y",
    predictor=Fixed("1") + BYM2("region", index="region", graph=graph, precision=1.0, phi=0.5),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.latent_marginals("region").mean  # posterior mean over the 6 regions
```

`precision` (τ) accepts a plain number or a `Hyperparameter`, exactly like
the other structured effects. `phi` accepts either a **fixed float** in
`(0, 1)` (shown above) or a `Hyperparameter(transform="logit")`, in which
case φ is **estimated** by empirical Bayes and, under
`hyperparameters="integrate"`, **integrated over jointly with τ** (INLA), the
same way `ProperCAR`'s ρ is:

```python
from pylgm import BYM2, Fixed, Gaussian, Hyperparameter, LGM, PCBYM2Phi
from pylgm.priors import PCPrecision

# A larger graph with a genuinely spatial signal: phi is weakly identified,
# so a handful of regions cannot distinguish structured from unstructured.
rng = np.random.default_rng(0)
n = 20
big_graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}
signal = np.zeros(n)
for i in range(1, n):
    signal[i] = 0.9 * signal[i - 1] + rng.normal(scale=0.4)
signal -= signal.mean()
big_frame = pd.DataFrame({
    "region": [str(i) for i in range(n)],
    "y": signal + rng.normal(scale=0.3, size=n),
})

phi = Hyperparameter("region.phi", initial=0.5, transform="logit", prior=PCBYM2Phi())
tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(
    response="y",
    predictor=Fixed("1") + BYM2("region", index="region", graph=big_graph, precision=tau, phi=phi),
    likelihood=Gaussian(sigma=0.3),
)

eb = model.fit(big_frame)                                # empirical Bayes
eb.hyperparameters["region.phi"]                         # point estimate of phi

post = model.fit(big_frame, hyperparameters="integrate")  # INLA over (tau, phi)
post.hyperparameter_marginals()["region.phi"].mean        # phi marginal
```

**φ is weakly identified, by nature of the model.** Structured and
unstructured components explain small datasets almost equally well, which is
precisely why Riebler et al. pair BYM2 with an informative PC prior. Two
practical consequences: the profile objective can be *bimodal* (mass near
"pure IID" and near "pure spatial", with a valley between), so empirical Bayes
— a local optimizer — may report whichever mode it descends into; and on small
graphs φ can collapse to its boundary even when the simulating truth was
interior. Keep the PC prior, and treat a boundary φ̂ as "not identified"
rather than as evidence about spatial structure.

**Integration does not rescue a boundary φ̂.** The INLA grid is built around
the empirical-Bayes mode and its curvature, so when φ̂ pins at a boundary the
grid degenerates there too: the reported φ marginal becomes a near point mass
with a spuriously tiny standard deviation (the example above returns a mean of
≈0.99996 with sd ≈2e-8, which is grid degeneracy, not posterior certainty).
Before believing a φ marginal, check `result.diagnostics["inla_grid_points"]`
and `["inla_collapsed"]` — a handful of points clustered at a bound means the
hyperparameter was not identified, whatever the interval width suggests. This
affects φ specifically because it is weakly identified; τ and ρ marginals on
the same fits behave normally.

`PCBYM2Phi(upper=0.5, alpha=2/3)` is Riebler et al.'s PC prior for φ,
calibrated so that `P(φ < upper) = alpha`. Because its distance scale
depends on the eigenvalues of the graph's scaled structure, it is declared
**unbound** and the compiler **binds it to the effect's graph** when a `BYM2`
`phi` hyperparameter references it; calling `.logpdf` on an unbound
`PCBYM2Phi` raises. An `alpha` at or below the graph's attainable floor
`d(upper)/d(1)` raises, naming the achievable range.

Because `BYM2`'s precision is unconstrained, like `ProperCAR` it works under
**all three latent strategies**, including full Laplace
(`latent_strategy="laplace"`) — unlike `Besag`, which full Laplace rejects
for carrying a sum-to-zero constraint.

**Not yet built:** the augmented (2n) representation exposing the structured
component `u*` separately, a config-file `bym2` effect type, graceful
isolated-node handling, and sparse/large-graph scaling — see the
[spatial roadmap](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md#4-spatial-effects)
for what's next.

## Bounded hyperparameters

Hyperparameter inference runs on an unconstrained internal scale chosen per
parameter by its `transform`:

| `transform` | Natural domain | Used for |
| --- | --- | --- |
| `"log"` (default) | `θ > 0` | `sigma`, effect precisions (τ) |
| `"logit"` | a bounded interval `(a, b)` | `ProperCAR` `rho`, `BYM2` `phi` |
| `"identity"` | `θ > 0` | accepted, but **not** wired into inference (treated as `log`) |

Both the empirical-Bayes optimizer and the INLA grid work in this internal
space, and the INLA importance weights carry the corresponding Jacobian
`Σ log|dθ/du|`. For log-scale parameters this is exactly the previous
behaviour, so existing models are unaffected; the abstraction is what makes a
bounded parameter such as ρ estimable at all.

A `"logit"` hyperparameter may declare any finite `initial` (zero or negative
included) and may leave `lower`/`upper` as `None`, letting the effect supply
the interval — which is how `ProperCAR` passes its graph-derived range for ρ.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

See the [approved design](docs/superpowers/specs/2026-07-22-pylgm-design.md).

## Optional PySpark input

`LGM.fit` accepts a Spark DataFrame when the optional extra is installed:

```bash
python -m pip install "pylgm[spark]"
```

Spark is a **data boundary only**: it validates required columns, non-null and
unique `(*panel, time)` keys, projects just the response, canonical keys,
Formulaic fixed-effect variables, and structured-effect indices, orders the rows
canonically, and collects that compact table to the driver. The exact-Gaussian
fit still runs on the driver through the same `CanonicalPanel` compiler and
reference engine as Pandas input; there is no distributed inference and no
separate Spark model language. The original Spark DataFrame is never collected
with `toPandas()`.

Spark input **requires an explicit `LGM.time`** — a Spark DataFrame has no stable
row order, so there is no synthetic row key. For the same reason, Spark
predictive arrays are returned in canonical `(*panel, time)` order (not caller
order), and the result carries immutable `result.prediction_keys` (one row per
prediction) for joining predictions back to the source table:

```python
result = model.fit(spark_df)              # engine="exact_gaussian" by default
keys = result.prediction_keys             # canonical (*panel, time) rows
means = result.predictive_mean            # aligned to keys
```

`model.fit(spark_df, max_driver_rows=100_000)` bounds the driver collection: the
adapter rejects oversized input **before** collecting it. Pass `max_driver_rows=None`
to disable this adapter preflight (the exact-Gaussian dense guards still apply).
Pandas input is unaffected: it keeps caller-row prediction order and no
`prediction_keys`. A runnable example lives at
[`examples/general_lgm/run_spark.py`](examples/general_lgm/README.md).

### On Databricks

The cluster runtime already ships PySpark, so install **plain `pylgm`** — the
`[spark]` extra would pull a second PySpark into the notebook environment and
can shadow the runtime's:

```python
%pip install pylgm
```

Pass the DataFrame straight to `fit` using the notebook's pre-provided `spark`
session. Inference runs on the **driver**, so the collected table must fit in
driver memory — keep `max_driver_rows` sane and size the driver node
accordingly:

```python
sdf = spark.read.table("catalog.schema.panel")
result = model.fit(sdf, max_driver_rows=1_000_000)
keys = result.prediction_keys            # join predictions back to the source table
```

## Internal compiled IR policy

`pylgm.ir` remains importable for engine and advanced integration work, but it
is internal and unstable and is not covered by the top-level compatibility
policy. Code that previously constructed `CompiledLGM(..., sigma=value, ...)`
must migrate to
`CompiledLGM(..., likelihood=CompiledGaussian(value), ...)`, importing
`CompiledGaussian` from `pylgm.likelihoods`. The read-only `compiled.sigma`
accessor remains available for compiled Gaussian models.

## Legacy forecasting utilities

The schema-v1 `Pipeline` and schema-v2 `Experiment` interfaces, including the
`fit` and `compare` commands, remain available for existing forecasting
workflows. `compare` evaluates configured candidates over rolling-origin,
leakage-safe forecast folds. `persistence` is reported as an autoregressive
benchmark but is never selectable, so inspect the benchmark rows in
`metrics.parquet` rather than interpreting selection as evidence that an LGM
beat the benchmark.

```bash
pylgm compare examples/predictive_selection/config.yaml \
  examples/predictive_selection/data.csv \
  --output comparison
```

The [general example](examples/predictive_selection/README.md) is committed and
domain-neutral. The [NIC-shaped configuration](examples/nic_backtest/README.md)
runs against a local Parquet source but does not redistribute any source data.
Legacy empirical-Bayes runs use plug-in optima rather than integrating
hyperparameter uncertainty. Prediction and metric rows carry an explicit
`evaluation_mode` value (`latest` or `vintage`). Expected candidate/fold failures
are exposed as immutable structured records in `ComparisonResult.failures` and
persisted with optimizer counts, numerical failure history, and root-cause
details in `failures.json`.

## Exact Gaussian example

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
```

The example combines fixed effects, an IID region effect, and an intrinsic RW1
time effect. Rows with missing responses are prediction targets. The exact
Gaussian engine conditions on declared effect precisions and observation
standard deviation; it does not integrate their uncertainty.

## Exact Gaussian reference limits

The exact Gaussian engine is a small/medium-model reference implementation. Its dense
posterior covariance requires O(p^2) memory and its dense factorization requires
O(p^3) work for latent dimension `p`. Conservative dimension and estimated-memory
preflights reject larger models before dense conversion. Advanced direct callers may
explicitly opt in with `fit_gaussian(model, allow_large_dense=True)`; `Pipeline`
deliberately remains safe by default. Schema-v2 experiments have the equivalent
strict configuration switch:

```yaml
inference:
  engine: exact_gaussian
  allow_large_dense: true
  hyperparameters: ...
```

The resolved value is recorded in experiment artifacts and is forwarded to every
preflight, optimizer evaluation, and prediction fit. It does not change or weaken
the 4,096-latent-dimension and 512 MiB default guards; it only provides an explicit
opt-in override for schema-v2 experiments.

## Atomic artifact publication

Run directories are published with operating-system no-replace primitives:
`renamex_np` on macOS, exported `renameat2` (or known-architecture syscall fallback)
on Linux, and `MoveFileExW` on Windows. Platforms without one of these atomic
primitives fail with an unsupported-operation error. pyLGM never weakens the
no-overwrite guarantee by falling back to a replacing rename.

## Release installation gate

A release must be verified from a clean environment with the declared build
backend available:

```bash
python -m pip install -e . --no-deps
pylgm --help
```

This repository currently has no CI workflow to extend cleanly. The offline
0.3 review environment also lacks `hatchling`, so a clean build or editable
installation cannot run there without downloading the build backend; that
environment uses `PYTHONPATH=src` only for tests, lint, and CLI smoke checks.
The clean install plus installed-CLI smoke remains a required release gate in a
provisioned, networked build environment.

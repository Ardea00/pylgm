# pyLGM

General-purpose latent Gaussian modeling foundations for Python.

Version 0.3 is a bounded foundation release: its stable inference engines are
exact Gaussian and Laplace (fixed plug-in hyperparameters), and its
declarative runtime accepts Pandas and Spark data. It provides `LGM`,
Gaussian/Poisson/Bernoulli likelihoods, fixed/IID/RW1/RW2 effects, fit-row
predictions, posterior marginals, and linear combinations. Schema-v1 and
schema-v2 forecasting interfaces remain available for compatibility.

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
still not copied into the materialized IR, and there is no Jacobian
correction for `transform`. `PCPrecision` and `GaussianPrior` therefore
already drive point estimation, but not yet integrated (marginal)
hyperparameter inference.

AR1 effects, `result.predict(new_data)`, parameterized IR metadata,
INLA-style hyperparameter marginals, sparse production
engines, and HMC are deferred to later slices. The spatial (CAR) effect
family is landing incrementally: `Besag` (intrinsic CAR/ICAR) has shipped
(see ["Besag / intrinsic CAR (ICAR) spatial effect"](#besag--intrinsic-car-icar-spatial-effect)
below); proper CAR and BYM2 remain deferred. Optional PySpark input is now
supported as a data boundary (see below). Pandas predictions in 0.3 cover
the rows supplied to `LGM.fit`; their arrays follow the caller's original
row order.

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
posterior variance and excludes response-scale observation noise, unlike
`GaussianResult.predictive_variance`, which adds it. A runnable example lives at
[`examples/count_glm/README.md`](examples/count_glm/README.md), including the
Spark data-boundary path (Spark only collects and canonicalizes data; the
Laplace fit itself still runs on the driver, same as `exact_gaussian`).

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

**Not yet built:** proper CAR (a spatial-dependence hyperparameter `ρ`,
unconstrained), BYM2, spatial autocorrelation diagnostics, PC-priors for
spatial hyperparameters, and shapefile/GeoJSON graph construction. The YAML
`ModelConfig` frontend also does not yet have a `besag` effect type — see
the [spatial roadmap](docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md#4-spatial-effects)
for what's next.

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

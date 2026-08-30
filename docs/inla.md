# INLA integration

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

This mode uses the **Gaussian latent strategy** at every grid point by
default (the exact-Gaussian posterior, or the Laplace approximation for
non-Gaussian likelihoods); INLA's simplified- and full-Laplace latent
corrections are available via `latent_strategy` — see
["Simplified-Laplace latent marginals"](#simplified-laplace-latent-marginals)
and ["Full-Laplace latent marginals"](#full-laplace-latent-marginals) below.
It is practical for a handful of declared
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
[`examples/inla/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/inla/README.md).

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
[`examples/inla_sla/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/inla_sla/README.md).

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
[`examples/inla_full_laplace/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/inla_full_laplace/README.md).

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
[`examples/inla_criteria/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/inla_criteria/README.md).


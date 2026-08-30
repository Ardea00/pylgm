# Likelihoods

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
["Empirical Bayes"](empirical-bayes.md#empirical-bayes) below) — INLA-style numerical
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
[`examples/count_glm/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/count_glm/README.md), including the
Spark data-boundary path (Spark only collects and canonicalizes data; the
Laplace fit itself still runs on the driver, same as `exact_gaussian`).

## The GLM families

Beyond `Gaussian`, `Poisson`, and `Bernoulli`, the Laplace engine fits four
more generalized-linear families. All reuse the same Newton mode-finder — they
differ only in their per-observation link and derivatives.

| Family | Link | Response support | Dispersion `phi` | `fitted_mean` |
|--------|------|------------------|------------------|---------------|
| `NegativeBinomial(phi=1.0)` | log | non-negative integers | `Var = mu + mu^2/phi` (NB2); `phi -> inf` is Poisson | `exp(mean + var/2)` |
| `Gamma(phi=1.0)` | log | positive reals | `Var = mu^2/phi`; shape `a = phi` | `exp(mean + var/2)` |
| `Beta(phi=1.0)` | logit | open interval `(0, 1)` | `Beta(mu*phi, (1-mu)*phi)` precision | point estimate `logit^-1(mean)` |
| `Binomial(trials="col")` | logit | integers `0 <= y <= n` | — (no free dispersion) | counts `n * p` |

```python
from pylgm import Beta, Binomial, Fixed, Gamma, LGM, NegativeBinomial

# Overdispersed counts (fixed dispersion phi=2.5)
LGM("y", NegativeBinomial(2.5), Fixed("1 + x"), time="t").fit(frame, engine="laplace")

# Positive-continuous (gamma) and proportions in (0, 1) (beta)
LGM("y", Gamma(3.0), Fixed("1 + x"), time="t").fit(frame, engine="laplace")
LGM("rate", Beta(10.0), Fixed("1 + x"), time="t").fit(frame, engine="laplace")

# Aggregated binomial: `trials` names the per-row count column n; predicts n*p
LGM("successes", Binomial("n"), Fixed("1 + x"), time="t").fit(frame, engine="laplace")
```

**Dispersion `phi`.** For NB/Gamma/Beta, `phi` is a strictly-positive
concentration parameter. Pass a fixed float, or a
[`Hyperparameter`](empirical-bayes.md#empirical-bayes) to estimate it by
type-II ML alongside the effect precisions (`hyperparameters="optimize"`). The
YAML frontend accepts a fixed value only: `likelihood: {family: nbinomial, phi:
2.5}` (phi is optional and defaults to 1.0); `gaussian` still requires `sigma`,
and non-φ families reject `phi`.

**Binomial `trials`.** `Binomial` takes the *name* of a per-row trials column
`n` (not a value). The response `y` is the success count, and both the fitted
values and out-of-sample `predict()` return **counts** `n * p` rather than the
probability `p` — so `predict(new_data)` requires `new_data` to carry its own
trials column, and scores each new row against its own `n`. The `n = 1` case
reduces exactly to `Bernoulli`. In YAML: `likelihood: {family: binomial,
trials: n}` (required for binomial, rejected for every other family).

`response_prediction` for `Beta` and `Binomial` follows the `Bernoulli`
convention — a point estimate that ignores the linear-predictor variance —
while the two log-link families (`NegativeBinomial`, `Gamma`) apply the same
exact lognormal correction `exp(mean + var/2)` as `Poisson`.

## Survival likelihoods

`WeibullSurv` and `ExponentialSurv` fit event-time (duration) data under a
**proportional-hazards (PH)** parameterization on the Laplace engine. Unlike
the GLM families above, the response is a follow-up *time*, not a plain
outcome, so they carry their own event/censoring data contract:

```python
from pylgm import Fixed, Hyperparameter, IID, LGM, WeibullSurv

model = LGM(
    response="spell_time",                      # follow-up time (must be > 0)
    likelihood=WeibullSurv(
        "event",                                 # 1 = observed, 0 = right-censored
        shape=Hyperparameter("alpha", initial=1.0, transform="log"),
        entry="entry",                           # optional: left-truncation time
    ),
    predictor=Fixed("1 + x") + IID("frailty", index="id", precision=2.0),
)
result = model.fit(frame, engine="laplace", hyperparameters="optimize")
```

**Hazard.** With linear predictor `eta`, the hazard is
`h(t) = alpha * t**(alpha-1) * exp(eta)`, cumulative hazard
`H(t) = t**alpha * exp(eta)`, survival `S(t) = exp(-H(t))`.
`ExponentialSurv` is exactly `WeibullSurv` with `alpha = 1` fixed (constant
hazard). `alpha` may be a fixed float or a `Hyperparameter` (estimated by
type-II ML, same mechanism as GLM dispersion `phi` above); the fitted value is
reported on `result.hyperparameters["alpha"]` (or whatever name the
`Hyperparameter` was given) and — critically — `result.predict(new_data)`
uses the **fitted** shape, not the `Hyperparameter`'s initial guess.

**Data contract.** `event` names a 0/1 column (1 = the event was observed at
`response`, 0 = the subject was right-censored — still at risk — at
`response`). `entry`, if given, names a left-truncation (delayed-entry) time
column: the subject is only observed to be at risk from `entry` onward, and
must satisfy `0 <= entry < response`; omit it (`entry=None`, the default) when
there is no delayed entry. Both apply per row — there is no separate `panel`
requirement, though `WeibullSurv`/`ExponentialSurv` compose with `panel`/`time`
like any other likelihood.

**Fitted mean & hazard ratios.** `response_mean` (and hence `fitted_mean`) is
the unconditional expected duration `E[T] = exp(-eta/alpha) * Gamma(1 +
1/alpha)` — a closed-form Weibull moment, not a censoring-adjusted estimator;
compare it against the raw (censored) sample mean with that caveat in mind.
Exponentiating a fixed-effect coefficient gives the familiar **hazard ratio**
`exp(beta)`: covariate effects multiply the hazard rather than shifting the
mean directly.

**Frailty (unobserved heterogeneity).** There is no dedicated "frailty"
construct — declare it as an ordinary `IID` effect indexed by the individual,
exactly like the `IID` overdispersion idiom in
[`examples/count_regression`](https://github.com/Ardea00/pylgm/tree/main/examples/count_regression):
one latent value per row/individual, identified through the survival
likelihood's nonlinearity rather than through repeated observations per
group.

**Not (yet) covered.** Interval-censoring (event known only to fall between
two visits — needs a person-period/discrete-time Bernoulli recoding instead),
time-varying covariates (needs episode-splitting into covariate-constant
sub-intervals), and competing risks (multiple distinct failure causes) are
outside this slice's scope.

A runnable example — a simulated unemployment-duration panel with
right-censoring, left-truncation, estimated shape, and an `IID` frailty term
— lives at
[`examples/survival_duration/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/survival_duration/README.md).
In YAML: `likelihood: {family: weibullsurv, event: d, shape: 1.5, entry: v}` /
`{family: exponentialsurv, event: d}` (`shape` accepts a fixed value only,
same as GLM `phi` above; `entry` is optional for both).

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


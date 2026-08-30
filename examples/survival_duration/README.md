# Unemployment-duration example (Weibull proportional hazards)

A simulated panel of individual unemployment spells, fit with `WeibullSurv`
through `engine="laplace"`. Run it from the repository root:

```bash
PYTHONPATH=src python examples/survival_duration/run.py
```

## The data

`build()` simulates 400 individuals, each with one completed-or-censored
spell:

- `spell_time` — the response: either the completed duration (job found) or
  the point at which the individual left observation.
- `event` — 1 if the spell ended in the observed job-finding time, 0 if it
  was still ongoing when the observation window closed at `CENSOR_AT`
  (right-censoring).
- `entry` — for spells already under way when the observation window opened
  (about 30% of the sample), the elapsed time before entry; 0 otherwise
  (left-truncation — the individual only enters the risk set at `entry`, not
  at time 0).
- `treated` — a 0/1 covariate (e.g. job-search assistance).

## The model

```python
model = LGM(
    response="spell_time",
    likelihood=WeibullSurv(
        "event",
        shape=Hyperparameter("alpha", initial=1.0, transform="log"),
        entry="entry",
    ),
    predictor=Fixed("1 + treated") + IID("frailty", index="id", precision=2.0),
)
result = model.fit(frame, engine="laplace", hyperparameters="optimize")
```

**Reading the PH parameterization.** `WeibullSurv` models the hazard as
`h(t) = alpha * t**(alpha-1) * exp(eta)`, where `eta` is the linear predictor
(here `Fixed("1 + treated") + frailty`). `alpha > 1` means the hazard of
finding a job rises the longer someone has been unemployed (positive duration
dependence); `alpha = 1` recovers a constant-hazard `ExponentialSurv`.
Exponentiating a fixed-effect coefficient gives a **hazard ratio**: a positive
`treated` coefficient means treated individuals leave unemployment faster, so
`hazard_ratio = exp(beta_treated) > 1`.

**The frailty term.** `IID("frailty", index="id", precision=2.0)` adds one
latent value per individual to `eta` — unobserved heterogeneity ("frailty")
in a spell's baseline hazard that isn't explained by `treated`. This is the
same idiom as `examples/count_regression`'s per-crab overdispersion effect:
one row per group, identified through the likelihood's nonlinearity rather
than through repeated observations. Its precision is fixed here (`2.0`) rather
than estimated, to keep the empirical-Bayes search over `alpha` alone and the
example fast; pass a `Hyperparameter(...)` instead to estimate it, as
`examples/count_regression` does for its overdispersion precision.

**Shape estimation.** `shape=Hyperparameter("alpha", ...)` estimates `alpha`
by empirical Bayes (type-II ML) alongside the fit — see
[`docs/likelihoods.md`](../../docs/likelihoods.md#survival-likelihoods) and
[`docs/empirical-bayes.md`](../../docs/empirical-bayes.md).

## What it prints

```
n=400 spells, censored=19.0%, left_truncated=30.0%
hazard_ratio=1.916 (planted=1.649)
shape=1.470 (planted=1.400)
fitted_mean_duration=0.775 observed_mean_duration=0.772 (observed is the raw,
censoring-truncated average; fitted is the model-implied unconditional E[T])
```

`hazard_ratio` and `shape` recover the planted values up to Monte Carlo noise
at this sample size (400 individuals, ~1/5 censored). `observed_mean_duration`
is the naive average of the (right-censored) response column — mechanically
biased low whenever spells are cut short by the observation window —
while `fitted_mean_duration` is the model's implied unconditional `E[T]`
(`response_mean`, i.e. `exp(-eta/alpha) * Gamma(1 + 1/alpha)`), which does not
share that mechanical bias.

## What this does not cover

pyLGM's survival support is proportional-hazards with a fixed functional
form of time (Weibull/exponential). It does not (yet) cover:

- **Interval-censoring** — only known-exact or right-censored event times are
  supported; interval-censored data (event known only to lie between two
  visits) needs a person-period / discrete-time (Bernoulli) recoding instead.
- **Time-varying covariates** — `eta` is constant over a spell; a covariate
  that changes during the spell needs episode-splitting (one row per
  covariate-constant sub-interval, each contributing its own entry/exit
  times) before it fits this contract.
- **Competing risks** — a single event type/censoring indicator; distinct
  failure causes are not separately modelled.

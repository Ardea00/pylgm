# Combining with gradient boosting

pyLGM and gradient boosting fail in opposite directions. A latent Gaussian
model represents structure — space, time, groups, networks — and returns a
posterior for each piece, but its predictor is **linear** in the covariates.
Boosting represents products, thresholds and deep interactions, but has no
notion of a spatial field and no posterior at all.

Panels routinely contain both kinds of signal. The two methods compose in one
place, and it is not a new kind of model: it is pyLGM's existing linear
predictor.

## The composition

The model pyLGM fits is

$$y_i \mid \eta_i \sim \pi(y \mid \eta_i), \qquad
\eta_i = o_i + \sum_k A_k[i,:]\, x_k, \qquad
x \sim \mathcal N\big(0, Q(\theta)^{-1}\big)$$

where \(o_i\) is the per-row `offset`. That term is a **known constant** — it
carries no parameters and enters no prior. So an external learner's output can
be dropped into it without touching anything else:

$$\eta \;=\; \underbrace{f(z)}_{\text{boosting: nonlinear, no posterior}}
\;+\; \underbrace{\textstyle\sum_k A_k x_k}_{\text{pyLGM: structured, with a posterior}}$$

Both orderings below are one block step of coordinate ascent on the joint
penalized log-likelihood
\(\ell(f,x) = \sum_i \log \pi(y_i \mid f(z_i) + a_i^\top x) - \tfrac12 x^\top Q(\theta) x\).

!!! tip "The two hooks"
    **Boosting first** enters through `LGM(offset="column")` — a column name,
    so the offset travels with the data into `predict()` as well as `fit()`.

    **Boosting second** leaves through `result.predictive_mean`, which is η on
    the **link scale, offset included** — exactly what XGBoost's `base_margin`
    and LightGBM's `init_score` expect.

Neither requires a boosting library to be installed, and neither is a pyLGM
dependency. Everything below is user code.

## Boosting first, pyLGM second

Fit the booster, put its **raw margin** in a column, declare that column as the
offset, and let the latent field explain what is left.

```python
import numpy as np
from xgboost import XGBRegressor
from pylgm import Besag, Fixed, Hyperparameter, LGM, Poisson

# Out-of-fold margins: every row scored by a booster that never saw it.
fold = np.random.default_rng(0).permutation(len(x_train)) % 5
margin = np.zeros(len(x_train))
for f in range(5):
    held_out = fold == f
    booster = XGBRegressor(objective="count:poisson").fit(x_train[~held_out], y_train[~held_out])
    margin[held_out] = booster.predict(x_train[held_out], output_margin=True)

model = LGM(
    response="y",
    likelihood=Poisson(),
    offset="boost",                                    # <- the hook
    predictor=Fixed("1") + Besag("area", index="area", graph=graph,
                                 precision=Hyperparameter("area.precision", initial=1.0)),
)
result = model.fit(train.assign(boost=margin), engine="laplace")

# At prediction time the offset comes from a booster fit on ALL training rows.
deployed = XGBRegressor(objective="count:poisson").fit(x_train, y_train)
prediction = result.predict(test.assign(boost=deployed.predict(x_test, output_margin=True)))
```

Because the offset is a known constant, every posterior downstream of it —
`latent_marginals`, `predictive_variance`, the estimated hyperparameters —
means exactly what it normally means.

## pyLGM first, boosting second

Fit the model, hand its linear predictor to the booster as a starting margin,
and let boosting add whatever nonlinear covariate structure the linear
predictor could not reach.

```python
result = model.fit(train, engine="laplace")

booster = XGBRegressor(objective="count:poisson")
booster.fit(x_train, y_train, base_margin=result.predictive_mean)   # <- the hook

eta_test = booster.predict(
    x_test, output_margin=True,
    base_margin=result.predict(test).predictive_mean,
)
```

For a Gaussian likelihood this reduces to boosting the raw residuals
\(y - \hat\eta\). For every other likelihood, do it on the **margin** scale via
`base_margin` / `init_score` — boosting raw response residuals under a log link
is wrong, since the working residual is \((y-\mu)/\mu\), not \(y-\mu\).

This direction costs the posterior. `predictive_variance` is the latent field's
variance only; a booster fit *on top* of η is estimated from the same rows and
appears nowhere in the latent covariance. After stage 2 the interval is a
conditional variance given \(\hat f\), treating an estimated quantity as known.

## Three rules

### 1. The training offset must be out-of-fold

This is the one that costs you something real, and it is nearly invisible.
In-sample boosting margins have already fit the training rows, so the Laplace
fit sees residuals that are too small and empirical Bayes answers with an
inflated precision — the structured effect shrinks toward zero.

[`examples/boosted_offset`](https://github.com/Ardea00/pylgm/tree/main/examples/boosted_offset)
runs the identical booster twice, differing only in whether the training offset
was out-of-fold:

| training offset | held-out η RMSE | field sd | 95% coverage | `area.precision` |
|---|---|---|---|---|
| out-of-fold | 0.3492 | 0.560 | **0.96** | 2.56 |
| in-sample | 0.3790 | 0.373 | **0.64** | 5.68 |

The true field sd is 0.566. Point accuracy barely moves — which is exactly why
this is dangerous. The damage is entirely in the posterior, the reason to use
this library at all. It is stacking leakage: the same failure mode as in-sample
target encoding.

In more extreme cases the optimizer drives the precision all the way into its
declared bound, and `result.diagnostics["hyperparameters_at_bound"]` names the
parameter (see [empirical Bayes](empirical-bayes.md)). Worth checking — but it
is a symptom, not a substitute for building the offset correctly.

### 2. Match the link scale

The offset lives on the linear-predictor scale, so the booster must yield a
*margin*, not a mean: `output_margin=True` (XGBoost) or `raw_score=True`
(LightGBM), with the objective matched to the pyLGM likelihood.

| pyLGM likelihood | link | XGBoost objective | LightGBM objective |
|---|---|---|---|
| `Gaussian()` | identity | `reg:squarederror` | `regression` |
| `Poisson()` | log | `count:poisson` | `poisson` |
| `Bernoulli()` / `Binomial()` | logit | `binary:logistic` | `binary` |

A plain `.predict()` returns μ. Used as an offset it produces a fit that
converges to nonsense without complaining.

### 3. Random K-fold is only correct without a time axis

On a time-indexed panel the margin for time \(t\) must not be built from
\(t+1\). Use rolling origins instead —
`pylgm.evaluation.folds.build_fold_definitions` already generates
`(origin, target, horizon)` triples with vintage-aware training windows, which
is the same scheme
[model comparison and backtesting](model-comparison.md) uses.

## Which ordering

Prefer **boosting first** unless you only need a point forecast.

In the worked example the two orderings are within noise of each other on point
accuracy (0.349 vs 0.333 held-out η RMSE, both roughly half the error of either
method alone), so the tie is broken by what survives: boosting-first keeps every
pyLGM posterior intact, boosting-second does not.

A second, quieter benefit of boosting first: the latent field it recovers is
*better* than the one pyLGM recovers unaided (field RMSE 0.128 vs 0.202). With
the covariate surface carried by the offset, the spatial effect stops absorbing
misfit that was never spatial.

## Iterating between the two

The two orderings are single sweeps of the same block-coordinate ascent, so they
can be alternated:

$$x^{(t+1)} \leftarrow \text{LGM fit with offset } f^{(t)}(z), \qquad
f^{(t+1)} \leftarrow f^{(t)} + \text{a few boosting rounds with } \texttt{base\_margin} = A x^{(t+1)}$$

\(\ell\) is concave in η for every likelihood pyLGM ships, hence concave in each
block separately — but boosting with early stopping is not an exact block
maximizer, so **monitor a held-out score, not \(\ell\)**, which will keep
climbing straight into overfitting. Three to ten sweeps in practice.

!!! warning "Identifiability"
    Two things collapse the split between \(f\) and the latent field:

    - **Grouping keys in the booster's features.** If the booster can see
      `area`, \(f\) and the `Besag` field compete for identical signal and the
      division between them is decided by the booster's implicit shrinkage
      against the pyLGM prior — not by the data. Exclude the index columns of
      every structured effect from the booster's feature set.
    - **Re-estimating θ every sweep.** Each sweep's residuals are slightly more
      in-sample than the last, so repeated empirical Bayes ratchets the
      precisions upward. Estimate θ on the first sweep, pass fixed
      hyperparameters after, and re-estimate once at the end if at all.

## Not shipped

There is no `Boost` effect, no `pylgm.boosting` module, and no boosting
dependency — deliberately. An effect implies participation in the latent
covariance, which is precisely the claim the coverage numbers above warn
against; keeping the booster in the offset, where it is constant by
construction, is what keeps the posteriors honest. Also absent: a backfitting
driver for the iteration above (the loop is a handful of lines of user code),
and any automatic out-of-fold offset construction.

# Boosted offset — gradient boosting and pyLGM in one linear predictor

`examples/method_comparison` shows the two regimes apart: structured smoothing
wins on small-area spatial signal, boosting wins on nonlinear covariate
interactions. Real panels have both. This example combines them.

## The composition

pyLGM's linear predictor already has a slot for a known per-row term:

```
eta[i] = offset[i] + sum_k A_k[i,:] x_k,     x ~ N(0, Q(theta)^-1)
```

So the hybrid needs no new machinery — put the booster's raw margin in the
offset and let the latent field explain what is left:

```
eta = f(z)            <- boosting: nonlinear, interaction-heavy, no posterior
    + sum_k A_k x_k   <- pyLGM: structured, with a posterior per component
```

Both are one block step of coordinate ascent on the joint objective
`sum_i log pi(y_i | f(z_i) + a_i' x) - 0.5 x' Q(theta) x`.

## The panel

`simulate()` generates Poisson counts over 100 areas on a ring graph, 20 rows
each, whose log-rate carries **both** signals:

```
eta[i] = 0.5 + 0.7*x1*x2 + 0.6*1{x3 > 0.5} + 0.4*(x1^2 - 0.75) + s[area[i]]
```

`s` is a smooth sinusoidal field over the ring (sd 0.566, centred — Besag is
sum-to-zero). The covariate term is a product, a threshold and a curvature: no
linear predictor represents it. The first 14 rows of each area train, the last
6 test, so every area is identified on both halves.

Scoring is RMSE against the **true** `eta`, not against noisy held-out counts —
the honest target for a smoothing method.

## Results

```
Held-out RMSE against the TRUE log-rate eta (lower is better)
     boosting alone                       0.6552
     pyLGM alone                          0.7410
     boost -> LGM (out-of-fold offset)    0.3492
     boost -> LGM (in-sample offset)      0.3790
     LGM -> boost (base margin)           0.3332

Spatial field recovery (true field sd = 0.566)
                                             rmse      sd   cover   precision
     pyLGM alone                           0.2024  0.6256    0.89        1.17
     boost -> LGM (out-of-fold offset)     0.1281  0.5601    0.96        2.56
     boost -> LGM (in-sample offset)       0.2165  0.3729    0.64        5.68
```

The combination roughly halves the error of either method alone, and the
spatial field it recovers is better than the one pyLGM recovers unaided
(0.128 vs 0.202): with the covariate surface handled by the booster, the field
no longer has to absorb misfit that was never spatial.

## Three things that will bite you

**1. The training offset must be out-of-fold.** Compare the last two rows of
the field table. Both use the *same* booster; they differ only in whether the
training offset came from a booster that had seen those rows. The in-sample
version has already fit the training data, so the Laplace fit sees residuals
that are too small, and empirical Bayes answers with a precision twice the
honest value. The field loses a third of its amplitude (sd 0.373 vs 0.560) and
its 95% intervals cover **0.64** instead of 0.96.

Point predictions barely move (0.379 vs 0.349) — which is exactly why this is
dangerous. The damage is entirely in the posterior, which is the reason to use
this library at all. This is stacking leakage; the same failure as in-sample
target encoding.

In extreme cases the optimizer pushes the precision all the way to its
declared bound, and `result.diagnostics["hyperparameters_at_bound"]` names it.
Worth checking; it is not a substitute for building the offset correctly.

**2. Link-scale consistency.** The offset lives on the linear-predictor scale,
so the booster must yield a *margin*, not a mean: `output_margin=True`
(XGBoost) / `raw_score=True` (LightGBM), with matched objectives —
`count:poisson` ↔ `Poisson()`, `binary:logistic` ↔ `Bernoulli()`,
`reg:squarederror` ↔ `Gaussian()`. A plain `predict()` returns `mu` and gives a
fit that converges to nonsense.

**3. Random K-fold is only right without a time axis.** On a time-indexed
panel the margin for time `t` must not be built from `t+1`. Use the
rolling-origin scheme this repo already has,
`pylgm.evaluation.folds.build_fold_definitions`, which generates
`(origin, target, horizon)` triples with vintage-aware training windows.

## Which ordering

`LGM -> boost` edges out `boost -> LGM` on point RMSE here (0.333 vs 0.349),
and that is a real result — but it costs the posterior. An offset is a known
constant, so every pyLGM marginal downstream of it means exactly what it says.
A booster fit *on top* of the latent predictor is estimated from the same rows
and appears nowhere in the latent covariance, so `predictive_variance` becomes
a conditional variance given `f_hat`, treating an estimated quantity as known.

Prefer `boost -> LGM` unless you only need a point forecast.

## See also

[Combining with gradient boosting](https://ardea00.github.io/pylgm/boosting/) —
the guide page this example backs, with the link-scale/objective table and the
backfitting variant.

## Running it

```bash
PYTHONPATH=src python examples/boosted_offset/run.py
```

Requires `xgboost`, which is deliberately **not** a pyLGM dependency:

```bash
pip install xgboost
```

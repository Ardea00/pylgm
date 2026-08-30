# pyLGM

General-purpose **latent Gaussian models** for Python — the model class behind
INLA. Declare fixed effects, random effects, temporal (RW/AR1/seasonal),
mixed-frequency (MIDAS), spatial (CAR), directed-network (SAR) and dynamic-panel
(SDPD) structure; fit them under Gaussian, Poisson, Bernoulli, Binomial,
negative-binomial, Gamma, Beta or survival likelihoods, from a Pandas or Spark
DataFrame, with exact-Gaussian and Laplace engines, empirical Bayes / MAP-II
hyperparameter estimation, INLA-style posterior integration, and a sparse solver
that carries the full uncertainty surface at network scale.

Deterministic, no MCMC, no new dependencies.

```bash
pip install pylgm
```

## Start here

A Poisson model with a per-region random intercept:

```python
import pandas as pd
from pylgm import Fixed, IID, LGM, Poisson

frame = pd.DataFrame({
    "region": ["north", "north", "north", "south", "south", "south"],
    "time":   [1, 2, 3, 1, 2, 3],
    "x":      [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
    "count":  [3, 5, 8, 2, 3, 5],
})

model = LGM(
    response="count",
    likelihood=Poisson(),
    predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
    panel=("region",),
    time="time",
)

result = model.fit(frame, engine="laplace")
print(result.fitted_mean.round(3).tolist())
```

## Guide

| Page | Covers |
|---|---|
| [How it works](how-it-works.md) | One `fit()` call end to end — compile, engine, solve, hyperparameters — then prediction and forecasting |
| [Likelihoods](likelihoods.md) | Gaussian (exact), Poisson, Bernoulli, Binomial, negative-binomial, Gamma, Beta, Weibull/exponential survival |
| [Effects](effects.md) | `Fixed`, `IID`, `RW1`/`RW2`, `AR1` (optionally group-wise), `Seasonal`, `MIDAS`, `SpaceTime` |
| [Spatial effects](spatial-effects.md) | `Besag`, `ProperCAR`, `BYM2`, weighted graphs, directed `SAR`, dynamic `DynamicSpatialPanel` |
| [Effects → Constraints](effects.md#linear-constraints-extraconstr) | Arbitrary linear constraints `A x = e` (R-INLA `extraconstr`) |
| [Empirical Bayes and priors](empirical-bayes.md) | Type-II ML, MAP-II priors, bounded hyperparameters |
| [INLA integration](inla.md) | Grid quadrature, simplified/full-Laplace marginals, DIC/WAIC/CPO/PIT |
| [Prediction](prediction.md) | Fit-row and out-of-sample prediction, and forecasting new levels |
| [Spark](spark.md) | Spark / Databricks data boundary |
| [Comparison](comparison.md) | Measured against a GLM, XGBoost and a Metropolis sampler — including where pyLGM loses |
| [Theory](theory.md) | The LGM class, Laplace/INLA inference, every structured effect, with references |
| [Internals & release policy](internals.md) | Scope, compiled-IR policy, artifact publication, release gate |
| [Roadmap](roadmap.md) | What's shipped and what's next |
| [Development](development.md) | Working on pyLGM itself |

## Worked examples

| Example | Shows |
|---|---|
| [Disease mapping](examples-disease-mapping.md) | `Besag` spatial smoothing on Scotland lip cancer — fit to observed counts jumps 0.63 → 0.96 vs a non-spatial GLM |
| [Count regression](examples-count-regression.md) | Horseshoe-crab Poisson GLM matching `statsmodels`, plus estimated overdispersion for honest uncertainty |
| [Method comparison](comparison.md) | pyLGM vs GLM vs XGBoost vs Metropolis, on problems chosen so each one wins somewhere |

The [examples gallery](examples.md) indexes all 19 runnable scripts.

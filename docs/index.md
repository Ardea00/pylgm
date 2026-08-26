# pyLGM

General-purpose **latent Gaussian models** for Python — the model class behind
INLA. Fit fixed effects, random effects, temporal (RW/AR1) and spatial (CAR)
structure under Gaussian, Poisson, or Bernoulli likelihoods, from a Pandas (or
Spark) DataFrame, with exact-Gaussian and Laplace inference engines, empirical
Bayes / MAP-II hyperparameter estimation, and INLA-style posterior integration.

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
| [Likelihoods](likelihoods.md) | Gaussian (exact), Poisson & Bernoulli (Laplace), the `predictive_variance` convention, exact-Gaussian reference limits |
| [Effects](effects.md) | `Fixed`, `IID`, `RW1`/`RW2`, stationary `AR1` |
| [Spatial effects](spatial-effects.md) | `Besag` (ICAR), `ProperCAR` (with ρ), `BYM2` (with φ) |
| [Empirical Bayes and priors](empirical-bayes.md) | Type-II ML, MAP-II priors, bounded hyperparameters |
| [INLA integration](inla.md) | Grid quadrature, simplified/full-Laplace marginals, DIC/WAIC/CPO/PIT |
| [Prediction](prediction.md) | Fit-row and out-of-sample prediction |
| [Spark](spark.md) | Spark / Databricks data boundary |
| [Theory](theory.md) | The LGM class, Laplace/INLA inference, the CAR family, PC priors, with references |
| [Internals & release policy](internals.md) | 0.3 scope, compiled-IR policy, artifact publication, release gate |
| [Roadmap](roadmap.md) | What's shipped and what's next |
| [Development](development.md) | Working on pyLGM itself |

## Worked examples, benchmarked against standard regression

| Example | Shows |
|---|---|
| [Disease mapping](examples-disease-mapping.md) | Besag spatial smoothing on Scotland lip cancer — fit to observed counts jumps 0.63 → 0.96 vs a non-spatial GLM |
| [Count regression](examples-count-regression.md) | Horseshoe-crab Poisson GLM matching `statsmodels`, plus estimated overdispersion for honest uncertainty |

Runnable scripts for every feature live under
[`examples/`](https://github.com/Ardea00/pylgm/tree/main/examples) in the repository.

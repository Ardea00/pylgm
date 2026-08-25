# pyLGM

General-purpose **latent Gaussian models** for Python — the model class behind
INLA. Fit fixed effects, random effects, temporal (RW/AR1) and spatial (CAR)
structure under Gaussian, Poisson, or Bernoulli likelihoods, from a Pandas (or
Spark) DataFrame, with exact-Gaussian and Laplace inference engines, empirical
Bayes / MAP-II hyperparameter estimation, and INLA-style posterior integration.

## Install

```bash
pip install pylgm
pip install "pylgm[spark]"   # optional: Spark data boundary
```

Requires Python ≥ 3.11.

## 30-second example

A Poisson model with a per-region random intercept, fit with the Laplace engine:

```python
import pandas as pd
from pylgm import Fixed, IID, LGM, Poisson

# Counts of events per region over time, with a covariate x.
frame = pd.DataFrame({
    "region": ["north", "north", "north", "south", "south", "south"],
    "time":   [1, 2, 3, 1, 2, 3],
    "x":      [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
    "count":  [3, 5, 8, 2, 3, 5],
})

model = LGM(
    response="count",
    likelihood=Poisson(),                                 # canonical log link
    predictor=Fixed("1 + x")                              # fixed effects
        + IID("region", index="region", precision=2.0),  # random intercept per region
    panel=("region",),
    time="time",
)

result = model.fit(frame, engine="laplace")
print("fitted_mean:", result.fitted_mean.round(3).tolist())
# fitted_mean: [3.254, 4.986, 8.15, 2.201, 3.373, 5.513]
```

The same model can be declared in YAML and loaded with `pylgm.config.load_model`.
See the [general LGM example](examples/general_lgm/README.md), which fits the
same data through both the Python and YAML frontends, and the other 13 runnable
scripts under [`examples/`](examples/).

## What's in the box

| Area | What you get | Docs |
|---|---|---|
| **Likelihoods** | Gaussian (exact), Poisson & Bernoulli (Laplace) | [likelihoods](docs/likelihoods.md) |
| **Effects** | `Fixed`, `IID`, `RW1`/`RW2`, stationary `AR1` | [effects](docs/effects.md) |
| **Spatial** | `Besag` (ICAR), `ProperCAR` (with ρ), `BYM2` (with φ) | [spatial effects](docs/spatial-effects.md) |
| **Hyperparameters** | Empirical Bayes (type-II ML), MAP-II priors, bounds | [empirical bayes](docs/empirical-bayes.md) |
| **Integration** | INLA-style grid quadrature, simplified/full-Laplace marginals, DIC/WAIC/CPO/PIT | [INLA](docs/inla.md) |
| **Prediction** | fit-row and out-of-sample `result.predict(new_data)` | [prediction](docs/prediction.md) |
| **Data boundary** | Pandas, or Spark / Databricks | [spark](docs/spark.md) |

## Documentation

Full docs: **https://ardea00.github.io/pylgm/** — or browse the
[`docs/`](docs/) folder. Start with the [index](docs/index.md).

## Scope and roadmap

pyLGM 0.3 is a bounded foundation release. What is and isn't in it, and where
it's going, are documented in the [roadmap](docs/roadmap.md) and the
[internals / release policy](docs/internals.md) page.

## Development

See [docs/development.md](docs/development.md). License: [MIT](LICENSE).

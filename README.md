<h1 align="center">pyLGM</h1>

<p align="center">
  <strong>General-purpose latent Gaussian models for Python — the model class behind INLA.</strong><br>
  Deterministic Bayesian inference for structured data. No MCMC, no new dependencies.
</p>

<p align="center">
  <a href="https://pypi.org/project/pylgm/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pylgm?color=3b82f6"></a>
  <a href="https://pypi.org/project/pylgm/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/pylgm"></a>
  <a href="https://github.com/Ardea00/pylgm/actions/workflows/test.yml"><img alt="CI" src="https://github.com/Ardea00/pylgm/actions/workflows/test.yml/badge.svg?branch=main"></a>
  <a href="https://ardea00.github.io/pylgm/"><img alt="Docs" src="https://img.shields.io/badge/docs-ardea00.github.io%2Fpylgm-blue"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/pylgm?color=green"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-261230"></a>
</p>

<p align="center">
  <a href="https://ardea00.github.io/pylgm/">Documentation</a> ·
  <a href="https://ardea00.github.io/pylgm/how-it-works/">How it works</a> ·
  <a href="https://ardea00.github.io/pylgm/comparison/">Comparison</a> ·
  <a href="https://ardea00.github.io/pylgm/examples/">Examples</a> ·
  <a href="https://ardea00.github.io/pylgm/theory/">Theory</a>
</p>

---

## What it does

Declare structure — groups, space, time, frequency, networks — and pyLGM
returns a **posterior for each component**, not just a prediction. Fits run in
seconds through Laplace approximations and INLA-style integration, and are
deterministic: same data, same answer, no convergence diagnostics.

<p align="center">
  <img src="docs/img/scotland_shrinkage.png" alt="Besag smoothing shrinks noisy rates toward local means" width="620">
</p>

<p align="center"><em>Partial pooling in one picture: noisy small-area rates (left, near zero) are
pulled toward their neighbours, while well-observed areas keep their signal.
Marker size is the expected count. From
<a href="https://ardea00.github.io/pylgm/examples-disease-mapping/">the disease-mapping example</a>.</em></p>

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

frame = pd.DataFrame({
    "region": ["north", "north", "north", "south", "south", "south"],
    "time":   [1, 2, 3, 1, 2, 3],
    "x":      [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
    "count":  [3, 5, 8, 2, 3, 5],
})

model = LGM(
    response="count",
    likelihood=Poisson(),                                # canonical log link
    predictor=Fixed("1 + x")                             # fixed effects
        + IID("region", index="region", precision=2.0),  # random intercept per region
    panel=("region",),
    time="time",
)

result = model.fit(frame, engine="laplace")
print("fitted_mean:", result.fitted_mean.round(3).tolist())
# fitted_mean: [3.254, 4.986, 8.15, 2.201, 3.373, 5.513]
```

The same model can be declared in YAML and loaded with `pylgm.config.load_model`.

## Reproducing a published result

[`examples/columbus_spatial_econometrics`](examples/columbus_spatial_econometrics)
reproduces the reference result of spatial econometrics — Anselin (1988),
Table 12.1, 49 Columbus OH neighbourhoods:

| model | const | INC | HOVAL | ρ / λ |
|---|---|---|---|---|
| published OLS | 68.619 | −1.5973 | −0.2739 | — |
| **OLS, recomputed** | **68.619** | **−1.5973** | **−0.2739** | — |
| published ML spatial error | 60.279 | −0.9573 | −0.3046 | 0.5468 |
| **pyLGM `SAR`** | **59.543** | **−0.9057** | **−0.3058** | **0.5946** |

OLS matching the published numbers exactly verifies the data and spec; the
`SAR` fit then lands next to the published ML spatial-error estimates, and
recovers the finding that ignoring spatial correlation **overstates the income
effect by 1.76×**.

## A network that changes every year

[`examples/state_income_dynamic_network`](examples/state_income_dynamic_network)
fits 48 US states over 1997–2007 with **one network per year**, then knocks out
20% of the panel and restores it:

| method | RMSE ↓ (log income) |
|---|---|
| **`DynamicSpatialPanel`** | **0.0246** |
| state mean | 0.1214 |
| year mean | 0.1484 |

~5× closer than the obvious baselines. The same example reports where it loses
— a last-value forecast beats it on level forecasts, because the fitted γ ≈ 1
says log income is near a random walk.

## Why not just use XGBoost?

Often you should — and the [comparison page](https://ardea00.github.io/pylgm/comparison/)
says so with measured numbers rather than adjectives. Both problems below are
simulated, so the *true* surface is known and scored directly:

| Problem | pyLGM | XGBoost | GLM |
|---|---|---|---|
| 200 areas, 3 counts each, smooth spatial signal (RMSE ↓) | **0.135** | 0.371 | 0.429 |
| 4000 rows, nonlinear covariate interactions (RMSE ↓) | 2.839 | **0.293** | — |

pyLGM wins where the per-unit sample is thin but the units are related, and
returns 95% intervals that cover the truth 99% of the time. Gradient boosting
wins where the signal is interactions among covariates and you have the rows to
learn them. Reproduce both with
[`examples/method_comparison`](examples/method_comparison).

## What's in the box

| Area | What you get | Docs |
|---|---|---|
| **Likelihoods** | Gaussian (exact), Poisson, Bernoulli, Binomial, negative-binomial, Gamma, Beta, Weibull/exponential survival | [likelihoods](https://ardea00.github.io/pylgm/likelihoods/) |
| **Effects** | `Fixed`, `IID`, `RW1`/`RW2`, `AR1` (optionally group-wise), `Seasonal`, `MIDAS`, `MIDASParametric`, `SpaceTime` | [effects](https://ardea00.github.io/pylgm/effects/) |
| **Spatial** | `Besag` (ICAR), `ProperCAR`, `BYM2`, weighted graphs | [spatial](https://ardea00.github.io/pylgm/spatial-effects/) |
| **Networks** | directed `SAR`, dynamic `DynamicSpatialPanel` (SDPD) with forward forecasting | [spatial](https://ardea00.github.io/pylgm/spatial-effects/) |
| **Scale** | sparse solver past the dense guard, with the full uncertainty surface | [internals](https://ardea00.github.io/pylgm/internals/) |
| **Hyperparameters** | Empirical Bayes (type-II ML), MAP-II with PC priors, bounds | [empirical bayes](https://ardea00.github.io/pylgm/empirical-bayes/) |
| **Integration** | INLA grid quadrature, simplified/full-Laplace marginals, DIC/WAIC/CPO/PIT | [INLA](https://ardea00.github.io/pylgm/inla/) |
| **Prediction** | fit-row, out-of-sample `predict`, and forecasting via `NaN` rows | [prediction](https://ardea00.github.io/pylgm/prediction/) |
| **Constraints** | arbitrary linear constraints `A x = e` (R-INLA `extraconstr`) | [effects](https://ardea00.github.io/pylgm/effects/#linear-constraints-extraconstr) |
| **Data boundary** | Pandas, or Spark / Databricks | [spark](https://ardea00.github.io/pylgm/spark/) |

## Documentation

Full docs: **https://ardea00.github.io/pylgm/**

- [How it works](https://ardea00.github.io/pylgm/how-it-works/) — one `fit()` call end to end, then prediction and forecasting
- [Comparison](https://ardea00.github.io/pylgm/comparison/) — against regression, gradient boosting and MCMC, including where pyLGM loses
- [Theory](https://ardea00.github.io/pylgm/theory/) — the model class and every structured effect, with references
- [Examples](https://ardea00.github.io/pylgm/examples/) — 21 runnable scripts

## Roadmap

What's shipped and what's next is on the
[roadmap](https://ardea00.github.io/pylgm/roadmap/); scope and compatibility
policy on the [internals](https://ardea00.github.io/pylgm/internals/) page.

## Contributing

Issues and pull requests welcome — see
[docs/development.md](docs/development.md) for the test setup. `main` is
protected: PRs need the full CI matrix green before merge.

## License

[MIT](LICENSE)

# How pyLGM compares

pyLGM is not a general-purpose predictor. It is a **structured-uncertainty**
method: it assumes the signal decomposes into smooth, spatial, temporal or
network-shaped pieces, and in exchange it returns calibrated uncertainty for
each piece. That assumption is what makes it strong in some problems and the
wrong tool in others.

Every number on this page comes from
[`examples/method_comparison`](https://github.com/Ardea00/pylgm/tree/main/examples/method_comparison),
which you can run yourself. Both problems are **simulated**, so the true latent
surface is known and scored directly — the honest target for a smoothing method,
rather than in-sample fit to noisy observations.

## The short version

| | classical regression / GLM | gradient boosting (XGBoost) | MCMC | **pyLGM** |
|---|---|---|---|---|
| Nonlinear interactions, many features | weak | **excellent** | depends on model | weak |
| Few observations per group | overfits or needs manual pooling | overfits | good | **excellent** |
| Spatial / temporal / network structure | manual basis functions | must be learned from data | good | **built in** |
| Calibrated uncertainty | for the fitted parameters | none natively | **exact in the limit** | **yes, per component** |
| Extrapolating past the data | linear only | **cannot** — constant beyond the hull | good | **yes**, via the structure |
| Runtime | seconds | seconds | minutes to hours | **seconds** |
| Reproducibility | exact | exact given a seed | needs convergence checks | **exact, deterministic** |

## Versus regression and GLMs

A GLM with a dummy per group is the honest baseline, and it fails in a specific
way: with few observations per group, each dummy is estimated from almost no
data, so the estimates are noisy and the extreme ones are the most wrong.
Nothing tells a rare-event area from a genuinely quiet one.

pyLGM's structured effects apply **partial pooling** — each area borrows
strength from its neighbours, at a rate the data itself chooses through the
estimated precision \(\tau\). A GLM can only choose between no pooling (dummies)
and complete pooling (drop the term).

**Problem A: 200 areas, 3 Poisson observations each, smooth spatial signal.**
RMSE against the *true* latent field:

| method | RMSE ↓ |
|---|---|
| **pyLGM (`Besag`)** | **0.1345** |
| XGBoost | 0.3708 |
| GLM, area dummies | 0.4288 |

pyLGM is about **3× closer to the truth**, and its 95% credible intervals
contain the true value **99%** of the time — slightly conservative, which is the
safe direction to be wrong. Neither baseline produces an interval at all.

This is the regime the method exists for: small-area estimation, disease
mapping, sparse panels, any setting where the per-unit sample is thin but the
units are related.

## Versus gradient boosting

Boosted trees are excellent at exactly what LGMs are bad at: discovering
nonlinear interactions among many covariates when you have enough rows to learn
them.

**Problem B: 4000 rows, response driven by products and thresholds of three
continuous covariates.** RMSE against the true signal on held-out rows:

| method | RMSE ↓ |
|---|---|
| **XGBoost** | **0.2934** |
| pyLGM, linear predictor | 2.8391 |

XGBoost is roughly **10× better**, and no amount of tuning a linear predictor
closes that gap — the model class simply does not contain the interaction. If
your problem looks like this, use gradient boosting.

Three differences matter beyond accuracy:

- **Uncertainty.** XGBoost returns a point prediction. Quantile objectives or
  conformal wrappers add intervals, but they are not a posterior and they do not
  decompose by component. pyLGM gives you the posterior for *each term*
  separately — `latent_marginals("region")` versus `latent_marginals("trend")`.
- **Extrapolation.** A tree is constant outside the range of its training data,
  so a boosted model cannot forecast a trend past the last observed period.
  A `RW1` or `AR1` extrapolates because the *structure* carries it forward; see
  [prediction](prediction.md).
- **Structure as an input, not a discovery.** If you know regions are adjacent
  or periods are ordered, pyLGM takes that as a graph or an index. A tree must
  rediscover it from data, which costs the data you did not have.

**They compose.** These are not exclusive: boosting the residuals of a fitted
LGM, or feeding an LGM's smoothed component in as a feature, is often better
than either alone.

## Versus MCMC and Monte Carlo

This is a comparison of *inference style*, not model class — MCMC can fit
latent Gaussian models too, and for a long time that was the standard way.

Monte Carlo error falls as \(1/\sqrt{S}\) in the number of draws \(S\). A
deterministic method has no such error. **Problem C** targets a small Gaussian
model whose posterior is available in closed form, and compares one exact solve
against a random-walk Metropolis sampler:

| method | distance from the exact posterior mean ↓ |
|---|---|
| **pyLGM (one solve)** | **0.0000** |
| Metropolis, 1 000 draws | 0.2066 |
| Metropolis, 10 000 draws | 0.0638 |
| Metropolis, 100 000 draws | 0.0161 |

Each 10× in draws buys about \(\sqrt{10} \approx 3.2\times\) accuracy — visible
in the table, and the reason sampling gets expensive when you want another
digit. For a Gaussian likelihood pyLGM's answer is not an approximation at all;
for non-Gaussian likelihoods it is a Laplace approximation whose error comes
from the posterior's shape rather than from a random seed.

What this buys, in practice:

- **No convergence diagnostics.** No burn-in, no thinning, no \(\hat{R}\), no
  divergences to interpret. A fit either converges or raises.
- **Determinism.** The same data give bit-identical results, which matters for
  a production pipeline and for testing.
- **Speed.** Seconds instead of minutes-to-hours for models of this shape.

What you give up:

- **Model class.** INLA-style inference requires the latent field to be
  Gaussian and the hyperparameters to be few. Outside that, use MCMC.
- **Exactness in hard posteriors.** For strongly non-Gaussian or multimodal
  posteriors, a Laplace approximation can be biased where MCMC, run long
  enough, is not. pyLGM's simplified- and full-Laplace latent marginals
  ([INLA integration](inla.md)) exist to narrow that gap, not to close it.
- **Many hyperparameters.** The INLA grid grows as \((2r+1)^d\) in the number
  \(d\) of hyperparameters, so it is practical for a handful, not dozens.

## When *not* to reach for pyLGM

Being clear about this is more useful than a feature list:

- The signal is dominated by **interactions between many covariates** → gradient
  boosting.
- You have **no structure to exploit** — no groups, no space, no time, no
  network — and plenty of rows per pattern → almost anything else is simpler.
- Your latent field is **genuinely non-Gaussian** (heavy-tailed, multimodal,
  discrete states) → MCMC, or a mixture/state-space tool.
- You need **dozens of hyperparameters** estimated jointly → the grid does not
  scale there.
- You want a **black-box predictor** and do not care how the prediction
  decomposes → the structure is overhead you are not using.

## Reproducing this page

```bash
pip install scikit-learn xgboost     # not pyLGM dependencies
PYTHONPATH=src python examples/method_comparison/run.py
```

The script prints exactly the tables above. It is excluded from the CI example
suite so that pyLGM itself keeps no dependency on scikit-learn or xgboost.

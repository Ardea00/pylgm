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

## On real data, against a published result

Simulations are a fair test of a smoothing method but a weak one for
credibility, so
[`examples/columbus_spatial_econometrics`](https://github.com/Ardea00/pylgm/tree/main/examples/columbus_spatial_econometrics)
reproduces the reference result of spatial econometrics: Anselin (1988),
Table 12.1 — 49 Columbus, OH neighbourhoods, crime on income and housing value.

| model | const | INC | HOVAL | ρ / λ |
|---|---|---|---|---|
| published OLS | 68.619 | −1.5973 | −0.2739 | — |
| **OLS, recomputed here** | **68.619** | **−1.5973** | **−0.2739** | — |
| published ML spatial error | 60.279 | −0.9573 | −0.3046 | 0.5468 |
| **pyLGM `SAR`** | **59.543** | **−0.9057** | **−0.3058** | **0.5946** |

The OLS row matching the published values *exactly* is what verifies the data
and specification. pyLGM's `SAR` then lands next to the published ML
spatial-error estimates, and the economic conclusion reproduces cleanly:
ignoring spatial correlation **overstates the income effect by 1.76×**.

pyLGM is not re-implementing ML estimation — it is Bayesian, and carries a
nugget the ML spatial-error model does not — so exact agreement would be
suspicious rather than reassuring. The residual gap shows up in ρ (0.595 vs
0.547).

### Where the structure earns its keep

[`examples/state_income_dynamic_network`](https://github.com/Ardea00/pylgm/tree/main/examples/state_income_dynamic_network)
fits 48 US states over 1997–2007 with **a different network every year** —
contiguity weighted by the previous year's income similarity — then knocks 20%
of the panel cells out and asks each method to restore them:

| method | RMSE ↓ (log income) |
|---|---|
| **pyLGM `DynamicSpatialPanel`** | **0.0246** |
| state mean | 0.1214 |
| year mean | 0.1484 |
| grand mean | 0.1786 |

About **5× closer**, because a missing cell is reconstructed from both its own
history and its neighbours' current values. Panel gaps are the ordinary case,
not a contrived one.

That example also reports where it **loses**: on a plain level forecast a
last-value benchmark wins (0.029 vs 0.040). That is not a defect — the fitted
γ ≈ 1.005 says log income is near a random walk, and for a random walk the last
observed value *is* the optimal forecast.

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

**They compose.** These are not exclusive, and the composition is concrete:
put the booster's raw margin in pyLGM's `offset`, so one linear predictor
carries a boosted nonlinear term *and* a structured latent field with its
posterior. On a panel built to contain both kinds of signal at once, that
halves the held-out error of either method alone — and recovers the spatial
field better than pyLGM does unaided. The reverse order works too, through
`base_margin`, but forfeits the posterior. See
[combining with gradient boosting](boosting.md).

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

## Versus INLA (pyINLA)

[pyINLA](https://pyinla.org/) drives the `inla` C program from Python, so it is
the reference implementation of the same method. The benchmark in
[`benchmarks/pyinla`](https://github.com/Ardea00/pylgm/tree/main/benchmarks/pyinla)
fits the same model, priors (PC(1, 0.01) on every precision, scaled Besag) and
data in both, with `hyperparameters="integrate"`; median of 3 runs, pyINLA
0.1.10, pyLGM in the Python 3.11 environment pyINLA requires (pyLGM runs about
1.3-2.5x faster on current NumPy/SciPy).

| model | size | pyLGM | pyINLA |
|---|---|---|---|
| Gaussian + IID | 5 000 rows | 1.80 s | 0.31 s |
| | 50 000 rows | 6.57 s | 1.66 s |
| Poisson + Besag | 100 areas | 0.11 s | 0.20 s |
| | 2 500 areas | 3.85 s | 0.44 s |
| | 4 900 areas | 9.13 s | 0.70 s |

**Accuracy is at parity**: hyperparameter posterior means agree to 4-5
significant figures (24.3396 vs 24.3404 at 2 500 areas) and latent means and
sds to within 2e-4 and 0.1%, *provided both apply the same mean correction*.
INLA's reported latent means include a variational (VB) correction from the
conditional mode toward the mean by default; pyLGM's equivalent is
`fit(..., mean_correction=True)`, which is off by default. Left off, pyLGM
reports the mode-centred mean and differs from INLA by exactly that shift
(0.02 on the intercept of the 100-area Poisson model; with INLA's VB turned off
the two agree to 1e-4). `mean_correction` and a skewed `latent_strategy` correct
the same gap, so they cannot be combined.

**Speed**: pyINLA is faster on one-off fits of large non-Gaussian models, by a
constant factor that now comes from re-analysing the sparse factorisation at
every Newton step (SciPy exposes no symbolic/numeric split).

**Where pyLGM is ahead: absorbing new data.** `result.update` conditions
a fitted posterior on new rows exactly, with no refactorisation
([prediction](prediction.md#sequential-updates)); INLA refits. Per new period of
a regional panel, at the same fixed hyperparameters:

| panel | latents | `update` | pyLGM refit | pyINLA refit |
|---|---|---|---|---|
| 100 x 60 | 162 | 0.004 s | 0.015 s | 0.32 s |
| 1 000 x 100 | 1 102 | 0.30 s | 0.61 s | 1.19 s |

`update` equals the refit to 1e-9. It pays off when a batch of new rows is small
next to the latent field; when every batch touches every latent, refit.

The same holds beyond fixed Gaussian fits:
- A **Laplace fit** (Poisson, binomial, negative binomial, ...) absorbs new rows through a
  Newton step in their own `k`-dimensional predictor space.
- An **integrated fit** also updates its hyperparameter posterior, by reweighting the INLA
  grid with each point's `p(y_new | y_old, theta)`, with no new hyperparameter search.

INLA has neither: every new batch is a full refit.

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

The script prints exactly the tables above; a CI test runs it when those
packages are installed and skips otherwise, so pyLGM itself keeps no dependency
on scikit-learn or xgboost.

The two real-data examples need nothing beyond pyLGM, and both are checked in
CI:

```bash
PYTHONPATH=src python examples/columbus_spatial_econometrics/run.py
PYTHONPATH=src python examples/state_income_dynamic_network/run.py   # ~40s
```

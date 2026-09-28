# Predicting new rows

`result.predict(new_data)` is available on `GaussianResult`, `LaplaceResult`,
and `INLAResult` (i.e. every result produced by `LGM.fit`). It scores rows
that were **not** passed to `fit` by rebuilding their design and reusing the
already-fitted latent posterior — no refit. It returns an immutable
`Prediction` with `predictive_mean`, `predictive_variance`, `fitted_mean`,
and `keys` (`new_data`'s own index), plus `.to_frame()`, a `DataFrame`
indexed like `new_data` with columns `predictive_mean`, `predictive_sd`, and
`fitted_mean`.

`new_data` must carry every column the fitted model reads — each effect's
`index` column, every variable used in a `Fixed` formula, and the `offset`
column when the model declares one — but **not** the response column.

!!! tip "Which array do I want?"
    Compare against **observed responses** (counts, 0/1, y) with
    `fitted_mean` — it is on the response scale (`exp` of the linear predictor
    for a Poisson log link, `logit⁻¹` for Bernoulli, identity for Gaussian).
    `predictive_mean`/`predictive_variance` are on the **linear-predictor**
    (`η`) scale. See ["scale conventions"](#scale-conventions) below.

```python
import pandas as pd
from pylgm import Fixed, Gaussian, IID, LGM

frame = pd.DataFrame({
    "claims": [0.5, 1.5, 2.5, 3.5],
    "x": [1.0, 2.0, 3.0, 4.0],
    "region": ["a", "b", "a", "b"],
})
model = LGM(
    response="claims",
    predictor=Fixed("1 + x") + IID("region_effect", index="region", precision=2.0),
    likelihood=Gaussian(sigma=0.5),
)
result = model.fit(frame)

# new_data carries "x" (the Fixed formula variable) and "region" (the IID
# effect's index), but not "claims" (the response).
new_scenarios = pd.DataFrame({"x": [10.0, -3.0], "region": ["b", "a"]})
prediction = result.predict(new_scenarios)
print(prediction.to_frame())
#    predictive_mean  predictive_sd  fitted_mean
# 0         9.499999       1.764396     9.499999
# 1        -3.499999       1.288687    -3.499999
```

`predict` **reuses** the fitted posterior; it cannot create a new latent
column. An index level absent from the fitted model — a region `predict`
never saw at `fit` time — raises `ValueError` naming both the offending block
and the unrecognized level(s), rather than silently falling back to a prior
or a zero contribution. The one exception is a spatial effect: it is keyed on
graph **nodes**, so a node present in the effect's `graph` but not referenced
by any fitted row already has a posterior and predicts without error.

Forecasting genuinely new levels — a future time point, a region never seen
at all — is a different operation, and it already worked before `predict`
existed: include those rows at `fit` time with a `NaN` response. They
contribute no likelihood but still receive a latent column (and, for a
structured effect such as `RW1`, an extrapolated posterior) plus a full
prediction, right alongside the observed rows.

```python
import numpy as np
import pandas as pd
from pylgm import Fixed, Gaussian, IID, LGM, RW1

n_hist = 30
history = pd.DataFrame({
    "y": 1.0 + 0.2 * np.sin(np.arange(n_hist) / 3.0),
    "region": (["a", "b"] * (n_hist // 2))[:n_hist],
    "t": range(n_hist),
})
# These 4 future periods never appeared at fit time. Give them a NaN
# response instead of calling predict() on them.
future = pd.DataFrame({
    "y": [np.nan, np.nan, np.nan, np.nan],
    "region": ["a", "b", "a", "b"],
    "t": [n_hist, n_hist + 1, n_hist + 2, n_hist + 3],
})
frame = pd.concat([history, future], ignore_index=True)

model = LGM(
    response="y",
    predictor=Fixed("1") + IID("region", index="region", precision=2.0)
    + RW1("trend", index="t", precision=2.0),
    likelihood=Gaussian(sigma=0.2),
)
result = model.fit(frame)

# The last 4 rows are the future periods; predictive_mean/_variance already
# cover them, in the caller's row order (same arrays fit() always returns).
print(np.round(result.predictive_mean[-4:], 3))    # [0.957 0.957 0.957 0.957]
print(np.round(result.predictive_variance[-4:], 3))  # [0.557 1.037 1.557 2.037]
```

An `RW1`/`RW2` forecast extrapolates **flat** from the last fitted level —
these examples show it exactly, because the region effect nets out to ~0 by
symmetry — and its variance strictly increases with each additional step
ahead, converging toward growth of `1/τ` per step as the fitted history
grows long relative to the forecast horizon (the finite-sample deviation
comes from the effect's sum-to-zero identifiability constraint, which
couples the whole chain including the unobserved tail).

| need | use |
| --- | --- |
| new covariate values / scenarios on **known** index levels | `result.predict(new_data)` |
| **new** time points, regions, or groups | `NaN`-response rows at `fit` time |

## Scale conventions

`predictive_variance` and `fitted_mean` follow exactly the same conventions
`predict` mirrors from the fit-row outputs: `predictive_variance` is the
linear-predictor variance `Var(eta)`, for every engine — see
["The `predictive_variance` convention"](likelihoods.md#the-predictive_variance-convention)
above for the Gaussian reconstruction of the response-scale value.
`fitted_mean` is identity for Gaussian, the exact
lognormal expectation `exp(μ + σ²_η/2)` for the Poisson log link, and the
documented **point estimate** `logit⁻¹(μ)` (ignoring linear-predictor
variance) for the Bernoulli logit link — see
["Non-Gaussian likelihoods (Laplace)"](likelihoods.md#non-gaussian-likelihoods-laplace)
above.

One exception is worth knowing about. Under `hyperparameters="integrate"`
with a **non-linear link**, the integrated `result.fitted_mean` mixes the
*transformed* per-hyperparameter values (`Σₖ wₖ·g(μₖ, σ²ₖ)`), while `predict`
transforms the *integrated* moments (`g(Σₖ wₖμₖ, Var)`). Jensen's inequality
separates the two by an amount that grows with the hyperparameter
uncertainty, so `predict().fitted_mean` is a moment-matched approximation of
`result.fitted_mean` there rather than an exact match (reproducing it exactly
would mean retaining every grid point's latent covariance). `predictive_mean`
and `predictive_variance` are exact, as is `fitted_mean` for the identity link
and for all plug-in and empirical-Bayes fits.

`predict` works on a result fitted from either Pandas or a Spark DataFrame,
but `new_data` itself must always be a Pandas DataFrame — Spark `new_data` is
not supported. **Not shipped**: a prior-based fallback for unseen levels,
predictive quantiles (use [`sample`](#joint-posterior-draws)), response-scale predictive variance for
non-Gaussian links, and an automatic future-frame construction helper (the
`NaN`-response rows above are built by hand).


## Joint posterior draws

`predictive_mean` and `predictive_variance` are marginal summaries. Targets that
are nonlinear in the predictor (shares of a total, growth rates, the probability
of a contraction, CRPS or PIT) need the joint posterior. `result.sample(n, rng)`
returns an `(n, grid_rows)` array of draws of the linear predictor `eta`, row-aligned
with `predictive_mean`, without observation noise:

```python
draws = result.sample(4000, rng=0)              # rng: anything default_rng accepts
share = draws[:, :20] / draws[:, :20].sum(axis=1, keepdims=True)
p_contraction = (draws[:, 4:] < draws[:, :-4]).mean(axis=0)
```

For a nonlinear target, score the draws directly:
`pylgm.evaluation.crps_from_draws(target_draws, actual)` is the CRPS of their
empirical distribution per column, and `(target_draws <= actual).mean(axis=0)`
is the PIT.

Every draw satisfies all exact constraints (`LinearConstraint`, intrinsic
sum-to-zero rows) to rounding. With `hyperparameters="integrate"` the draws mix
the conditional posteriors across the hyperparameter grid with the integration
weights, so they carry hyperparameter uncertainty. `sample` is available for
exact-Gaussian fits, dense or sparse, and for Laplace fits, where the draws come
from the Gaussian approximation at the mode (centred on the corrected mean when
`mean_correction=True`), so they inherit its accuracy rather than the exact
posterior's skewness.

## Sequential updates

When new observations arrive for rows already on the fitted grid, a result
absorbs them without refitting:

```python
grid.loc[grid["t"] >= 12, "y"] = np.nan        # future periods on the grid, NaN response
result = model.fit(grid)
result = result.update(new_rows)                # rows for t == 12, response observed
result = result.update(more_rows)               # updates chain
```

At fixed hyperparameters, `k` new rows `y = A x + offset + e` condition the
posterior exactly:

\[
S = A\Sigma A^\top + \sigma^2 I,\quad
\mu' = \mu + \Sigma A^\top S^{-1} r,\quad
\Sigma' = \Sigma - \Sigma A^\top S^{-1} A \Sigma,\quad
r = y - \text{offset} - A\mu,
\]

and `log_marginal_likelihood` gains \(\log \mathcal N(r; 0, S) = \log p(y_\text{new}\mid y_\text{old})\).
The cost is `k` solves against the factor the fit already holds — no new
factorisation — and \(\Sigma'\) is kept as the base posterior minus a rank-`k`
term, so marginals, `predict()`, `linear_combinations()` and `sample()` (via
Matheron's rule) all reflect the new rows, on dense and sparse fits alike. The
result is identical to a refit on all rows at the same hyperparameters.

### Non-Gaussian likelihoods

A Laplace result updates the same way. The new rows see the latent field only
through their predictor \(\eta = \text{offset} + A x\), whose prior under the
fitted posterior is \(\mathcal N(m, S_0)\) with \(m = \text{offset} + A\mu\),
\(S_0 = A\Sigma A^\top\). The mode is a `k`-dimensional Newton problem, and at
it, with \(W = -\partial^2_\eta \log p(y\mid\eta)\) and \(a = \partial_\eta \log p(y\mid\eta)\),

\[
B = I + W^{1/2} S_0 W^{1/2},\quad
\mu' = \mu + \Sigma A^\top a,\quad
\Sigma' = \Sigma - \Sigma A^\top W^{1/2} B^{-1} W^{1/2} A \Sigma,
\]

with `log_marginal_likelihood` gaining the Laplace approximation of
\(\log p(y_\text{new}\mid y_\text{old})\). Binomial rows bring their own trials
column. This is not identical to a refit: a refit re-linearises the *old* rows
around the new mode, while `update` keeps their curvature where the fit left it.
The gap is second order in the mode shift (on a 180-row Poisson panel, 1e-3 on
the latent mean and 1–2% on its standard deviations after two updates).
Survival likelihoods do not support `update`.

### Hyperparameters

With `hyperparameters="integrate"`, `update` moves the hyperparameters too.
Every retained grid point's conditional is updated as above, and its
integration weight is multiplied by that point's
\(p(y_\text{new}\mid y_\text{old}, \theta_k)\):

\[
w_k' \propto w_k\, p(y_\text{new}\mid y_\text{old}, \theta_k),\qquad
\log p(y_\text{new}\mid y_\text{old}) = \log \textstyle\sum_k w_k\, p(y_\text{new}\mid y_\text{old}, \theta_k).
\]

This is Bayes' rule on the fitted grid, and it is exact there for a Gaussian
likelihood. Latent moments, predictions, draws and the hyperparameter
marginals are re-mixed with the new weights, and the prediction context's
plug-in hyperparameters follow them. The first update refits each grid point's
conditional once, which is cheaper than a new INLA search. Later updates reuse
those conditionals. The grid itself does not move: a tighter posterior sheds
effective points (`diagnostics["inla_effective_weight"]`), and `update` warns
when fewer than three remain — refit then to re-centre it. `criteria` score
the originally fitted rows, so an updated result raises on `criteria` rather
than report stale values. A skewed `latent_strategy` needs the old rows' data
to refit its marginals, so `update` warns and reports the Gaussian grid-mixture
marginals instead.

With all the data at hand — a backtest over rolling or expanding windows —
refit each window with `fit(..., warm_start=previous)` instead
([empirical Bayes](empirical-bayes.md#rolling-and-expanding-windows)): each
window is then a real fit, rows may leave as well as arrive, and new levels
are fine.

Limits: an empirical-Bayes result keeps its hyperparameters fixed (fit with
`"integrate"` to have them move); every latent level the new rows touch must
already be on the grid (a new level raises, as in `predict()`); rows with a NaN
response are skipped; joint models do not support `update` yet. On a dense fit
the accumulated low-rank terms collapse into one factor once they hold more
rows than latents. On a sparse fit they grow by `k` columns per update, so
refit after many large updates.

### News decomposition

`result.news(release, at=targets)` breaks the revision a release causes into
one contribution per released row:

```python
news = result.news(release, at=targets)   # targets: rows to track, default the fitted grid
news.releases     # per released row: actual, expected (before the release), news
news.prediction   # linear predictor at each target row x one column per released row
news.latent       # every latent effect, indexed by (block, label) x released row
news.updated      # the posterior after the release, as result.update(release)

news.prediction.sum(axis=1)                    # = the total revision of each target
news.latent.loc["trend"]                       # each trend level's revision, by release
news.prediction.T.groupby(release["series"]).sum().T   # per series, or any other grouping
```

The **news** of a Gaussian row is its forecast error: the actual value minus
what the posterior expected, \(y_j - \mathrm E[\eta_j \mid y_\text{old}]\). The
update moves every target by a fixed linear combination of the news:

\[
\Delta\mu_t = \sum_j K_{tj}\, \text{news}_j,\qquad
K = G\,\Sigma A^\top S^{-1}.
\]

Here \(G\) is the target design, so column \(j\) depends on row \(j\)'s news
alone. The columns add up to the revision exactly.

For a non-Gaussian row the news is the working response at the new mode, minus
\(\mathrm E[\eta_j]\). This is the linearisation the Laplace update makes, so
the decomposition is still exact. `expected` is reported on the response scale
(\(\mathrm E[y_j]\) under the pre-release posterior), and `news` on the
predictor scale, where the contributions add.

On an integrated result every grid point's revision is decomposed, and the
results are mixed with the post-release weights. A final `hyperparameters`
column holds what the release changes by moving the hyperparameter posterior
itself, \(\sum_k (w_k' - w_k)\,\mu_k\). Impacts are on the linear predictor;
for a nonlinear link the response-scale revision does not split additively.

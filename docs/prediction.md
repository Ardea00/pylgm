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
predictive quantiles or simulation, response-scale predictive variance for
non-Gaussian links, and an automatic future-frame construction helper (the
`NaN`-response rows above are built by hand).


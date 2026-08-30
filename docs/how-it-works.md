# How a fit works

This page follows one `LGM.fit(frame)` call from a DataFrame to a posterior, and
then follows a forecast out past the last observation. It is the operational
companion to [theory](theory.md): what actually runs, in what order, and what
each stage can fail on.

## The pipeline

```
frame ──▶ canonical panel ──▶ compile ──▶ engine ──▶ hyperparameters ──▶ result
          (validate rows)     (build Q)   (solve)    (EB or INLA grid)
```

### 1. Canonical panel

The DataFrame is validated into a canonical form: the response and every column
an effect references must exist, index levels must have a total order, and rows
keep their original order so predictions come back aligned to what you passed
in. A Spark DataFrame is collected here, at the boundary
([Spark](spark.md)).

Rows with a `NaN` response survive this stage deliberately — they contribute no
likelihood but still create latent columns. That single rule is what makes
forecasting work; see [below](#forecasting).

### 2. Compile

Each declarative effect becomes a **latent block**: a design matrix mapping rows
to latent columns, a sparse precision \(Q\), and any constraints the effect
carries.

| effect | design | precision |
|---|---|---|
| `Fixed` | the formula's model matrix | ridge \(\varepsilon I\) |
| `IID` | one-hot over levels | \(\tau I\) |
| `RW1`/`RW2` | one-hot over ordered levels | \(\tau D^\top D\), sum-to-zero constrained |
| `AR1` | one-hot (optionally per group) | tridiagonal, proper |
| `Seasonal` | one-hot over ordered levels | \(\tau S^\top S + \delta P_0\) |
| `Besag`/`ProperCAR`/`BYM2` | one-hot over graph nodes | graph Laplacian variants |
| `SAR`/`DynamicSpatialPanel` | one-hot over nodes / (unit, period) cells | \(\tau M^\top M\) |

The blocks are then stacked into one design and one block-diagonal \(Q\). This
is also where a mistyped column is caught, naming the column and what is
available.

### 3. Pick an engine

Two independent choices:

**Exact or approximate.** A Gaussian likelihood gets the *exact* Gaussian
engine — the posterior is available in closed form, no approximation. Everything
else gets the Laplace engine.

**Dense or sparse.** Below the dense-reference guard the posterior covariance is
materialised directly. Past it, a sparse constrained-Gaussian solver takes over,
which never forms the dense covariance and instead answers marginal, predictive
and linear-combination variances through a **Takahashi selected inverse**. The
switch is automatic; what changes is that `result.covariance` raises rather than
returning a matrix that would not fit in memory
([internals](internals.md)).

### 4. Solve for the latent field

For a Gaussian likelihood the posterior precision is
\(Q + X^\top X / \sigma^2\), and one Cholesky solve gives the mean.

For a non-Gaussian likelihood the mode is found by **Newton iteration**: at each
step the likelihood is replaced by its local quadratic, giving a Gaussian
subproblem of exactly the same shape, which is solved and repeated until the
gradient vanishes. Constraints are imposed by null-space reparametrisation
throughout, so every solve happens in the reduced coordinates.

### 5. Estimate hyperparameters

With `hyperparameters="optimize"` (the default) the marginal likelihood is
maximised over \(\theta\) — type-II ML, or MAP-II when priors are declared. Each
evaluation re-runs stage 4, which is why an effect whose precision must be
rebuilt per \(\theta\) (`SAR`, `MIDAS`, `Seasonal`) is more expensive than one
that merely rescales.

With `hyperparameters="integrate"` a grid is laid over \(\theta\), every node is
solved, and the per-node marginals are mixed by posterior weight — this is what
propagates hyperparameter uncertainty into the latent marginals
([INLA integration](inla.md)).

!!! warning "Check whether an estimate pinned at its bound"
    A `Hyperparameter`'s default bounds are derived from `initial`
    (`initial × 1e-3` to `initial × 1e3`). If the reported estimate sits at one
    of them, the optimizer wanted to go further and the fit is being shaped by
    the bound, not the data. Compare `result.hyperparameters[name]` against the
    declared `lower`/`upper` and widen them if it is pinned.

### 6. What comes back

`result.mean` is the posterior mean of the whole latent vector;
`latent_marginals(block)` scopes it to one effect;
`predictive_mean`/`predictive_variance` are the linear predictor per row;
`hyperparameters` and `diagnostics` record what was estimated and whether it
converged.

## Prediction

Two genuinely different operations, and picking the wrong one is the most common
confusion:

| you want | use |
|---|---|
| new covariate values on **levels the model already saw** | `result.predict(new_data)` |
| **new** time points, regions or groups | `NaN`-response rows at `fit` time |

`predict` reuses the fitted latent posterior, so it can score any row whose
index levels were in the fit — but it cannot *create* a latent component, and
says so rather than guessing.

## Forecasting

A forecast is not a separate mechanism. Include the future periods at fit time
with a `NaN` response: they contribute nothing to the likelihood, but they get
latent columns, and the effect's structure couples those columns to the observed
ones. The posterior then extrapolates through the prior.

```python
frame = pd.concat([history, future], ignore_index=True)   # future has y = NaN
result = model.fit(frame)
result.predictive_mean[-h:]        # the forecast
result.predictive_variance[-h:]    # its uncertainty, per horizon
```

**Why the uncertainty grows.** For an `RW1`, the future cells are connected to
the last observed cell through a chain of increments, each contributing variance
\(1/\tau\). Fitting a random-walk series and forecasting six steps:

| horizon | forecast | 95% interval width | actual | covered |
|---|---|---|---|---|
| T+1 | 5.153 | 1.506 | 4.482 | ✅ |
| T+2 | 5.153 | 1.864 | 4.370 | ✅ |
| T+3 | 5.153 | 2.164 | 4.434 | ✅ |
| T+4 | 5.153 | 2.426 | 4.108 | ✅ |
| T+5 | 5.153 | 2.663 | 4.155 | ✅ |
| T+6 | 5.153 | 2.881 | 4.065 | ✅ |

Two things to read here. The **mean is flat** — the optimal forecast of a random
walk *is* the last level, and a model that invented a trend would be
overfitting. The **interval widens monotonically**, from 1.51 to 2.88, because
each extra step adds another increment's variance. This is the behaviour a
boosted tree cannot produce: outside the training range it is constant, with no
notion of accumulating uncertainty ([comparison](comparison.md)).

Different structures extrapolate differently:

- **`RW1`** — flat mean, variance growing linearly in the horizon.
- **`RW2`** — continues the local *slope*, variance growing faster.
- **`AR1`** — decays toward the mean at rate \(\rho^h\), variance saturating at
  the stationary variance rather than growing without bound.
- **`Seasonal`** — repeats the estimated pattern forward, so seasonality
  survives the horizon while the trend term handles level.
- **`Besag`/`BYM2`** — a region with no observations borrows from its
  neighbours, which is forecasting in space rather than time.

### Forecasting a network forward

`DynamicSpatialPanel` is the exception that needs its own call, because a future
period needs a future *network*, which is data you have to supply:

```python
from pylgm import forecast_dynamic_spatial_panel

forecast_dynamic_spatial_panel(result, effect, {5: graph_t5, 6: graph_t6})
```

It propagates \(\hat{x}_{t+1} = A_{t+1}^{-1} B_{t+1} \hat{x}_t\) forward,
carrying marginal variances with it, and returns a tidy frame of
`unit, time, mean, variance`. See
[spatial effects](spatial-effects.md#dynamic-spatial-panel-sdpd).

## Where time goes

Profiling a fit, in rough order:

1. **Hyperparameter estimation** dominates — every evaluation is a full solve.
   Fixing a hyperparameter instead of estimating it is the single biggest
   speedup available.
2. **`hyperparameters="integrate"`** multiplies that by the grid size,
   \((2r+1)^d\) for \(d\) hyperparameters.
3. **Sparse factorisation** grows superlinearly in the latent dimension, so a
   graph twice as large costs more than twice as much.
4. **Compilation** is negligible unless an effect rebuilds its precision per
   \(\theta\).

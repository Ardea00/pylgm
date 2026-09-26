# Linear observations

`LinearObservation` fits Gaussian data that observe sums or other linear
combinations of a finer prediction grid. This covers temporal disaggregation,
geographical reconciliation and benchmarked nowcasting without inventing a
pseudo-response at the fine level.

If the model's predictor on the supplied grid is

\[
\eta = o + Zx,
\]

a linear observation declares

\[
y_j = (C\eta)_j + \varepsilon_j,
\qquad \varepsilon_j \sim N(0,\sigma_j^2).
\]

```python
from pylgm import Gaussian, LGM, LinearConstraint, LinearObservation

result = model.fit(
    grid,
    observations=[
        LinearObservation(national_quarterly, C_quarter, sigma=1.0),
        LinearObservation(regional_annual, C_region_year, sigma=2.0),
    ],
    constraints=[
        LinearConstraint(C_accounting, national_annual),
    ],
)
```

Each operator has one row per aggregate value and one column per row of
`grid`, in the order in which the caller supplied those rows. Sparse SciPy
matrices are accepted and are preferable for large aggregation systems. pyLGM
realigns the columns if canonical panel sorting changes the internal row order.

The response column named by `LGM.response` may be absent when linear
observations or constraints are supplied. If it is present, its non-null rows
are combined with the linear observations, using the model's Gaussian `sigma`;
each `LinearObservation` uses its own scalar or row-specific `sigma`.

## Soft observations versus exact constraints

Use `LinearObservation` for published estimates, preliminary releases and
other measurements with uncertainty. Its `sigma` is the standard deviation of
the aggregate measurement, not of each fine-grid cell.

`sigma` may also be a `Hyperparameter`: one scalar standard deviation for the
whole block, estimated by empirical Bayes or integrated by INLA like any effect
hyperparameter (`transform="log"`; a `prior` makes it MAP-II). This calibrates
observation error, or measures the discrepancy between two sources, such as a
national total against the sum of its regions:

```python
discrepancy = Hyperparameter("national.sigma", initial=1.0, lower=1e-3, upper=1e3)
result = model.fit(grid, observations=[LinearObservation(national, C_national, discrepancy)])
result.hyperparameters["national.sigma"]
```

Comparing it against a `LinearConstraint` on the same rows gives the stochastic
versus exact aggregation ablation. Hyperparameter names must be unique across
the model and its observations.

Without row responses, the model's own Gaussian `sigma` is a placeholder: it
moves neither the predictions nor the log marginal likelihood. Declaring it as a
`Hyperparameter` then raises `ModelValidationError` rather than handing the
optimizer a flat direction.

Use `LinearConstraint` only for an identity that must hold exactly. pyLGM
translates `C @ eta = e` into a constraint on the latent field,
`C @ Z @ x = e - C @ o`, rejects incompatible systems and removes redundant
rows before inference.

An exact constraint is *data*: its right-hand side enters the log marginal
likelihood as

$$
\log p(y, e \mid \theta) = \log p(y \mid \theta)
  + \log \mathcal{N}\bigl(e;\ A\mu_{\text{post}},\ A Q_{\text{post}}^{-1} A^\top\bigr),
$$

where $\mu_{\text{post}}$ and $Q_{\text{post}}$ condition on $y$ and on the
structural constraints only (intrinsic sum-to-zero rows and `LGM(constraints=...)`
label rows, which remain pure conditioning, as R-INLA's `extraconstr`). Empirical
Bayes and INLA therefore learn hyperparameters from the aggregates, and an AR1
fitted to annual totals alone recovers the Chow-Lin likelihood. The density is
that of the kept rows: supplying `2 C, 2 e` instead of `C, e` shifts the log
marginal likelihood by `-rows * log 2`, a constant in $\theta$ that leaves the
hyperparameter estimates unchanged.

The returned `predictive_mean` and `predictive_variance` remain aligned with
the original fine grid, rather than with the shorter aggregate-observation
vector. The likelihood is standardized internally, including the Jacobian
normalization, so posterior inference and log marginal likelihood retain the
declared heterogeneous observation variances.

## Building operators

`pylgm.operators` builds common operators directly from a pandas `DataFrame`,
with columns in the caller's row order, ready to pass to `LinearObservation`
or `LinearConstraint`.

```python
from pylgm.operators import aggregation_operator, compose, cumulation_operator, difference_operator
```

`aggregation_operator(frame, by, *, weights=None, rows=None)` sums (or
weighted-sums) rows into one operator row per distinct value of `by`, sorted
ascending; it returns `(operator, keys)`, where `keys` is a `DataFrame` naming
the group for each operator row.

`difference_operator(frame, time, panel=None, *, lag=1, order=1)` applies
`(1 - L^lag)^order` within each panel unit, after sorting that unit's rows by
`time`; it returns `(operator, keys)`, where `keys` names the unit and time of
each row's target position.

`cumulation_operator(frame, time, panel=None, *, lag=1)` is the strided
cumulative sum within each unit (`lag=1` is the running sum); for `lag=1,
order=1`, `difference_operator(frame) @ cumulation_operator(frame)` selects
the identity rows at position `i >= 1` of each unit, so cumulation is a right
inverse of differencing past the first observation.

`compose(*operators)` chains them left to right into one sparse operator,
converting any dense arguments and raising on shape mismatches.

### Data scale

What is linear in the predictor, and so expressible with `LinearObservation`,
depends on which scale the predictor lives on:

| Data                                              | Linear in the predictor?                                                   |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Levels, observed as sums                            | Yes — `aggregation_operator`.                                                       |
| Levels, observed as changes                         | Yes — `difference_operator`.                                                        |
| Predictor on differences, aggregates on levels      | Yes — `compose(aggregation_operator(...), cumulation_operator(...))`, with the starting level of each unit supplied by an intercept or offset. |
| Predictor on log levels, observed growth rates      | Yes — `difference_operator` on the log scale, since `log x_t - log x_{t-lag}` is linear. |
| Predictor on log levels, aggregates on levels        | Yes, with `scale="log"` (below).                                                    |

## Aggregates of exponentiated predictors

`scale="log"` on `LinearObservation` or `LinearConstraint` declares the
operator on `exp(eta)` rather than on `eta` itself:

\[
y_j = (C\exp(\eta))_j + \varepsilon_j, \qquad C\exp(\eta) = e.
\]

```python
result = model.fit(
    grid,
    observations=[LinearObservation(values, C, sigma, scale="log")],
    constraints=[LinearConstraint(C_annual, rhs, scale="log")],
)
```

pyLGM fits this by Gauss-Newton relinearization: each `scale="log"` item is
replaced by its tangent at a trial grid predictor `eta0`, `C @ diag(exp(eta0))`
with a matching offset shift, giving an ordinary identity-scale item; the
inner fit's grid predictor becomes the next `eta0`, and the loop repeats to a
fixed point. At that point the gradient (and constraint Jacobian) of the
linearized problem equals the true one, so the fixed point is the exact
conditional mode; the log marginal likelihood reported is the Gauss-Newton
Laplace approximation at that mode. This costs a few inner fits per
hyperparameter value evaluated, each warm-started from the previous
relinearization.

Growth rates on log levels still use `difference_operator` with the default
`scale="identity"` (that observation is already linear); it is aggregates of
levels, `C @ exp(eta)`, that need `scale="log"`.

Limitations: the model criteria (CPO/PIT/WAIC) of `scale="log"` rows are
evaluated on the linearized model at the fixed point, not on the original
nonlinear one. `latent_strategy="laplace"` still rejects constraints, as
without `scale="log"`. Non-convergence of the relinearization raises
`InferenceError`.

## Scope

Linear observations currently support Gaussian `LGM` models with pandas input.
They work with fixed fits, empirical-Bayes optimization and INLA integration.
Exact constraints run on both the dense and the sparse Gaussian paths; on the
sparse path a `LinearConstraint` row may span several latent blocks (label
constraints passed to `LGM(constraints=...)` must still touch a single block).
A `Joint` model accepts them too, per outcome — see
[Linear observations and constraints](joint-models.md#linear-observations-and-constraints).

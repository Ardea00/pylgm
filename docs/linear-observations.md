# Linear observations

`LinearObservation` fits Gaussian data that observe sums or other linear
combinations of a finer prediction grid, under any row likelihood. This covers temporal disaggregation,
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
| Chain-linked volumes, aggregates across groups | Not additive; linear after weighting by annual-overlap factors (below). |

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

## Chain-linked volumes

Under annual overlap, each sub-period of period `p` is valued at the prices
of period `p-1` and linked to the chain by one factor per group and period.
Conversely, a chain-linked volume `CL` becomes a value at previous-period
prices when multiplied by `k[p] = current_total[p-1] / volume_total[p-1]`,
built from the previous period's totals. Values at previous-period prices
add up across groups; chain-linked volumes do not, since each group carries
its own chain of factors. A temporal sum does stay linear: within one
period every sub-period of a group shares the same factor, so the group's
chain-linked sub-periods sum to its chain-linked period total.

`pylgm.index_numbers` builds the rescaling factors from published totals and
applies them:

```python
from pylgm.index_numbers import align_factors, chain, overlap_factors, unchain

factors = overlap_factors(
    annual_totals, volume="volume", current="current", period="year", group="group",
)
previous_period_prices = unchain(quarterly, "volume", factors, period="year", group="group")
volumes_again = chain(quarterly, previous_period_prices, factors, period="year", group="group")
```

`overlap_factors` reads one row per `(group, period)` of `annual_totals` and
emits, for every period whose totals are both finite and positive, the
factor for the *next* period. `align_factors` looks up the matching factor
for each row of a finer frame; `unchain`/`chain` multiply or divide a column
(or an array whose last axis matches the frame) by that factor, so posterior
draws of shape `(n_draws, n_rows)` work directly.

Because rescaled values are additive but chain-linked ones are not, an
accounting identity across groups only holds after weighting each group's
chain-linked volume by its own factor:

\[
\sum_g k[g, p]\, CL[g, t] = k[\text{agg}, p]\, CL[\text{agg}, t],
\]

for a sub-period `t` of period `p`. This is an `aggregation_operator` with
`weights = k_g / k_agg`, and `scale="log"` when the predictor lives on log
chain-linked levels:

```python
weight = align_factors(frame, factors, period="year", group="group")
weight_aggregate = align_factors(aggregate_frame, factors, period="year", group="group")
operator, keys = aggregation_operator(frame, "quarter", weights=weight / weight_aggregate)
constraint = LinearConstraint(operator, aggregate_chain_linked_values, scale="log")
```

**Publication lags.** The factor for period `p` only needs the totals of
`p-1`. If group totals are published with a lag of `L` periods, the temporal
constraint (the sum of a group's own sub-periods to its own period total)
applies only to periods whose totals are published, the cross-group
constraint above is exact — its weights are known — up to the first
unpublished period plus one, and beyond that the weights are unknown.
`overlap_factors(..., through=...)` carries the last known factor forward
(flagged `carried=True`) for those periods; rows using a carried factor
should enter as a soft `LinearObservation` with an estimated `sigma` (a
`Hyperparameter`) absorbing the error of the carried weight, not as a hard
`LinearConstraint`.

For example, with a publication lag of 2, and the current period called `T`:
periods `<= T-2` get both constraints exactly (their factors use fully
published totals); period `T-1` keeps the exact cross-group constraint,
since its factor only needs the totals of `T-2`; period `T` has no group
totals to build its own factor from, so `through=T` carries the `T-1` factor
forward with `carried=True`, and the cross-group identity for period `T`
should be declared as a `LinearObservation` rather than a
`LinearConstraint`.

## Scope

Linear observations and constraints work with any `LGM` likelihood on pandas
input, and with fixed fits, empirical-Bayes optimization and INLA integration.
A Gaussian model folds them into one exact Gaussian projection. Any other
likelihood keeps its own rows and fits on the Laplace engine: each
`LinearObservation` row becomes a Gaussian pseudo-row, and a `LinearConstraint`
enters as `log p(y, e) = log p(e) + log p(y | e)` -- the exact prior density of
`A x` at `e`, plus a Laplace step on the prior conditioned on `A x = e`, so the
reported mode is the exact constrained mode.

`scale="log"` items are fitted by relinearization: each pass replaces
`C g(eta)` by its tangent. On the Laplace engines (any non-Gaussian likelihood,
and every `Joint`) a `LinearObservation` also adds the second-order term the
tangent drops, `-sum_k r_k / sigma_k^2 grad^2 (C_k g)`, as a curvature
correction centred at the linearization point: the mode is unchanged, each
pass is a full Newton step, and the Laplace evidence uses the exact Hessian. A
fixed point where that Hessian is indefinite is a saddle and raises, rather
than being reported as a mode -- give the predictor an intercept, as any
aggregate model should have. Two cases keep the Gauss-Newton curvature, so
their evidence is approximate when the aggregates are not fitted exactly: a
`LinearConstraint` (the missing term needs its Lagrange multipliers), and a
Gaussian `LGM` on the `exact_gaussian` engine (the correction is not a
Gaussian row).

Exact constraints on a non-Gaussian model need dense `c x latent` workspace
for `c` constraint rows. For many aggregates -- one per node of a network --
prefer a `LinearObservation` with a small `sigma`: its pseudo-rows stay sparse.

Exact constraints run on both the dense and the sparse paths; on the
sparse path a `LinearConstraint` row may span several latent blocks (label
constraints passed to `LGM(constraints=...)` must still touch a single block).
A `Joint` model accepts them too, per outcome — see
[Linear observations and constraints](joint-models.md#linear-observations-and-constraints).

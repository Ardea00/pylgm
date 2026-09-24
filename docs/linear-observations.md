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

## Scope

Linear observations currently support Gaussian `LGM` models with pandas input.
They work with fixed fits, empirical-Bayes optimization and INLA integration.
Exact constraints run on both the dense and the sparse Gaussian paths; on the
sparse path a `LinearConstraint` row may span several latent blocks (label
constraints passed to `LGM(constraints=...)` must still touch a single block).

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

Use `LinearConstraint` only for an identity that must hold exactly. pyLGM
translates `C @ eta = e` into a constraint on the latent field,
`C @ Z @ x = e - C @ o`, rejects incompatible systems and removes redundant
rows before inference.

The returned `predictive_mean` and `predictive_variance` remain aligned with
the original fine grid, rather than with the shorter aggregate-observation
vector. The likelihood is standardized internally, including the Jacobian
normalization, so posterior inference and log marginal likelihood retain the
declared heterogeneous observation variances.

## Scope

Linear observations currently support Gaussian `LGM` models with pandas input.
They work with fixed fits, empirical-Bayes optimization and INLA integration.
Exact cross-block constraints use the dense Gaussian path; models large enough
to route to the sparse solver still inherit its existing restriction against
cross-block constraint rows.

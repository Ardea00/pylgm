# Empirical Bayes and priors

## Empirical Bayes

Declaring an effect's precision (or a Gaussian likelihood's `sigma`) as a
`Hyperparameter` — instead of passing a plain number — makes `LGM.fit`
estimate it by **type-II maximum likelihood**: it optimizes the marginal
likelihood over the declared hyperparameter(s), starting from `.initial` and
respecting any declared `lower`/`upper` bounds, for both the
`exact_gaussian` and `laplace` engines:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM

model = LGM(
    response="y",
    likelihood=Gaussian(0.5),
    predictor=Fixed("1")
    + IID("region", index="region", precision=Hyperparameter("region_precision", initial=1.0)),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="exact_gaussian")
result.hyperparameters["region_precision"]  # the type-II ML estimate
result.diagnostics["empirical_bayes_converged"]
```

The fitted value is exposed on `result.hyperparameters`; a model with no
declared `Hyperparameter` leaves `result.hyperparameters` as `None`.

When a declared `Hyperparameter` also carries a `prior` (e.g. `PCPrecision`,
`GaussianPrior`), `LGM.fit` estimates it by **MAP-II** instead of pure
type-II ML: the same marginal-likelihood objective is penalized by the
prior's log density evaluated on the hyperparameter's native scale (no
Jacobian correction for `transform`), for both the `exact_gaussian` and
`laplace` engines. A prior-free `Hyperparameter` stays pure type-II ML.
`result.diagnostics["hyperparameter_penalized"]` records whether any
declared hyperparameter was penalized this way:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, PCPrecision

model = LGM(
    response="y",
    likelihood=Gaussian(0.5),
    predictor=Fixed("1")
    + IID(
        "region", index="region",
        precision=Hyperparameter(
            "region_precision", initial=1.0,
            prior=PCPrecision(upper_sd=1.0, alpha=0.01),
        ),
    ),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="exact_gaussian")
result.hyperparameters["region_precision"]  # the MAP-II estimate
result.diagnostics["hyperparameter_penalized"]  # True
```

This is still a point estimate, not a marginal over the hyperparameter —
full posterior *integration* over hyperparameters is the separate
`hyperparameters="integrate"` mode described below. YAML declaration of
`Hyperparameter`s is not yet supported — this is a Python-API-only
capability. Runnable examples live at
[`examples/empirical_bayes/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/empirical_bayes/README.md)
(prior-free, type-II ML) and
[`examples/map_ii/README.md`](https://github.com/Ardea00/pylgm/blob/main/examples/map_ii/README.md) (prior'd, MAP-II).

## Bounded hyperparameters

Hyperparameter inference runs on an unconstrained internal scale chosen per
parameter by its `transform`:

| `transform` | Natural domain | Used for |
| --- | --- | --- |
| `"log"` (default) | `θ > 0` | `sigma`, effect precisions (τ) |
| `"logit"` | a bounded interval `(a, b)` | `ProperCAR` `rho`, `BYM2` `phi` |
| `"identity"` | `θ > 0` | accepted, but **not** wired into inference (treated as `log`) |

Both the empirical-Bayes optimizer and the INLA grid work in this internal
space, and the INLA importance weights carry the corresponding Jacobian
`Σ log|dθ/du|`. For log-scale parameters this is exactly the previous
behaviour, so existing models are unaffected; the abstraction is what makes a
bounded parameter such as ρ estimable at all.

A `"logit"` hyperparameter may declare any finite `initial` (zero or negative
included) and may leave `lower`/`upper` as `None`, letting the effect supply
the interval — which is how `ProperCAR` passes its graph-derived range for ρ.


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
parameter by its `transform`. Both the empirical-Bayes optimizer and the INLA
grid work in that internal space, and the INLA importance weights carry the
corresponding Jacobian `Σ log|dθ/du|`.

### Choosing a transform

| `transform` | Natural domain | Internal scale | Use it for |
| --- | --- | --- | --- |
| `"log"` (default) | `θ > 0` | `u = log θ` | `sigma`, every effect **precision** `τ`, the `phi` of NegativeBinomial/Gamma/Beta, the Weibull `shape` |
| `"logit"` | a bounded interval `(a, b)` | logit on that interval | `AR1` `rho`, `ProperCAR` `rho`, `SAR` `rho`, `DynamicSpatialPanel` `rho`, `BYM2` `phi` |
| `"identity"` | the whole real line | `u = θ` | `MIDASParametric` `shape1`/`shape2`, `DynamicSpatialPanel` `gamma` and `eta` |

The rule of thumb is the parameter's **domain**, not its typical value:

- **Strictly positive and unbounded above** → `"log"`. This is the default, so
  precisions need nothing declared.
- **Confined to an interval** → `"logit"`. Any correlation-like parameter is
  here. Under `"logit"` the `initial` may be any finite real — including `0.0`
  or a negative value — and `lower`/`upper` may be left `None` for the effect to
  supply, which is how `ProperCAR` passes its graph-derived range for ρ.
- **Real line** → `"identity"`. Requires finite `lower`/`upper`; they default to
  a symmetric window around `initial`.

!!! tip "The most common first error"
    Declaring a correlation with the default transform:

    ```python
    Hyperparameter("a.rho", initial=0.0)          # ValueError
    Hyperparameter("a.rho", initial=0.0, transform="logit")   # correct
    ```

    `initial=0.0` is the natural starting guess for a correlation but is
    invalid under `"log"`, whose domain is `θ > 0`. The error says so and names
    the fix.

### When an estimate hits its bound

An estimate that lands on the edge of its interval is being set by **the bound,
not the data** — the optimizer wanted to keep going. Empirical Bayes detects
this, warns, and records it:

```python
result.diagnostics["hyperparameters_at_bound"]   # "trend.precision", or "" if none
```

This matters because `Hyperparameter` **derives its bounds from `initial`** when
you do not give them: `initial × 1e-3` to `initial × 1e3`. An `initial` chosen
casually therefore silently constrains the fit. Widen `lower`/`upper` and refit:

```python
Hyperparameter("trend.precision", initial=10.0, lower=1e-4, upper=1e9)
```

Closeness is judged on the transform's own scale, which is where the optimizer
works — a precision of `9999.98` against an upper bound of `10000` counts as
pinned, which it is in every sense that matters, even though it is `0.02` away
in natural units.

A precision driven to its **upper** bound usually has a specific meaning: that
effect is being estimated away. An infinite precision is a zero-variance field,
i.e. "the data prefers no such term at all". Widening the bound will not change
that conclusion — it will just let the estimate run further. The fix there is to
drop the term or accept it, not to loosen the interval.

Under `hyperparameters="integrate"` the bounds define the grid rather than a
search region, so this diagnostic is reported for the empirical-Bayes path only.

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

## Parallel evaluations

`LGM.fit(..., num_workers=N)` (and `Joint.fit`) runs a hyperparameter
search's independent conditional fits — a finite-difference gradient's
evaluation points, or an INLA grid (see docs/inla.md) — concurrently on a
thread pool instead of one at a time. While the workers run, BLAS is limited
to `blas_threads` threads (default `1`) so the workers do not oversubscribe
the machine's cores.

The fits are independent computations recombined in the serial order, so a
run with `num_workers=N` is **bit-identical** to `num_workers=1,
blas_threads=1`. It is not bit-identical to the default `num_workers=1,
blas_threads=None`, which leaves BLAS threading untouched: a different BLAS
thread count changes the summation order, giving differences of the order
of 1e-9 relative.

`blas_threads` also works on its own, with `num_workers=1`. On models with
many small dense blocks, letting BLAS spawn a thread per core often makes a
single fit slower rather than faster, and `blas_threads=1` can be several
times quicker. Measure it on your own model.

More workers cost memory: a gradient batch keeps up to twice as many
compiled models as there are hyperparameters, plus `num_workers` fits, alive
at once. Choose `num_workers` from the machine's physical cores and the
model's size, and measure rather than assume that more is better.

BLAS threads are limited through `threadpoolctl`, which controls OpenBLAS,
MKL and BLIS. It cannot control Apple Accelerate, which the macOS NumPy and
SciPy wheels typically use. There `blas_threads` has no effect, and
`VECLIB_MAXIMUM_THREADS`, set before Python starts, is the only control.
The fits still run in parallel; only the BLAS thread cap is missing.

## Rolling and expanding windows

A backtest fits the same model on a sequence of windows. It can grow the window
(expanding) or move it (sliding: old periods leave, new ones arrive). Each
window's fit is a good starting point for the next one:

```python
previous = None
for window in windows:
    result = model.fit(window, hyperparameters="optimize", warm_start=previous)
    previous = result
```

`warm_start` works with both `"optimize"` and `"integrate"` (and in `Joint.fit`):

- The hyperparameter search starts at the previous estimates. For an integrated
  result that is its INLA mode.
- A Laplace mode starts at the previous latent mean, matched by label. Levels new
  to the window start at zero, and levels that left it are dropped.

An estimate the previous window left pinned at a bound is not used as a start,
because the objective is flat there and the search would stall. The warm fit
keeps only a label-to-value map, never the previous result, so a long loop does
not chain every window into memory.

A warm start changes where the search begins, not the objective, so each
window's result is a fresh fit of that window. It matches a cold fit to the
optimizer's stopping tolerance, which is a relative plateau of `1e-5` in the
objective (about 0.01 nats on a log marginal likelihood near -1000). A 20 × 80
panel with eight windows of 40 periods gave these results:

| model | windows | cold | warm | evaluations |
|---|---|---|---|---|
| Gaussian, `"optimize"` | expanding | 2.0 s | 1.3 s | 931 → 525 |
| Gaussian, `"optimize"` | sliding | 2.1 s | 1.4 s | 1071 → 637 |
| Poisson, `"optimize"` | sliding | 1.9 s | 1.4 s | 465 → 260 |
| Gaussian, `"integrate"` | sliding | 19.8 s | 18.9 s | — |
| Poisson, `"integrate"` | sliding | 7.2 s | 6.8 s | — |

These timings come from macOS. The saving is typical but not guaranteed: the
search path depends on floating-point details of the platform's BLAS, and on
single windows run on Linux and Windows, the warm search took more evaluations
than the cold one. Measure it on your own experiment. Integration gains little
either way: the search a warm start shortens is a small part of the work, and
exploring the grid is most of it.

`result.update(new_rows)` (see [prediction](prediction.md#sequential-updates))
is the other tool. It absorbs new rows without refitting and needs no old data.
When the data are at hand, as in a backtest, a warm-started fit is the one to
use: it is a real fit of the window, so it handles sliding windows, new levels
and moving hyperparameters.

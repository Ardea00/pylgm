# Effects

pyLGM's predictor is a **sum of effects**: `Fixed("1 + x") + IID(...) + RW1(...)`.
Each structured effect takes a fixed `precision` or a declared
`Hyperparameter` to be estimated (see
[Empirical Bayes and priors](empirical-bayes.md)).

## Every effect at a glance

| Effect | What it models | Documented |
| --- | --- | --- |
| `Fixed(formula)` | Fixed effects from a formulaic formula | [core](#core-building-blocks) |
| `IID(name, index)` | Exchangeable random effects per level | [core](#core-building-blocks) |
| `RW1` / `RW2` | Smooth temporal trends (1st / 2nd difference penalty) | [core](#core-building-blocks) |
| `AR1(name, index, rho=)` | Stationary first-order autoregression | [AR1](#ar1-effect) |
| `AR1(..., group=)` | One independent AR1 series **per panel unit** | [group-wise AR1](#group-wise-ar1) |
| `Seasonal(name, index, period=)` | A slowly-drifting periodic pattern | [Seasonal](#seasonal-effect) |
| `Weighted(effect, by)` | Modulates an indexed effect by a numeric column — spatially-varying coefficients | [Weighted](#weighted-effects) |
| `Copy(name, index, scale=)` | A second occurrence of an existing field at another index, optionally rescaled | [Copy](#copy) |
| `MIDAS(name, columns)` | Mixed-frequency distributed lag, smoothness-penalised | [MIDAS](#midas-smooth-lag-effect) |
| `MIDASParametric(...)` | Restricted lag curve (exp-Almon / Beta kernel) | [restricted MIDAS](#restricted-midas-effect-parametric-lag-weights) |
| `SpaceTime(name, space, time, interaction=)` | Knorr-Held space-time interaction, types I–IV | [SpaceTime](#spacetime-effect-knorr-held-interaction) |
| `Besag(name, index, graph)` | Intrinsic CAR (ICAR) over a neighbour graph | [spatial](spatial-effects.md#besag-intrinsic-car-icar-spatial-effect) |
| `ProperCAR(..., rho=)` | Proper CAR with a spatial-dependence parameter | [spatial](spatial-effects.md#proper-car-spatial-effect) |
| `BYM2(..., phi=)` | Structured + unstructured convolution, interpretable `φ` | [spatial](spatial-effects.md#bym2-spatial-effect) |
| `SAR(name, index, graph, rho=)` | **Directed** network autoregression `(I−ρW)ᵀ(I−ρW)` | [spatial](spatial-effects.md#directed-spatial-autoregressive-sar-effect) |
| `DynamicSpatialPanel(...)` | **One network per period** — SDPD, with `ρ`, `γ`, `η` | [spatial](spatial-effects.md#dynamic-spatial-panel-sdpd) |
| `LGM(constraints=...)` | Arbitrary linear constraints `A x = e` on the latent field | [constraints](#linear-constraints-extraconstr) |

Graphs may be **weighted**, so the CAR family also models ownership,
interbank-exposure and supply-chain networks rather than only geography — see
[weighted neighbour graphs](spatial-effects.md#weighted-neighbour-graphs).

Everything on the spatial and network side lives on the
[spatial and network effects](spatial-effects.md) page; the rest is documented
below. Runnable scripts for all of them are in the
[examples gallery](examples.md).

## Core building blocks

### `Fixed(formula, prior_precision=1e-6)`

An ordinary fixed-effect block, built from a
[formulaic](https://matthewwardrop.github.io/formulaic/) formula, so
`"1 + x"`, `"1 + x + I(x**2)"` and categorical expansion all behave as they do
in R-style formulas. `prior_precision` is a ridge on the coefficients — the
default is deliberately tiny, i.e. an almost-flat prior, so the estimates match
what an unpenalised regression would give.

Every model needs at most one `Fixed` term; its columns are recomputed from the
formula when `predict` scores new rows, which is how a categorical level unseen
at fit time is caught rather than silently mis-encoded.

### `IID(name, index, precision=1.0)`

Exchangeable random effects: one latent value per level of `index`, independent
given the precision, `Q = τI`. This is the workhorse for "a per-group offset I
want shrunk toward zero" — regions, firms, individuals.

Unlike a `Fixed` dummy per group, an `IID` term **pools**: the estimated `τ`
decides how much each level is pulled toward the overall mean, so levels with
few observations are shrunk more. That is the difference the
[method comparison](comparison.md#versus-regression-and-glms) measures.

`IID` is also how you add a **frailty** to a survival model, and the
unstructured half of a `BYM2` convolution.

### `RW1(name, index, precision=1.0)` and `RW2(...)`

Smoothness priors over an **ordered** index. `RW1` penalises first differences
(the level wanders, the trend is locally flat), `RW2` penalises second
differences (the *slope* wanders, so the fitted trend is smoother and
extrapolates linearly rather than flat):

\[
Q_{\text{RW1}} = \tau D_1^\top D_1,
\qquad
Q_{\text{RW2}} = \tau D_2^\top D_2 .
\]

Both are **intrinsic**: rank-deficient by 1 and 2 respectively, with the null
space (level, and level+slope) removed by sum-to-zero constraints. That null
space is a nuisance absorbed by the intercept — which is exactly the opposite of
[`Seasonal`](#seasonal-effect), whose null space is the signal.

Choose by what you believe about the trend, and note the forecasting
consequence: past the last observation `RW1` projects a flat mean with variance
growing linearly in the horizon, while `RW2` continues the local slope with
variance growing faster. See
[how forecasting works](how-it-works.md#forecasting).

**Regular spacing is assumed.** Both relate *consecutive* levels with no notion
of the gap between them, so a missing period should enter as a `NaN`-response
row to keep the grid regular.

## AR1 effect

`AR1(name, index, precision=1.0, rho=0.5)` declares a stationary first-order
autoregressive latent effect on the ordered, observed levels of `index` (the
same level-ordering rule as `RW1`/`RW2`; at least two levels are required).
Its precision is `Q = τ/(1−ρ²)·T`, with `T` tridiagonal (unit corners,
`1+ρ²` on the interior diagonal, `−ρ` off it), which inverts to
`Cov[i, j] = (1/τ)·ρ^|i−j|`. **`τ` is the marginal precision** — INLA's
convention — so the marginal variance of every level is exactly `1/τ`,
directly comparable to an `IID` effect's precision or `BYM2`'s `τ`, *not* an
innovation precision. `ρ = 0` collapses `Q` to exactly `τ·I` (independent
levels); as `ρ → 1` the effect approaches the intrinsic `RW1` limit.

```python
import numpy as np
import pandas as pd
from pylgm import AR1, Fixed, Gaussian, LGM

frame = pd.DataFrame({
    "t": range(10),
    "y": [1.0, 1.4, 1.1, 1.6, 1.3, 1.8, 1.5, 2.0, 1.7, 2.2],
})
model = LGM(
    response="y",
    predictor=Fixed("1") + AR1("trend", index="t", precision=2.0, rho=0.7),
    likelihood=Gaussian(sigma=0.3),
)
result = model.fit(frame)
print(np.round(result.latent_marginals("trend").mean, 3))
# [-0.466 -0.244 -0.358 -0.065 -0.166  0.129  0.027  0.322  0.22   0.506]
```

`rho` accepts either a fixed float strictly inside `(-1, 1)` (plug-in, shown
above) or a declared `Hyperparameter` with `transform="logit"`, in which case
ρ is **estimated** by empirical Bayes and, under
`hyperparameters="integrate"`, integrated over jointly with τ by INLA — a
non-logit `Hyperparameter` for `rho` raises `CompilationError`. The interval
is the fixed `(-1, 1)` inset by `1e-6` (no graph-derived bound, unlike
`ProperCAR`'s ρ); a `Hyperparameter`'s own `lower`/`upper` narrow it further if
you set them.

One caveat on that interval: **conditioning degrades as ρ approaches the bound
and as the series grows** — `cond(Q)` runs roughly `2e6·n` at the inset edge, so
a 1000-level series fitted at ρ ≈ 1 retains only about seven significant digits.
Nothing fails, but treat an ρ̂ pinned at the bound on a long series as "the data
want a random walk" and consider `RW1` instead.

```python
from pylgm import AR1, Fixed, Gaussian, Hyperparameter, LGM
from pylgm.priors import PCPrecision

rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
tau = Hyperparameter("trend.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(
    response="y",
    predictor=Fixed("1") + AR1("trend", index="t", precision=tau, rho=rho),
    likelihood=Gaussian(sigma=0.3),
)
eb = model.fit(frame)                                 # empirical Bayes
eb.hyperparameters["trend.rho"]                        # point estimate of rho

post = model.fit(frame, hyperparameters="integrate")   # INLA over (tau, rho)
post.hyperparameter_marginals()["trend.rho"].mean      # rho marginal
```

Unlike `RW1`/`RW2`, `AR1` is proper/full-rank, so it carries **no
sum-to-zero constraint**: it works under all three latent strategies,
including `latent_strategy="laplace"` (full Laplace), where `RW1`/`RW2` are
rejected because they are intrinsic/constrained. `result.predict(new_data)`
works on `AR1` exactly as described in
["Predicting new rows"](prediction.md#predicting-new-rows) above; an unseen time level
raises the same `ValueError` pointing at the `NaN`-response workflow.

**Regular spacing is assumed.** `AR1` relates *consecutive* ordered levels —
it has no notion of the gap between them — so a missing period (a skipped
month, say) is silently treated as a single step, understating the true
elapsed time and its correlation decay. If a period is absent from the data
but should count as a step, include it as a row with a `NaN` response at
`fit` time: it contributes no likelihood but still creates its latent column
and restores regular spacing, exactly like the future-period rows in
["Predicting new rows"](prediction.md#predicting-new-rows) above. **Not shipped**:
irregular-spacing support (`ρ^Δt`) and AR(p) effects.

**From YAML:** the standalone `load_model` frontend declares `type: ar1`,
indexed by a single `index`, with optional fixed `rho` (default `0.5`),
`precision` (default `1.0`), and a `group` column for the group-wise variant.
Estimating `rho`/`precision` from YAML stays Python-API-only.

```yaml
predictor:
  effects:
    - {name: trend, type: ar1, index: period, rho: 0.5, precision: 1.0, group: unit}
```

### Group-wise AR1

`AR1(name, index, precision, rho, group=None)` with `group` set declares **one
independent AR1 series per level of that column** — a separate series per firm,
country, or region — sharing `precision` and `rho` across groups but not their
realizations. This is the panel case: a common persistence parameter estimated
from all units at once, with each unit keeping its own trajectory.

```python
model = LGM(
    response="y",
    predictor=Fixed("1") + AR1(
        "dyn", index="t", group="firm",
        rho=Hyperparameter("dyn.rho", initial=0.0, transform="logit"),
        precision=Hyperparameter("dyn.precision", initial=1.0),
    ),
    likelihood=Gaussian(sigma=0.3),
)
```

The latent field has one cell per `(group, index)` pair, laid out group-major
and labelled `"<group>@<level>"`, so the precision is block diagonal —
`I_G ⊗ T(ρ)`, one contiguous AR1 band per group. Pooling is in the
hyperparameters only: fitting `G` groups jointly is *exactly* `G` separate AR1
fits at the same `ρ` and `precision` (the test suite pins this equality), so no
group borrows latent strength from another.

The time levels are the union of levels observed anywhere in the frame, so the
grid is balanced across groups: a `(group, period)` cell absent from the data
still gets a latent column and a prediction, exactly like the `NaN`-response
future rows above. `result.predict(new_data)` scores grouped rows; an unseen
*group* raises, pointing at the same `NaN`-response workflow as an unseen level.

**Regular spacing is assumed within each group**, for the same reason it is for
the ungrouped effect.

## Seasonal effect

`Seasonal(name, index, period, precision=1.0, ridge=1e-6)` declares a
**slowly-drifting seasonal pattern** of the declared `period` (4 for quarterly
data, 12 for monthly). With `RW1`/`RW2` for trend it gives the classic
trend + seasonal + irregular decomposition:

```python
model = LGM(
    response="y",
    predictor=Fixed("1")
    + RW1("trend", index="t", precision=Hyperparameter("trend.precision", initial=10.0))
    + Seasonal("seas", index="t", period=4,
               precision=Hyperparameter("seas.precision", initial=10.0)),
    likelihood=Gaussian(sigma=0.25),
)
result = model.fit(frame)
result.latent_marginals("seas").mean   # the seasonal component
```

The penalty sums every `period` consecutive levels and shrinks that sum toward
zero, so `precision` controls **how fast the seasonal pattern may drift**, not
its size: a pattern that repeats exactly is unpenalized by `precision`, and a
large `precision` means "the shape is nearly fixed from cycle to cycle".

**Why there is a `ridge` and no constraint.** The directions `precision` leaves
unpenalized are exactly the fixed seasonal patterns — the signal, not a
nuisance. This is the opposite of `RW1`/`RW2`, whose unpenalized level and
slope *are* nuisances absorbed by the intercept, and so are removed by
sum-to-zero constraints. Constraining them away here would annihilate a stable
seasonal pattern entirely. Instead, as in
[`MIDAS`](#midas-smooth-lag-effect), those directions carry a fixed,
`precision`-independent `ridge` on their orthogonal projector: the pattern
stays estimable, the block stays positive definite, and the effect declares no
constraints. Raise `ridge` to shrink the seasonal amplitude toward zero; lower
it to free the pattern further.

This differs from R-INLA's `seasonal`, which leaves the model rank-deficient
and applies a single sum-to-zero constraint for intercept confounding. The
ridge reaches the same place — a weakly-identified fixed pattern — while
keeping the conditioned precision positive definite, which is what the engines
here require.

**Regular spacing is assumed**, as for `RW1`/`RW2`/`AR1`: the effect relates
*consecutive* ordered levels and has no notion of the gap between them. A
missing period should enter as a `NaN`-response row so the grid stays regular.
Future periods included that way also receive an extrapolated seasonal
posterior, so the pattern projects forward — see
["Predicting new rows"](prediction.md#predicting-new-rows).

**From YAML:** the standalone `load_model` frontend declares `type: seasonal`,
indexed by a single `index`, with a **required** `period` and optional fixed
`precision` (default `1.0`) and `ridge` (default `1e-6`). Estimating `precision`
from YAML stays Python-API-only. Write floats in full YAML form (`1.0e-6`, not
`1e-6`, which YAML reads as a string).

```yaml
predictor:
  effects:
    - {name: seas, type: seasonal, index: month, period: 12, precision: 1.0, ridge: 1.0e-6}
```

## Weighted effects

`Weighted(effect, by)` modulates an indexed effect by a numeric column: the
design becomes `diag(by) A` instead of the plain incidence `A`, so the
effect's contribution to the predictor at row `i` is `by_i * u_{index(i)}`
rather than plain `u_{index(i)}`. This is R-INLA's `f(index, weights,
model=...)`.

The model this unlocks is a **spatially-varying coefficient**: a covariate
whose slope is itself a latent field instead of one shared number.

```
log mu_i = alpha + z_i * u_{s(i)},    u ~ IID(tau)
```

```python
from pylgm import Fixed, IID, LGM, Poisson, Weighted
from pylgm.parameters import Hyperparameter

# ... a frame whose `region` is the index, `z` the covariate whose slope
# varies, and `y` the Poisson response ...
result = LGM(
    response="y", likelihood=Poisson(),
    predictor=Fixed("1") + Weighted(
        IID("u", index="region", precision=Hyperparameter("tau", initial=1.0)), by="z"
    ),
).fit(frame, engine="laplace")
```

The effect of `z` now varies by `region`: region `s`'s slope is `u_s`, shrunk
toward zero by the estimated `tau` exactly as an ordinary `IID` would shrink
an intercept. Any indexed effect can be wrapped this way, not only `IID` — a
`Weighted(RW1(...), by=...)` gives a smoothly-varying-in-time coefficient, a
`Weighted(Besag(...), by=...)` a spatially-smooth one.

**Precision, labels and constraints are the inner effect's, untouched.**
Wrapping changes how the field enters the predictor, not the field itself, so
`result.latent_marginals("u")` and the `u:0`, `u:1`, ... labels in
`result.labels` are exactly what the unwrapped `IID("u", ...)` would produce.
The weighting is design-only.

**`by` must be numeric, finite, and not all-zero.** A missing or non-numeric
column is a data contract error. An all-zero column is rejected too, rather
than silently accepted: it would make the effect contribute nothing to the
predictor while still consuming latent dimensions, fitting happily and
reporting a field the data never actually informed.

**`Fixed` cannot be wrapped.** `Weighted` requires an indexed effect — it
scales rows of an incidence matrix `A` — and `Fixed` has no index; its design
comes from a formula instead. To weight a fixed-effect column, multiply the
covariate into the formula directly (`Fixed("z:region")` or similar), rather
than wrapping.

## Copy

`Copy(name, index, scale=1.0)` is a second occurrence of an existing latent
field at a different index, R-INLA's `f(index, copy="name")`. It adds
`scale * A_index` to the target block's design — `A_index` the incidence
matrix of `index` over the target's own levels — and produces **no block of
its own**.

The model this expresses is a field entering one predictor twice, once
unscaled at its declaring index and once rescaled at a second index:

```
log mu_k = alpha + u_{i(k)} + beta * u_{j(k)},    u ~ IID(tau)
```

```python
from pylgm import Copy, Fixed, IID, LGM, Poisson
from pylgm.parameters import Hyperparameter

# ... a frame whose `i` indexes u's own declaration, `j` a second index over
# the same levels, and `y` the Poisson response ...
result = LGM(
    response="y", likelihood=Poisson(),
    predictor=Fixed("1")
    + IID("u", index="i", precision=1.0)
    + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
).fit(frame, engine="laplace")
```

**A copy contributes no labels or block of its own.** `IID("u", ...) +
Copy("u", index="j", ...)` produces exactly the blocks and `result.labels`
that `IID("u", ...)` alone would — the copy only changes what values land in
`u`'s existing columns. `result.latent_marginals("u")` is `u` as declared;
there is nothing named after the copy to query separately.

**`scale` may be a fixed number or a `Hyperparameter`.** A fixed scale bakes
into the design once; an estimated scale makes the design a function of the
hyperparameter, re-formed on every draw during optimisation, and the fitted
value is reported under its own name in `result.hyperparameters` exactly like
any other estimated parameter.

**The copy's index values must already be levels of the target.** A copy
reuses an existing latent field — it has no mechanism to create a level in
it — so an `index` column containing a value the target was never declared
over is a `CompilationError` at compile time, not a silently-added level.

**`Weighted(Copy(...))` and `Copy` inside a `Joint` are both rejected.** A
copy is a term referencing another effect's block, not an indexed effect with
a design of its own, so wrapping it in `Weighted` raises `TypeError` at
construction — weight the target effect instead. A `Joint` sub-model
containing a `Copy` fails to compile with a `CompilationError`, because
`Joint` compiles each sub-model's effects independently and has no target
block, from an earlier sub-model or the same one, for the copy to fold into.

## MIDAS smooth-lag effect

`MIDAS(name, columns, precision=1.0, order=2, ridge=1e-6)` declares a
mixed-frequency distributed-lag effect. Each entry of `columns` is one column
of the high-frequency (HF) covariate at a fixed lag — lag 0, lag 1, … — so the
design is just those columns stacked (U-MIDAS: every lag enters as its own
regressor). The lag *coefficients* are then tied together by a random-walk
smoothness prior over the lag index, so the estimated lag curve is smooth
instead of the noisy, overfit shape unrestricted OLS gives when the lags are
many and collinear.

```python
import numpy as np
import pandas as pd
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, MIDAS

columns = tuple(f"x_lag{k}" for k in range(6))
# ... a frame whose `columns` hold the HF covariate at lags 0..5 and `y` the LF target ...
model = LGM(
    response="y",
    predictor=Fixed("1")
    + MIDAS("lag", columns=columns, precision=Hyperparameter("lag.precision", initial=1.0)),
    likelihood=Gaussian(sigma=0.5),
)
result = model.fit(frame)
print(np.round(result.latent_marginals("lag").mean, 3))  # the fitted lag curve
```

**The penalty is a random walk over the lag index**, exactly the `RW1`/`RW2`
operator (`order=1` or `order=2`, default `2`) but applied to the coefficient
vector rather than to a time index. `order=2` penalises curvature (favouring a
smooth, gently bending decay); `order=1` penalises steps (favouring a flat or
monotone shape). `precision` (`τ`) is the smoothing strength: a fixed float
plugs it in, a declared `Hyperparameter` has it **estimated** by empirical
Bayes and, under `hyperparameters="integrate"`, integrated over by INLA — the
data choose how much to smooth.

**The lag curve's level and slope stay free of `τ`.** A random-walk penalty is
rank-deficient: it says nothing about the curve's overall level (order 1) or
level *and* slope (order 2) — the part that carries the covariate's actual
effect size. Those null-space directions instead get a fixed, `τ`-independent
precision `ridge` (`δ`, default `1e-6`, the same diffuse prior every
[`Fixed`](#) coefficient gets), so tightening `τ` smooths the *shape* without
ever shrinking the *magnitude* toward zero. Concretely the block precision is
`Q(τ) = τ·DᵀD + δ·P₀`, with `D` the order-`order` difference operator and `P₀`
the projector onto its null space; because the two terms act on orthogonal
subspaces, `τ` touches only the curvature directions. The effect is proper and
carries no constraint, so it runs under every latent strategy including full
Laplace (`latent_strategy="laplace"`).

**Aligning the HF data to the LF target is the caller's job.** `MIDAS` takes
the lag columns as given; building them (e.g. `frame[f"x_lag{k}"] =
hf.shift(k)` after resampling the HF series onto the LF rows) is upstream data
prep. See [`examples/midas_nowcast`](https://github.com/Ardea00/pylgm/tree/main/examples/midas_nowcast)
for an end-to-end nowcasting run that recovers a known decaying kernel.
Parametric lag kernels (exp-Almon / Beta weight functions) are now shipped —
see [Restricted MIDAS](#restricted-midas-effect-parametric-lag-weights) below.

**From YAML:** the standalone `load_model` frontend declares `type: midas`,
indexed by the HF lag `columns` (in place of a single `index`), with optional
fixed `precision`, `order` (`1` or `2`), and `ridge`. Estimating `precision`
from YAML stays Python-API-only. Write floats in full YAML form (`1.0e-6`, not
`1e-6`, which YAML reads as a string).

```yaml
predictor:
  effects:
    - {name: lag, type: midas, columns: [x0, x1, x2, x3], precision: 1.0, order: 2, ridge: 1.0e-6}
```

## SpaceTime effect (Knorr-Held interaction)

`SpaceTime(name, space, time, graph=None, interaction="IV", order=1, precision=1.0, scale=True)`
builds the **interaction** term `δ(s,t)` of a Knorr-Held space-time model — one
latent block over the `S·T` area×period cells with precision `τ·(K_s ⊗ K_t)`.
The spatial (`Besag`) and temporal (`RW1`/`RW2`) main effects are composed with
`+`; `SpaceTime` supplies only the interaction, keeping each term its own
variance component.

| `interaction` | `K_s` | `K_t` | Reading |
|------|-------|-------|---------|
| `"I"`   | `I_s` | `I_t` | unstructured cell-wise interaction (proper) |
| `"II"`  | `I_s` | `R_t` | each area its own independent temporal trend |
| `"III"` | `R_s` | `I_t` | each period its own independent spatial pattern |
| `"IV"`  | `R_s` | `R_t` | inseparable: neighbours in *both* space and time tied |

`R_s` is the (weighted) Sørbye–Rue-scaled Besag Laplacian; `R_t` the scaled RW
structure of the given `order`. `graph` is **required for III/IV** and optional
for I/II (the area universe is then read from the observed `space` column).
`precision` is a single `τ` — a float (plug-in) or a `Hyperparameter`
(estimated by EB, integrated by INLA).

```python
from pylgm import Besag, Fixed, Gaussian, LGM, RW1, SpaceTime

model = LGM(
    response="y",
    predictor=(
        Fixed("1")
        + Besag("area", index="area", graph=W)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=W, interaction="IV", order=1)
    ),
    likelihood=Gaussian(sigma=0.5),
)
result = model.fit(frame)
```

Type I is proper and carries no constraint, so it runs under all latent
strategies including `latent_strategy="laplace"`. Types II–IV carry the
Knorr-Held / Schrödle–Held sum-to-zero constraints (derived from the Kronecker
null space), so — like `RW2` and `Besag` — they require
`hyperparameters="integrate"` and are rejected under full Laplace. The compiler
emits a one-line warning if a `SpaceTime` effect is present without its spatial
and temporal main effects, since the constraints assume those absorb the
marginals. The latent field is dense `S·T`, so the >4096-dim preflight guard
bites at large grids — real economic scale is gated on the sparse backend.

**From YAML:** the standalone `load_model` frontend declares `type: spacetime`,
indexed by **two** columns — `space` and `time` (both required) — instead of a
single `index`. `graph` (inline) or `graph_file` (an R-INLA `.graph` or `.json`
neighbour dict) supplies the spatial neighbours, at most one and required for
`interaction: III`/`IV`. `interaction` (default `IV`), `order` (`1`/`2`, default
`1`), fixed `precision` (default `1.0`), and `scale` (default `true`) are
optional. Estimating `precision` from YAML stays Python-API-only.

```yaml
predictor:
  effects:
    - {name: st, type: spacetime, space: area, time: t, graph: {a: [b], b: [a]}, interaction: IV, order: 1, precision: 1.0}
```

## Restricted MIDAS effect (parametric lag weights)

`MIDASParametric(name, columns, kernel="beta", shape1=None, shape2=None, prior_precision=1e-6)`
collapses the high-frequency lag `columns` into a **single** regressor
`β · Σ_k w(k; θ) · x_{t,k}`, where `w(·; θ)` is a parametric lag-weight kernel.
This is the *restricted* counterpart to the U-MIDAS
[`MIDAS`](#midas-smooth-lag-effect) effect, which keeps every lag as its own
smoothed coefficient.

| `kernel` | shape params | weight `log w_k` (normalized to Σ=1) |
|------|------|------|
| `"beta"` | `a, b > 0` | `(a−1)·log x_k + (b−1)·log(1−x_k)`, `x_k=(k+1)/(K+1)` |
| `"exp_almon"` | `θ1, θ2` real | `θ1·k + θ2·k²` (θ2<0 ⇒ decay) |

Lags are indexed `columns[0]` (shallowest) … `columns[K-1]` (deepest); weights
are softmax-normalized. Each shape is a float (fixed θ) or a `Hyperparameter`
(estimated by EB, integrated by INLA); omit them for kernel-appropriate
defaults (Beta `a=b=2`; exp-Almon `θ1=0, θ2=−0.1`). The loading `β` is a single
coefficient with a fixed vague Gaussian prior (`prior_precision`), so the block
is proper and unconstrained — it runs under every latent strategy, including
`latent_strategy="laplace"`. `predict()` rebuilds the aggregate for new rows at
the fitted weights.

```python
from pylgm import Fixed, Gaussian, LGM, MIDASParametric

model = LGM(
    response="y",
    predictor=Fixed("1") + MIDASParametric("m", ("x0", "x1", "x2", "x3"), kernel="beta"),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.hyperparameters["m.shape1"]  # estimated Beta a
```

**From YAML:** the frontend declares `type: midas_parametric`, indexed by the
lag `columns` with an optional `kernel` (`beta` or `exp_almon`, default `beta`).
The kernel shapes are estimated — the effect's default — and fixed shape
overrides stay Python-API-only.

```yaml
predictor:
  effects:
    - {name: m, type: midas_parametric, columns: [x0, x1, x2, x3], kernel: exp_almon}
```

## Linear constraints (`extraconstr`)

`LGM(..., constraints=...)` imposes arbitrary linear constraints `A x = e` on
the assembled latent field — the label-keyed equivalent of R-INLA's
`extraconstr`. Each entry names a linear combination of latent elements by
their **qualified labels** (`"effect:level"`, the same strings that appear in
`result.labels`) and either constrains it to zero (a bare mapping) or to a
given value (a `(mapping, rhs)` pair):

```python
from pylgm import Fixed, Gaussian, IID, LGM

model = LGM(
    response="y",
    likelihood=Gaussian(1.0),
    predictor=Fixed("1") + IID("region", "region", 1.0),
    panel=("region",),
    time="time",
    constraints=[
        {"region:oslo": 1.0, "region:bergen": -1.0},  # force two effects to coincide
        ({"region:tromso": 1.0}, 2.5),                 # pin one effect to 2.5
    ],
)
```

The first row (a bare mapping) is `A x = 0`; the second (a `(mapping, rhs)`
pair) is `A x = 2.5`. Constraints compose with any intrinsic constraints an
effect already carries — a `Besag` sum-to-zero, a random-walk anchor — which
are appended automatically. They hold **exactly** in the posterior mean under
both engines (`exact_gaussian` and `laplace`) and under `hyperparameters=
"optimize"`/`"integrate"`.

An unknown label raises `CompilationError` at fit time (labels are only known
once the latent field is assembled); a malformed constraint — an empty mapping,
all-zero coefficients, a non-finite coefficient or right-hand side — raises at
`LGM` construction.

A nonzero right-hand side is imposed by **conditioning the prior on `A x = e`**
(conditioning by kriging), matching R-INLA's semantics; see
[Theory](theory.md#linear-constraints). Contradictory constraints are not
rejected — they degrade to the least-squares closest satisfiable field.


# Effects

pyLGM's predictor is a sum of effects. The core building blocks are
`Fixed` (a formulaic fixed-effects formula), `IID` (exchangeable random
effects), and the random walks `RW1`/`RW2` (smooth temporal trends); each
structured effect takes a fixed `precision` or a declared `Hyperparameter`
(see [Empirical Bayes and priors](empirical-bayes.md)). Runnable examples of
these live under [`examples/`](https://github.com/Ardea00/pylgm/tree/main/examples). The stationary `AR1` effect is
documented below; spatial (CAR) effects have their own
[page](spatial-effects.md). Model-level [linear constraints](#linear-constraints-extraconstr)
on the assembled latent field are documented at the bottom of this page.

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
irregular-spacing support (`ρ^Δt`), a config-file `ar1` effect type, and
AR(p) effects.

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
**Not shipped**: a hybrid HF/LF nowcasting frontend and a config-file `midas`
effect type.

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


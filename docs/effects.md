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
irregular-spacing support (`ρ^Δt`), a config-file `ar1` effect type,
AR(p)/seasonal effects, and group-wise AR1 (a separate series per panel
unit).

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
for an end-to-end nowcasting run that recovers a known decaying kernel. **Not
shipped**: parametric lag kernels (exp-Almon / Beta weight functions), a hybrid
HF/LF nowcasting frontend, and a config-file `midas` effect type.

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


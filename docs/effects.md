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


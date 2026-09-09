# Joint multi-likelihood models

> **Research-grade.** Joint models live on the `research-tier` branch, not `main`.
> They are tested and reviewed, and validated against MCMC ground truth — but
> on *simulated* data. No published result on real data has been reproduced,
> and `latent_strategy="laplace"` is known to degrade on them. See
> [research status](research-status.md) before relying on this for published work.

`LGM` takes exactly one `response` column and one `likelihood`. `Joint` stacks
several `LGM` sub-models — each with its own response, likelihood, offset, and
predictor — into a single fit, optionally letting one latent field enter more
than one sub-model with a per-sub-model scaling. This covers shared-component
disease mapping, joint PD/LGD, longitudinal-plus-survival, and any model where
several outcomes are believed to share part of their latent structure.

## The stacking model

A `Joint` compiles to an ordinary `CompiledLGM` with more rows — the
underlying IR does not change. Responses stack as `y = (y⁽¹⁾, ..., y⁽ᴷ⁾)`,
with sub-model `k` occupying a contiguous row slice `R_k`. Both `CompiledLGM`
invariants survive exactly: the design is still `hstack(blocks)` and the
precision still `block_diag(blocks)`.

- A **sub-model-private block** (declared in one sub-model's own `predictor`)
  is zero-padded to zero rows outside `R_k`. Its precision, labels, and
  constraints are untouched — a Besag sum-to-zero constraint on one sub-model
  still means what it meant standalone.
- A **shared block** (declared once via `Shared`, entering several
  sub-models) carries nonzero rows in every slice it enters, scaled per
  slice: `Shared(u, scale=(s_1, ..., s_K))` contributes `s_k * u_i` to slice
  `k`'s linear predictor.
- The likelihood becomes a row-dispatching mixture: each sub-model's own
  likelihood applies to its own row slice, so a Poisson sub-model's rows never
  see a Gaussian sub-model's likelihood or vice versa.

Sharing is expressed on the **design** side, not as off-block-diagonal
precision coupling — see [Not supported yet](#not-supported-yet) for what that
restriction rules out.

## The `Joint` / `Shared` API

```python
Joint(submodels, shared=())
```

`submodels` is a list of at least two `LGM` instances, each declaring its own
`response` (must be unique across the joint), `likelihood`, `predictor`, and
optionally `offset`/`panel`/`time`. `shared` is a list of `Shared` entries.

```python
Shared(effect, scale=1.0, allow_ragged=False)
```

`effect` is any latent effect spec (`IID`, `Besag`, `RW1`, ...), built once
against the **union** of its index's levels across sub-models — a level seen
in only some sub-models still gets a latent entry, informed by fewer rows.
`scale` is one of:

- a **float**, broadcast to every sub-model;
- a **`Hyperparameter`** — shorthand for the Knorr-Held & Best `(delta,
  delta⁻¹)` pairing (below), valid only when the joint has exactly two
  sub-models;
- an explicit **tuple** of floats and/or `Hyperparameter`s, one entry per
  sub-model, always allowed regardless of sub-model count.

`allow_ragged=True` silences the report raised when the index's level set
differs between sub-models (off by default: an unintended mismatch weakens
the shared field without any visible symptom otherwise).

### Worked example: Knorr-Held & Best shared component

Knorr-Held & Best (2001) model two disease outcomes over the same districts
sharing one spatial component `u`, scaled oppositely so its overall level
stays identified:

```
log mu_1i = alpha_1 + delta * u_i + v_1i
log mu_2i = alpha_2 + delta^-1 * u_i + v_2i
```

```python
import numpy as np
import pandas as pd

from pylgm import Besag, Fixed, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter

rng = np.random.default_rng(0)

# A small connected chain graph over districts "0".."19".
n = 20
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}
districts = [str(i) for i in range(n)]

u_true = rng.normal(0.0, 0.6, size=n)
delta_true = 1.6
log_E = np.log(rng.uniform(50, 200, size=n))  # expected-count offset, per district

eta_oral = -0.2 + delta_true * u_true
eta_larynx = 0.1 + u_true / delta_true

frame = pd.DataFrame({
    "district": districts * 2,
    "log_E_oral": np.concatenate([log_E, log_E]),
    "log_E_larynx": np.concatenate([log_E, log_E]),
    "oral": list(rng.poisson(np.exp(eta_oral + log_E))) + [np.nan] * n,
    "larynx": [np.nan] * n + list(rng.poisson(np.exp(eta_larynx + log_E))),
})

joint = Joint(
    [
        LGM(response="oral", likelihood=Poisson(), offset="log_E_oral", predictor=Fixed("1")),
        LGM(response="larynx", likelihood=Poisson(), offset="log_E_larynx", predictor=Fixed("1")),
    ],
    shared=[Shared(
        Besag("u", index="district", graph=graph, precision=1.0),
        scale=Hyperparameter("delta", initial=1.0),
    )],
)
result = joint.fit(frame, engine="laplace")
result.hyperparameters["delta"]  # -> 1.62, close to the simulated 1.6
```

`u`'s labels are qualified with the shared name (`"u:0"`, `"u:1"`, ...); a
sub-model-private block is qualified with its outcome instead
(`"oral:fixed:Intercept"`, distinguishing it from `"larynx:fixed:Intercept"`).

## The `(delta, delta^-1)` shorthand

Passing a bare `Hyperparameter` as `scale` — as in the example above — is
shorthand for `scale=(delta, ("delta", "inverse"))`: sub-model 1 gets
`delta * u`, sub-model 2 gets `u / delta`. This is accepted **only when the
joint has exactly two sub-models**, because with three or more the pairing has
no canonical meaning:

```python
Joint(
    [oral_model, larynx_model, third_model],
    shared=[Shared(shared_effect, scale=Hyperparameter("delta", initial=1.0))],
)
# ValueError: Shared 'u' has a scalar Hyperparameter scale, which is the
# (delta, delta^-1) shorthand and requires exactly two sub-models; this
# joint has 3. Pass an explicit per-sub-model tuple instead.
```

With three or more sub-models — or whenever you want scales other than the
KHB pairing — pass an explicit per-sub-model tuple:
`Shared(u, scale=(delta_1, delta_2, delta_3))`, mixing floats and
`Hyperparameter`s freely.

## `engine="laplace"` only

`Joint.fit` accepts only `engine="laplace"`:

```python
joint.fit(frame, engine="exact_gaussian")
# UnsupportedEngineError: Joint models require engine='laplace'; the
# exact_gaussian engine needs a single CompiledGaussian likelihood, and a
# mixture is not one. Laplace is exact for an all-Gaussian stack anyway.
```

The exact-Gaussian engine (`inference/gaussian.py`) is built around a single
`CompiledGaussian` likelihood; a joint's likelihood is a row-dispatching
mixture, which is not one, even when every sub-model happens to be Gaussian.
Laplace needs no such restriction, and for an all-Gaussian stack its answer is
exact anyway (the Newton step converges in one iteration), so nothing is lost
by routing every joint fit through Laplace.

## Prediction

```python
result.predict(new_data, outcome="oral")
```

`outcome` selects which sub-model's likelihood and predictor to score
`new_data` against. It is:

- **required** on a joint result — `predict(new_data)` without it raises
  `ValueError` naming the valid outcomes;
- **rejected** on a single-response result (one produced by `LGM.fit`) —
  passing `outcome=` there raises `ValueError`.

`new_data` must be homogeneous: predicting two outcomes means two separate
calls, each against rows meant for that outcome. There is no mixed-outcome
`predict` in a single call.

```python
oral_rows = frame[frame["oral"].notna()].reset_index(drop=True)
prediction = result.predict(oral_rows, outcome="oral")
prediction.predictive_mean  # scored on `oral`'s Poisson likelihood

result.predict(oral_rows)
# ValueError: predict() on a joint result requires outcome=, one of
# ('oral', 'larynx')
```

Each outcome is scored against **its own** compiled likelihood, not the
mixture, so `predictive_mean`, trial counts (`Binomial`), and survival
auxiliaries (`event`/`entry`) behave exactly as they do for a standalone
`LGM` of that outcome's family.

## Hyperparameter namespace

Every sub-model's hyperparameters and every `Shared` scale draw from **one
flat namespace across the whole joint** — hyperparameter names are not
qualified by outcome the way block/label names are. A name reused by two
*different* `Hyperparameter` declarations is rejected at compile time rather
than silently aliased:

```python
Joint(
    [LGM(response="oral", likelihood=Gaussian(sigma=Hyperparameter("sigma", initial=1.0)), ...),
     LGM(response="larynx", likelihood=Gaussian(sigma=Hyperparameter("sigma", initial=1.0)), ...)],
).fit(frame, engine="laplace")
# CompilationError: hyperparameter name 'sigma' is declared by more than one
# sub-model. Joint sub-models share one hyperparameter namespace, so give
# each its own name (e.g. 'tau_oral', 'tau_larynx').
```

Two Gaussian sub-models must therefore be given explicitly distinct `sigma`
names (`Gaussian(sigma=Hyperparameter("sigma_oral", ...))` /
`Gaussian(sigma=Hyperparameter("sigma_larynx", ...))`), and a `Shared` scale's
name must not collide with any sub-model hyperparameter's name either:

```python
# oral's own `delta` IID precision collides with the shared scale's `delta`.
# CompilationError: hyperparameter name 'delta' is declared by more than one
# sub-model/shared entry, with a different Hyperparameter object for each. ...
```

The one exception is deliberate reuse: passing the **same** `Hyperparameter`
object to more than one `Shared` entry (or letting the `(delta, delta⁻¹)`
shorthand produce the same object twice) dedups instead of raising, since it
is the same declaration, not a collision.

## Not supported yet

- **`latent_strategy="laplace"` is not recommended on joint models.** Under
  `hyperparameters="integrate"`, the full-Laplace (tabulated) strategy was
  measured against NUTS ground truth on a shared-component model and came out
  *worse* than the plain `gaussian` baseline — mean |z| 0.337 against 0.072, and
  worst-case 1.14 against 0.40 — with skewness estimates that barely track the
  truth (correlation +0.23). A single-response control over the same data shows
  no such degradation (`laplace` slightly *improves* there), so this is specific
  to joint models and not a general property of the strategy.

  `simplified_laplace` behaves as designed: its skewness estimates correlate
  +0.87 with NUTS, and it more than halves the worst-case error (0.19 against
  0.40). Prefer it when you want a skew correction on a joint. Note that in a
  near-Gaussian regime neither correction has anything to fix and both can add
  a little error — `gaussian` is already close to exact there.

  See `examples/joint_mcmc_crosscheck/` for the measurement setup. This is
  reported as a limitation rather than pinned by a test, because a test would
  cement behaviour we believe is wrong.

- **NaN-response hold-out.** `LGM.fit` keeps NaN-response rows as *unobserved*
  — excluded from the likelihood, but still assigned fitted values on the
  predictor. `Joint.fit` instead drops each sub-model's NaN-response rows before
  compiling, so that idiom does nothing on a `Joint`.

  This is deliberate, not an oversight. In the long-stacked layout joint models
  are normally given — one row per (outcome, unit) pair — every row is NaN for
  every *other* outcome, so a NaN means "this row belongs to another outcome",
  not "hold this observation out". Keeping those rows would double the stacked
  design and produce fitted values for observations that do not exist. To hold a
  row out of a joint fit, drop it from the frame and score it afterwards with
  `result.predict(new_data, outcome=...)`.

- **Off-block-diagonal precision coupling** (coregionalization). Sharing is
  expressed entirely on the design side; `precision == block_diag(blocks)`
  still holds exactly, so two shared fields cannot be given a correlated
  cross-outcome precision the way a full coregionalized model would.
- **`copy` between effects within a single `LGM`.** `Shared` covers sharing
  *across* `Joint` sub-models, not two effects sharing a hyperparameter inside
  one sub-model's own predictor.
- **`replicate`** — conditionally independent copies of an effect sharing
  hyperparameters.
- **Latent fields in the likelihood's scale.** Effects still sum into the
  mean's linear predictor only; a shared or private field cannot enter a
  likelihood's dispersion/scale parameter.
- **A `Hyperparameter` on a `Shared` effect's own structural fields**
  (`precision`, `rho`, `phi`, `gamma`, `eta`, the `MIDASParametric` shapes).
  Only `Shared.scale` may be a `Hyperparameter`; the effect's own fields must
  be fixed values, or compilation raises `CompilationError`:

  ```python
  Shared(IID("u", index="district", precision=Hyperparameter("tau_u", initial=1.0)),
         scale=Hyperparameter("delta", initial=1.0))
  # CompilationError: shared effect 'u' declares Hyperparameter(s) tau_u on
  # its own precision/rho/phi/gamma/eta/shape -- estimating a shared effect's
  # own structural parameters is not supported yet. Pass a fixed value for
  # that field for now; only the Shared `scale` may be a Hyperparameter.
  ```

- **A YAML frontend for `Joint`.** The Python API lands first; a declarative
  surface is a natural follow-on once the shape is settled.
- **Mixed-outcome `predict` in a single call.** `outcome=` selects exactly one
  sub-model per call.
- **A shared spatial effect's graph must contain exactly the observed
  regions.** `Besag`/`ProperCAR`/`SAR`/`BYM2` take their latent domain from
  the graph, not the data; a node in the graph with no observed row for the
  shared index (a common case when a graph ships from a shapefile with more
  regions than the data covers) raises `CompilationError` rather than
  silently padding in an unobserved region.
- **Spark input.** `LGM.fit` accepts a Spark DataFrame as well as pandas;
  `Joint.fit` accepts only a pandas DataFrame.

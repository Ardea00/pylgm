# Spatial (CAR) effects

## Besag / intrinsic CAR (ICAR) spatial effect

`Besag(name, index, graph, precision=1.0, scale=True)` declares an intrinsic
conditional-autoregressive (ICAR) spatial latent effect: precision
`τ·(D−W)`, the graph Laplacian of the adjacency, with density proportional to
`exp(−τ/2 · Σ_{i~j}(xᵢ−xⱼ)²)`. This structure is rank-deficient (it is
invariant to adding a constant within a connected component), so the effect
carries one sum-to-zero constraint per connected component of the graph.

`graph` is a neighbour dict `{region: [neighbours, ...]}` keyed by the same
labels as the data's `index` column, or the result of `load_graph_file(path)`,
which parses the R-INLA/latte `.graph` text format into that same dict shape.
The graph's nodes are the effect's domain, so it may include regions absent
from the observed data (e.g. to predict at unobserved regions); every
observed region must be a node in the graph, or `fit` raises a clear error.
A node with zero neighbours (an isolated region) has no ICAR structure to
borrow strength from, so it is handled gracefully as an independent
unit-variance `IID` singleton: its structure diagonal is 1 and it carries no
sum-to-zero constraint (R-INLA's `adjust.for.con.comp` default), rather than
aborting the fit. A multi-node disconnected component (an "island" group) is
likewise supported and gets its own sum-to-zero constraint.

`scale=True` (the default) applies the Sørbye–Rue (2014) scaling
convention: each connected component's structure matrix is rescaled to a
unit geometric-mean marginal variance, so `τ` is comparable across different
graphs and to an `IID` effect's precision on the same scale. Pass
`scale=False` to use the raw, unscaled `D−W` structure instead.

```python
import numpy as np
import pandas as pd

from pylgm import Besag, Fixed, Gaussian, LGM, load_graph_file

# A small connected chain graph over regions "0".."5": i <-> i-1, i <-> i+1.
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5] for i in range(6)}

frame = pd.DataFrame({"region": [str(i) for i in range(6)], "y": np.sin(np.arange(6) / 2.0)})
model = LGM(
    response="y",
    predictor=Fixed("1") + Besag("region", index="region", graph=graph, precision=1.0),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.latent_marginals("region")  # GaussianMarginals over the 6 regions

# Loading a graph from an R-INLA / latte .graph file instead of an inline dict:
graph_from_file = load_graph_file("regions.graph")  # -> {"1": ["2"], "2": ["1", "3"], ...}
```

`precision` accepts a plain number or a declared `Hyperparameter`, and
participates in `hyperparameters="optimize"`/`"integrate"` exactly like
`IID`/`RW1`/`RW2` precisions. `Besag` composes with other effects the same
way (`Fixed(...) + Besag(...) + IID(...)`) and works under both the default
`latent_strategy="gaussian"` and `latent_strategy="simplified_laplace"`.
Because it is a constrained effect (like `RW1`/`RW2`), `latent_strategy="laplace"`
(full Laplace) rejects it with `UnsupportedEngineError`, the same restriction
already documented above for `RW`/intrinsic effects.

**From YAML:** the standalone `load_model` frontend declares `besag`,
`proper_car`, and `bym2` effects, with the neighbour graph given inline as
`graph:` or as a path via `graph_file:` (resolved relative to the YAML file).
`graph_file:` accepts an R-INLA `.graph` file or a `.json` neighbour dict
(`{node: [neighbour, ...]}`, or `{node: {neighbour: weight}}` when weighted) —
the same shape the inline `graph:` and Python `graph=` arguments take, so an
existing JSON adjacency is reusable without conversion. Fixed values only — `precision`, `rho` (required
for `proper_car`), `phi` (`bym2`), and `scale` (`besag`); estimating a
hyperparameter stays Python-API-only. Example:

```yaml
response: cases
likelihood: {family: poisson}
data: {panel: [region], time: year}
predictor:
  fixed: "1 + urbanicity"
  effects:
    - {name: spatial, type: besag, index: region, precision: 1.0, graph_file: adjacency.graph}
```

The legacy `ModelConfig` frontend (used by the `fit`/`compare` CLI) does not
declare spatial effects.

**Not yet built:** spatial autocorrelation diagnostics and shapefile/GeoJSON
graph construction — see the
[spatial roadmap](roadmap.md)
for what's next.

## Weighted neighbour graphs

The whole CAR family (`Besag`/`ProperCAR`/`BYM2`) accepts a **weighted** graph:
instead of a bare neighbour list, give each node a `{neighbour: weight}` mapping
and the adjacency `W` carries those weights (`D` is the weighted degree, and the
ICAR density becomes `exp(−τ/2 · Σ_{i~j} wᵢⱼ(xᵢ−xⱼ)²)`). Nothing else changes —
Sørbye–Rue scaling, constraints, ρ-validity and `φ` all derive from `(nodes, W)`
and carry over untouched.

```python
# Geographic adjacency reads as: "a and b are neighbours." A weighted graph
# reads as: "a and b are coupled with strength 5, a and c with strength 0.1" —
# so W can encode firm-ownership / interbank-exposure / supply-chain / IO
# dependence strength rather than only 0/1 contiguity.
graph = {"a": {"b": 5.0, "c": 0.1},
         "b": {"a": 5.0, "c": 0.1},
         "c": {"a": 0.1, "b": 0.1}}
```

For `BYM2` this makes `φ` read as *"fraction of variance explained by network
dependence vs. idiosyncratic."* Requirements, enforced at declaration:

- **Weights must be finite and strictly positive.** A weighted ICAR/CAR is a
  valid GMRF only for `wᵢⱼ ≥ 0`; a zero weight means "no edge" — omit it.
- **The graph must be symmetric.** Every edge `i→j` needs its reverse `j→i`
  with an equal weight (within `rtol=1e-9, atol=1e-12`; the stored value is
  their mean). A missing reverse or a weight mismatch raises, naming the pair.
- Directed / signed economic matrices (exposure, correlations) must be made
  symmetric-nonnegative **before** building the graph; `normalize_graph` is a
  strict validator, it does not silently symmetrize. To model the directed
  relation itself rather than a symmetrized shadow of it, use the
  [`SAR` effect](#directed-spatial-autoregressive-sar-effect), whose precision
  `(I−ρW)ᵀ(I−ρW)` accepts an asymmetric `W`.

A bare-label list (`{node: [neighbours]}`) is exactly a weighted graph with all
weights `1.0`, so existing unweighted graphs, `.graph` files, and YAML are
unchanged. The weighted mapping also works inline in YAML `graph:`
(`{a: {b: 5.0}, ...}`). See `examples/weighted_network/` for a runnable
ownership-exposure `BYM2` fit.

## Proper CAR spatial effect

`ProperCAR(name, index, graph, rho, precision=1.0)` declares a proper
conditional-autoregressive spatial latent effect: precision
`Q = τ·(D − ρW)`, where `D` is the degree matrix and `W` the adjacency of
`graph`. Unlike `Besag`/ICAR, this precision is full-rank (proper), so the
effect carries **no sum-to-zero constraint**.

`graph` uses the same neighbour-dict or `load_graph_file(path)` input as
`Besag`. `rho` (ρ) must lie in the open interval `(1/μ_min, 1/μ_max)`, where
`μ` are the eigenvalues of the normalized adjacency, for `D − ρW` to be
positive definite; for a graph with edges the upper bound is `1`. `ρ = 0`
recovers a spatially-independent, degree-weighted precision `τD`. An
out-of-range `rho` raises a `ValueError` naming the valid interval.

`rho` accepts either a **fixed float** (plug-in, shown below) or a declared
`Hyperparameter`, in which case ρ is **estimated** — see
["Estimating ρ"](#estimating).

```python
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, LGM, ProperCAR

# The same small connected chain graph over regions "0".."5" used above.
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5] for i in range(6)}

frame = pd.DataFrame({"region": [str(i) for i in range(6)], "y": np.sin(np.arange(6) / 2.0)})
model = LGM(
    response="y",
    predictor=Fixed("1") + ProperCAR("region", index="region", graph=graph, rho=0.9),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.latent_marginals("region").mean  # posterior mean over the 6 regions
```

`precision` (τ) accepts a plain number or a declared `Hyperparameter`, and
participates in `hyperparameters="optimize"`/`"integrate"` exactly like the
other structured effects' precisions. Because it is unconstrained,
`ProperCAR` is the **first spatial effect to work under
`latent_strategy="laplace"`** (full Laplace) — `Besag` is rejected there
since it carries a sum-to-zero constraint.

### Estimating ρ

Passing a `Hyperparameter` for `rho` estimates the spatial dependence instead
of fixing it. Declare it with `transform="logit"`: ρ lives on a bounded
interval, so it is inferred on a scaled-logit scale rather than the log scale
used for positive parameters. The interval itself is resolved **from the
graph** at compile time — leave `lower`/`upper` as `None` and the compiler
supplies `(1/μ_min, 1/μ_max)` — and `initial` may be any finite value inside
it, including `0.0`.

```python
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, ProperCAR
from pylgm.priors import PCPrecision

rho = Hyperparameter("region.rho", initial=0.0, transform="logit")
tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(
    response="y",
    predictor=Fixed("1")
    + ProperCAR("region", index="region", graph=graph, rho=rho, precision=tau),
    likelihood=Gaussian(sigma=0.1),
)

eb = model.fit(frame)                                   # empirical Bayes
eb.hyperparameters["region.rho"]                        # point estimate of rho

post = model.fit(frame, hyperparameters="integrate")    # INLA over (tau, rho)
post.hyperparameter_marginals()["region.rho"].mean      # rho marginal
```

ρ and τ are estimated (or integrated over) **jointly**. Under the hood the
effect's precision is no longer a scalar multiple of a fixed matrix, so it is
carried as a parametric block that rebuilds `τ(D − ρW)` at each hyperparameter
value; see ["Bounded hyperparameters"](empirical-bayes.md#bounded-hyperparameters).

**Not yet built:** Sørbye–Rue scaling of the proper-CAR structure itself, and
shapefile/GeoJSON graph construction — see the
[spatial roadmap](roadmap.md)
for what's next.

## BYM2 spatial effect

`BYM2(name, index, graph, precision=1.0, phi=0.5)` declares Riebler et al.'s
(2016) BYM2 convolution model: a marginal-precision reparameterization of the
classic BYM structured-plus-unstructured mixture,

```
x = τ^(-1/2) · ( √(1-φ)·v + √φ·u* )
```

with `v ~ N(0, I)` unstructured noise and `u*` the **Sørbye–Rue-scaled**
ICAR component (`Besag`'s scaling, always applied — there is no `scale` flag
on `BYM2`). `τ` is the **marginal** precision of `x` (not the ICAR
precision), and `φ ∈ (0, 1)` is the fraction of the marginal variance
attributed to the spatially structured component: `φ → 0` behaves like plain
IID noise, `φ → 1` like `Besag`/ICAR.

Rather than the classical augmented `2n`-dimensional representation
(structured and unstructured components stacked and reported separately),
this is implemented in the **marginal, `n`-dimensional** parameterization:
`Q = τ·[(1-φ)·I + φ·R*⁻]⁻¹`, built once per graph via an eigendecomposition
of the scaled structure's generalized inverse and then reassembled cheaply at
any `(τ, φ)`. Because this precision is full-rank, `BYM2` carries **no
constraint at all** — unlike `Besag` (one sum-to-zero constraint per
connected component). `graph` takes the same neighbour-dict or
`load_graph_file(path)` input as `Besag`/`ProperCAR`, with the same graceful
isolated-node handling (an isolated region becomes an independent IID unit)
and disconnected-graph support — in both the dense spectral and the augmented
large-graph paths.

```python
import numpy as np
import pandas as pd

from pylgm import BYM2, Fixed, Gaussian, LGM

# The same small connected chain graph over regions "0".."5" used above.
graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5] for i in range(6)}

frame = pd.DataFrame({"region": [str(i) for i in range(6)], "y": np.sin(np.arange(6) / 2.0)})
model = LGM(
    response="y",
    predictor=Fixed("1") + BYM2("region", index="region", graph=graph, precision=1.0, phi=0.5),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.latent_marginals("region").mean  # posterior mean over the 6 regions
```

`precision` (τ) accepts a plain number or a `Hyperparameter`, exactly like
the other structured effects. `phi` accepts either a **fixed float** in
`(0, 1)` (shown above) or a `Hyperparameter(transform="logit")`, in which
case φ is **estimated** by empirical Bayes and, under
`hyperparameters="integrate"`, **integrated over jointly with τ** (INLA), the
same way `ProperCAR`'s ρ is:

```python
from pylgm import BYM2, Fixed, Gaussian, Hyperparameter, LGM, PCBYM2Phi
from pylgm.priors import PCPrecision

# A larger graph with a genuinely spatial signal: phi is weakly identified,
# so a handful of regions cannot distinguish structured from unstructured.
rng = np.random.default_rng(0)
n = 20
big_graph = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}
signal = np.zeros(n)
for i in range(1, n):
    signal[i] = 0.9 * signal[i - 1] + rng.normal(scale=0.4)
signal -= signal.mean()
big_frame = pd.DataFrame({
    "region": [str(i) for i in range(n)],
    "y": signal + rng.normal(scale=0.3, size=n),
})

phi = Hyperparameter("region.phi", initial=0.5, transform="logit", prior=PCBYM2Phi())
tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
model = LGM(
    response="y",
    predictor=Fixed("1") + BYM2("region", index="region", graph=big_graph, precision=tau, phi=phi),
    likelihood=Gaussian(sigma=0.3),
)

eb = model.fit(big_frame)                                # empirical Bayes
eb.hyperparameters["region.phi"]                         # point estimate of phi

post = model.fit(big_frame, hyperparameters="integrate")  # INLA over (tau, phi)
post.hyperparameter_marginals()["region.phi"].mean        # phi marginal
```

**φ is weakly identified, by nature of the model.** Structured and
unstructured components explain small datasets almost equally well, which is
precisely why Riebler et al. pair BYM2 with an informative PC prior. Two
practical consequences: the profile objective can be *bimodal* (mass near
"pure IID" and near "pure spatial", with a valley between), so empirical Bayes
— a local optimizer — may report whichever mode it descends into; and on small
graphs φ can collapse to its boundary even when the simulating truth was
interior. Keep the PC prior, and treat a boundary φ̂ as "not identified"
rather than as evidence about spatial structure.

**Integration does not rescue a boundary φ̂.** The INLA grid is built around
the empirical-Bayes mode and its curvature, so when φ̂ pins at a boundary the
grid degenerates there too: the reported φ marginal becomes a near point mass
with a spuriously tiny standard deviation (the example above returns a mean of
≈0.99996 with sd ≈2e-8, which is grid degeneracy, not posterior certainty).
Before believing a φ marginal, check `result.diagnostics["inla_grid_points"]`
and `["inla_collapsed"]` — a handful of points clustered at a bound means the
hyperparameter was not identified, whatever the interval width suggests. This
affects φ specifically because it is weakly identified; τ and ρ marginals on
the same fits behave normally.

`PCBYM2Phi(upper=0.5, alpha=2/3)` is Riebler et al.'s PC prior for φ,
calibrated so that `P(φ < upper) = alpha`. Because its distance scale
depends on the eigenvalues of the graph's scaled structure, it is declared
**unbound** and the compiler **binds it to the effect's graph** when a `BYM2`
`phi` hyperparameter references it; calling `.logpdf` on an unbound
`PCBYM2Phi` raises. An `alpha` at or below the graph's attainable floor
`d(upper)/d(1)` raises, naming the achievable range.

Because `BYM2`'s precision is unconstrained, like `ProperCAR` it works under
**all three latent strategies**, including full Laplace
(`latent_strategy="laplace"`) — unlike `Besag`, which full Laplace rejects
for carrying a sum-to-zero constraint.

**Structured component at network scale.** On large graphs `BYM2` fits
through the augmented (`2n`) representation, which carries the structured
ICAR component `u*` explicitly alongside `x`. Both are reported:
`latent_marginals("region")` gives the `n` `x`-marginals (matching the dense
path) and `latent_marginals("region.structured")` gives the `n` `u*`
marginals — so the spatially-structured part is recoverable separately from
the total effect. (The dense `n`-dimensional path parameterizes `u*` away and
reports `x` only.)

## Directed spatial-autoregressive (SAR) effect

`SAR(name, index, graph, rho, precision=1.0)` declares a **directed**
spatial-autoregressive latent effect, for influence relations that a
symmetric CAR neighbour graph cannot represent — interbank exposure,
ownership, or supply-chain dependence, where "`a` is exposed to `b`" does not
imply "`b` is exposed to `a`". Unlike `Besag`/`ProperCAR`/`BYM2`, `graph`
need not be symmetric: `graph[node]` lists that node's own counterparties
(row `i` = who `i` depends on), and edge weights, if given, need not match
their reverse.

The precision is built from a directed operator rather than a Laplacian:

```
W row-standardized (each row rescaled to sum to 1)
M = I - ρW
Q = precision · MᵀM = precision · (I - ρW)ᵀ(I - ρW)
```

`MᵀM` is symmetric and positive-definite for any `ρ` in the open interval
`(-1, 1)` — regardless of `W`'s asymmetry — which is why `ρ` (unlike
`ProperCAR`'s graph-dependent bound) always validates against the fixed
`(-1, 1)` interval. `ρ = 0` recovers an independent, unit-scaled precision.
Row-standardization means `ρ` reads as *how much of a node's latent value is
explained by the (equally-weighted-average) counterparties' values versus its
own idiosyncratic shock* — `ρ → 0` independent nodes, `ρ → 1` near-total
pass-through.

```python
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, SAR

graph = {"bank_0": ["bank_1", "bank_2"], "bank_1": ["bank_2"], "bank_2": ["bank_0"]}
model = LGM(
    response="y",
    predictor=Fixed("1") + SAR(
        "influence", "bank", graph,
        rho=Hyperparameter("influence.rho", initial=0.0, transform="logit"),
        precision=Hyperparameter("influence.precision", initial=1.0),
    ),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
result.hyperparameters["influence.rho"]  # estimated contagion strength
```

`rho` and `precision` each accept a plain float (fixed) or a `Hyperparameter`
(estimated by empirical Bayes / integrated under `hyperparameters="integrate"`),
exactly like `ProperCAR`'s ρ — declare `rho` with `transform="logit"` to
estimate it. `SAR`'s precision is full-rank, so it carries **no constraint**
and works under the default `latent_strategy="gaussian"`. See
[`examples/directed_network_sar`](https://github.com/Ardea00/pylgm/tree/main/examples/directed_network_sar)
for a runnable interbank-exposure fit that recovers a known `ρ`.

**From YAML:** the standalone frontend declares `type: sar` alongside
`besag`/`proper_car`/`bym2`, with `graph`/`graph_file` and a required fixed
`rho` (estimating `rho` from YAML stays Python-API-only, like `proper_car`).

```yaml
predictor:
  effects:
    - {name: influence, type: sar, index: bank, rho: 0.5, precision: 1.0, graph: {...}}
```

## Dynamic spatial panel (SDPD)

`DynamicSpatialPanel(name, unit, time, graphs, rho, gamma=0.0, eta=0.0, precision=1.0)`
is the time-varying generalization of `SAR`: a balanced `unit x time` grid
with one directed network `W_t` per period (`graphs` is a `{time: graph}`
mapping), and three coefficients reading the diffusion mechanism:

- **`ρ` (contemporaneous)** — same-period spatial pass-through, as in `SAR`.
- **`γ` (temporal)** — a unit's own persistence from the previous period
  (an AR(1)-like term, but per-unit rather than pooled).
- **`η` (spatio-temporal diffusion)** — how much of a unit's *previous*
  period's counterparties' values carry forward into it this period.

The latent field is stacked period-major-then-unit (`f"{unit}@{time}"`
labels) with a block-bidiagonal operator: the diagonal block for period `t`
is `A_t = I - ρW_t`, and the sub-diagonal block coupling period `t` to `t-1`
is `-B_t = -(γI + ηW_t)`; the first period has no sub-diagonal block
(conditional-on-initial). The precision is `Q = precision · MᵀM`, exactly the
same `_gram_precision` construction `SAR` uses — indeed **`T = 1` reduces
exactly to `SAR`**. Every `graphs[t]` is aligned onto the union of all
periods' units, so a unit absent from a given period's network is simply an
all-zero row that period; the grid itself must still be balanced (every
`(unit, time)` pair observed).

```python
from pylgm import DynamicSpatialPanel, Fixed, Gaussian, Hyperparameter, LGM

graphs = {"2021": graph_2021, "2022": graph_2022, "2023": graph_2023}
model = LGM(
    response="y",
    predictor=Fixed("1") + DynamicSpatialPanel(
        "d", "bank", "year", graphs,
        rho=Hyperparameter("d.rho", initial=0.0, transform="logit"),
        gamma=Hyperparameter("d.gamma", initial=0.0, transform="identity"),
        eta=Hyperparameter("d.eta", initial=0.0, transform="identity"),
        precision=Hyperparameter("d.precision", initial=1.0),
    ),
    likelihood=Gaussian(sigma=0.1),
)
result = model.fit(frame)
```

`rho`, `gamma`, `eta`, and `precision` each independently accept a fixed
float or a `Hyperparameter`; `rho` should be declared `transform="logit"`
(bounded to `(-1, 1)`) and `gamma`/`eta` `transform="identity"` (unbounded).
Like `SAR`, `DynamicSpatialPanel`'s precision is full-rank and unconstrained.

**Forecasting future periods.** `forecast_dynamic_spatial_panel(result,
effect, future_graphs)` propagates the fitted last-period latent mean and
marginal variance forward through new periods' networks via the SDPD forward
recursion `x̂_{t+1} = A_{t+1}⁻¹B_{t+1}x̂_t`, with the matching variance
propagation carried diagonal-only (marginal, consistent with the gaussian
latent strategy). It returns a frame with columns `unit, time, mean,
variance` for the requested future periods:

```python
from pylgm import forecast_dynamic_spatial_panel

forecast = forecast_dynamic_spatial_panel(
    result, model.predictor.effects[-1], {"2024": graph_2024, "2025": graph_2025}
)
```

**Sparse (E-sparse) scale.** Both `SAR` and `DynamicSpatialPanel` fit past the
dense-reference guard through the same sparse constrained-Gaussian solver as
`Besag`/`BYM2`, with posterior mean, marginal variance, estimated
hyperparameters, and predictions available — but only under the default
`latent_strategy="gaussian"`: `simplified_laplace`/`laplace` marginals are not
yet available above the sparse guard for any effect (a general restriction,
not SAR/SDPD-specific), so use the default strategy at network scale.

**Not yet built:** raw (non-row-standardized) `W`, unbalanced `unit x time`
grids, non-Gaussian likelihoods' sparse marginals for `SAR`/`DynamicSpatialPanel`,
and a YAML `dynamic_spatial_panel` type (Python-API-only for now) — see the
[spatial roadmap](roadmap.md) for what's next.


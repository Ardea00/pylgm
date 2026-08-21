# pyLGM Latte-Parity and PySpark Architecture

**Status:** Approved 2026-08-08

**Scope:** This is the target-state architecture and staged roadmap, not a
claim that every example or component is runnable in pyLGM 0.3. The narrower
0.3 contract and a runnable Python/YAML example are documented in the
[project README](../../../README.md) and
[general LGM example](../../../examples/general_lgm/README.md).

## Purpose

pyLGM is a general-purpose Python library for latent Gaussian models (LGMs). Its
functional reference is Latte.jl, while its data boundary is designed for Pandas
and PySpark. Inflation forecasting, nowcasting, rolling evaluation, and economic
benchmarks are applications of the library rather than its organizing concern.

The library models

\[
y_i \mid x, \theta \sim \pi(y_i \mid \eta_i, \theta), \qquad
\eta = A x + o, \qquad
x \mid \theta \sim \mathcal N(\mu, Q(\theta)^{-1}),
\]

where the observation operator \(A\) and latent precision \(Q\) remain sparse.

## Design Principles

- Follow Latte.jl semantically without copying Julia-specific macro syntax.
- Keep one inference-independent LGM intermediate representation (IR).
- Compile Python objects and YAML configuration to the same IR.
- Use PySpark for distributed data preparation and sparse operator assembly.
- Run global sparse LGM inference on the driver.
- Never collect the full source DataFrame when a compact model representation is
  sufficient.
- Keep PySpark and heavy numerical backends optional.
- Preserve current Gaussian behavior while generalizing the architecture.
- Do not silently fall back from one inference engine to another.

## Target-State Public API

The intended primary interface is a declarative Python model. The following is
a target-state example: prior metadata is declared correctly through a
`Hyperparameter`, but 0.3 only materializes each parameter's `.initial` value,
and its `spark_df` line requires the deferred PySpark adapter.

```python
from pylgm import Gaussian, Fixed, Hyperparameter, IID, LGM, PCPrecision, RW2

region_precision = Hyperparameter(
    "region_precision",
    initial=1.0,
    prior=PCPrecision(upper_sd=1.0, alpha=0.01),
)
trend_precision = Hyperparameter(
    "trend_precision",
    initial=1.0,
    prior=PCPrecision(upper_sd=1.0, alpha=0.01),
)

model = LGM(
    response="inflation",
    likelihood=Gaussian(),
    predictor=(
        Fixed("1 + energy_price")
        + IID("region", index="region", precision=region_precision)
        + RW2("trend", index="month", precision=trend_precision)
    ),
    panel=("region",),
    time="month",
)

result = model.fit(spark_df, engine="exact_gaussian")
```

In 0.3, use a Pandas DataFrame and fixed numeric hyperparameters. YAML is a
serializable frontend for standard components, not a separate modeling system.
A runnable file using this complete schema is published as
[`examples/general_lgm/config.yaml`](../../../examples/general_lgm/config.yaml):

```yaml
response: inflation
likelihood:
  family: gaussian
  sigma: 0.5
data:
  panel: [region]
  time: month
predictor:
  fixed: "1 + energy_price"
  effects:
    - {name: region, type: iid, index: region, precision: 2.0}
    - {name: trend, type: rw2, index: month, precision: 3.0}
```

The target state gives every inference engine a common `LGMResult` surface:

```python
result.latent_marginals("trend")
result.hyperparameter_marginals()
result.predict(new_data)
result.linear_combinations(...)
result.log_marginal_likelihood
result.diagnostics
```

In 0.3, fit-row predictions are exposed as `predictive_mean` and
`predictive_variance`, while `result.predict(new_data)` is deferred.
`hyperparameter_marginals()` is present but empty for the conditional exact
fit because hyperparameters are not integrated.

Custom latent effects implement a small compilation protocol that returns a
validated latent block. Standard users should not need this protocol.

## Target Data and Inference Boundary

Pandas and PySpark are adapters into the same compiler. An adapter is responsible
for validating semantic columns, establishing stable categorical/index mappings,
and producing the response, observed mask, offsets, and sparse design operators.
The compiled IR owns no DataFrame and has no Spark dependency.

PySpark performs distributed reads, filtering, transformation, indexing, and
aggregation. Only the compact arrays, index maps, sparse triplets, and graph
operators required by the model cross to the driver. Global factorizations and
hyperparameter integration run on the driver because they operate on a coupled
sparse precision system and do not decompose naturally by Spark row partitions.

The 0.3 adapter is Pandas-only. It preserves a positional source-row mapping
through canonical sorting and realigns declarative fit-row predictions to the
caller's original order. PySpark input remains a later adapter slice.

## Target Core Intermediate Representation

The generalized IR contains:

- response values and observation mask;
- likelihood and link specifications;
- offset, exposure, and optional observation weights;
- named latent blocks and their sparse observation operators;
- sparse precision operators parameterized by hyperparameters;
- identifiability constraints;
- hyperparameters, transforms, and priors;
- stable labels and source-index mappings.

`sigma` is a Gaussian likelihood parameter, not a field on the general compiled
model. Precision matrices may be intrinsic and constrained. The compiler must not
assume that every valid global precision is merely a fixed block diagonal matrix.

The 0.3 declarative compiler is an interim conditional implementation: it
materializes each `Hyperparameter.initial` and does not preserve parameter names,
transforms, or priors in `CompiledLGM`. Retaining that metadata in a parameterized,
inference-independent IR is a target-state requirement, not a completed 0.3
capability.

## Components and Inference Roadmap

Development proceeds in complete vertical slices.

### 1. General LGM foundation (target state)

- fixed, IID, RW1, RW2, and AR1 effects;
- identifiability constraints;
- ordinary and penalized-complexity priors;
- Gaussian likelihood;
- Pandas and PySpark adapters;
- posterior marginals, prediction, and linear combinations;
- exact Gaussian reference inference.

The delivered 0.3 slice is narrower: fixed/IID/RW1/RW2 effects, Gaussian
likelihood, Pandas input, fit-row prediction, posterior marginals, linear
combinations, and the exact Gaussian reference engine. The optional PySpark
data-boundary adapter has since shipped (see below). Declarative empirical
Bayes has since shipped for the exact-Gaussian engine: a `Hyperparameter`
declared on `sigma` or an effect precision is estimated by type-II ML rather
than merely materialized at `.initial` (see below and the
[project README](../../../README.md#empirical-bayes)). Penalized MAP-II has
since shipped on top of that: when the declared `Hyperparameter` also
carries a `prior`, its native-scale log density penalizes the same marginal
likelihood, and `result.diagnostics["hyperparameter_penalized"]` records it
(no Jacobian correction for `transform`). This is still a point estimate,
not a marginal. AR1, `result.predict(new_data)`, and parameterized IR
metadata are explicitly deferred. The hyperparameter integration core (grid
quadrature, section 3a) and model-assessment criteria (DIC/WAIC/CPO/PIT,
section 3b) have since shipped; richer latent strategies beyond Gaussian
(section 3c) remain deferred. The target foundation is therefore not
complete in 0.3.

### 2. Non-Gaussian LGM

The Laplace core of this slice has shipped: `Poisson()` (canonical log link)
and `Bernoulli()` (canonical logit link) likelihoods, an `LGM.offset` column
added to the linear predictor before the link, and a Newton-Raphson
`engine="laplace"` mode-finder dispatched through `LGM.fit`, exposed through
both the Python and YAML frontends and the existing Pandas/PySpark data
boundary. It fits at plug-in hyperparameters by default (declared effect
precisions are conditioned on, not integrated); `result.fitted_mean` is exact
for the log link and a documented point estimate for the logit link.
Declarative empirical Bayes has since shipped for this engine too: a
`Hyperparameter` declared on an effect precision is estimated by type-II ML
at the Laplace mode, dispatched the same way through `LGM.fit` (see the
[project README](../../../README.md#empirical-bayes)). Penalized MAP-II has
since shipped for this engine as well, the same way as for exact Gaussian
(see section 1 above). Still deferred:

- Binomial(n) likelihood;
- non-canonical links;
- exposures and observation weights;
- richer latent strategies for INLA integration (section 3c below);
- spatial effects.

### 3. INLA-style inference

**3a. INLA integration core — shipped.** Penalized MAP-II (section 1 and 2
above) was the interim step short of full integration. `LGM.fit(...,
hyperparameters="integrate")` has since shipped for both the
exact-Gaussian and Laplace engines: grid quadrature in log space around the
empirical-Bayes mode, oriented and scaled by a finite-difference Hessian,
weighted by the marginal likelihood and the log-space Jacobian, with a
`max_grid_points` guard. It gives an integrated marginal likelihood,
populated `result.hyperparameter_marginals()`, and integrated latent
marginals via `result.latent_marginals(...)`, using the Gaussian latent
strategy at every grid point (see the
[project README](../../../README.md#inla-integration)).

**3b. Model-assessment criteria — shipped.** Every integrated fit
(`result.criteria`, an attached `ModelCriteria`) reports DIC, WAIC (each with
an effective-parameter count), per-observation CPO via the leave-one-out
harmonic-mean identity with a `cpo_failures` reliability flag, a
`log_cpo_sum` leave-one-out log score, and per-observation PIT, for both the
exact-Gaussian and Laplace engines (see the
[project README](../../../README.md#model-assessment-criteria)). Still
deferred within 3b: a randomized PIT for discrete-response likelihoods (the
shipped `pit` is the non-randomized `P(Y <= y)`), and criteria for plug-in
fits (`hyperparameters="optimize"` or the default) rather than only
integrated ones.

Still deferred:

- **3c. Richer latent strategies** — the simplified- and full-Laplace
  corrections to the latent marginals that give INLA its name; 3a uses only
  the Gaussian approximation (exact-Gaussian posterior or Laplace mode) at
  each grid point.

### 4. Spatial effects

- Besag/ICAR, proper CAR, and BYM2;
- graph construction through either data adapter;
- optional Matérn-SPDE support.

### 5. HMC-Laplace

- sampling over hyperparameters;
- conditional Laplace approximation of the latent field;
- optional advanced backend after INLA-style inference is stable.

## Reuse of the Existing Codebase

The following implementation is retained and generalized:

- immutable latent blocks, sparse designs, precisions, and constraints;
- fixed, IID, RW1, and RW2 builders;
- formula and configuration compilation;
- exact Gaussian posterior and marginal-likelihood calculations;
- empirical-Bayes optimization of observation scale and effect precisions;
- strict validation, CLI conventions, and existing regression tests.

The target compatibility-preserving refactor comprises:

1. separate likelihood-specific state from the general IR;
2. introduce likelihood, link, prior, and hyperparameter abstractions;
3. introduce a common posterior marginal and result API;
4. add the PySpark adapter without coupling Spark to the core;
5. preserve legacy configuration and Gaussian outputs;
6. move forecasting-specific evaluation behind `pylgm.contrib.forecasting`.

Version 0.3 delivers the likelihood separation, modeling vocabulary, common
result surface, Pandas compiler, and compatibility preservation. The PySpark
data-boundary adapter (Gaussian-only, driver-side inference) has since shipped
per the [PySpark adapter design](2026-08-17-pylgm-pyspark-adapter-design.md); the
`contrib.forecasting` relocation and distributed sparse assembly remain follow-up
work.

The current dense Gaussian implementation remains a small/medium reference engine.
A sparse backend replaces it as the scalable default in a later vertical slice.

## Target Package Boundaries

```text
pylgm/
  model.py
  ir/
  data/
  effects/
  likelihoods/
  links/
  priors/
  inference/
  results/
  config/
  contrib/forecasting/
```

Forecasting, nowcasting, rolling backtests, Ridge or persistence benchmarks, and
economic report artifacts do not belong to the inference core. Existing features
remain available during migration and move to `contrib` with compatibility imports
where practical.

The current 0.3 tree has not completed the `results/` split or forecasting
relocation; the diagram is the intended boundary.

## Failure Model

- Data errors identify invalid columns, keys, values, or Spark mappings.
- Compilation errors identify invalid model structure or unsupported components.
- Identifiability errors identify unresolved intrinsic-model constraints.
- Numerical errors identify factorization or finite-arithmetic failures.
- Convergence errors carry optimizer state and diagnostics.
- Unsupported-engine errors list the model feature that prevents execution.

An inference result always records its engine, convergence state, effective model
dimensions, warnings, and numerical diagnostics. No failed fit is presented as a
successful result.

## Validation Strategy

Correctness is established at three levels:

1. analytic Gaussian cases with known posterior moments and marginal likelihood;
2. reproducible parity fixtures against Latte.jl and R-INLA;
3. simulation studies with known parameters, posterior coverage, and calibrated
   predictive quantities.

Pandas and PySpark compilation of the same logical data must yield equivalent IRs,
including index maps and sparse operators. Each inference engine has documented
numerical tolerances and remains experimental until it passes its parity suite.
The Italian NIC dataset is a real application example, not the primary oracle for
mathematical correctness.

## Non-goals for the First Refactor

- AR1 effects;
- the PySpark adapter (was a non-goal of the first refactor; has since shipped
  as its own Gaussian-only data-boundary slice);
- `result.predict(new_data)`;
- parameterized IR metadata and prior-aware hyperparameter inference;
- non-Gaussian likelihoods;
- INLA integration (was a non-goal of the first refactor; the integration
  core and model-assessment criteria have since shipped as sub-slices 3a and
  3b, with richer latent strategies (3c) remaining);
- spatial effects and SPDE meshes;
- HMC-Laplace;
- distributed sparse factorization on Spark executors;
- a new forecasting evaluation framework.

These remain explicit later vertical slices rather than placeholders inside the
foundation refactor.

# pyLGM Latte-Parity and PySpark Architecture

**Status:** Approved 2026-08-08

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

## Public API

The primary interface is a declarative Python model:

```python
from pylgm import Gaussian, Fixed, IID, LGM, PCPrecision, RW2

model = LGM(
    response="inflation",
    likelihood=Gaussian(),
    predictor=(
        Fixed("1 + energy_price")
        + IID("region", index="region", precision=PCPrecision(1, alpha=0.01))
        + RW2("trend", index="month", precision=PCPrecision(1, alpha=0.01))
    ),
)

result = model.fit(spark_df, engine="exact_gaussian")
```

YAML is a serializable frontend for standard components, not a separate modeling
system:

```yaml
response: inflation
likelihood:
  family: gaussian
predictor:
  fixed: "1 + energy_price"
  effects:
    - {name: region, type: iid, index: region}
    - {name: trend, type: rw2, index: month}
```

Every inference engine returns a common `LGMResult` surface:

```python
result.latent_marginals("trend")
result.hyperparameter_marginals()
result.predict(new_data)
result.linear_combinations(...)
result.log_marginal_likelihood
result.diagnostics
```

Custom latent effects implement a small compilation protocol that returns a
validated latent block. Standard users should not need this protocol.

## Data and Inference Boundary

Pandas and PySpark are adapters into the same compiler. An adapter is responsible
for validating semantic columns, establishing stable categorical/index mappings,
and producing the response, observed mask, offsets, and sparse design operators.
The compiled IR owns no DataFrame and has no Spark dependency.

PySpark performs distributed reads, filtering, transformation, indexing, and
aggregation. Only the compact arrays, index maps, sparse triplets, and graph
operators required by the model cross to the driver. Global factorizations and
hyperparameter integration run on the driver because they operate on a coupled
sparse precision system and do not decompose naturally by Spark row partitions.

## Core Intermediate Representation

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

## Components and Inference Roadmap

Development proceeds in complete vertical slices.

### 1. General LGM foundation

- fixed, IID, RW1, RW2, and AR1 effects;
- identifiability constraints;
- ordinary and penalized-complexity priors;
- Gaussian likelihood;
- Pandas and PySpark adapters;
- posterior marginals, prediction, and linear combinations;
- exact Gaussian reference inference.

### 2. Non-Gaussian LGM

- Poisson, Binomial, and Bernoulli likelihoods;
- identity, log, and logit links;
- offsets, exposures, and weights;
- latent-mode optimization and Laplace approximation;
- fast inference at the hyperparameter mode.

### 3. INLA-style inference

- numerical integration over hyperparameters;
- latent and hyperparameter marginals;
- log marginal likelihood, DIC, WAIC, CPO, and PIT;
- explicit approximation-quality diagnostics.

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

The first implementation milestone is a compatibility-preserving refactor:

1. separate likelihood-specific state from the general IR;
2. introduce likelihood, link, prior, and hyperparameter abstractions;
3. introduce a common posterior marginal and result API;
4. add the PySpark adapter without coupling Spark to the core;
5. preserve legacy configuration and Gaussian outputs;
6. move forecasting-specific evaluation behind `pylgm.contrib.forecasting`.

The current dense Gaussian implementation remains a small/medium reference engine.
A sparse backend replaces it as the scalable default in a later vertical slice.

## Package Boundaries

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

- non-Gaussian likelihoods;
- INLA integration;
- spatial effects and SPDE meshes;
- HMC-Laplace;
- distributed sparse factorization on Spark executors;
- a new forecasting evaluation framework.

These remain explicit later vertical slices rather than placeholders inside the
foundation refactor.

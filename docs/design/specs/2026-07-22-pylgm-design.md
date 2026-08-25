# pyLGM design specification

Date: 2026-07-22  
Repository: `Ardea00/pyLGM`  
Python package: `pylgm`  
CLI: `pylgm`  
License: MIT

## 1. Purpose

pyLGM is an open-source Python library for configuration-driven Bayesian inference and forecasting with latent Gaussian models (LGMs). It targets structured additive models for temporal, spatial, and general panel data.

Italian national and regional inflation nowcasting is the reference application, but no CPI-specific behavior belongs in the statistical core. The same core must support other panel problems through declared dimensions, likelihoods, covariates, latent effects, and interactions.

The main user workflow is local and reproducible:

```bash
pylgm compare configs/italian_inflation.yaml
```

The equivalent local Python interface is:

```python
from pylgm import Pipeline

result = Pipeline.from_yaml("configs/italian_inflation.yaml").run()
```

“API” means this importable Python interface, not a network service. A server, hosted platform, scheduler, and HTTP API are outside the initial scope.

## 2. Design principles

1. **One model, multiple inference engines.** A model is compiled once into an immutable intermediate representation (IR). Every inference engine consumes that representation.
2. **Independent statistical engine.** pyLGM owns its model graph, sparse-precision algebra, constraints, posterior approximations, NUTS, INLA, HMC-Laplace, posterior reconstruction, and forecasting behavior.
3. **Foundational numerical dependencies only.** NumPy, SciPy/SuiteSparse, and JAX may provide arrays, sparse factorization, and automatic differentiation. They do not define pyLGM's inference algorithms.
4. **Correctness before speed.** Analytical tests, simulation-based calibration, and cross-engine agreement gate releases before performance benchmarks.
5. **Explicit statistical behavior.** Defaults are resolved into persisted configuration. Silent statistical fallbacks are forbidden.
6. **Domain-neutral core.** Source-specific ingestion and inflation conventions live in application adapters and configurations.
7. **Local-first operation.** Pandas, local Spark, Spark Connect, and externally managed Spark sessions such as Databricks are supported without platform-specific deployment machinery.
8. **Rue-based formulation.** GMRF, intrinsic-field, scaling, BYM2, penalized-complexity-prior, and INLA conventions follow the cited work of Håvard Rue and collaborators.

## 3. Scope

### 3.1 Initial scope

- Structured additive panel LGMs with temporal, spatial, group, covariate, and separable interaction effects.
- Gaussian, Student-t, Poisson, and Binomial observation models.
- Exact Gaussian inference, repository-owned full-field NUTS, and conditional Laplace approximation.
- Configuration-driven candidate comparison and rolling-origin probabilistic evaluation.
- Vintage-aware evaluation when release metadata is supplied, with an explicit latest-available fallback.
- A canonical Spark/Pandas panel interface and optional source adapters.
- Italian inflation as a complete reference application targeting national and regional monthly CPI inflation, horizons 0, 1, 3, 6, and 12 months.

### 3.2 Planned extensions supported by the architecture

- INLA posterior marginals using Gaussian, simplified-Laplace, and full-Laplace strategies.
- HMC-Laplace with repository-owned NUTS over hyperparameters and conditional latent reconstruction.
- Product-category × region panels.
- Automated candidate generation and model-structure search.
- Additional likelihoods and latent-effect plugins.
- Conditional-LGM extensions such as learned factor loadings.

### 3.3 Excluded from the initial scope

- Arbitrary probabilistic programs.
- Unknown dynamic-factor loadings, nonlinear latent transitions, and regime switching.
- Distributed statistical inference on Spark executors.
- Web services, hosted dashboards, job schedulers, and production deployment infrastructure.
- Automatic source-specific data cleaning expressed as an unrestricted YAML transformation language.

## 4. System architecture

The dependency direction is strictly downward:

```text
YAML configuration + canonical panel
                |
                v
       validation and compiler
                |
                v
       immutable LGM model IR
                |
       +--------+--------+----------+
       |        |        |          |
     NUTS    Laplace   INLA   HMC-Laplace
       +--------+--------+----------+
                |
                v
       common posterior result
                |
       +--------+---------+
       |                  |
   forecasting        evaluation
```

Boundary rules:

- Inference engines never parse YAML or access Spark objects.
- The compiler never runs an inference algorithm.
- Application adapters produce a canonical panel and metadata; they never contain inference logic.
- Forecasting and evaluation consume the common posterior/result contract, not engine-specific objects.
- Engine-specific diagnostics may be attached under a namespaced diagnostics field while common diagnostics retain stable names.

## 5. Package structure

```text
src/pylgm/
  config/       versioned schema and resolved configuration
  data/         canonical panel and adapter interfaces
  effects/      structured latent-effect definitions
  ir/           immutable compiled LGM representation
  linalg/       constraints and sparse numerical algebra
  inference/    exact Gaussian, NUTS, Laplace, INLA, HMC-Laplace
  forecast/     posterior prediction and scenario handling
  evaluation/   rolling and vintage-aware backtesting
  artifacts/    reproducible run storage and fingerprints
  cli/          local command-line entry points

applications/
  italian_inflation/
    adapters/
    graphs/
    configs/
    reports/

tests/
  unit/
  integration/
  calibration/
  reference/
  performance/
```

Modules communicate through typed public contracts. Effect implementations construct IR blocks; they do not call inference code. Numerical kernels operate on arrays and sparse matrices without knowing application or configuration concepts.

## 6. Statistical model and intermediate representation

Every supported model compiles to

\[
y_i \mid \eta_i, \theta_y \sim p(y_i \mid \eta_i, \theta_y),
\qquad
\eta = o + X\beta + \sum_k A_k x_k,
\]

with latent field

\[
x=(\beta,x_1,\ldots,x_K)
\sim \mathcal N\!\left(0,Q(\theta)^{-1}\right).
\]

The immutable IR contains:

- observation values, missing-target mask, likelihood, link, and likelihood hyperparameters;
- offset vector and optional known observation weights;
- fixed-effect design matrix `X`;
- sparse observation mapping `A_k` for each structured effect;
- parameterized sparse precision block `Q_k(theta)` for each latent effect;
- transformed hyperparameters, priors, support constraints, and initial values;
- linear identifiability constraints;
- latent and observation coordinates for time, space, groups, and panel dimensions;
- rank, structural-zero, and sparsity metadata required by numerical kernels;
- prediction mappings for observed and future rows.

The compiler produces deterministic latent ordering and stable names. Identical resolved configuration and canonical data fingerprints must produce structurally identical IR.

## 7. Effect vocabulary

The initial registry contains:

- fixed intercepts and covariates;
- IID group intercepts and varying slopes;
- RW1 and RW2 temporal effects;
- cyclic seasonal effects;
- AR(1) temporal effects;
- Besag/ICAR spatial effects;
- scaled BYM2 spatial effects;
- separable space-time effects based on Kronecker precision structure;
- offsets and known observation weights.

The initial `space_time` effect combines one registered spatial block and one registered temporal block with `Q_st = Q_time ⊗ Q_space`. Its constraints are derived from the component null spaces. Non-separable interactions and the alternative Knorr-Held interaction types are extensions rather than implicit behaviors of this initial term.

Intrinsic effects remain intrinsically defined. pyLGM represents their rank deficiency, null space, generalized determinant, scaling, and constraints explicitly rather than adding an arbitrary ridge term. RW and ICAR components default to sum-to-zero constraints when an intercept is present. Incompatible or redundant constraints are compiler errors.

Intrinsic-field scaling follows Sørbye and Rue so precision priors have comparable marginal-scale interpretations across graphs and time grids. BYM2 follows the scaled structured/unstructured parameterization of Riebler, Sørbye, Simpson, and Rue. Penalized-complexity priors are the default for standard deviation, correlation, and BYM2 mixing parameters; all resolved prior parameters are recorded.

## 8. Likelihood interface

Each likelihood supplies:

- elementwise log likelihood;
- derivatives required by Newton and Laplace calculations;
- parameter-support transforms;
- simulation for prior/posterior predictive checks;
- validation rules for response values and observation weights.

Initial likelihoods are Gaussian, Student-t, Poisson, and Binomial. Conditional independence by observation is required by the initial IR. Missing responses are prediction targets, not zero-weight pseudo-observations.

## 9. Inference roadmap

### 9.1 Exact Gaussian foundation

For Gaussian likelihoods with fixed hyperparameters, pyLGM computes exact posterior means, selected marginal variances, conditional simulations, and conditional log marginal likelihoods through constrained sparse Gaussian algebra. It does not label plug-in or numerically integrated hyperparameter uncertainty as exact. This stage is the analytical foundation for testing all later engines.

### 9.2 Repository-owned NUTS

The reference sampler implements:

- unconstrained parameter transforms and Jacobian adjustments;
- leapfrog Hamiltonian integration;
- multinomial NUTS tree expansion;
- no-u-turn termination;
- dual-averaging step-size adaptation;
- online diagonal mass-matrix adaptation by default, with dense adaptation allowed only below a configured dimension threshold;
- deterministic chain seeding;
- energy, acceptance, tree-depth, and divergence diagnostics.

The first NUTS implementation samples the complete latent field and hyperparameters. It is intended as the accuracy reference, not the fastest engine.

### 9.3 Conditional Laplace

For a hyperparameter vector `theta`, pyLGM finds the conditional latent mode using damped sparse Newton iterations. The negative conditional Hessian has the form

\[
Q^*(\theta)=Q(\theta)+A^\top W A.
\]

The engine computes the conditional Gaussian approximation, sparse log determinant, and Laplace-adjusted marginal log posterior for `theta`. Failure to find a valid mode or positive curvature is diagnostic output, not a trigger for a hidden alternative algorithm.

### 9.4 INLA

INLA development follows Rue, Martino, and Chopin:

1. approximate the joint hyperparameter posterior using the conditional Gaussian approximation to the latent field;
2. locate and scale the hyperparameter mode;
3. explore the low-dimensional hyperparameter posterior and assign numerical-integration weights;
4. compute conditional latent marginals using Gaussian, simplified-Laplace, and subsequently full-Laplace strategies;
5. integrate conditional marginals over hyperparameters;
6. expose marginal likelihood and predictive quantities with approximation metadata.

INLA is released only for bounded hyperparameter dimensionality established by benchmarks. Unsupported dimensionality produces a clear error suggesting NUTS or HMC-Laplace.

### 9.5 HMC-Laplace

HMC-Laplace applies the repository-owned NUTS algorithm to the Laplace-marginalized posterior of the low-dimensional hyperparameters. Conditional latent-field posteriors are reconstructed for retained hyperparameter draws. Gradients through the optimized latent mode and sparse factorization follow explicit implicit-differentiation identities and are checked against finite differences.

## 10. Data boundary and Spark integration

The reusable core starts from a canonical panel supplied as a Pandas or Spark DataFrame plus metadata. Required semantic roles are:

- response column;
- ordered time index;
- zero or more panel/entity indexes;
- declared covariates;
- optional spatial unit index and graph;
- optional offset and observation weights;
- optional `available_at` timestamp and vintage identifier.

The data layer supports:

- Pandas DataFrames;
- a locally created SparkSession;
- an externally supplied SparkSession;
- Spark Connect;
- managed sessions such as Databricks through the same adapter contract.

Spark performs ingestion, joins, filtering, aggregation, and declared standard transformations. Statistical inference is local. Before collection, pyLGM validates estimated row count, dimensional cardinalities, memory limits, duplicate panel keys, and aggregation level. The collected model panel is converted to typed NumPy/Xarray structures.

Optional application adapters may declare Spark tables, SQL, mappings, joins, and standard transformations. Complex source-specific behavior remains ordinary tested adapter code rather than becoming a general YAML data language.

## 11. Configuration

Configuration is YAML validated against a versioned strict schema. Unknown keys are errors. Defaults are expanded into a persisted resolved configuration.

Ordinary covariates use a formula string. Structured effects use explicit blocks:

```yaml
schema_version: 1

data:
  adapter: italian_inflation
  time: month
  response: inflation_yoy
  panel: [region]
  spatial_graph: italy_regions
  available_at: release_date
  vintage: vintage_date

model:
  likelihood: gaussian
  fixed: "1 + fuel_price + energy_price"
  effects:
    - name: national_trend
      type: rw2
      index: month
    - name: regional_spatial
      type: bym2
      index: region
      graph: italy_regions
    - name: regional_dynamics
      type: space_time
      space: region
      time: month

candidates:
  - name: temporal
    disable: [regional_spatial, regional_dynamics]
  - name: spatial_temporal
  - name: robust
    likelihood: student_t

inference:
  engine: nuts
  chains: 4
  draws: 1000
  warmup: 1000

evaluation:
  mode: auto
  horizons: [0, 1, 3, 6, 12]
  metrics: [log_score, crps, rmse, coverage]
```

Candidate entries inherit the base model and may override only schema-approved fields. The resolved configuration stores the complete model for every candidate so results do not depend on inheritance interpretation after execution.

## 12. Pipeline behavior

A comparison run performs these steps:

1. load or receive the Spark/Pandas panel;
2. validate schema, time order, panel uniqueness, missingness, spatial graph alignment, and release metadata;
3. apply declared standard transformations;
4. materialize the model-sized canonical panel locally;
5. resolve and compile every candidate into IR;
6. run prior predictive and identifiability checks;
7. fit and diagnose each candidate;
8. run rolling-origin backtests without future-data leakage;
9. rank candidates by predictive log score and calibration, with CRPS and RMSE as supporting metrics;
10. refit the selected candidate through the requested cutoff;
11. generate probabilistic forecasts and nowcasts;
12. persist reproducibility and result artifacts.

`evaluation.mode: auto` selects vintage-aware folds when both availability and vintage metadata are usable. Otherwise it uses latest-available expanding-window folds and emits a prominent warning. Every result and metric is labeled `vintage_aware` or `latest_available`; rankings never mix the two modes.

Candidate comparison is explicit in the initial version. Automated candidate generation is a later layer that emits ordinary resolved candidates and uses the same evaluation machinery.

## 13. Forecasting semantics

Horizon 0 means nowcasting the current unreleased or partially observed period. Positive horizons mean forecasting after the configured information cutoff. Future response rows are represented through prediction mappings and missing-target masks.

Covariates must declare one of these future-data policies:

- known at forecast origin;
- lagged so only released observations are used;
- externally supplied scenario;
- jointly forecast by a declared model, introduced after the initial release.

The compiler rejects a backtest fold if a covariate value violates its availability policy. Forecast outputs retain region, time, horizon, candidate, engine, evaluation mode, posterior summary, and interval-level coordinates.

## 14. Italian inflation reference application

The initial target is official monthly CPI inflation for Italy at national and regional levels. It supports current-month nowcasts and 1-, 3-, 6-, and 12-month forecasts.

The first reference model contains:

- a national RW2 or local-trend component;
- monthly seasonality;
- persistent regional deviations;
- scaled BYM2 regional structure based on the Italian regional adjacency graph;
- regional time-varying deviations or separable space-time interaction;
- timely observed regressors such as energy, fuel, commodity, and online-price indicators when available;
- observation/revision uncertainty represented explicitly when vintage data permit estimation.

Official aggregation weights are treated as known inputs where applicable. The initial application does not estimate unrestricted factor loadings. The data schema reserves `product_category`, allowing later product × region expansion without changing result coordinate conventions.

## 15. Results and artifacts

Every run writes a self-contained local artifact directory containing:

- submitted and resolved configuration;
- canonical data and spatial-graph fingerprints;
- package, Python, dependency, platform, and Spark versions;
- random seeds;
- compiled-model structural summary;
- posterior draws or deterministic marginal representations;
- engine diagnostics;
- fold definitions and information cutoffs;
- forecasts and evaluation metrics;
- candidate failures and model-selection decision;
- human-readable run summary.

Large data sources are fingerprinted rather than copied by default. Posterior and forecast arrays use labeled Xarray-compatible storage. Artifact schema versions are independent of configuration schema versions.

## 16. Error handling

Errors are typed by stage: configuration, data contract, compilation, numerical algebra, inference convergence, diagnostics, forecasting, and artifacts.

The following are hard errors unless the user explicitly changes the model:

- unknown configuration keys;
- duplicate panel keys;
- graph nodes inconsistent with observed spatial indexes;
- disconnected spatial components without an explicit policy;
- inconsistent rank or null-space calculations;
- redundant or insufficient identifiability constraints;
- non-finite log densities or derivatives;
- failure to obtain a valid sparse factorization;
- indefinite conditional curvature where positive curvature is required;
- future-data leakage in an evaluation fold.

During comparison, a candidate failure is isolated and recorded. Other candidates may continue. A candidate is ineligible for final selection when configured convergence, divergence, calibration, or numerical-stability thresholds fail.

## 17. Verification strategy

### 17.1 Mathematical and numerical tests

- analytical Gaussian posterior and log-marginal-likelihood fixtures;
- dense-versus-sparse equivalence;
- constrained-versus-reduced-coordinate equivalence for intrinsic fields;
- generalized determinant and rank tests;
- automatic-differentiation versus finite-difference gradients;
- sparse solve and selected-inverse checks;
- deterministic reproducibility within documented tolerances.

### 17.2 Inference tests

- NUTS on independent and correlated Gaussians;
- funnels and hierarchical models for divergence behavior;
- simulation-based calibration and parameter recovery;
- exact Gaussian versus NUTS agreement;
- NUTS versus Laplace agreement where the posterior is near Gaussian;
- exact or high-precision quadrature comparisons for small non-Gaussian LGMs;
- INLA and HMC-Laplace comparisons against full-field NUTS.

### 17.3 External reference tests

Published Rue/INLA examples become reproducible benchmark fixtures. Latte.jl and R-INLA may be used outside the runtime dependency graph as test oracles. Reference outputs and tolerances are stored so normal CI remains open-source and self-contained.

### 17.4 Pipeline tests

- Pandas, local Spark, Spark Connect, and externally managed-session adapter contracts;
- rolling-fold cutoff and leakage tests;
- vintage selection and revision handling;
- candidate failure isolation;
- artifact round trips and schema migration;
- end-to-end synthetic national/regional forecasting.

### 17.5 Performance tests

Benchmarks track sparse factorization time, symbolic-factorization reuse, fill-in, peak memory, Newton iterations, gradient cost, NUTS effective samples per second, and end-to-end fold runtime. Performance improvements cannot weaken numerical or calibration tolerances without an explicit documented decision.

## 18. Delivery phases

### Phase 1: foundations

- project packaging, MIT license, schemas, canonical panel, immutable IR;
- fixed, IID, RW1/RW2, AR1, ICAR/BYM2, and separable interaction blocks;
- constrained exact Gaussian engine;
- repository-owned NUTS;
- Gaussian and Student-t likelihoods;
- CLI/Python pipeline, artifacts, and basic rolling evaluation;
- synthetic examples and Italian inflation adapter skeleton.

### Phase 2: fast approximation and application

- conditional Laplace engine;
- Poisson and Binomial likelihoods;
- full candidate comparison and selection;
- vintage-aware evaluation;
- Italian national/regional inflation reference pipeline;
- analytical, calibration, and external reference benchmark suite.

### Phase 3: nested inference

- hyperparameter exploration and numerical integration;
- Gaussian and simplified-Laplace INLA marginals;
- full-Laplace marginals for supported likelihoods;
- HMC-Laplace using repository-owned NUTS;
- conditional latent reconstruction and joint approximation output.

### Phase 4: broader panels and automation

- product-category × region inflation models;
- candidate-generation policies and bounded automatic search;
- additional structured effects, likelihoods, and conditional-LGM extensions;
- performance work guided by stored benchmark histories.

Each phase must be independently usable and must satisfy its verification gates before the next engine becomes a supported user-facing option.

## 19. Success criteria

The initial public research-library milestone succeeds when:

1. a user can prepare a canonical national/regional panel in Pandas or Spark and compare configured LGM candidates locally;
2. the same compiled Gaussian model produces consistent exact and NUTS posterior results within documented tolerances;
3. spatial and intrinsic temporal effects are correctly scaled, constrained, and validated;
4. rolling and vintage-aware evaluations cannot access observations or covariates unavailable at the fold cutoff;
5. all outputs contain sufficient configuration, data, environment, and seed metadata for local reproduction;
6. the Italian inflation application generates probabilistic current-month and multi-horizon regional forecasts;
7. failed numerical or diagnostic checks are visible and cannot silently select a final model;
8. the core contains no CPI-specific assumptions and a synthetic non-economic panel example runs through the same pipeline.

## 20. Foundational and comparative references

- Rue, H. and Held, L. (2005). *Gaussian Markov Random Fields: Theory and Applications*. Chapman & Hall/CRC.
- Rue, H., Martino, S., and Chopin, N. (2009). “Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations.” *JRSS B*, 71(2), 319–392.
- Sørbye, S. H. and Rue, H. (2014). “Scaling intrinsic Gaussian Markov random field priors in spatial modelling.” *Spatial Statistics*, 8, 39–51.
- Simpson, D., Rue, H., Riebler, A., Martins, T. G., and Sørbye, S. H. (2017). “Penalising model component complexity: A principled, practical approach to constructing priors.” *Statistical Science*, 32(1), 1–28.
- Riebler, A., Sørbye, S. H., Simpson, D., and Rue, H. (2016). “An intuitive Bayesian spatial model for disease mapping that accounts for scaling.” *Statistical Methods in Medical Research*, 25(4), 1145–1165.
- Ruiz-Cárdenas, R., Krainski, E. T., and Rue, H. (2012). “Direct fitting of dynamic models using integrated nested Laplace approximations—INLA.” *Computational Statistics & Data Analysis*, 56(6), 1808–1828.
- Weiland, T. (2026). *Latte.jl: Probabilistic programming for latent Gaussian models in Julia*. [Documentation](https://lattejl.org/) and [source repository](https://github.com/timweiland/Latte.jl).

The Rue and collaborator references are normative for mathematical conventions. Deviations require an explicit rationale in code documentation and validation against the reference formulation. Latte.jl is the comparative software reference for the one-model/multiple-engines architecture and an external validation oracle; pyLGM remains an independent Python implementation.

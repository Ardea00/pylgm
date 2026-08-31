# Roadmap

pyLGM 0.6 is a bounded, correct-by-construction foundation. This page is the
honest map of what's shipped, what's next, and what's deliberately deferred.
For the precise semantics of each shipped feature, follow the links into the
[guide](index.md).

## Shipped in 0.6

- **Likelihoods** — Gaussian (exact engine), Poisson, Bernoulli, Binomial
  (counts `n·p` with a per-row trials column), NegativeBinomial (overdispersed
  counts), Gamma (positive-continuous), and Beta (proportions in `(0,1)`), all
  on the Laplace engine, with fixed or estimated dispersion `φ`. See
  [likelihoods](likelihoods.md).
- **Effects** — `Fixed`, `IID`, `RW1`/`RW2`, stationary `AR1` (optionally
  **group-wise**: one independent series per panel unit, sharing `ρ` and
  `precision`), the drifting `Seasonal` pattern, the `MIDAS`
  mixed-frequency smooth-lag effect and its restricted (parametric)
  `MIDASParametric` counterpart (exp-Almon / Beta lag kernels), and the
  Knorr-Held `SpaceTime` interaction (Types I–IV). See [effects](effects.md).
- **Hybrid composition** — a mixed-frequency `MIDAS` term, a spatial `BYM2`
  term, and a temporal `AR1` term sum through `+` into one latent field that
  fits and predicts, demonstrated end to end in
  [`examples/hybrid_nowcast`](https://github.com/Ardea00/pylgm/tree/main/examples/hybrid_nowcast).
- **Spatial (CAR) family** — `Besag` (ICAR), `ProperCAR` (with `ρ` fixed or
  estimated), and `BYM2` (with `φ` fixed or estimated), complete for the dense
  reference regime. Graphs may be **weighted** (`{node: {neighbour: weight}}`),
  so the same family models firm-ownership / interbank-exposure / supply-chain
  networks, not only geographic adjacency. Isolated regions (no neighbours) are
  handled gracefully as independent `IID` singletons in `Besag`/`BYM2` (dense
  and augmented paths), and the augmented large-graph `BYM2` supports
  multi-component (island) graphs. See [spatial effects](spatial-effects.md).
- **Hyperparameter estimation** — type-II ML empirical Bayes and MAP-II with
  PC/Gaussian priors, with bounded hyperparameters. See
  [empirical Bayes](empirical-bayes.md).
- **Posterior integration** — INLA-style grid quadrature with gaussian,
  simplified-Laplace, and full-Laplace latent marginals, plus DIC/WAIC/CPO/PIT
  model-assessment criteria. See [INLA integration](inla.md).
- **Prediction** — fit-row and out-of-sample `result.predict(new_data)`. See
  [prediction](prediction.md).
- **Data boundary** — Pandas, or Spark / Databricks input. See [Spark](spark.md).
- **Declarative frontend** — models expressible in YAML via
  `pylgm.config.load_model`, including the temporal `ar1` (optionally group-wise)
  and `seasonal` effects, the spatial `besag`/`proper_car`/`bym2` families with
  an inline or file graph (`graph_file` accepts an R-INLA `.graph` or a `.json`
  neighbour dict), the Knorr-Held `spacetime` interaction (types I–IV, indexed by
  a `space`+`time` pair), the directed `sar` and its time-varying
  `dynamicspatialpanel` (SDPD, indexed by a `unit`+`time` pair with per-period
  inline `graphs` or a `graph_files` mapping), and the mixed-frequency `midas`
  (smooth-lag) and `midas_parametric` (exp-Almon / Beta kernel) effects, indexed
  by their HF lag `columns`. **Every effect is now reachable from YAML** — no
  Python-only effects remain. See [spatial effects](spatial-effects.md) and
  [effects](effects.md#midas-smooth-lag-effect).
- **Sparse large-graph scaling (E-sparse)** — network and space-time models
  whose latent dimension exceeds the dense reference regime now fit past the
  dense guard through a sparse constrained-Gaussian solver, delivering posterior
  mean, marginal likelihood, estimated hyperparameters, and point predictions
  (A+B), and the full posterior-uncertainty surface at network scale (C):
  selected-inverse marginal, predictive, and linear-combination variances with
  constrained corrections, `predict`, sparse Sørbye-Rue scaling, augmented BYM2,
  and diagonal INLA grid integration — no new dependency, deterministic. The
  augmented BYM2 also **exposes its structured component `u*` as a separately
  reported latent**: `latent_marginals("region")` returns the `x`-marginals and
  `latent_marginals("region.structured")` the `u*` marginals. See
  [spatial effects](spatial-effects.md).
- **Survival likelihoods** — `WeibullSurv` (Weibull proportional hazards,
  shape `alpha` fixed or estimated) and `ExponentialSurv` (`alpha = 1`), with
  right-censoring (`event`) and left-truncation (`entry`) support, `alpha`
  estimation by empirical Bayes, and unobserved heterogeneity via an ordinary
  `IID` frailty term over individuals. See
  [likelihoods](likelihoods.md#survival-likelihoods) and the runnable
  [unemployment-duration example](https://github.com/Ardea00/pylgm/tree/main/examples/survival_duration).
- **Directed & dynamic network structure** — a directed `SAR` effect
  (`(I−ρW)ᵀ(I−ρW)` on a row-standardized, generally-asymmetric `W`) for
  economic influence relations that symmetric CAR discards, and its
  time-varying generalization `DynamicSpatialPanel` (contemporaneous `ρ`,
  temporal `γ`, spatio-temporal-diffusion `η` over a balanced `unit x time`
  grid, `T=1` reducing exactly to `SAR`). Both fit past the dense guard
  through the E-sparse solver, support forward forecasting
  (`forecast_dynamic_spatial_panel`) for future periods' networks, and both are
  declarable from YAML (`type: sar` / `type: dynamicspatialpanel`). See
  [spatial effects](spatial-effects.md)
  and [`examples/directed_network_sar`](https://github.com/Ardea00/pylgm/tree/main/examples/directed_network_sar).
- **Diagnostics** — an empirical-Bayes estimate that lands on the edge of its
  declared interval is now reported in
  `result.diagnostics["hyperparameters_at_bound"]` and warned about, because a
  pinned estimate means the bound (often a default derived from `initial`)
  rather than the data is setting the value.

## Breaking changes in 0.6

- `forecast_dynamic_spatial_panel` returns **`latent_mean` / `latent_variance`**
  instead of `mean` / `variance`. The values are unchanged; the old names read
  as a response-scale forecast, but they exclude the fixed effects, so
  comparing them with observations was wrong by the response's mean level. See
  [spatial effects](spatial-effects.md#dynamic-spatial-panel-sdpd).

## Next

Ordered roughly by expected value to users. Nothing here is committed to a date.

1. **Matérn / SPDE spatial fields** as an alternative to CAR neighbour graphs.
2. **Hybrid HF/LF nowcasting frontend** — a higher-level mixed-frequency
   nowcasting API that handles the HF→LF lag alignment (still the caller's job
   today). The underlying effects ship both in the Python API and as the
   declarative `midas` / `midas_parametric` YAML types.

## Deferred (not planned for the near term)

- Parameterized IR metadata and sparse production engines.
- Full MCMC/HMC inference — pyLGM is deterministic-approximation-first by design.

## How to influence this

Open an issue at
[github.com/Ardea00/pylgm/issues](https://github.com/Ardea00/pylgm/issues)
describing the model you're trying to fit. Real use cases reorder this list.

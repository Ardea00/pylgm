# Roadmap

pyLGM 0.7 is a bounded, correct-by-construction foundation. This page is the
honest map of what's shipped, what's next, and what's deliberately deferred.
For the precise semantics of each shipped feature, follow the links into the
[guide](index.md).

## Shipped in 0.7

New since 0.6 (the `research-tier` line, first released as `0.7.0rc1`):

- **Linear observations and exact aggregates** — `LinearObservation` (noisy
  aggregates, with a fixed or estimated `sigma`) and `LinearConstraint` (exact
  aggregates) on the predictor grid, for temporal disaggregation, benchmarking
  and nowcasting. Exact aggregates are data: they enter the log marginal
  likelihood as `log p(e | y)`, so empirical Bayes and INLA learn from them.
  `Joint.fit` takes them per outcome; `scale="log"` ties a log-scale predictor to
  aggregates on levels (Gauss-Newton relinearization to the exact mode); and
  `pylgm.operators` builds aggregation, difference and cumulation operators;
  `pylgm.index_numbers` converts chain-linked volumes to additive
  previous-year-price values (annual overlap). See
  [linear observations](linear-observations.md).
- **Sequential updates** — `result.update(new_rows)` conditions a fitted
  posterior on new observations with no refactorisation: exactly for an
  exact-Gaussian fit (identical to a refit), through a `k`-dimensional Laplace
  step for non-Gaussian likelihoods, and with the hyperparameters reweighted on
  the INLA grid for `hyperparameters="integrate"`. See
  [prediction](prediction.md#sequential-updates).
- **News decomposition** — `result.news(release, at=targets)` splits the
  revision a release causes, in every latent effect and in the linear
  predictor at chosen rows or aggregates, into one exact contribution per
  released row, per revised row (`revisions=`) and, for integrated fits, the
  hyperparameters. It also splits the variance drop sequentially, and it works
  on joint models (`Joint.fit(..., hold_out=...)` keeps future rows). See
  [prediction](prediction.md#news-decomposition).
- **Warm starts** — `fit(..., warm_start=previous)` starts a rolling or
  expanding backtest's fit at the previous window's hyperparameters and latent
  mode (by label), for `"optimize"` and `"integrate"`, pandas, Spark and
  `Joint`. See [empirical Bayes](empirical-bayes.md#rolling-and-expanding-windows).
- **Joint posterior draws** — `result.sample(n, rng)` for exact-Gaussian (dense
  and sparse), Laplace and integrated fits, every draw meeting the exact
  constraints; `pylgm.evaluation.crps_from_draws` scores nonlinear targets built
  from them, and scored predictions carry `crps` and `pit`. See
  [prediction](prediction.md#joint-posterior-draws).
- **Likelihoods** — zero-inflated counts (`ZeroInflated`: ZIP, ZINB, ZIB);
  negative binomial, gamma and zero-inflated Laplace fits now use the observed
  curvature at the mode. See [likelihoods](likelihoods.md).
- **Inference** — `mean_correction=True` moves a Laplace mean from the mode
  toward the posterior mean (R-INLA's default VB correction); INLA explores the
  hyperparameter grid by density and switches to CCD or a Korobov lattice past a
  handful of hyperparameters; simulation-based calibration in `pylgm.validation`.
- **Sparse Laplace engine** — non-Gaussian fits above ~1 000 latents run on the
  partitioned sparse solver (Newton warm-started across the INLA grid and EB
  search), including confounded intrinsic effects (Besag + RW1, BYM2 + RW1,
  Knorr-Held with main effects), constraints coupling blocks, and
  simplified-Laplace marginals; predictive variances come from the selected
  inverse. See [INLA integration](inla.md#large-models).
- **Parallel evaluations and EB at scale** — `num_workers`/`blas_threads` fan an
  EB search's or INLA grid's independent conditional fits out over threads
  (results identical to `num_workers=1`); the EB search keeps memory flat, uses
  a central-difference gradient and stops on a plateau (`objective_tolerance`,
  `stall_iterations`). See [empirical Bayes](empirical-bayes.md#parallel-evaluations).
- **Effects** — `RW1`/`RW2` take `scale=True` (Sørbye-Rue, R-INLA's
  `scale.model`), and `RW1Structure`/`RW2Structure` take the same flag with the
  same default; `Fixed(prior_precision=Hyperparameter(...))` learns the ridge on
  the coefficients; `Replicated(Grouped(...))` is R-INLA's `group` + `replicate`
  on one term.

## Shipped through 0.6

- **Likelihoods** — Gaussian (exact engine), Poisson, Bernoulli, Binomial
  (counts `n·p` with a per-row trials column), NegativeBinomial (overdispersed
  counts), Gamma (positive-continuous), and Beta (proportions in `(0,1)`), all
  on the Laplace engine, with fixed or estimated dispersion `φ`. See
  [likelihoods](likelihoods.md).
- **Effects** — `Fixed`, `IID`, `RW1`/`RW2`, stationary `AR1` (optionally
  **replicated**: one independent series per panel unit, sharing `ρ` and
  `precision`), the drifting `Seasonal` pattern, the `MIDAS`
  mixed-frequency smooth-lag effect and its restricted (parametric)
  `MIDASParametric` counterpart (exp-Almon / Beta lag kernels), and the
  Knorr-Held `SpaceTime` interaction (Types I–IV). See [effects](effects.md).
- **Effect modifiers** — R-INLA expresses `weights`, `copy`, `replicate` and
  `group` as arguments of one `f()` call; pyLGM ships them as composable
  wrappers, so any indexed effect gains them without each effect
  reimplementing four validation paths. `Weighted(effect, by)` scales the
  design by a column (spatially-varying coefficients); `Copy(name, index,
  scale)` lets one latent field enter a predictor twice at different indices,
  optionally under an estimated scale; `Replicated(effect, over)` gives `R`
  independent copies sharing every hyperparameter (`I_R ⊗ Q`); and
  `Grouped(effect, over, structure)` gives `G` *correlated* copies tied by a
  between-group precision (`Q_S ⊗ Q_E`), with `IIDStructure`, `AR1Structure`,
  `RW1Structure`/`RW2Structure` and `BesagStructure` available for `Q_S`. The
  three that map onto Knorr-Held's interaction types reproduce `SpaceTime`
  I–IV, which is how the composition is checked. Each modifier still produces
  exactly one latent block, so inference is untouched. See
  [effects](effects.md#weighted-effects), and
  [research status](research-status.md) for what is and is not verified.

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

> **Joint models are research-grade and live on the `research-tier` branch, not
> `main`.** They are tested and reviewed, but validated only internally and
> against MCMC on *simulated* data -- no published result on real data has been
> reproduced, and `latent_strategy="laplace"` is known to degrade on them. See
> [research status](research-status.md) for exactly what is and is not
> established.

- **Joint multi-likelihood models** — `Joint` stacks several `LGM` sub-models
  (each with its own response, likelihood, offset, and predictor) into one
  fit, with `Shared` letting one latent field enter more than one sub-model
  under a per-sub-model scaling, including the Knorr-Held & Best `(delta,
  delta⁻¹)` shared-component pairing for exactly two sub-models. Fits by
  `engine="laplace"` only; per-outcome prediction via
  `result.predict(new_data, outcome=...)`. This covers the *scaling* half of
  R-INLA's `copy` — a shared field entering another sub-model under an
  estimated multiplicative scale — but the copied field's own hyperparameters
  (precision, rho, phi, ...) must stay fixed, unlike `copy`, which estimates
  those too. It also does **not** cover off-block-diagonal precision coupling
  (coregionalization) — see [joint models](joint-models.md#not-supported-yet).
  `copy` and `replicate` *within* a single sub-model shipped separately, as the
  `Copy` and `Replicated` modifiers above.

## Breaking changes in 0.6

- `forecast_dynamic_spatial_panel` returns **`latent_mean` / `latent_variance`**
  instead of `mean` / `variance`. The values are unchanged; the old names read
  as a response-scale forecast, but they exclude the fixed effects, so
  comparing them with observations was wrong by the response's mean level. See
  [spatial effects](spatial-effects.md#dynamic-spatial-panel-sdpd).

## Next

Ordered roughly by expected value to users. Nothing here is committed to a date.

1. **Matérn / SPDE spatial fields** as an alternative to CAR neighbour graphs.

## Deferred (not planned for the near term)

- Parameterized IR metadata and sparse production engines.
- Full MCMC/HMC inference — pyLGM is deterministic-approximation-first by design.

## How to influence this

Open an issue at
[github.com/Ardea00/pylgm/issues](https://github.com/Ardea00/pylgm/issues)
describing the model you're trying to fit. Real use cases reorder this list.

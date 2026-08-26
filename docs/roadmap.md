# Roadmap

pyLGM 0.3 is a bounded, correct-by-construction foundation. This page is the
honest map of what's shipped, what's next, and what's deliberately deferred.
For the precise semantics of each shipped feature, follow the links into the
[guide](index.md).

## Shipped in 0.3

- **Likelihoods** — Gaussian (exact engine), Poisson and Bernoulli (Laplace
  engine). See [likelihoods](likelihoods.md).
- **Effects** — `Fixed`, `IID`, `RW1`/`RW2`, stationary `AR1`, and the
  `MIDAS` mixed-frequency smooth-lag effect. See [effects](effects.md).
- **Spatial (CAR) family** — `Besag` (ICAR), `ProperCAR` (with `ρ` fixed or
  estimated), and `BYM2` (with `φ` fixed or estimated), complete for the dense
  reference regime. Graphs may be **weighted** (`{node: {neighbour: weight}}`),
  so the same family models firm-ownership / interbank-exposure / supply-chain
  networks, not only geographic adjacency. See
  [spatial effects](spatial-effects.md).
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
  `pylgm.config.load_model`.

## Next

Ordered roughly by expected value to users. Nothing here is committed to a date.

1. **Config-file spatial effects** — declare `besag`/`proper_car`/`bym2` (with
   an inline or file graph) in the YAML frontend, not just the Python API.
2. **Sparse / large-graph scaling** — sparse precision assembly and solves so
   spatial models scale beyond the dense reference regime.
3. **Graceful isolated-node handling** in graphs, and the augmented 2n CAR
   representation.
4. **Additional likelihoods** — e.g. Binomial and negative-Binomial counts.
5. **Matérn / SPDE spatial fields** as an alternative to CAR neighbour graphs.
6. **Directed & dynamic network structure** — a SAR effect
   (`(I−ρW)ᵀ(I−ρW)`) for directed economic influence that symmetrized CAR
   discards, and time-varying `W_t` — building on the weighted-graph support
   above.
7. **Parametric MIDAS lag kernels** — exp-Almon / Beta weight functions and a
   hybrid HF/LF nowcasting frontend, extending the shipped `MIDAS` smooth-lag
   effect; a config-file `midas` effect type.

## Deferred (not planned for the near term)

- Parameterized IR metadata and sparse production engines.
- Full MCMC/HMC inference — pyLGM is deterministic-approximation-first by design.

## How to influence this

Open an issue at
[github.com/Ardea00/pylgm/issues](https://github.com/Ardea00/pylgm/issues)
describing the model you're trying to fit. Real use cases reorder this list.

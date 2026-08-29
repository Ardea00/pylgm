# Directed interbank-exposure network (SAR)

This directory fits a seeded 30-bank interbank-exposure network through a
directed spatial-autoregressive (`SAR`) latent effect. Run it from the
repository root:

```bash
PYTHONPATH=src python examples/directed_network_sar/run.py
```

## Reading a directed `W`

`Besag`/`ProperCAR`/`BYM2` require a **symmetric** neighbour graph — "a and b
are neighbours" is a statement about the pair, not a direction. Real economic
exposure is rarely symmetric: bank `a` being exposed to `b` (`a` holds `b`'s
debt) does not imply `b` is exposed to `a`. `SAR` accepts a **directed**
graph instead: `graph[bank]` lists that bank's own counterparties, i.e. *row
`i` = who `i` depends on*, not who depends on `i`.

```python
graph = {
    "bank_0": ["bank_7", "bank_12", "bank_3"],  # bank_0 is exposed to these three
    "bank_1": ["bank_5", "bank_9", "bank_14"],
    ...
}
```

Each row is **row-standardized** (rescaled to sum to 1) before use, so a
bank's stress is the *weighted average* of its counterparties' stress,
regardless of how many counterparties it has. The latent precision is then
built from the directed operator `M = I - rho*W`:

```
Q = precision * MᵀM = precision * (I - rho*W)ᵀ(I - rho*W)
```

which is symmetric and positive-definite for any `rho` in `(-1, 1)` even
though `W` itself is not symmetric — the asymmetry lives in `W`, not in the
resulting precision.

## `rho` as contagion strength

`rho` measures how much of a bank's latent value is explained by its
counterparties versus its own idiosyncratic shock: `rho -> 0` is independent
banks, `rho -> 1` is near-total pass-through of counterparty stress. This
example simulates a true `rho = 0.6` and declares it as an estimated
`Hyperparameter` (`transform="logit"`, since `rho` is bounded), fit by
empirical Bayes alongside the effect's `precision`.

## Time-varying extension

This example is a single cross-section (one snapshot of the network). When
the network itself evolves — counterparty relationships change period to
period, or stress persists and diffuses through time as well as space — the
time-varying generalization is `DynamicSpatialPanel`: it adds a temporal
persistence coefficient `gamma` and a spatio-temporal diffusion coefficient
`eta` on top of `rho`, over a balanced `unit x period` grid of per-period
directed networks `W_t`. Once fit, `forecast_dynamic_spatial_panel` propagates
the fitted last-period latent state forward through future periods' networks.
See [`docs/spatial-effects.md`](../../docs/spatial-effects.md) for the full
`(rho, gamma, eta)` contract.

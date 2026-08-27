# Hybrid nowcast — MIDAS + BYM2 + AR1 in one latent field

This example proves the core composition claim: `+` sums a high-frequency
mixed-frequency term, a spatial-network term, and a temporal term into a
single latent Gaussian field that fits and predicts.

## The panel

`build()` generates a deterministic regional-GDP nowcast keyed by
`(region, quarter)`: `R = 8` regions on a ring graph, `T = 40` quarters, and a
monthly high-frequency indicator aligned into `K = 12` lag columns
`ind_lag0..ind_lag11` (via `pandas.Series.shift`, the same alignment
`examples/midas_nowcast` uses). The target is

```
y[r,t] = mu + sum_k w_k * x[r,t,k] + s[r] + a[t] + noise
```

with a known exp-Almon lag kernel `w_k`, a spatial signal `s[r]` that varies
smoothly over the ring, and an AR1 temporal signal `a[t]`.

## The two composed models

Both share the same spatial + temporal common term:

```python
common = (
    BYM2("region", index="region", graph=graph, precision=..., phi=0.5)
    + AR1("quarter", index="quarter", precision=..., rho=0.6)
)
```

- **A — U-MIDAS:** `Fixed("1") + MIDAS("lag", columns=cols, precision=...) + common`.
  Spends a smoothed coefficient per lag; recovers the kernel as
  `result.latent_marginals("lag").mean`.
- **B — parametric MIDAS:** `Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon") + common`.
  Spends only two shape parameters; recovers the kernel as
  `midas_weights("exp_almon", K, theta_hat)` from the fitted
  `result.hyperparameters`.

Both recover the true kernel and track the target (`corr > 0.98` on the fixed
seed). See the [effects guide](../../docs/effects.md).

## Fixed mixing parameters

`phi` (BYM2) and `rho` (AR1) are passed as fixed floats to keep the example
fast and deterministic — both are weakly identified on a panel this short.
Both accept a `Hyperparameter(...)` instead to estimate them; that is a
one-line change per term.

## Run

```
PYTHONPATH=src python examples/hybrid_nowcast/run.py
```

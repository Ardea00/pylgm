# A network that changes every year: US state income (SDPD)

Most spatial models fix the network once. `DynamicSpatialPanel` takes **one
network per period**, so the graph itself is a time series.

```bash
PYTHONPATH=src python examples/state_income_dynamic_network/run.py   # ~40s
```

48 US states, 1997–2007. The topology is real geography (state contiguity); the
edge *weights* are economic — neighbouring states link more strongly when their
income levels are close. The weight for year `t` is built from year `t−1`
income, so the network never sees the value being modelled.

## What it shows

**Panels have holes.** The example knocks out 20% of the cells and asks each
method to put them back:

```
Recovering 102 knocked-out panel cells -- RMSE (log income):
     SDPD (dynamic network)    0.0246
     state mean                0.1214
     year mean                 0.1484
     grand mean                0.1786
```

Roughly **5× closer** than the obvious baselines, because a missing cell is
reconstructed from both its own history and its neighbours' current values.

**The coefficients separate what a static model conflates:**

```
  rho   (neighbour spillover, same year) = +0.6393
  gamma (own persistence, one year back) = +1.0052
  eta   (neighbour spillover, lagged)    = -0.5960
```

`γ ≈ 1` is a near-unit root — log income is close to a random walk. `ρ > 0` with
`η < 0` is the familiar spatial error-correction shape: strong contemporaneous
co-movement with neighbours, partly unwound the following year.

## Where it does *not* win

The script also forecasts 2008–09 and reports the honest answer:

```
  year          SDPD  last value
  2008        0.0397      0.0291
  2009        0.1242      0.0255
```

A last-value forecast beats it. That follows directly from `γ ≈ 1`: for a random
walk the last observed value *is* the optimal forecast, and no structure beats
it at its own game. The structure pays off on the gaps, not here — and a page
that hid this would be worth less than one that shows it.

## Note on the forecast helper

`forecast_dynamic_spatial_panel` returns the marginals of the **latent field**,
not the response scale. Add the fixed-effect contribution before comparing with
observations:

```python
beta = dict(zip(result.labels, result.mean))
forecast["predicted"] = forecast["mean"] + beta["fixed:Intercept"]
```

## Cost

SDPD is expensive: the block-bidiagonal operator is rebuilt at every
hyperparameter evaluation, and cost grows superlinearly in `units × periods`
(481 cells ≈ 6s, 961 cells ≈ 72s with two estimated coefficients). Fix
coefficients you do not need estimated.

## Data

Per-capita income by US state and the 48-state first-order contiguity graph,
redistributed with PySAL/libpysal (BSD-3-Clause); original source is the US
Bureau of Economic Analysis.

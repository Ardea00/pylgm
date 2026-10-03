# Network reconstruction from a censored credit register

A seeded mask-and-reconstruct study: 1500 firms x 15 banks (22 500 candidate
edges). Links follow `expit(a0 + firm + bank effects)`, log amounts
`Normal(b0 + firm + bank effects, sigma)`; the threshold `c` is set so about
35% of existing links fall below it. A link at or above `c` is *reported*; 30%
of non-links are *known non-links* (a stand-in for information identifying the
link intercept); every other edge is *censored*. Each firm's total
below-threshold debt is observed with ~2% multiplicative noise.

```bash
PYTHONPATH=src python examples/network_reconstruction/run.py
```

Only the censored edges are scored (truth = does the edge exist below `c`, and
its amount): reported and known edges are data, not reconstruction targets.

- **pyLGM**: `Joint` + `CensoredHurdle` with the margins as a
  `LinearObservation(scale="below_threshold")`; sigma is estimated.
- **RAS**: maximum-entropy/proportional baseline, each firm's margin spread
  over its censored edges by bank size (Upper 2011; Mastromatteo et al. 2012).
- **dcGM**: density-corrected gravity model (Cimini et al. 2015, *Systemic
  risk analysis on reconstructed economic and financial networks*). Its density
  parameter `z` is calibrated on the **true** number of below-threshold links,
  an oracle advantage pyLGM does not get.

```
                         auc  precision_at_k  weighted_cosine  weighted_jaccard  fit_seconds
pylgm                  0.812           0.594            0.643             0.274       28.687
ras                    0.462           0.253            0.537             0.243        0.004
dcgm (oracle density)  0.738           0.483            0.570             0.252        0.004
```

The run takes about 30 s, almost all of it the pyLGM fit (about 8 s with sigma fixed).

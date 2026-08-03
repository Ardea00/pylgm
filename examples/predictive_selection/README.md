# Predictive selection example

This committed, domain-neutral panel has three regions over ten periods. It
compares a region-only Gaussian model with a region-plus-RW1-trend model at
one- and two-period horizons. Both candidates use empirical-Bayes optimization
within the declared bounds; `persistence` is automatically reported as a
non-selectable autoregressive benchmark.

Run it locally:

```bash
pylgm compare examples/predictive_selection/config.yaml \
  examples/predictive_selection/data.csv \
  --output comparison
```

The output contains the selected model, fold-level predictions, aggregate
metrics, and schema-v2 artifacts. The exact Gaussian engine is intended for
small/medium latent dimensions; it conditions on empirical-Bayes hyperparameter
estimates and does not integrate hyperparameter uncertainty.

# NIC-shaped backtest configuration

This is a configuration-only example for a regional monthly inflation panel.
It uses only the generic data roles `date`, `region_code`, `coicop`, and
`yoy_pct`, so it can run directly against a local NIC Parquet file without any
data transformation. The candidates separately test region-plus-time,
COICOP-plus-time, and combined region/COICOP/time effects. `persistence` is
included automatically as the non-selectable autoregressive benchmark.
It uses a 36-month rolling estimation window to keep the small/medium exact
Gaussian reference workload practical while adapting to recent inflation
regimes; the CLI still reads the full source Parquet directly.

Run it locally with a data file you are entitled to use:

```bash
pylgm compare examples/nic_backtest/config.yaml \
  /path/to/nic_regionale_mensile.parquet \
  --output nic-comparison
```

The NIC source data is not redistributed or committed by this repository. The
comparison uses empirical Bayes only: hyperparameter uncertainty is not
integrated. pyLGM's exact Gaussian engine is a small/medium latent-dimension
reference implementation, so use this configuration locally only where its
dimension and memory guards accept the data.

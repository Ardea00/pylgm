# Runnable declarative LGM example

This directory fits `data.csv` through both supported 0.3 declarative
frontends: a Python `LGM` declaration and the standalone `config.yaml` model.
Run it from the repository root:

```bash
PYTHONPATH=src python examples/general_lgm/run.py
```

The YAML frontend does not use the legacy run or experiment schemas. Its
equivalent loading call is:

```python
from pathlib import Path

from pylgm.config import load_model

model = load_model(Path("config.yaml"))
result = model.fit(frame)
```

The supported YAML vocabulary in 0.3 is a Gaussian likelihood, one fixed
formula, and IID, RW1, or RW2 structured effects. `sigma` and every structured
effect's `precision` are required finite positive numeric plug-in values.
AR1, new-data prediction, and hyperparameter-prior inference are not
part of this runnable example.

## Optional PySpark frontend

With the `pylgm[spark]` extra and Java installed, the same YAML model fits a
Spark DataFrame. Spark validates and canonically orders the data; the fit runs
on the driver. Run it from the repository root:

```bash
PYTHONPATH=src python examples/general_lgm/run_spark.py
```

It prints `result.prediction_keys` (canonical `(*panel, time)` rows) alongside
`result.predictive_mean`. Spark predictions are in canonical key order rather
than caller order, so the keys are how you join predictions back to source rows.

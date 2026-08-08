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
PySpark, AR1, new-data prediction, and hyperparameter-prior inference are not
part of this runnable example.

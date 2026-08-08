# Standalone declarative LGM YAML

`config.yaml` defines a Gaussian latent Gaussian model without using the legacy
run or experiment configuration schemas. Load it into the same public `LGM`
declaration used by the Python frontend, then fit a Pandas data frame:

```python
from pathlib import Path

from pylgm.config import load_model

model = load_model(Path("config.yaml"))
result = model.fit(frame)
```

The supported YAML vocabulary in this milestone is a Gaussian likelihood, one
fixed formula, and IID, RW1, or RW2 structured effects. `sigma` and every
structured effect's `precision` are explicit, finite positive values.

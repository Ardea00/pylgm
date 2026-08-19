# Runnable Poisson count example (Laplace)

This directory fits `data.csv` — a small two-region count panel with a log
exposure offset — through the standalone `config.yaml` model and the
`engine="laplace"` inference path. Run it from the repository root:

```bash
PYTHONPATH=src python examples/count_glm/run.py
```

The YAML declares a Poisson likelihood (canonical log link), a top-level
`offset: logexp` column, a fixed `1 + x` formula, and one IID `region_effect`.
Its equivalent loading call is:

```python
from pathlib import Path

from pylgm.config import load_model

model = load_model(Path("config.yaml"))
result = model.fit(frame, engine="laplace")
```

`family: poisson` and `family: bernoulli` forbid a `sigma` key — that field
only applies to `family: gaussian`. `result.fitted_mean` is the Poisson
response-scale prediction (exact under the log link: `exp(mean + variance/2)`
of the linear predictor). `result.diagnostics["newton_iterations"]` reports
how many Newton steps the Laplace mode-finding took. See the
["Non-Gaussian likelihoods (Laplace)" section](../../README.md) in the
project README for the full engine documentation, including its fixed
plug-in hyperparameters and the Bernoulli point-estimate caveat.

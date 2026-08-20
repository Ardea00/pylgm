# Runnable INLA integration example (exact Gaussian)

This directory fits `data.csv` — a twenty-four-region, six-timepoint Gaussian
panel with a genuine per-region offset — through a Python `LGM` declaration
that gives the region `IID` effect a declared `Hyperparameter` precision, and
resolves it by **INLA-style grid integration** instead of plugging in a
single optimum. Run it from the repository root:

```bash
PYTHONPATH=src python examples/inla/run.py
```

The model is:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM

model = LGM(
    response="y",
    likelihood=Gaussian(0.5),
    predictor=Fixed("1")
    + IID("region", index="region", precision=Hyperparameter("region_precision", initial=1.0)),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
```

Passing `hyperparameters="integrate"` (instead of the default `"optimize"`)
runs quadrature over a grid of `region_precision` values around the
empirical-Bayes mode, in log space, weighted by the marginal likelihood and
the log-space Jacobian. The result carries:

- `result.hyperparameter_marginals()["region_precision"]` — mean and sd of
  the *integrated* hyperparameter posterior, not a single point estimate;
- `result.latent_marginals("region")` — latent region-effect marginals
  averaged over that hyperparameter grid, so they are **wider** than the
  plug-in (`hyperparameters="optimize"`) fit because they also carry
  hyperparameter uncertainty;
- `result.diagnostics["inla_grid_points"]` — how many grid points were kept
  after the density-drop filter (this example keeps several, confirming the
  posterior is genuinely integrated rather than collapsed to one point).

## A limitation worth knowing

Grid quadrature assumes the hyperparameter posterior is reasonably interior
and near-Gaussian in log space. This example is built that way on purpose:
24 regions with several observations each give the region precision a real
signal, so its posterior mode sits away from the declared `[lower, upper]`
bounds. A poorly identified model (too few groups, or a pure-noise panel
with no real per-region effect) instead drives the estimate toward a bound,
the grid degenerates toward a single point, and the integration quietly
reduces to the plug-in fit. `result.diagnostics["inla_active_bounds"]`
reports which hyperparameters (if any) hit a bound at the mode — empty here.

See the ["INLA integration" section](../../README.md#inla-integration) of
the project README for the full contract, including what is still out of
scope (model-assessment criteria such as DIC/WAIC/CPO/PIT, and richer latent
strategies beyond the Gaussian approximation).

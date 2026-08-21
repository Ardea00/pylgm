# Runnable INLA simplified-Laplace latent-marginals example

This directory fits `data.csv` — a twenty-two-region, six-timepoint Poisson
panel with a genuine per-region log-rate effect — through a Python `LGM`
declaration that gives the region `IID` effect a declared `Hyperparameter`
precision, resolves it by INLA-style grid integration
(`hyperparameters="integrate"`), and requests **skew-aware latent
marginals** via `latent_strategy="simplified_laplace"`. Run it from the
repository root:

```bash
PYTHONPATH=src python examples/inla_sla/run.py
```

The model is:

```python
from pylgm import Fixed, Hyperparameter, IID, LGM, Poisson

model = LGM(
    response="y",
    likelihood=Poisson(),
    predictor=Fixed("1")
    + IID("region", index="region", precision=Hyperparameter("region_precision", initial=1.0)),
    panel=("region",),
    time="time",
)
sla = model.fit(
    frame, engine="laplace", hyperparameters="integrate",
    latent_strategy="simplified_laplace",
)
region = sla.latent_marginals("region")  # a SkewNormalMarginals
```

`region.mean`, `region.std`, and `region.skewness` are per-region summaries
of a grid-mixture-of-skew-normals posterior; `region.quantile(0.025)` and
`region.quantile(0.975)` give a (generally asymmetric) 95% interval per
region. The script contrasts that interval with the one built from
`latent_strategy="gaussian"` (the default) — `mean +/- 1.96 * std` — which is
symmetric by construction. For a Poisson likelihood with non-trivial counts
the two intervals are close but not identical: the simplified-Laplace
correction shifts probability mass in the direction the Laplace
approximation's third-derivative correction implies (see the ["Simplified-Laplace
latent marginals"](../../README.md#simplified-laplace-latent-marginals)
section of the project README for the full contract).

## What to look for in the output

```
sla mean[:3]: [...]
sla std[:3]: [...]
sla skewness[:3]: [...]
sla 95% interval[:3]: lo=[...] hi=[...]
gaussian 95% interval[:3]: lo=[...] hi=[...]
```

The reported `skewness` values are non-zero (small but genuine, since this
is a mildly informative Poisson panel), and the `sla` interval is not
symmetric about `sla mean` the way the `gaussian` interval is symmetric
about `gaussian mean` — that asymmetry is exactly what the simplified-Laplace
strategy adds over the Gaussian one.

See the ["INLA integration"](../../README.md#inla-integration) section of
the project README for the grid-quadrature contract this example builds on,
and ["Simplified-Laplace latent marginals"](../../README.md#simplified-laplace-latent-marginals)
for what the skew correction does and does not cover (it is the
*simplified*, not full, Laplace correction, and is validated against a
brute-force true-marginal oracle rather than R-INLA).

# Runnable INLA full-Laplace latent-marginals example

This directory fits `data.csv` — the same twenty-two-region, six-timepoint
Poisson panel used by [`examples/inla_sla`](../inla_sla/README.md) — through
a Python `LGM` declaration that gives the region `IID` effect a declared
`Hyperparameter` precision, resolves it by INLA-style grid integration
(`hyperparameters="integrate"`), and requests the **most accurate latent
strategy**, `latent_strategy="laplace"` (INLA's full-Laplace correction).
Run it from the repository root:

```bash
PYTHONPATH=src python examples/inla_full_laplace/run.py
```

The model is:

```python
from pylgm import Fixed, Hyperparameter, IID, LGM, Poisson

model = LGM(
    response="y",
    likelihood=Poisson(),
    # Fixed + IID only -- no RW effect. Full Laplace rejects constrained
    # (intrinsic/RW) effects; see "Scope" below.
    predictor=Fixed("1")
    + IID("region", index="region", precision=Hyperparameter("region_precision", initial=1.0)),
    panel=("region",),
    time="time",
)
full = model.fit(
    frame, engine="laplace", hyperparameters="integrate",
    latent_strategy="laplace",
)
region = full.latent_marginals("region")  # a TabulatedMarginals
```

`region.mean`, `region.std`, and `region.skewness` are per-region summaries
of a numerically tabulated posterior density (the grid-mixture of the
full-Laplace-corrected per-hyperparameter-grid-point marginals);
`region.quantile(0.025)` and `region.quantile(0.975)` give a (generally
asymmetric) 95% interval per region, and `region.pdf`/`region.cdf` evaluate
the tabulated density directly. The script contrasts the full-Laplace
skewness and 95% interval with the `"gaussian"` (default, symmetric by
construction) and `"simplified_laplace"` strategies on the *same* fit, so
you can see the three-tier ladder move in the same direction with increasing
magnitude.

## What to look for in the output

```
full-laplace mean[:3]: [...]
full-laplace std[:3]: [...]
full-laplace skewness[:3]: [...]
full-laplace quantile(0.025)[:3]: [...]
full-laplace quantile(0.975)[:3]: [...]

strategy       skewness[:3]                  95% interval[:3]
gaussian       [0.0, 0.0, 0.0]   lo=[...] hi=[...]
simplified-LA  [...]   lo=[...] hi=[...]
full-laplace   [...]   lo=[...] hi=[...]
```

The `gaussian` row is symmetric about its mean by construction (skewness
reported as zero); `simplified-LA` and `full-laplace` are both skewed, and
the full-Laplace magnitude is typically larger than the simplified-Laplace
one for the same component, since full Laplace additionally re-fits the
denominator per latent component (RMC 2009 §3.2.2) instead of only
correcting the numerator (§3.2.3, what the simplified strategy applies).

See the ["INLA integration"](../../README.md#inla-integration) section of
the project README for the grid-quadrature contract this example builds on,
["Simplified-Laplace latent marginals"](../../README.md#simplified-laplace-latent-marginals)
for the intermediate skew-correction tier, and
["Full-Laplace latent marginals"](../../README.md#full-laplace-latent-marginals)
for the full contract this example exercises — including its scope
(unconstrained models only; the `IID` effect above has no `RW` sibling for
that reason) and validation status (a brute-force true-marginal oracle, not
R-INLA parity).

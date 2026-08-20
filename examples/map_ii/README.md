# Runnable penalized MAP-II example (exact Gaussian)

This directory fits `data.csv` — the same four-region, twenty-timepoint
Gaussian panel used by [`examples/empirical_bayes`](../empirical_bayes/README.md)
— through a Python `LGM` declaration that gives the region `IID` effect a
`Hyperparameter` precision carrying a `PCPrecision` prior. Run it from the
repository root:

```bash
PYTHONPATH=src python examples/map_ii/run.py
```

The model is:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, PCPrecision

model = LGM(
    response="y",
    likelihood=Gaussian(0.5),
    predictor=Fixed("1")
    + IID(
        "region",
        index="region",
        precision=Hyperparameter(
            "region_precision", initial=1.0,
            prior=PCPrecision(upper_sd=1.0, alpha=0.01),
        ),
    ),
    panel=("region",),
    time="time",
)
result = model.fit(frame, engine="exact_gaussian")
```

Declaring `region_precision` as a `Hyperparameter` with a `prior` makes
`LGM.fit` estimate it by **MAP-II**: the exact-Gaussian marginal likelihood
plus the PC prior's log density evaluated on the precision's native scale
(no Jacobian correction), optimized starting from `initial` and respecting
any declared `lower`/`upper` bounds. A prior-free `Hyperparameter` would
instead stay pure type-II ML (see the empirical-Bayes example). This is
still a point estimate, not a full Bayesian marginal over the
hyperparameter — see the ["Empirical Bayes" section](../../README.md) of the
project README for the full contract, including the Laplace-engine case and
what remains out of scope (full hyperparameter integration/marginals is the
later INLA slice).

The fitted precision is exposed on `result.hyperparameters["region_precision"]`
and the penalization flag on `result.diagnostics["hyperparameter_penalized"]`,
which is `True` whenever any declared `Hyperparameter` carries a `prior`.
Note that the reported `PCPrecision` estimate is the precision-scale posterior
mode, which differs from a standard-deviation-scale mode because the PC prior's
σ→τ change of variables is built into its density.

# Runnable declarative empirical-Bayes example (exact Gaussian)

This directory fits `data.csv` — a four-region, twenty-timepoint Gaussian
panel — through a Python `LGM` declaration that gives the region `IID`
effect a declared `Hyperparameter` precision instead of a fixed number. Run
it from the repository root:

```bash
PYTHONPATH=src python examples/empirical_bayes/run.py
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
result = model.fit(frame, engine="exact_gaussian")
```

Declaring `region_precision` as a `Hyperparameter` with no `prior` (instead
of passing a plain number) makes `LGM.fit` estimate it by **type-II maximum
likelihood**: it optimizes the exact-Gaussian marginal likelihood over the
hyperparameter, starting from `initial` and respecting any declared
`lower`/`upper` bounds. This is an unpenalized point estimate, not a full
Bayesian marginal over the hyperparameter — see the
["Empirical Bayes" section](../../README.md) of the project README for the
full contract, including the Laplace-engine case, the penalized MAP-II case
(see [`examples/map_ii`](../map_ii/README.md) for a `Hyperparameter` declared
with a `prior`), and what is still out of scope (YAML declaration and
hyperparameter marginals/INLA).

The fitted precision is exposed on `result.hyperparameters["region_precision"]`
and the optimizer's convergence flag on
`result.diagnostics["empirical_bayes_converged"]`.

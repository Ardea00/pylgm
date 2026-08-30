# Method comparison: pyLGM vs GLM vs gradient boosting vs Monte Carlo

Three deliberately different problems, each simulated so the **true** latent
surface is known and can be scored directly.

| Problem | What it shows |
|---|---|
| **A.** 200 areas, 3 Poisson counts each, smooth spatial signal | pyLGM's partial pooling beats both unpooled dummies and XGBoost (~3× lower RMSE), with 99% interval coverage |
| **B.** 4000 rows, response from products/thresholds of 3 covariates | XGBoost beats a linear predictor by ~10× — the model class does not contain the interaction |
| **C.** Small Gaussian model with a closed-form posterior | One exact solve vs random-walk Metropolis; MC error falls as 1/√S |

## Run

```bash
pip install scikit-learn xgboost     # NOT pyLGM dependencies
PYTHONPATH=src python examples/method_comparison/run.py
```

## Expected output

```
A. Small-area spatial counts -- RMSE against the TRUE latent field
     pyLGM (Besag)            0.1345
     GLM (area dummies)       0.4288
     XGBoost                  0.3708
     pyLGM 95% interval coverage: 0.99

B. Nonlinear covariate interactions -- RMSE against the TRUE signal
     pyLGM (linear predictor) 2.8391
     XGBoost                  0.2934

C. Deterministic vs Monte Carlo -- distance from the EXACT posterior mean
     pyLGM (one exact solve)  0.0000
     Metropolis,   1,000 draws 0.2066
     Metropolis,  10,000 draws 0.0638
     Metropolis, 100,000 draws 0.0161
```

Discussion of what these mean: [How pyLGM compares](../../docs/comparison.md).

## A note on node ordering

Graph nodes are sorted **as strings**, so areas are labelled `0000`, `0001`, …
rather than `0`, `1`, …. With bare `str(i)` the sorted order would be
`0, 1, 10, 100, …` and the latent field would silently misalign against the
simulated truth — which produced a wrong, pyLGM-looks-bad result the first time
this benchmark was written.

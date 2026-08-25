# Count regression: horseshoe crabs

Poisson regression on genuine over-dispersed count data (Agresti/Brockmann
horseshoe crabs, 173 females), benchmarked against a standard `statsmodels`
Poisson GLM.

```bash
PYTHONPATH=src python examples/count_regression/run.py
```

Shows that pyLGM's plain Poisson fit matches statsmodels to three decimals,
that adding an estimated IID overdispersion effect widens the coefficient CI
honestly (the counts are over-dispersed), and that out-of-sample prediction is
on par with the GLM. Writes `docs/img/crabs_fit.png` and
`docs/img/crabs_prediction.png`, used on the
[count-regression docs page](../../docs/examples-count-regression.md).

`data.csv` is the Agresti categorical-data horseshoe-crab dataset
(`crab, sat, y, weight, width, color, spine`).

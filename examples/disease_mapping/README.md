# Disease mapping: Scotland lip cancer

Poisson disease mapping with a Besag (ICAR) spatial effect on the canonical
GeoBUGS Scotland lip-cancer data (56 districts), benchmarked against a standard
non-spatial `statsmodels` Poisson GLM with the same offset and covariate.

```bash
PYTHONPATH=src python examples/disease_mapping/run.py
```

Shows that the spatial random effect shrinks noisy raw rates toward local means
and tracks observed counts far better than the covariate-only GLM
(r ≈ 0.96 vs 0.63), while the agriculture/fishing/forestry covariate stays
positive. Writes `docs/img/scotland_shrinkage.png` and
`docs/img/scotland_fit.png`, used on the
[disease-mapping docs page](../../docs/examples-disease-mapping.md).

`data.csv` has `area, observed, expected, aff`; `graph.json` is the district
adjacency graph (neighbour lists keyed by area id).

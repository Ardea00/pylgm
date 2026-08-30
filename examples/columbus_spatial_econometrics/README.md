# Reproducing Anselin (1988): Columbus crime

The reference dataset of spatial econometrics. 49 contiguous Columbus, OH
neighbourhoods; residential burglaries and vehicle thefts per 1000 households
regressed on household income and housing value.

```bash
PYTHONPATH=src python examples/columbus_spatial_econometrics/run.py
```

## Why it matters

Anselin's point is that **OLS is biased here**: neighbourhoods are not
independent draws, and ignoring the spatial correlation in the residuals
overstates the income effect. pyLGM's `SAR` effect puts a spatial-autoregressive
field on the latent scale — `Q = τ(I − ρW)ᵀ(I − ρW)`, the spatial *error* model
structure — and should move the coefficient back toward the published
maximum-likelihood estimate.

## Expected output

```
model                           const      INC    HOVAL     rho
published OLS                  68.619  -1.5973  -0.2739    --
OLS (pyLGM repo)               68.619  -1.5973  -0.2739    --
published ML spatial error     60.279  -0.9573  -0.3046  0.5468
pyLGM SAR                      59.543  -0.9057  -0.3058  0.5946

Ignoring spatial correlation overstates the income effect by 1.76x (-1.597 -> -0.906).
```

The OLS row **matches the published values exactly**, which is what verifies the
data and specification. pyLGM's SAR then lands next to the published ML
spatial-error estimates: `HOVAL` −0.3058 vs −0.3046, the constant 59.5 vs 60.3,
and income −0.906 vs −0.957.

## On the remaining gap

pyLGM is not re-implementing ML estimation, so exact agreement is not expected
and would be suspicious. Two differences account for the spread:

- pyLGM is **Bayesian**: the latent field is shrunk by its prior, and `ρ` is a
  posterior estimate rather than an ML maximiser.
- The model carries a **nugget** (`Gaussian(sigma=...)`) alongside the spatial
  field, so observation noise and spatial noise compete for the same variance.
  ML spatial error has no such term. Here the nugget is squeezed toward zero,
  which is why the estimates land close.

`ρ = 0.595` against `λ = 0.547` is the visible consequence. The economically
relevant conclusion — spatial dependence roughly halves the apparent income
effect — reproduces cleanly.

## Data

Anselin, L. (1988). *Spatial Econometrics: Methods and Models*, Table 12.1,
p. 189. Redistributed with PySAL/libpysal (BSD-3-Clause). `graph.json` is the
first-order contiguity graph shipped there as `columbus.gal`; ids are
zero-padded so the graph's sorted node order matches the data.

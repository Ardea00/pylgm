# Examples

Every example is a runnable script under
[`examples/`](https://github.com/Ardea00/pylgm/tree/main/examples), each with a
README stating what it shows and its expected output. Run any of them from the
repo root:

```bash
PYTHONPATH=src python examples/<name>/run.py
```

Most are covered by the CI suite, so the numbers in their READMEs are checked
rather than remembered.

## Reproducing published results

| Example | Shows |
|---|---|
| [`columbus_spatial_econometrics`](https://github.com/Ardea00/pylgm/tree/main/examples/columbus_spatial_econometrics) | **Anselin (1988), Table 12.1** — the reference dataset of spatial econometrics. OLS reproduces the published values exactly; pyLGM's `SAR` lands next to the published ML spatial-error estimates, halving the apparent income effect |
| [`state_income_dynamic_network`](https://github.com/Ardea00/pylgm/tree/main/examples/state_income_dynamic_network) | A **network that changes every year** — real US state income and contiguity, recovering knocked-out panel cells ~5× better than the obvious baselines |

## Start here

| Example | Shows |
|---|---|
| [`general_lgm`](https://github.com/Ardea00/pylgm/tree/main/examples/general_lgm) | The same model through both the Python and the YAML frontend |
| [`count_glm`](https://github.com/Ardea00/pylgm/tree/main/examples/count_glm) | A Poisson count model on the Laplace engine |
| [`method_comparison`](https://github.com/Ardea00/pylgm/tree/main/examples/method_comparison) | pyLGM against a GLM, XGBoost and a Metropolis sampler — see [comparison](comparison.md) |

## Spatial and network structure

| Example | Shows |
|---|---|
| [`disease_mapping`](https://github.com/Ardea00/pylgm/tree/main/examples/disease_mapping) | `Besag` spatial smoothing on Scotland lip cancer; fit to observed counts goes 0.63 → 0.96 against a non-spatial GLM. Walkthrough: [disease mapping](examples-disease-mapping.md) |
| [`weighted_network`](https://github.com/Ardea00/pylgm/tree/main/examples/weighted_network) | `BYM2` on a **weighted** firm-exposure graph — the CAR family on an economic network, not a map |
| [`directed_network_sar`](https://github.com/Ardea00/pylgm/tree/main/examples/directed_network_sar) | `SAR` on a directed interbank-exposure network, estimating contagion strength ρ |
| [`columbus_spatial_econometrics`](https://github.com/Ardea00/pylgm/tree/main/examples/columbus_spatial_econometrics) | `SAR` reproducing Anselin's published Columbus results on real contiguity data |
| [`state_income_dynamic_network`](https://github.com/Ardea00/pylgm/tree/main/examples/state_income_dynamic_network) | `DynamicSpatialPanel` with one network per year, on 48 US states |

## Time, frequency and forecasting

| Example | Shows |
|---|---|
| [`midas_nowcast`](https://github.com/Ardea00/pylgm/tree/main/examples/midas_nowcast) | `MIDAS` smooth-lag regression of a low-frequency target on high-frequency lags |
| [`hybrid_nowcast`](https://github.com/Ardea00/pylgm/tree/main/examples/hybrid_nowcast) | `MIDAS` + `BYM2` + `AR1` composed into one latent field that fits and predicts |
| [`synthetic_panel`](https://github.com/Ardea00/pylgm/tree/main/examples/synthetic_panel) | A panel with both unit and time structure |
| [`nic_backtest`](https://github.com/Ardea00/pylgm/tree/main/examples/nic_backtest) | A rolling-origin backtest configuration with covariate-availability rules |

## Counts, durations and model choice

| Example | Shows |
|---|---|
| [`count_regression`](https://github.com/Ardea00/pylgm/tree/main/examples/count_regression) | Horseshoe-crab Poisson GLM matching `statsmodels`, plus estimated overdispersion. Walkthrough: [count regression](examples-count-regression.md) |
| [`survival_duration`](https://github.com/Ardea00/pylgm/tree/main/examples/survival_duration) | `WeibullSurv` unemployment durations with right-censoring and an `IID` frailty |
| [`predictive_selection`](https://github.com/Ardea00/pylgm/tree/main/examples/predictive_selection) | Choosing between candidate models by out-of-sample score |

## Hyperparameters and posterior integration

| Example | Shows |
|---|---|
| [`empirical_bayes`](https://github.com/Ardea00/pylgm/tree/main/examples/empirical_bayes) | Type-II ML estimation of a precision, prior-free |
| [`map_ii`](https://github.com/Ardea00/pylgm/tree/main/examples/map_ii) | The same fit penalised by a PC prior (MAP-II) |
| [`inla`](https://github.com/Ardea00/pylgm/tree/main/examples/inla) | Grid integration over hyperparameters instead of a point estimate |
| [`inla_sla`](https://github.com/Ardea00/pylgm/tree/main/examples/inla_sla) | Simplified-Laplace latent marginals, with skewness |
| [`inla_full_laplace`](https://github.com/Ardea00/pylgm/tree/main/examples/inla_full_laplace) | Full-Laplace latent marginals and their quantiles |
| [`inla_criteria`](https://github.com/Ardea00/pylgm/tree/main/examples/inla_criteria) | DIC / WAIC / CPO / PIT model assessment |

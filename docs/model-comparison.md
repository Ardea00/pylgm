# Model comparison and backtesting

Fitting one model tells you what it thinks. Choosing between models needs
out-of-sample evidence, and on time-series or panel data that means a
**rolling-origin backtest** rather than a random split — a random split leaks
the future into the training set.

`Experiment` runs that comparison: several candidate models, evaluated at
declared horizons from declared origins, scored on the same folds, with a
deterministic selection rule and publishable artifacts.

## The shape of it

```python
import pandas as pd
from pylgm import Experiment

experiment = Experiment.from_yaml("examples/predictive_selection/config.yaml")
result = experiment.compare(pd.read_csv("examples/predictive_selection/data.csv"))
# result is a ComparisonResult: immutable, with defensive copies of its tables

result.decision.selected      # 'region_and_trend'
result.decision.ranking       # ('region_and_trend', 'region_only', 'persistence')
result.metrics                # per candidate x origin x horizon
result.predictions            # every fold prediction, with its origin and horizon
result.failures               # candidates that could not be fit, and why
```

Candidates are declared as **overrides on one base model**, so the thing being
compared is the change and not two separately-written specs:

```yaml
model:
  likelihood: gaussian
  fixed: "1"
  sigma: 0.5
  effects:
    - {name: region, type: iid, index: region, precision: 1.0}

candidates:
  - {name: region_only}
  - name: region_and_trend
    overrides:
      effects:
        - {name: region, type: iid, index: region, precision: 1.0}
        - {name: trend, type: rw1, index: month, precision: 1.0}
    optimize:
      sigma:              {initial: 0.5, lower: 0.1, upper: 3.0}
      region.precision:   {initial: 1.0, lower: 0.05, upper: 20.0}
      trend.precision:    {initial: 1.0, lower: 0.05, upper: 20.0}
```

Note the `optimize` keys: on the config path hyperparameter names are
**derived** as `<effect>.<parameter>`, so you never invent them.

## Folds are rolling-origin

```yaml
evaluation:
  horizons: [1, 2]
  origins: {last: 2}          # or {values: [6, 7]}
  window: {type: expanding}   # or {type: rolling, length: N}
  interval_levels: [0.8]
```

For each origin, everything at or before it is training data and the response
at `origin + h` is held out. The held-out rows stay in the frame with a `NaN`
response — the same mechanism as
[forecasting](prediction.md#predicting-new-rows) — so the latent structure is
built once and the future simply contributes no likelihood.

**Covariate availability** is declared, not assumed:

```yaml
data:
  covariates:
    - {name: x, availability: known_future}        # e.g. a calendar variable
    - {name: z, availability: observed_with_lag, lag: 1}
    - {name: w, availability: unavailable_future}
```

A covariate marked `unavailable_future` raises if a fold would need it past the
origin, rather than quietly scoring a model that could not have been run at the
time. This is the difference between a backtest and a look-ahead.

## What gets scored

`result.metrics` carries, per candidate × origin × horizon:

| group | columns |
| --- | --- |
| point accuracy | `rmse`, `mae`, `bias` |
| density | `log_predictive_density`, `mean_log_predictive_density` |
| calibration | `coverage_<level>`, `average_width_<level>` |
| bookkeeping | `prediction_count`, `evaluation_mode`, `is_benchmark` |

Coverage is the one people skip and shouldn't: a model can win on RMSE while
its intervals are badly calibrated, and `coverage_0_8` says whether an 80%
interval actually contained the truth 80% of the time.

A **`persistence` benchmark** (carry the last observed value forward) is scored
automatically and marked `is_benchmark`. It is never selectable — it is there so
you can see whether any candidate is beating "do nothing". For a near-unit-root
series it often is not, which is a real result rather than a bug; the
[dynamic-network case study](examples-dynamic-network.md#where-it-does-not-win)
walks through exactly that situation.

## Selection is a rule, not a judgement call

```python
result.decision.selected     # the winner, or None
result.decision.ranking      # every candidate, best first
result.decision.eligible     # {'persistence': False, 'region_only': True, ...}
result.decision.reasons      # {'persistence': ('benchmark',), ...}
```

`reasons` is the part worth reading: a candidate that was excluded says why —
it was the benchmark, or it failed a coverage tolerance, or it did not fit.
Ties are broken deterministically within `rmse_tie_tolerance`, so the same data
gives the same choice.

Candidates that fail to fit do **not** abort the comparison. They land in
`result.failures` as structured `CandidateFailure` records with a
`FailureCause`, and the others are still scored — one badly-specified candidate
should not cost you the whole run.

## From the command line

```bash
pylgm compare config.yaml data.csv --output comparison/
```

writes the selected model, fold predictions, aggregate metrics, and versioned
artifacts (`summary.json`, `decision.json`, `failures.json`,
`resolved_config.json`, prediction tables). The artifacts carry an
**experiment fingerprint** over the data, the resolved config and the fold
definitions, so a rerun that should be identical is verifiably identical — see
[internals](internals.md).

`pylgm fit` is the single-model counterpart.

## Worked examples

- [`examples/predictive_selection`](https://github.com/Ardea00/pylgm/tree/main/examples/predictive_selection)
  — a three-region panel comparing a region-only model against region + RW1
  trend at horizons 1 and 2.
- [`examples/nic_backtest`](https://github.com/Ardea00/pylgm/tree/main/examples/nic_backtest)
  — a larger rolling-origin configuration with covariate-availability rules.

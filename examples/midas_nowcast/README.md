# Runnable MIDAS smooth-lag nowcasting example

This directory nowcasts a quarterly target from a monthly indicator using the
[`MIDAS`](../../docs/effects.md#midas-smooth-lag-effect) mixed-frequency
smooth-lag effect. The data are generated inline (no `data.csv`): a monthly
indicator is aligned to quarterly rows with `pandas.Series.shift`, so each row
carries the indicator at lags 0..11, and the target is that lag window dotted
with a known smooth decaying kernel plus noise. Run it from the repository
root:

```bash
PYTHONPATH=src python examples/midas_nowcast/run.py
```

The model is:

```python
from pylgm import Fixed, Gaussian, Hyperparameter, LGM, MIDAS

model = LGM(
    response="y",
    predictor=Fixed("1")
    + MIDAS("lag", columns=columns, precision=Hyperparameter("lag.precision", initial=1.0)),
    likelihood=Gaussian(sigma=0.5),
)
result = model.fit(frame)
```

`columns` are the twelve monthly lag columns. Declaring the `MIDAS` precision
as a `Hyperparameter` lets empirical Bayes choose the smoothing strength `τ`;
the order-2 random-walk penalty over the lag index then pulls the twelve lag
coefficients into a smooth curve while leaving the curve's level and slope
free of `τ` (see the [effects guide](../../docs/effects.md#midas-smooth-lag-effect)).
The script prints the estimated `τ`, the true vs recovered lag curves, and
their correlation — which lands near 1.0, whereas unrestricted OLS on the same
twelve collinear lags would be visibly noisier.

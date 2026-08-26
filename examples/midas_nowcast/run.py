"""Nowcast a quarterly target from a monthly indicator with a MIDAS smooth-lag effect.

Self-contained: a synthetic monthly indicator is aligned to quarterly rows via
`shift`, then a `MIDAS(order=2)` effect with an estimated smoothing precision
recovers the known decaying lag kernel that generated the target.
"""

import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, MIDAS


LAGS = 12  # months of the monthly indicator feeding each quarterly target


def _nowcast_frame(n_quarters=200, seed=0):
    """Build quarterly rows carrying LAGS monthly lag columns of an HF indicator."""
    rng = np.random.default_rng(seed)
    # A monthly indicator, one value per quarterly row's most recent month.
    monthly = pd.Series(rng.normal(size=n_quarters + LAGS))
    # Align HF to LF: lag k column is the indicator shifted back k months.
    columns = tuple(f"ind_lag{k}" for k in range(LAGS))
    lagged = {name: monthly.shift(k) for k, name in enumerate(columns)}
    frame = pd.DataFrame(lagged).dropna().reset_index(drop=True)

    # True lag kernel: a smooth exp-Almon-shaped decay over the 12 months.
    k = np.arange(LAGS)
    kernel = np.exp(-0.4 * k) * (1.0 + 0.2 * k)
    frame["y"] = frame[list(columns)].to_numpy() @ kernel + rng.normal(scale=0.5, size=len(frame))
    return frame, columns, kernel


def main() -> None:
    frame, columns, kernel = _nowcast_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + MIDAS("lag", columns=columns, precision=Hyperparameter("lag.precision", initial=1.0)),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame)

    fitted = result.latent_marginals("lag").mean
    print(f"smoothing precision (estimated): {result.hyperparameters['lag.precision']:.3f}")
    print(f"lag:      {[f'{k:>5}' for k in range(LAGS)]}")
    print(f"true:     {np.round(kernel, 3)}")
    print(f"recovered:{np.round(fitted, 3)}")
    print(f"correlation with true kernel: {np.corrcoef(fitted, kernel)[0, 1]:.3f}")


if __name__ == "__main__":
    main()

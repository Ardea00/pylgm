import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.sdpd_forecast import forecast_dynamic_spatial_panel
from pylgm.effects.spec import DynamicSpatialPanel


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def test_forecast_returns_future_grid():
    n, periods = 6, 4
    rng = np.random.default_rng(0)
    rows = [
        {"unit": str(i), "period": str(t), "y": rng.standard_normal()}
        for t in range(periods)
        for i in range(n)
    ]
    frame = pd.DataFrame(rows)
    graphs = {str(t): _ring(n) for t in range(periods)}
    effect = DynamicSpatialPanel(
        "d", "unit", "period", graphs, rho=0.4, gamma=0.3, eta=0.1,
        precision=Hyperparameter("d.prec", initial=1.0),
    )
    model = LGM(response="y", predictor=Fixed("1") + effect, likelihood=Gaussian(sigma=0.2))
    result = model.fit(frame)
    # NB: pass the *compiled* effect (graphs canonicalized). Rebuild it from the
    # predictor to get the frozen spec the model actually holds.
    fitted_effect = model.predictor.effects[1]
    forecast = forecast_dynamic_spatial_panel(result, fitted_effect, {"4": _ring(n)})
    assert set(forecast.columns) == {"unit", "time", "mean", "variance"}
    assert len(forecast) == n
    assert (forecast["time"] == "4").all()
    assert np.isfinite(forecast["mean"]).all()
    assert (forecast["variance"] > 0).all()

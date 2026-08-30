import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.spec import SAR


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def _simulate(n, rho, seed=0):
    from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize

    rng = np.random.default_rng(seed)
    nodes, w = normalize_directed_graph(_ring(n))
    w = row_standardize(w).toarray()
    m = np.eye(n) - rho * w
    x = np.linalg.solve(m, rng.standard_normal(n))  # draw from the SAR field
    y = x + 0.1 * rng.standard_normal(n)
    return pd.DataFrame({"region": list(nodes), "y": y})


def test_sar_fixed_rho_fits():
    frame = _simulate(12, 0.5)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR("s", "region", _ring(12), rho=0.5, precision=Hyperparameter("s.prec", initial=1.0)),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.mean.shape[0] > 0
    assert np.isfinite(result.mean).all()


def test_sar_estimates_rho():
    frame = _simulate(20, 0.7, seed=3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR(
            "s", "region", _ring(20),
            rho=Hyperparameter("s.rho", initial=0.0, transform="logit"),
            precision=Hyperparameter("s.prec", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert 0.3 < result.hyperparameters["s.rho"] < 0.95  # recovers a positive rho

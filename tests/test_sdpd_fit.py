import numpy as np
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.spec import DynamicSpatialPanel


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def _simulate(n, periods, rho, gamma, eta, seed=0):
    from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize

    rng = np.random.default_rng(seed)
    nodes, w = normalize_directed_graph(_ring(n))
    w = row_standardize(w).toarray()
    a = np.eye(n) - rho * w
    b = gamma * np.eye(n) + eta * w
    x_prev = np.linalg.solve(a, rng.standard_normal(n))
    rows = []
    xs = [x_prev]
    for _ in range(1, periods):
        x = np.linalg.solve(a, b @ xs[-1] + rng.standard_normal(n))
        xs.append(x)
    for t, x in enumerate(xs):
        for i, node in enumerate(nodes):
            rows.append({"unit": node, "period": str(t), "y": x[i] + 0.1 * rng.standard_normal()})
    graphs = {str(t): _ring(n) for t in range(periods)}
    return pd.DataFrame(rows), graphs


def test_sdpd_fixed_coefficients_fits_and_predicts():
    frame, graphs = _simulate(8, 4, rho=0.4, gamma=0.3, eta=0.1)
    model = LGM(
        response="y",
        predictor=Fixed("1") + DynamicSpatialPanel(
            "d", "unit", "period", graphs, rho=0.4, gamma=0.3, eta=0.1,
            precision=Hyperparameter("d.prec", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert np.isfinite(result.mean).all()
    prediction = result.predict(frame.head(4))
    assert prediction.predictive_mean.shape == (4,)
    assert np.isfinite(prediction.predictive_mean).all()


def test_sdpd_estimates_all_three_coefficients():
    frame, graphs = _simulate(12, 6, rho=0.5, gamma=0.4, eta=0.2, seed=7)
    model = LGM(
        response="y",
        predictor=Fixed("1") + DynamicSpatialPanel(
            "d", "unit", "period", graphs,
            rho=Hyperparameter("d.rho", initial=0.0, transform="logit"),
            gamma=Hyperparameter("d.gamma", initial=0.0, transform="identity"),
            eta=Hyperparameter("d.eta", initial=0.0, transform="identity"),
            precision=Hyperparameter("d.prec", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.hyperparameters["d.rho"] > 0.2
    assert result.hyperparameters["d.gamma"] > 0.15
    assert np.isfinite(result.hyperparameters["d.eta"])

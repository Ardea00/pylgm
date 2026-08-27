import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, LGM, MIDASParametric
from pylgm.effects.midas import midas_weights


def _data(kernel="beta", theta=(2.0, 5.0), beta=2.0, n=200, K=6, seed=3):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, K))
    w = midas_weights(kernel, K, theta)
    y = 0.3 + beta * (V @ w) + rng.normal(scale=0.08, size=n)
    cols = {f"x{k}": V[:, k] for k in range(K)}
    cols["y"] = y
    return pd.DataFrame(cols), tuple(f"x{k}" for k in range(K))


def test_predict_recovers_the_field():
    frame, cols = _data()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta"),
        likelihood=Gaussian(sigma=0.08),
    )
    result = model.fit(frame)
    eta = result.predict(frame).predictive_mean
    c_truth = frame["y"].to_numpy() - frame["y"].mean()
    c_eta = eta - eta.mean()
    assert np.corrcoef(c_eta, c_truth)[0, 1] > 0.9


def test_predict_uses_fitted_weights_not_initial():
    # exp-Almon truth decays strongly; the initial (0, -0.1) is nearly flat.
    frame, cols = _data(kernel="exp_almon", theta=(0.0, -0.8))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon"),
        likelihood=Gaussian(sigma=0.08),
    )
    result = model.fit(frame)
    # Score a fresh block of rows; the prediction context must aggregate them
    # with the ESTIMATED weights, so the fit should track the true signal.
    hold, _ = _data(kernel="exp_almon", theta=(0.0, -0.8), seed=99)
    eta = result.predict(hold).predictive_mean
    c_truth = hold["y"].to_numpy() - hold["y"].mean()
    assert np.corrcoef(eta - eta.mean(), c_truth)[0, 1] > 0.85

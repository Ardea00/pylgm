import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, LGM, MIDASParametric
from pylgm.effects.midas import midas_weights


def _frame(kernel="beta", theta=(2.0, 4.0), beta=2.0, n=160, K=6, seed=1):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, K))
    w = midas_weights(kernel, K, theta)
    y = 0.5 + beta * (V @ w) + rng.normal(scale=0.1, size=n)
    cols = {f"x{k}": V[:, k] for k in range(K)}
    cols["y"] = y
    return pd.DataFrame(cols), tuple(f"x{k}" for k in range(K))


def test_fixed_theta_fit_runs():
    frame, cols = _frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta", shape1=2.0, shape2=4.0),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert np.isfinite(result.latent_marginals("m").mean).all()


def test_estimated_beta_shapes_are_reported_and_in_domain():
    frame, cols = _frame(kernel="beta", theta=(2.0, 5.0))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.hyperparameters["m.shape1"] > 0.0
    assert result.hyperparameters["m.shape2"] > 0.0


def test_estimated_exp_almon_shapes_are_reported():
    frame, cols = _frame(kernel="exp_almon", theta=(0.0, -0.4))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert np.isfinite(result.hyperparameters["m.shape1"])
    assert np.isfinite(result.hyperparameters["m.shape2"])


def test_proper_block_accepted_under_full_laplace():
    # beta has a proper vague prior and no constraints -> full Laplace is fine
    frame, cols = _frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDASParametric("m", cols, kernel="beta"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, hyperparameters="integrate", latent_strategy="laplace")
    assert np.isfinite(result.latent_marginals("m").mean).all()

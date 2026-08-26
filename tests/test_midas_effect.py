import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, MIDAS
from pylgm.priors import PCPrecision

K = 6
COLUMNS = tuple(f"lag{j}" for j in range(K))


def _midas_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, K))
    kernel = np.exp(-0.6 * np.arange(K))  # smooth decaying lag weights
    y = x @ kernel + rng.normal(scale=0.1, size=n)
    frame = pd.DataFrame(x, columns=list(COLUMNS))
    frame["y"] = y
    return frame, kernel


def test_fixed_tau_fit_returns_lag_curve():
    frame, _ = _midas_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDAS("lag", columns=COLUMNS, precision=1.0),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    marginals = result.latent_marginals("lag")
    assert marginals.mean.shape == (K,)
    assert np.all(np.isfinite(marginals.mean))


def test_estimate_tau_empirical_bayes():
    frame, _ = _midas_frame()
    hp = Hyperparameter("lag.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDAS("lag", columns=COLUMNS, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    estimate = result.hyperparameters["lag.precision"]
    assert np.isfinite(estimate) and estimate > 0


def test_integrate_tau():
    frame, _ = _midas_frame()
    hp = Hyperparameter("lag.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDAS("lag", columns=COLUMNS, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, hyperparameters="integrate")
    assert "lag.precision" in result.hyperparameter_marginals()


def test_full_laplace_accepts_midas():
    # Unlike RW2/Besag (rejected under full Laplace for carrying a constraint),
    # MIDAS is proper and unconstrained, so full Laplace runs.
    frame, _ = _midas_frame()
    hp = Hyperparameter("lag.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDAS("lag", columns=COLUMNS, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, hyperparameters="integrate", latent_strategy="laplace")
    assert result.latent_marginals("lag").mean.shape == (K,)


def test_predict_on_new_rows():
    frame, _ = _midas_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDAS("lag", columns=COLUMNS, precision=1.0),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    new_frame, _ = _midas_frame(n=5, seed=1)
    prediction = result.predict(new_frame.drop(columns="y"))
    assert prediction.predictive_mean.shape == (5,)
    assert np.all(np.isfinite(prediction.predictive_mean))


def _second_difference_energy(curve):
    second = curve[:-2] - 2 * curve[1:-1] + curve[2:]
    return float(second @ second)


def test_penalised_lag_curve_recovers_smooth_kernel():
    # Simulate y = sum_k w_k * x_{t-k} + noise from a smooth decaying kernel over
    # 12 lags; the RW2-penalised MIDAS curve should recover the kernel and be far
    # smoother than the unpenalised OLS lag coefficients.
    lags = 12
    columns = tuple(f"lag{j}" for j in range(lags))
    rng = np.random.default_rng(7)
    n = 160
    x = rng.normal(size=(n, lags))
    k = np.arange(lags)
    kernel = np.exp(-0.35 * k) * (1.0 + 0.15 * k)  # smooth exp-Almon-like shape
    y = x @ kernel + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(x, columns=list(columns))
    frame["y"] = y

    hp = Hyperparameter("lag.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(
        response="y",
        predictor=Fixed("1") + MIDAS("lag", columns=columns, precision=hp, order=2),
        likelihood=Gaussian(sigma=0.5),
    )
    fitted = model.fit(frame).latent_marginals("lag").mean

    ols = np.linalg.lstsq(x, y, rcond=None)[0]  # unpenalised lag coefficients

    # (a) far smoother than OLS
    assert _second_difference_energy(fitted) < 0.25 * _second_difference_energy(ols)
    # (b) recovers the true kernel shape
    assert np.corrcoef(fitted, kernel)[0, 1] > 0.9

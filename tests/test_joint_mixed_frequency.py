"""A higher-frequency indicator measuring a lower-frequency shared latent.

The indicator arrives every subperiod (e.g. three releases per period); the
target arrives once per period, with the latest period's target row absent
altogether -- the ragged edge. ``Shared`` links them through one AR1 latent
indexed on ``period``, and ``allow_ragged=True`` accepts that the target's
index set is missing the latest level.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Fixed, Gaussian, LGM
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def _frame(months_in_latest, seed=0, periods=12):
    rng = np.random.default_rng(seed)
    rho, precision = 0.7, 1.0

    # Stationary AR1 draw over `periods` levels: cov[i, j] = rho**|i-j| / precision.
    idx = np.arange(periods)
    cov = (rho ** np.abs(idx[:, None] - idx[None, :])) / precision
    u = rng.multivariate_normal(np.zeros(periods), cov)

    target_periods = np.arange(periods - 1)
    target = 0.5 + u[target_periods] + rng.normal(0.0, 0.3, size=len(target_periods))
    target_rows = pd.DataFrame({
        "period": target_periods.astype(np.int64),
        "target": target,
        "indicator": np.nan,
    })

    ind_periods = []
    ind_subperiods = []
    for period in range(periods):
        count = 3 if period < periods - 1 else months_in_latest
        ind_periods.extend([period] * count)
        ind_subperiods.extend(range(count))
    ind_periods = np.array(ind_periods, dtype=np.int64)
    ind_subperiods = np.array(ind_subperiods, dtype=np.int64)
    indicator = -0.2 + 0.8 * u[ind_periods] + rng.normal(0.0, 0.4, size=len(ind_periods))
    indicator_rows = pd.DataFrame({
        "period": ind_periods,
        "subperiod": ind_subperiods,
        "indicator": indicator,
        "target": np.nan,
    })

    frame = pd.concat([target_rows, indicator_rows], ignore_index=True)
    frame["period"] = frame["period"].astype(np.int64)
    return frame, u


def _joint():
    return Joint(
        [
            LGM(response="target", likelihood=Gaussian(sigma=0.3), predictor=Fixed("1")),
            LGM(response="indicator", likelihood=Gaussian(sigma=0.4), predictor=Fixed("1")),
        ],
        shared=[Shared(AR1("u", index="period", precision=1.0, rho=0.7),
                       scale=(1.0, 0.8), allow_ragged=True)],
    )


def _oracle(frame, periods, sigma_target, sigma_indicator, rho, precision, loading):
    """Hand-computed exact Gaussian posterior for x = (a_target, a_indicator, u_0..u_{P-1})."""
    idx = np.arange(periods)
    C = (rho ** np.abs(idx[:, None] - idx[None, :])) / precision
    Qu = np.linalg.inv(C)

    dim = 2 + periods
    Qprior = np.zeros((dim, dim))
    Qprior[0, 0] = 1e-6
    Qprior[1, 1] = 1e-6
    Qprior[2:, 2:] = Qu

    rows_A = []
    rows_y = []
    rows_w = []
    for _, row in frame.iterrows():
        p = int(row["period"])
        e_p = np.zeros(periods)
        e_p[p] = 1.0
        if not np.isnan(row["target"]):
            a = np.zeros(dim)
            a[0] = 1.0
            a[2:] = e_p
            rows_A.append(a)
            rows_y.append(row["target"])
            rows_w.append(1.0 / sigma_target**2)
        if not np.isnan(row["indicator"]):
            a = np.zeros(dim)
            a[1] = 1.0
            a[2:] = loading * e_p
            rows_A.append(a)
            rows_y.append(row["indicator"])
            rows_w.append(1.0 / sigma_indicator**2)

    A = np.array(rows_A)
    y = np.array(rows_y)
    w = np.array(rows_w)
    W = np.diag(w)

    Qpost = Qprior + A.T @ W @ A
    mean = np.linalg.solve(Qpost, A.T @ (w * y))
    cov = np.linalg.inv(Qpost)

    c = np.zeros(dim)
    c[0] = 1.0
    c[2 + periods - 1] = 1.0
    m = float(c @ mean)
    v = float(c @ cov @ c)

    # Marginal (integrating out the latent) Gaussian LML: y ~ N(0, A Qprior^-1 A^T + diag(sigma^2)).
    Qprior_inv = np.zeros((dim, dim))
    Qprior_inv[0, 0] = 1.0 / 1e-6
    Qprior_inv[1, 1] = 1.0 / 1e-6
    Qprior_inv[2:, 2:] = C
    marg_cov = A @ Qprior_inv @ A.T + np.diag(1.0 / w)
    sign, logdet = np.linalg.slogdet(marg_cov)
    n = len(y)
    lml = -0.5 * (n * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(marg_cov, y))

    return m, v, lml


@pytest.mark.parametrize("months_in_latest", [1, 2, 3])
def test_shared_latent_matches_the_exact_gaussian_posterior(months_in_latest):
    periods = 12
    frame, u = _frame(months_in_latest, seed=0, periods=periods)
    joint = _joint()
    result = joint.fit(frame)

    new_data = pd.DataFrame({
        "period": [periods - 1],
        "target": [np.nan],
        "indicator": [np.nan],
    })
    prediction = result.predict(new_data, outcome="target")

    m, v, lml = _oracle(
        frame, periods, sigma_target=0.3, sigma_indicator=0.4, rho=0.7, precision=1.0, loading=0.8,
    )

    assert prediction.predictive_mean[0] == pytest.approx(m, abs=1e-8)
    assert prediction.predictive_variance[0] == pytest.approx(v, rel=1e-6)
    assert result.log_marginal_likelihood == pytest.approx(lml, abs=1e-5)


def test_more_subperiods_shrink_the_latest_period_uncertainty():
    # Only the variance is deterministic at fixed hyperparameters; the mean
    # need not move toward the truth with every extra release of one draw.
    periods = 12
    variances = []
    for months_in_latest in (1, 2, 3):
        frame, u = _frame(months_in_latest, seed=0, periods=periods)
        joint = _joint()
        result = joint.fit(frame)
        new_data = pd.DataFrame({
            "period": [periods - 1],
            "target": [np.nan],
            "indicator": [np.nan],
        })
        prediction = result.predict(new_data, outcome="target")
        variances.append(prediction.predictive_variance[0])

    assert variances[0] > variances[1] > variances[2]


def test_loading_and_noise_scales_are_estimated():
    periods = 80
    frame, u = _frame(months_in_latest=3, seed=1, periods=periods)

    joint = Joint(
        [
            LGM(response="target", likelihood=Gaussian(sigma=Hyperparameter("sigma_target", initial=0.5)),
                predictor=Fixed("1")),
            LGM(response="indicator", likelihood=Gaussian(sigma=Hyperparameter("sigma_indicator", initial=0.5)),
                predictor=Fixed("1")),
        ],
        shared=[Shared(
            AR1("u", index="period", precision=1.0, rho=0.7),
            scale=(1.0, Hyperparameter("loading", initial=1.0)),
            allow_ragged=True,
        )],
    )
    result = joint.fit(frame)

    assert result.hyperparameters["loading"] == pytest.approx(0.8, abs=0.15)
    assert result.hyperparameters["sigma_target"] == pytest.approx(0.3, rel=0.30)
    assert result.hyperparameters["sigma_indicator"] == pytest.approx(0.4, rel=0.30)

    new_data = pd.DataFrame({
        "period": [periods - 1],
        "target": [np.nan],
        "indicator": [np.nan],
    })
    prediction = result.predict(new_data, outcome="target")
    truth = 0.5 + u[periods - 1]
    assert abs(prediction.predictive_mean[0] - truth) <= 3 * np.sqrt(prediction.predictive_variance[0])

"""A data-carrying ``LinearConstraint`` enters the log marginal likelihood.

The target is ``log p(y, e | theta) = log p(y | theta) + log p(e | y, theta)``:
exact aggregates are data, so empirical Bayes and INLA must learn from them.
Structural (intrinsic sum-to-zero) and model-level label constraints stay pure
conditioning.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import multivariate_normal

import pylgm.inference.gaussian as gaussian_engine
from pylgm import AR1, RW1, Fixed, Gaussian, Hyperparameter, LGM
from pylgm import LinearConstraint, LinearObservation

T = 24
ANNUAL = np.kron(np.eye(T // 4), np.ones((1, 4)))


@pytest.fixture(params=["dense", "sparse"])
def engine(request, monkeypatch):
    """Route the exact Gaussian fit through the dense or the sparse engine."""
    if request.param == "sparse":
        monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    return request.param


def _grid():
    return pd.DataFrame({"u": ["a"] * T, "t": range(T)})


def _series():
    return np.cumsum(np.random.default_rng(0).normal(size=T)) * 0.5


def _ar1_covariance(rho, tau=1.0):
    # pyLGM's AR1 precision is the marginal precision.
    i = np.arange(T)
    return rho ** np.abs(i[:, None] - i[None]) / tau


def _ar1_model(rho):
    return LGM(
        response="y", likelihood=Gaussian(1.0),
        predictor=AR1("ar", "t", precision=1.0, rho=rho), panel=("u",), time="t",
    )


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8, 0.95])
def test_constraint_lml_is_the_density_of_the_aggregates(rho, engine):
    e = ANNUAL @ _series()
    result = _ar1_model(rho).fit(_grid(), constraints=[LinearConstraint(ANNUAL, e)])

    covariance = ANNUAL @ _ar1_covariance(rho) @ ANNUAL.T
    expected = multivariate_normal(np.zeros(e.size), covariance).logpdf(e)
    assert result.log_marginal_likelihood == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_redundant_data_rows_do_not_double_count(engine):
    e = ANNUAL @ _series()
    doubled = LinearConstraint(np.vstack([ANNUAL, ANNUAL]), np.concatenate([e, e]))
    once = _ar1_model(0.7).fit(_grid(), constraints=[LinearConstraint(ANNUAL, e)])
    twice = _ar1_model(0.7).fit(_grid(), constraints=[doubled])
    assert twice.log_marginal_likelihood == pytest.approx(once.log_marginal_likelihood)


def test_constraint_is_the_zero_noise_limit_of_a_linear_observation(engine):
    """Fixed (vague) + intrinsic RW1 + row data: the structural sum-to-zero and
    the vague intercept must not make the aggregate term improper."""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=T)) + 5.0
    frame = _grid().assign(y=x + rng.normal(scale=0.3, size=T))
    frame.loc[16:, "y"] = np.nan  # the last two years are nowcast from aggregates only
    model = LGM(
        response="y", likelihood=Gaussian(0.3),
        predictor=Fixed("1") + RW1("rw", "t", precision=2.0), panel=("u",), time="t",
    )
    e = ANNUAL @ x
    exact = model.fit(frame, constraints=[LinearConstraint(ANNUAL, e)])
    noisy = model.fit(frame, observations=[LinearObservation(e, ANNUAL, sigma=1e-4)])

    assert np.isfinite(exact.log_marginal_likelihood)
    assert exact.log_marginal_likelihood == pytest.approx(
        noisy.log_marginal_likelihood, rel=1e-6
    )
    np.testing.assert_allclose(ANNUAL @ exact.predictive_mean, e, rtol=1e-9)
    np.testing.assert_allclose(exact.predictive_mean, noisy.predictive_mean, atol=1e-4)


def test_label_constraints_stay_pure_conditioning():
    """``LGM(constraints=...)`` is R-INLA's extraconstr: it conditions, it is not data."""
    frame = _grid().assign(y=_series())
    base = dict(response="y", likelihood=Gaussian(0.5), panel=("u",), time="t")
    pinned = LGM(**base, predictor=AR1("ar", "t", precision=1.0, rho=0.5),
                 constraints=[({"ar:0": 1.0}, 0.0)])
    as_data = LGM(**base, predictor=AR1("ar", "t", precision=1.0, rho=0.5))
    first = np.zeros((1, T))
    first[0, 0] = 1.0

    label_lml = pinned.fit(frame).log_marginal_likelihood
    data_lml = as_data.fit(frame, constraints=[LinearConstraint(first, [0.0])])
    # The data version adds log p(eta_0 = 0 | y), which the label version must not.
    assert data_lml.log_marginal_likelihood != pytest.approx(label_lml)


def test_empirical_bayes_recovers_the_rho_that_maximises_the_aggregate_likelihood():
    e = ANNUAL @ _series()

    def true_loglik(rho):
        covariance = ANNUAL @ _ar1_covariance(rho) @ ANNUAL.T
        return multivariate_normal(np.zeros(e.size), covariance).logpdf(e)

    best = minimize_scalar(lambda r: -true_loglik(r), bounds=(-0.99, 0.99), method="bounded",
                           options={"xatol": 1e-8}).x
    rho = Hyperparameter("ar.rho", initial=0.3, transform="logit")
    result = _ar1_model(rho).fit(_grid(), constraints=[LinearConstraint(ANNUAL, e)])
    assert result.hyperparameters["ar.rho"] == pytest.approx(best, abs=1e-3)


# --- Chow-Lin ---------------------------------------------------------------

def _chow_lin(indicator, annual, rho):
    """Independent GLS Chow-Lin: y = X b + u, u stationary AR1(rho), C y = Y."""
    n = indicator.shape[0]
    aggregation = np.kron(np.eye(n // 4), np.ones((1, 4)))
    i = np.arange(n)
    sigma = rho ** np.abs(i[:, None] - i[None]) / (1 - rho**2)
    v = aggregation @ sigma @ aggregation.T
    xa = aggregation @ indicator
    v_inv = np.linalg.inv(v)
    info = xa.T @ v_inv @ xa
    beta = np.linalg.solve(info, xa.T @ v_inv @ annual)
    residual = annual - xa @ beta
    series = indicator @ beta + sigma @ aggregation.T @ v_inv @ residual
    return series, v, info, residual


def _chow_lin_reml_rho(indicator, annual):
    """REML over (rho, sigma^2): beta integrated under a flat prior."""
    def negative(params):
        rho, log_s2 = np.tanh(params[0]), params[1]
        _, v, info, residual = _chow_lin(indicator, annual, rho)
        v = v * np.exp(log_s2)
        info = info / np.exp(log_s2)
        return 0.5 * (np.linalg.slogdet(v)[1] + np.linalg.slogdet(info)[1]
                      + residual @ np.linalg.solve(v, residual))
    fit = minimize(negative, [0.5, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000})
    return float(np.tanh(fit.x[0]))


def _textbook_series():
    # A small quarterly indicator with trend and seasonality, eight years.
    n = 32
    t = np.arange(n)
    rng = np.random.default_rng(11)
    indicator = 100 + 0.8 * t + 3 * np.sin(np.pi * t / 2) + rng.normal(scale=1.0, size=n)
    truth = 5 + 1.5 * indicator + np.cumsum(rng.normal(scale=0.7, size=n))
    annual = np.kron(np.eye(n // 4), np.ones((1, 4))) @ truth
    return indicator, annual


def _chow_lin_lgm(rho, tau=1.0):
    return LGM(
        response="y", likelihood=Gaussian(1.0),
        predictor=Fixed("1 + x", prior_precision=1e-10) + AR1("ar", "t", precision=tau, rho=rho),
        panel=("u",), time="t",
    )


def test_chow_lin_series_at_fixed_rho_matches_gls(engine):
    indicator, annual = _textbook_series()
    n = indicator.size
    frame = pd.DataFrame({"u": ["a"] * n, "t": range(n), "x": indicator})
    aggregation = np.kron(np.eye(n // 4), np.ones((1, 4)))
    x = np.column_stack([np.ones(n), indicator])

    expected, *_ = _chow_lin(x, annual, 0.6)
    result = _chow_lin_lgm(0.6).fit(frame, constraints=[LinearConstraint(aggregation, annual)])
    np.testing.assert_allclose(result.predictive_mean, expected, rtol=1e-6)


def test_chow_lin_reml_rho_is_reproduced_by_empirical_bayes():
    indicator, annual = _textbook_series()
    n = indicator.size
    frame = pd.DataFrame({"u": ["a"] * n, "t": range(n), "x": indicator})
    aggregation = np.kron(np.eye(n // 4), np.ones((1, 4)))
    x = np.column_stack([np.ones(n), indicator])

    expected = _chow_lin_reml_rho(x, annual)
    model = _chow_lin_lgm(
        Hyperparameter("ar.rho", initial=0.3, transform="logit"),
        Hyperparameter("ar.precision", initial=1.0, lower=1e-6, upper=1e6),
    )
    result = model.fit(frame, constraints=[LinearConstraint(aggregation, annual)])
    assert result.hyperparameters["ar.rho"] == pytest.approx(expected, abs=2e-3)

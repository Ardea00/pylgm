"""An estimable ``sigma`` on ``LinearObservation``."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

from pylgm import AR1, Gaussian, Hyperparameter, LGM, LinearConstraint, LinearObservation
from pylgm.exceptions import ModelValidationError

T = 48
TRUE_SIGMA = 0.7


def _grid():
    return pd.DataFrame({"u": ["a"] * T, "t": range(T)})


def _covariance(rho=0.6):
    i = np.arange(T)
    return rho ** np.abs(i[:, None] - i[None])  # AR1, marginal precision 1


def _aggregates():
    """Noisy two-quarter sums of an AR1 path: 47 overlapping windows."""
    rng = np.random.default_rng(5)
    path = np.linalg.cholesky(_covariance()) @ rng.normal(size=T)
    operator = np.zeros((T - 1, T))
    for row in range(T - 1):
        operator[row, row:row + 2] = 1.0
    values = operator @ path + rng.normal(scale=TRUE_SIGMA, size=T - 1)
    return operator, values


def _model(sigma=1.0):
    return LGM(
        response="y", likelihood=Gaussian(sigma),
        predictor=AR1("ar", "t", precision=1.0, rho=0.6), panel=("u",), time="t",
    )


def _closed_form(operator, values, sigma):
    covariance = operator @ _covariance() @ operator.T + sigma**2 * np.eye(values.size)
    return multivariate_normal(np.zeros(values.size), covariance).logpdf(values)


def test_empirical_bayes_recovers_the_observation_sigma_and_the_exact_lml():
    operator, values = _aggregates()
    sigma = Hyperparameter("agg.sigma", initial=1.0, lower=0.05, upper=5.0)
    result = _model().fit(_grid(), observations=[LinearObservation(values, operator, sigma)])

    best = minimize_scalar(
        lambda s: -_closed_form(operator, values, s), bounds=(0.05, 5.0), method="bounded",
        options={"xatol": 1e-8},
    ).x
    estimate = result.hyperparameters["agg.sigma"]
    assert estimate == pytest.approx(best, rel=1e-3)
    assert estimate == pytest.approx(TRUE_SIGMA, rel=0.3)
    assert result.log_marginal_likelihood == pytest.approx(
        _closed_form(operator, values, estimate), rel=1e-8
    )


def test_estimated_sigma_composes_with_effect_hyperparameters_and_fixed_blocks():
    operator, values = _aggregates()
    rho = Hyperparameter("ar.rho", initial=0.3, transform="logit")
    model = LGM(
        response="y", likelihood=Gaussian(1.0),
        predictor=AR1("ar", "t", precision=1.0, rho=rho), panel=("u",), time="t",
    )
    fixed = LinearObservation(values[:10], operator[:10], sigma=TRUE_SIGMA)
    estimated = LinearObservation(
        values[10:], operator[10:], Hyperparameter("agg.sigma", initial=1.0)
    )
    result = model.fit(_grid(), observations=[fixed, estimated])
    assert set(result.hyperparameters) == {"ar.rho", "agg.sigma"}


def test_estimated_sigma_integrates():
    operator, values = _aggregates()
    sigma = Hyperparameter("agg.sigma", initial=1.0, lower=0.05, upper=5.0)
    result = _model().fit(
        _grid(), observations=[LinearObservation(values, operator, sigma)],
        hyperparameters="integrate",
    )
    marginal = result.hyperparameter_marginals()["agg.sigma"]
    assert 0.3 < float(marginal.mean[0]) < 1.2


def test_a_logit_sigma_is_rejected():
    with pytest.raises((ValueError, TypeError)):
        LinearObservation([1.0], [[1.0]], Hyperparameter("s", initial=0.5, transform="logit"))


def test_duplicate_hyperparameter_names_are_rejected():
    operator, values = _aggregates()
    rho = Hyperparameter("dup", initial=0.3, transform="logit")
    model = LGM(
        response="y", likelihood=Gaussian(1.0),
        predictor=AR1("ar", "t", precision=1.0, rho=rho), panel=("u",), time="t",
    )
    with pytest.raises(ModelValidationError, match="dup"):
        model.fit(_grid(), observations=[
            LinearObservation(values, operator, Hyperparameter("dup", initial=1.0))
        ])


def test_a_dummy_row_sigma_cannot_be_estimated_without_row_responses():
    operator, values = _aggregates()
    model = _model(sigma=Hyperparameter("sigma", initial=1.0))
    with pytest.raises(ModelValidationError, match="no row responses"):
        model.fit(_grid(), observations=[LinearObservation(values, operator, sigma=0.5)])
    with pytest.raises(ModelValidationError, match="no row responses"):
        model.fit(_grid(), constraints=[LinearConstraint(operator[:2], values[:2])])

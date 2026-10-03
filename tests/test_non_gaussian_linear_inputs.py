"""Linear observations and constraints under a non-Gaussian row likelihood.

The oracle is the exact log-posterior written in plain numpy and optimised by
scipy, with no pyLGM in the loop.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from pylgm import Bernoulli, Fixed, Hyperparameter, IID, LGM, LinearConstraint, LinearObservation, Poisson

TAU = 2.0
SIGMA = 0.5
CELLS = 8
DIFFUSE = 1e-6  # Fixed's default prior precision


def _frame():
    # Rows 0-5 observed counts; rows 6-7 only reach the fit through the aggregate.
    y = [3.0, 1.0, 4.0, 2.0, 6.0, 0.0, np.nan, np.nan]
    return pd.DataFrame({"cell": [f"c{i}" for i in range(CELLS)], "y": y})


def _model(likelihood=None, precision=TAU):
    return LGM(
        response="y", likelihood=likelihood or Poisson(),
        predictor=Fixed("1") + IID("u", index="cell", precision=precision), panel=("cell",),
    )


OPERATOR = np.array([
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
])
TOTALS = np.array([20.0, 9.0])


def _eta(z):
    return z[0] + z[1:]


def _negative_log_posterior(z, aggregate=False):
    """``z = (intercept, u)``; returns the value and its gradient."""
    eta, y = _eta(z), _frame()["y"].to_numpy()[:6]
    value = DIFFUSE / 2 * z[0] ** 2 + TAU / 2 * z[1:] @ z[1:]
    value += np.sum(np.exp(eta[:6]) - y * eta[:6])
    d_eta = np.concatenate([np.exp(eta[:6]) - y, np.zeros(2)])
    if aggregate:
        residual = (TOTALS - OPERATOR @ np.exp(eta)) / SIGMA
        value += 0.5 * residual @ residual
        d_eta -= (residual / SIGMA) @ OPERATOR * np.exp(eta)
    gradient = np.concatenate([[DIFFUSE * z[0] + d_eta.sum()], TAU * z[1:] + d_eta])
    return value, gradient


def test_poisson_log_constraint_is_the_constrained_mode():
    constraint = LinearConstraint(OPERATOR, TOTALS, scale="log")
    result = _model().fit(_frame(), engine="laplace", constraints=[constraint])

    oracle = minimize(
        _negative_log_posterior, np.zeros(CELLS + 1), jac=True, method="SLSQP",
        constraints={"type": "eq", "fun": lambda z: OPERATOR @ np.exp(_eta(z)) - TOTALS},
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert oracle.success
    np.testing.assert_allclose(result.predictive_mean, _eta(oracle.x), atol=1e-5)
    np.testing.assert_allclose(OPERATOR @ np.exp(result.predictive_mean), TOTALS, rtol=1e-8)


def test_poisson_log_observation_is_the_mode():
    observation = LinearObservation(TOTALS, OPERATOR, sigma=SIGMA, scale="log")
    result = _model().fit(_frame(), engine="laplace", observations=[observation])

    oracle = minimize(
        _negative_log_posterior, np.zeros(CELLS + 1), args=(True,), jac=True, method="BFGS",
        options={"gtol": 1e-9},
    )
    assert oracle.success
    np.testing.assert_allclose(result.predictive_mean, _eta(oracle.x), atol=1e-6)


def test_identity_constraint_holds_on_bernoulli():
    frame = _frame().assign(y=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, np.nan, np.nan])
    constraint = LinearConstraint(OPERATOR[:1], [0.5])
    result = _model(Bernoulli()).fit(frame, engine="laplace", constraints=[constraint])
    assert OPERATOR[0] @ result.predictive_mean == pytest.approx(0.5, abs=1e-9)
    np.testing.assert_allclose(
        result.predict(frame).predictive_mean, result.predictive_mean, atol=1e-12,
    )


def test_constraints_only_without_a_response_column():
    frame = _frame().drop(columns="y")
    constraint = LinearConstraint(OPERATOR, TOTALS, scale="log")
    result = _model().fit(frame, engine="laplace", constraints=[constraint])
    np.testing.assert_allclose(OPERATOR @ np.exp(result.predictive_mean), TOTALS, rtol=1e-8)


def test_estimated_observation_sigma_and_precision():
    observation = LinearObservation(
        TOTALS, OPERATOR, Hyperparameter("s", initial=0.5, lower=1e-2, upper=10.0), scale="log",
    )
    model = _model(precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2))
    result = model.fit(_frame(), engine="laplace", observations=[observation])
    assert set(result.hyperparameters) == {"s", "tau"}
    assert np.isfinite(result.log_marginal_likelihood)

    integrated = model.fit(
        _frame(), engine="laplace", observations=[observation], hyperparameters="integrate",
    )
    assert np.isfinite(integrated.log_marginal_likelihood)

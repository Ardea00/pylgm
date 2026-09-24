"""INLA model criteria on the original scale for projected (linear-observation) fits.

Projection divides every row by its standard deviation; the criteria must add
the Jacobian ``-log sigma_i`` back, or DIC/WAIC/CPO shift by ``sum log sigma_i``.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Gaussian, Hyperparameter, LGM, LinearObservation

T = 30


def _data():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=T)) * 0.3 + rng.normal(scale=2.0, size=T)
    return pd.DataFrame({"u": ["a"] * T, "t": range(T), "y": y})


def _model(sigma):
    rho = Hyperparameter("rho", initial=0.5, transform="logit")
    return LGM(response="y", likelihood=Gaussian(sigma),
               predictor=AR1("ar", "t", precision=1.0, rho=rho), panel=("u",), time="t")


def _assert_same_criteria(rows, projected):
    for name in ("dic", "dic_effective_parameters", "waic", "waic_effective_parameters",
                 "log_cpo_sum"):
        assert getattr(projected.criteria, name) == pytest.approx(
            getattr(rows.criteria, name), rel=1e-6), name
    np.testing.assert_allclose(projected.criteria.cpo, rows.criteria.cpo, rtol=1e-5)
    np.testing.assert_allclose(projected.criteria.pit, rows.criteria.pit, atol=1e-6)


def test_criteria_of_a_linear_observation_fit_match_the_row_fit():
    frame = _data()
    rows = _model(2.0).fit(frame, hyperparameters="integrate")
    projected = _model(2.0).fit(
        frame[["u", "t"]], observations=[LinearObservation(frame["y"], np.eye(T), sigma=2.0)],
        hyperparameters="integrate",
    )
    _assert_same_criteria(rows, projected)


def test_criteria_with_an_estimated_observation_sigma_match_the_row_fit():
    frame = _data()
    rows = _model(Hyperparameter("s", initial=1.5)).fit(frame, hyperparameters="integrate")
    projected = _model(1.0).fit(
        frame[["u", "t"]],
        observations=[LinearObservation(frame["y"], np.eye(T), Hyperparameter("s", initial=1.5))],
        hyperparameters="integrate",
    )
    assert projected.log_marginal_likelihood == pytest.approx(rows.log_marginal_likelihood)
    _assert_same_criteria(rows, projected)

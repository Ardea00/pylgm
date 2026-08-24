"""Pins the linear-predictor-variance convention for ``predictive_variance``.

``predictive_variance`` means ``Var(eta)`` -- the linear-predictor posterior
variance -- for every result type and for ``predict()``. Before this change,
``inference/gaussian.py`` folded the observation variance (sigma^2) into it,
while ``inference/laplace.py`` (and every non-Gaussian likelihood, which has
no sigma^2 to fold in) did not. ``GaussianResult.observation_variance`` now
carries what used to be folded in, so ``predictive_variance +
observation_variance`` reconstructs the old Gaussian value.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson
from test_result_surface import _gaussian_iid_integrate


def _frame():
    return pd.DataFrame({
        "y": [0.5, 1.5, 2.5, 3.5],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })


def _gaussian_result(sigma=0.5):
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Gaussian(sigma=sigma),
    )
    return model.fit(_frame())


def test_gaussian_predictive_variance_is_the_linear_predictor_variance():
    sigma = 0.5
    result = _gaussian_result(sigma)
    design = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0],
                       [1.0, 3.0, 1.0, 0.0], [1.0, 4.0, 0.0, 1.0]])
    expected = np.einsum("ij,jk,ik->i", design, result.covariance, design)
    assert np.allclose(result.predictive_variance, expected)
    # ...and specifically NOT the old value
    assert not np.allclose(result.predictive_variance, expected + sigma**2)


def test_observation_variance_reconstructs_the_previous_value():
    sigma = 0.5
    result = _gaussian_result(sigma)
    assert result.observation_variance == pytest.approx(sigma**2)
    old = result.predictive_variance + result.observation_variance
    assert np.all(old > result.predictive_variance)


def test_predict_matches_the_fit_rows_under_the_new_convention():
    result = _gaussian_result()
    prediction = result.predict(_frame())
    assert np.allclose(prediction.predictive_variance, result.predictive_variance)


def test_laplace_predictive_variance_is_unchanged():
    frame = _frame().assign(y=[2.0, 3.0, 5.0, 4.0])
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")
    design = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0],
                       [1.0, 3.0, 1.0, 0.0], [1.0, 4.0, 0.0, 1.0]])
    expected = np.einsum("ij,jk,ik->i", design, result.covariance, design)
    assert np.allclose(result.predictive_variance, expected)


def test_integrated_gaussian_criteria_are_unchanged_by_the_convention_switch():
    """DIC/WAIC/log-CPO must not move: they are recomputed independently of
    ``predictive_variance`` in ``optimization/inla.py``'s ``_model_criteria``,
    which uses ``fit.mean``/``fit.covariance`` directly rather than the
    per-hyperparameter conditional's ``predictive_variance`` field. These
    literals were read from ``git show HEAD:tests/inference/result_surface_baseline.json``
    (key ``gaussian_iid_integrate.methods.criteria``) BEFORE this change, so
    the assertion holds regardless of whether the baseline file itself has
    since been regenerated.
    """
    model, frame, fit_kwargs = _gaussian_iid_integrate()
    result = model.fit(frame, **fit_kwargs)
    criteria = result.criteria
    assert criteria.dic == pytest.approx(-1.80770112057)
    assert criteria.waic == pytest.approx(-3.32670111458)
    assert criteria.log_cpo_sum == pytest.approx(1.37245533855)
    assert criteria.dic_effective_parameters == pytest.approx(3.64255561446)
    assert criteria.waic_effective_parameters == pytest.approx(1.67917813323)

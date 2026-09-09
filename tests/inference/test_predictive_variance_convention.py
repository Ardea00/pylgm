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
    """DIC/WAIC/log-CPO are recomputed independently of ``predictive_variance``
    in ``optimization/inla.py``'s ``_model_criteria``, which uses
    ``fit.mean``/``fit.covariance`` directly rather than the per-hyperparameter
    conditional's ``predictive_variance`` field.

    The literals are a snapshot, so they track the *integrator* -- criteria are
    an integral over the hyperparameter grid, and a deliberate change to how that
    grid is weighted moves them. They were last updated when
    ``log_density_drop`` rose from 2.5 to 12, and that update was verified rather
    than accepted: against an integration converged by refinement (fine step, no
    truncation), the previous values were off by 0.42 (DIC), 0.53 (WAIC) and 0.29
    (log-CPO), and these are off by 0.0063, 0.0080 and 0.0047 -- roughly sixty
    times closer. A snapshot updated without that check would be a test that
    cannot fail.
    """
    model, frame, fit_kwargs = _gaussian_iid_integrate()
    result = model.fit(frame, **fit_kwargs)
    criteria = result.criteria
    assert criteria.dic == pytest.approx(-1.37765973045)
    assert criteria.waic == pytest.approx(-2.79337410463)
    assert criteria.log_cpo_sum == pytest.approx(1.08009148250)
    assert criteria.dic_effective_parameters == pytest.approx(3.76442820499)
    assert criteria.waic_effective_parameters == pytest.approx(1.84917528791)

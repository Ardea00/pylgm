import numpy as np
import pandas as pd
import pytest

from pylgm import Copy, Fixed, IID, LGM, Poisson
from pylgm.parameters import Hyperparameter


def _data(seed=9, n=60):
    rng = np.random.default_rng(seed)
    levels = [f"L{k}" for k in range(10)]
    return pd.DataFrame({
        "i": [levels[k % 10] for k in range(n)],
        "j": [levels[(k * 3) % 10] for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


def _fitted(scale):
    frame = _data()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=scale),
    )
    return frame, model.fit(frame, engine="laplace")


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means_fixed_scale():
    frame, result = _fitted(2.0)
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means_estimated_scale():
    frame, result = _fitted(Hyperparameter("beta", initial=1.0))
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_prediction_uses_the_copy_index_from_new_data():
    """The copy's index is data. Permuting it on new rows must move the
    prediction; a predict path that rebuilt only the base design would not."""
    frame, result = _fitted(2.0)
    permuted = frame.assign(j=frame["j"].values[::-1])
    assert not np.allclose(
        result.predict(frame).predictive_mean,
        result.predict(permuted).predictive_mean,
    )


def test_predict_rejects_new_data_missing_the_copy_index():
    frame, result = _fitted(2.0)
    with pytest.raises(ValueError, match="j"):
        result.predict(frame.drop(columns=["j"]))


def test_predict_rejects_an_unseen_copy_level():
    frame, result = _fitted(2.0)
    unseen = frame.copy()
    unseen.loc[0, "j"] = "ZZ"
    with pytest.raises(ValueError, match="ZZ"):
        result.predict(unseen)


def _simulate_with_true_beta(seed=31, n_levels=12, n_rows=600, true_beta=1.8, sigma_u=0.5):
    """A fixture where beta is actually identifiable from the data, unlike
    ``_data`` above (whose response is unrelated to u/beta) -- needed so the
    integrate-path marginal mean below materially differs from the initial
    guess. Mirrors ``tests/test_copy_model.py``'s ``_simulate``."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, sigma_u, n_levels)
    i = rng.integers(0, n_levels, n_rows)
    j = rng.integers(0, n_levels, n_rows)
    eta = 0.3 + u[i] + true_beta * u[j]
    return pd.DataFrame({
        "i": [f"L{k}" for k in i],
        "j": [f"L{k}" for k in j],
        "y": rng.poisson(np.exp(eta)).astype(float),
        "row": range(n_rows),
    })


def test_predicting_after_integrate_uses_the_marginal_scale_not_the_initial():
    """Finding C1: ``result.hyperparameters`` is None on the INLA integrate
    path (the point estimate lives in ``result.hyperparameter_marginals()``
    instead), so ``predict()`` used to fall back to the copy scale's
    *initial* value with no error or warning -- freezing the wrong scale into
    every predicted row.
    """
    frame = _simulate_with_true_beta()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1")
        + IID("u", index="i", precision=1.0 / 0.5**2)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    )
    result = model.fit(frame, engine="laplace", hyperparameters="integrate")
    assert result.hyperparameters is None  # confirms this is the integrate path

    beta_mean = result.hyperparameter_marginals()["beta"].mean[0]
    assert beta_mean != pytest.approx(1.0, rel=1e-2)  # materially unlike the initial guess

    predicted = result.predict(frame).predictive_mean
    fitted = result.predictive_mean
    rel = np.abs(predicted - fitted) / np.maximum(np.abs(fitted), 1e-12)
    # The residual gap here is the genuine plug-in-vs-mixture approximation
    # for a hyperparameter-dependent design (the same approximation already
    # accepted for midas_parametric) -- not the ~65x blowup the initial-value
    # bug produced.
    assert rel.max() < 1.0


def test_no_copy_integrate_baseline_round_trips_to_machine_precision():
    """Control for the test above: a hyperparameter-dependent design (Copy)
    only round-trips approximately under integrate; a design with no
    hyperparameter dependence must still round-trip exactly, confirming the
    0.40-ish gap above is real design dependence and not a leftover bug."""
    frame = _data()
    model = LGM(
        response="y", likelihood=Poisson(),
        # A Hyperparameter is required to exercise the integrate path, but an
        # IID precision affects only the covariance, not the design -- so the
        # rebuilt prediction design here has no hyperparameter dependence at
        # all, unlike the Copy scale case above.
        predictor=Fixed("1") + IID("u", index="i", precision=Hyperparameter("tau", initial=1.0)),
    )
    result = model.fit(frame, engine="laplace", hyperparameters="integrate")
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-8, abs=1e-8
    )


def test_copy_over_a_datetime_index_compiles_and_predicts():
    """Finding I1: ``_copy_incidence``/``_copied_block`` used ``.astype(str)``
    while every other index-to-label site (``_structured_block``, e.g.) uses
    ``.map(str)``. On ``datetime64`` these disagree -- ``astype(str)`` gives
    ``'2020-01-02'`` where the block's labels are the ``.map(str)`` form
    ``'2020-01-02 00:00:00'`` -- so a copy over a valid, present datetime
    level was rejected with a CompilationError claiming the target lacked a
    level it actually has.
    """
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    n = 60
    i = rng.integers(0, 10, n)
    j = rng.integers(0, 10, n)
    frame = pd.DataFrame({
        "i": dates[i],
        "j": dates[j],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=2.0),
    )
    result = model.fit(frame, engine="laplace")
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-8, abs=1e-8
    )

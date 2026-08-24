import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson
from pylgm.priors import PCPrecision


def _frame():
    return pd.DataFrame({
        "y": [0.5, 1.5, 2.5, 3.5],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })


def _gaussian_model():
    return LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Gaussian(sigma=0.5),
    )


def test_predicting_the_fit_rows_reproduces_the_fit_predictions():
    # The strongest oracle: predict() on the training frame must reproduce
    # exactly what fit already reported for those rows.
    frame = _frame()
    result = _gaussian_model().fit(frame)
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)
    assert np.allclose(prediction.predictive_variance, result.predictive_variance)


def test_predicts_new_covariate_values_on_known_levels():
    frame = _frame()
    result = _gaussian_model().fit(frame)
    new = pd.DataFrame({"x": [10.0, -3.0], "region": ["b", "a"]})
    prediction = result.predict(new)
    assert prediction.predictive_mean.shape == (2,)
    assert np.all(np.isfinite(prediction.predictive_mean))
    assert np.all(prediction.predictive_variance > 0)


def test_poisson_fit_row_round_trip():
    frame = pd.DataFrame({
        "y": [2.0, 3.0, 5.0, 4.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + x") + IID("region", index="region", precision=2.0),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)
    assert np.allclose(prediction.fitted_mean, result.fitted_mean)


def test_categorical_subset_reproduces_the_fitted_encoding():
    # New data containing only ONE of the fitted categorical levels must still
    # produce the fitted columns. Re-running model_matrix here would not.
    frame = pd.DataFrame({
        "y": [0.5, 1.5, 2.5, 3.5],
        "g": ["a", "b", "c", "a"],
        "region": ["a", "b", "a", "b"],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + g") + IID("region", index="region", precision=2.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame)
    only_b = pd.DataFrame({"g": ["b"], "region": ["a"]})
    prediction = result.predict(only_b)
    assert prediction.predictive_mean.shape == (1,)
    assert np.all(np.isfinite(prediction.predictive_mean))


def test_unseen_level_raises_and_points_at_the_nan_workflow():
    result = _gaussian_model().fit(_frame())
    with pytest.raises(ValueError, match="NaN response"):
        result.predict(pd.DataFrame({"x": [1.0], "region": ["zzz"]}))


def test_spatial_effect_predicts_a_graph_node_absent_from_the_fit_data():
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    frame = pd.DataFrame({"y": [0.5, 1.5], "region": ["a", "b"]})
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", precision=1.0, graph=graph),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame)
    # "c" is a graph node, so it has a fitted latent component even though no
    # row referenced it.
    prediction = result.predict(pd.DataFrame({"region": ["c"]}))
    assert np.all(np.isfinite(prediction.predictive_mean))


def test_inla_result_predicts_with_the_integrated_posterior():
    frame = _frame()
    model = LGM(
        response="y",
        predictor=Fixed("1 + x")
        + IID("region", index="region",
              precision=Hyperparameter("region.precision", initial=1.0,
                                       prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(frame, hyperparameters="integrate")
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)


def test_offset_is_read_from_the_new_rows():
    frame = pd.DataFrame({
        "y": [2.0, 3.0, 5.0, 4.0],
        "region": ["a", "b", "a", "b"],
        "logexp": [0.0, 0.1, 0.2, 0.3],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1") + IID("region", index="region", precision=2.0),
        likelihood=Poisson(),
        offset="logexp",
    )
    result = model.fit(frame, engine="laplace")
    low = result.predict(pd.DataFrame({"region": ["a"], "logexp": [0.0]}))
    high = result.predict(pd.DataFrame({"region": ["a"], "logexp": [1.0]}))
    assert high.predictive_mean[0] == pytest.approx(low.predictive_mean[0] + 1.0)


def test_integrated_nonlinear_link_fitted_mean_is_a_documented_approximation():
    # integrate_inla mixes the TRANSFORMED per-theta values,
    #   fitted = sum_k w_k * g(mu_k, s_k),
    # while predict() transforms the INTEGRATED moments,
    #   fitted = g(sum_k w_k mu_k, Var_mixture).
    # For a non-linear g (the Poisson lognormal mean) Jensen makes these differ.
    # eta mean/variance stay exact; only the response transform is approximate.
    frame = pd.DataFrame({
        "y": [2.0, 3.0, 5.0, 4.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "region": ["a", "b", "a", "b"],
    })
    model = LGM(
        response="y",
        predictor=Fixed("1 + x")
        + IID("region", index="region",
              precision=Hyperparameter("region.precision", initial=1.0,
                                       prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace", hyperparameters="integrate")
    prediction = result.predict(frame)

    # the linear-predictor quantities remain exact
    assert np.allclose(prediction.predictive_mean, result.predictive_mean, atol=1e-12)
    assert np.allclose(prediction.predictive_variance, result.predictive_variance, atol=1e-12)

    # the response scale agrees closely but NOT exactly, by construction
    assert not np.allclose(prediction.fitted_mean, result.fitted_mean, atol=1e-12)
    assert np.allclose(prediction.fitted_mean, result.fitted_mean, rtol=5e-3)

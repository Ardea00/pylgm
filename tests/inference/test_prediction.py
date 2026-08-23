import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix

from pylgm.inference.prediction import Prediction, PredictionContext, predict_from
from pylgm.likelihoods import CompiledGaussian


def _context(frame, formula="1 + x"):
    spec = model_matrix(formula, frame).model_spec
    return PredictionContext(
        entries=(("fixed", spec), ("structured", ("region", "region", ("a", "b")))),
        likelihood=CompiledGaussian(0.5),
        offset=None,
        width=len(spec.column_names) + 2,
    )


def test_predicts_linear_predictor_mean_and_variance():
    frame = pd.DataFrame({"x": [1.0, 2.0], "region": ["a", "b"]})
    context = _context(frame)
    width = context.width
    mean = np.arange(1.0, width + 1.0)
    covariance = np.eye(width) * 0.25
    prediction = predict_from(context, mean, covariance, frame)
    assert isinstance(prediction, Prediction)

    # design: [1, x, region==a, region==b]
    design = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 2.0, 0.0, 1.0]])
    expected_mean = design @ mean
    expected_var = np.einsum("ij,jk,ik->i", design, covariance, design) + 0.5**2
    assert np.allclose(prediction.predictive_mean, expected_mean)
    assert np.allclose(prediction.predictive_variance, expected_var)


def test_to_frame_is_aligned_and_has_sd():
    frame = pd.DataFrame({"x": [1.0, 2.0], "region": ["a", "b"]}, index=[10, 20])
    context = _context(frame)
    prediction = predict_from(context, np.ones(context.width), np.eye(context.width), frame)
    table = prediction.to_frame()
    assert list(table.index) == [10, 20]
    assert list(table.columns) == ["predictive_mean", "predictive_sd", "fitted_mean"]
    assert np.allclose(table["predictive_sd"], np.sqrt(prediction.predictive_variance))


def test_unseen_level_names_the_block_and_the_workflow():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    unseen = pd.DataFrame({"x": [1.0], "region": ["zzz"]})
    with pytest.raises(ValueError, match="region"):
        predict_from(context, np.ones(context.width), np.eye(context.width), unseen)
    with pytest.raises(ValueError, match="NaN"):
        predict_from(context, np.ones(context.width), np.eye(context.width), unseen)


def test_missing_index_column_raises():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    with pytest.raises(ValueError, match="region"):
        predict_from(context, np.ones(context.width), np.eye(context.width),
                     pd.DataFrame({"x": [1.0]}))


def test_empty_frame_raises():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    with pytest.raises(ValueError, match="empty"):
        predict_from(context, np.ones(context.width), np.eye(context.width),
                     frame.iloc[:0])


def test_prediction_arrays_are_read_only():
    frame = pd.DataFrame({"x": [1.0], "region": ["a"]})
    context = _context(frame)
    prediction = predict_from(context, np.ones(context.width), np.eye(context.width), frame)
    with pytest.raises(ValueError):
        prediction.predictive_mean[0] = 5.0

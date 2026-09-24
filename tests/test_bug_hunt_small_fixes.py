"""Small guards from the bug-hunt pass."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import AR1, RW1, RW2, Gaussian, LGM, Seasonal


@pytest.mark.parametrize("effect", [
    RW1("trend", "t"), RW2("trend", "t"), AR1("trend", "t", rho=0.5),
    Seasonal("season", "t", period=2),
])
def test_an_irregular_numeric_index_warns(effect):
    """RW/AR1/Seasonal relate consecutive levels; a gap would silently be one step."""
    frame = pd.DataFrame({"t": [0, 1, 2, 5, 6, 7], "y": np.arange(6.0)})
    model = LGM(response="y", likelihood=Gaussian(1.0), predictor=effect)
    with pytest.warns(UserWarning, match="not evenly spaced"):
        model.fit(frame)


def test_a_regular_index_does_not_warn():
    frame = pd.DataFrame({"t": [0, 2, 4, 6, 8], "y": np.arange(5.0)})
    model = LGM(response="y", likelihood=Gaussian(1.0), predictor=RW1("trend", "t"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(frame)


@pytest.mark.parametrize("weights", [np.ones((1, 3)), np.array([[1.0, np.nan, 0.0, 0.0]])])
def test_sparse_linear_combinations_validate_weights_like_dense(weights, monkeypatch):
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    frame = pd.DataFrame({"t": range(4), "y": np.arange(4.0)})
    result = LGM(response="y", likelihood=Gaussian(1.0), predictor=RW1("trend", "t")).fit(frame)
    with pytest.raises(ValueError):
        result.linear_combinations(weights)

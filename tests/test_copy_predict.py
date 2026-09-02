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

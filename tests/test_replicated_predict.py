import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Replicated


def _data(seed=13, n=72):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": [f"t{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


def _fitted():
    frame = _data()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(IID("u", index="t", precision=1.0), over="firm"),
    )
    return frame, model.fit(frame, engine="laplace")


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means():
    frame, result = _fitted()
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_prediction_uses_the_replicate_column_from_new_data():
    """Reassigning rows to different replicates must move the prediction; a
    predict path that ignored the replicate column would not."""
    frame, result = _fitted()
    rotated = frame.assign(firm=frame["firm"].map({"f0": "f1", "f1": "f2",
                                                   "f2": "f3", "f3": "f0"}))
    assert not np.allclose(
        result.predict(frame).predictive_mean,
        result.predict(rotated).predictive_mean,
    )


def test_predict_rejects_new_data_missing_the_replicate_column():
    frame, result = _fitted()
    with pytest.raises(ValueError, match="firm"):
        result.predict(frame.drop(columns=["firm"]))


def test_predict_rejects_an_unseen_replicate():
    frame, result = _fitted()
    unseen = frame.copy()
    unseen.loc[0, "firm"] = "f99"
    with pytest.raises(ValueError, match="f99"):
        result.predict(unseen)


def test_predict_rejects_an_unseen_level():
    frame, result = _fitted()
    unseen = frame.copy()
    unseen.loc[0, "t"] = "t99"
    with pytest.raises(ValueError, match="t99"):
        result.predict(unseen)


def test_round_trip_holds_when_sorted_and_first_seen_level_order_differ():
    """Levels named t1..t11 sort as t1, t10, t11, t2, ... so the fitted label
    order and the frame's first-seen order genuinely diverge. This project has
    shipped two silent misalignment bugs of exactly this class."""
    rng = np.random.default_rng(21)
    n = 110
    frame = pd.DataFrame({
        "t": [f"t{(k % 11) + 1}" for k in range(n)],
        "firm": [f"f{k % 5}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(IID("u", index="t", precision=1.0), over="firm"),
    ).fit(frame, engine="laplace")
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )

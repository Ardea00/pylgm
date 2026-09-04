import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Replicated, Weighted


def _data(seed=13, n=72):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "t": [f"t{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


def _weighted_data(seed=13, n=72):
    """Same shape as _data(), plus a `z` column that varies per row so a predict
    path that silently drops the weighting is caught rather than hidden by a
    constant weight."""
    frame = _data(seed=seed, n=n)
    rng = np.random.default_rng(seed + 1)
    frame["z"] = rng.uniform(0.5, 2.5, n)
    return frame


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


def test_replicated_weighted_round_trips_on_the_fit_rows():
    """Replicated(Weighted(...)) predict path must apply the weighting: a
    dropped `by` column would still round-trip trivially at fit time (both
    sides use the same fitted linear predictor by construction, so this alone
    would not catch it) -- see test_replicated_weighted_predict_uses_z_from_new_data
    below for the test that actually exercises the weight column."""
    frame = _weighted_data()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            Weighted(IID("u", index="t", precision=1.0), by="z"), over="firm"
        ),
    )
    result = model.fit(frame, engine="laplace")
    assert result.predict(frame).predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_replicated_and_weighted_orderings_agree_at_predict_time():
    """Predict-time counterpart of test_replicated_commutes_with_weighted
    (tests/test_replicated_compile.py): the spec asserts the two orderings
    commute, and that must hold at predict() time too, not only at compile
    time. Before the fix, Replicated(Weighted(...)) silently dropped the
    weighting here even though the compiled designs were identical."""
    frame = _weighted_data()
    inner_first = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            Weighted(IID("u", index="t", precision=1.0), by="z"), over="firm"
        ),
    ).fit(frame, engine="laplace")
    outer_first = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            Replicated(IID("u", index="t", precision=1.0), over="firm"), by="z"
        ),
    ).fit(frame, engine="laplace")
    assert inner_first.predict(frame).predictive_mean == pytest.approx(
        outer_first.predict(frame).predictive_mean, rel=1e-9, abs=1e-9
    )


def test_replicated_weighted_predict_uses_z_from_new_data():
    """A predict path that baked the weights in at fit time (instead of reading
    `by` from new_data) would not move when z changes."""
    frame = _weighted_data()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            Weighted(IID("u", index="t", precision=1.0), by="z"), over="firm"
        ),
    )
    result = model.fit(frame, engine="laplace")
    changed = frame.copy()
    changed["z"] = changed["z"] * 3.0 + 1.0
    assert not np.allclose(
        result.predict(frame).predictive_mean,
        result.predict(changed).predictive_mean,
    )


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

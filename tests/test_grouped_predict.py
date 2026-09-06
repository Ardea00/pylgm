import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Fixed, Gaussian, Grouped, IID, LGM, Weighted,
)

GRAPH = {"r1": ["r2"], "r2": ["r1", "r3"], "r3": ["r2"]}


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        {"region": r, "t": f"t{t}", "z": 1.0 + 0.1 * t}
        for r in ("r1", "r2", "r3") for t in range(6)
    ]
    frame = pd.DataFrame(rows)
    frame["y"] = rng.standard_normal(len(frame))
    return frame


def _fit(predictor, frame):
    return LGM(response="y", predictor=predictor, likelihood=Gaussian(sigma=0.5)).fit(frame)


def test_prediction_round_trips_on_the_fit_rows():
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(IID("u", index="t"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    predicted = result.predict(frame).predictive_mean
    assert np.allclose(predicted, result.predictive_mean, rtol=1e-12, atol=1e-12)


def test_prediction_round_trips_with_weights_inside_the_group():
    """The form slice 3 shipped broken: Replicated(Weighted(...)) dropped weights."""
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(Weighted(IID("u", index="t"), by="z"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    predicted = result.predict(frame).predictive_mean
    assert np.allclose(predicted, result.predictive_mean, rtol=1e-12, atol=1e-12)


def test_prediction_round_trips_with_weights_outside_the_group():
    frame = _frame()
    result = _fit(
        Fixed("1") + Weighted(
            Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
            by="z",
        ),
        frame,
    )
    predicted = result.predict(frame).predictive_mean
    assert np.allclose(predicted, result.predictive_mean, rtol=1e-12, atol=1e-12)


def test_a_subset_of_groups_still_scores():
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(IID("u", index="t"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    subset = frame[frame["region"] == "r2"]
    assert np.isfinite(result.predict(subset).predictive_mean).all()


def test_an_unseen_level_is_rejected():
    frame = _frame()
    result = _fit(
        Fixed("1") + Grouped(IID("u", index="t"), over="region",
                             structure=BesagStructure(GRAPH)),
        frame,
    )
    unseen = frame.head(3).copy()
    unseen["t"] = "t99"
    with pytest.raises(ValueError, match="group/level"):
        result.predict(unseen)

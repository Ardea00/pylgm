import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Fixed, Gaussian, Grouped, IID, LGM, Weighted,
)
from pylgm.parameters import Hyperparameter

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
    predicted = result.predict(subset).predictive_mean
    assert np.isfinite(predicted).all()
    # `isfinite` alone survives every mutation of the design, including zeroing
    # it, and so does comparing against the full frame's *prediction* -- both
    # sides would go through the same broken path. The fitted values do not:
    # they come from the fit, never from _design_block_for.
    fitted = result.predictive_mean[frame["region"].to_numpy() == "r2"]
    assert np.allclose(predicted, fitted, rtol=1e-12, atol=1e-12)


def test_family_path_applies_weighted_like_the_plain_path():
    """F4: with a Hyperparameter on the inner effect, ``Grouped(Weighted(...))``
    compiles through ``_grouped_family_block`` and its own re-application of
    the weighting (compiler.py, guarded by ``if isinstance(inner_spec,
    Weighted):`` right after the family-block loop), not through the plain
    ``_build_effect_block`` path that
    ``test_prediction_round_trips_with_weights_inside_the_group`` above
    exercises. Guarding that re-application with ``if False and ...`` drops
    the weighting silently -- the hyperparameter stays estimated and the
    shape is unchanged, only the spatially-varying coefficient itself goes
    missing. There is no live defect today: this pins that the plain and
    family-path designs agree, weight column and all.
    """
    from pylgm.compiler import _build_effect_block, compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame()
    frame["row"] = range(len(frame))
    plain, _ = _build_effect_block(
        Grouped(Weighted(IID("u", index="t", precision=1.0), by="z"),
                over="region", structure=BesagStructure(GRAPH)),
        frame,
    )
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.5),
        predictor=Fixed("1") + Grouped(
            Weighted(IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)), by="z"),
            over="region", structure=BesagStructure(GRAPH),
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    family = compile_family(model, panel)
    materialized = family.materialize({"tau": 1.0})
    family_block = [b for b in materialized.blocks if b.name == "u"][0]
    assert np.allclose(family_block.design.toarray(), plain.design.toarray())
    # Not a vacuous scaling-by-one comparison: z is non-constant across rows.
    assert frame["z"].nunique() > 1


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

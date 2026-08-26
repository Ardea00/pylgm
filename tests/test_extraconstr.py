"""Model-level linear constraints (A x = e), the label-keyed extraconstr equivalent."""

import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson
from pylgm.exceptions import CompilationError


def _labelled_mean(result) -> dict[str, float]:
    return dict(zip(result.labels, result.mean, strict=True))


def _gaussian_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["a", "b", "c", "a", "b", "c"],
            "time": [1, 1, 1, 2, 2, 2],
            "y": [10.0, 2.0, 5.0, 11.0, 3.0, 6.0],
        }
    )


def _model(constraints=()) -> LGM:
    return LGM(
        response="y",
        likelihood=Gaussian(1.0),
        predictor=Fixed("1") + IID("region", "region", 1.0),
        panel=("region",),
        time="time",
        constraints=constraints,
    )


def test_constraint_makes_two_effects_coincide_exact_gaussian():
    frame = _gaussian_frame()

    free = _labelled_mean(_model().fit(frame, engine="exact_gaussian"))
    assert abs(free["region:a"] - free["region:b"]) > 1.0  # unconstrained: clearly differ

    tied = _labelled_mean(
        _model([{"region:a": 1.0, "region:b": -1.0}]).fit(frame, engine="exact_gaussian")
    )
    assert tied["region:a"] == pytest.approx(tied["region:b"], abs=1e-9)


def test_constraint_holds_under_laplace_engine():
    frame = pd.DataFrame(
        {
            "region": ["a", "b", "c", "a", "b", "c"],
            "time": [1, 1, 1, 2, 2, 2],
            "y": [12, 3, 6, 14, 2, 7],
        }
    )
    model = LGM(
        response="y",
        likelihood=Poisson(),
        predictor=Fixed("1") + IID("region", "region", 1.0),
        panel=("region",),
        time="time",
        constraints=[{"region:a": 1.0, "region:c": -1.0}],
    )

    mean = _labelled_mean(model.fit(frame, engine="laplace"))
    assert mean["region:a"] == pytest.approx(mean["region:c"], abs=1e-8)


def test_nonzero_rhs_pins_effect_exact_gaussian():
    frame = _gaussian_frame()
    tied = _labelled_mean(
        _model([({"region:a": 1.0}, 2.5)]).fit(frame, engine="exact_gaussian")
    )
    assert tied["region:a"] == pytest.approx(2.5, abs=1e-9)


def test_nonzero_rhs_pins_sum_exact_gaussian():
    frame = _gaussian_frame()
    tied = _labelled_mean(
        _model([({"region:a": 1.0, "region:b": 1.0}, 4.0)]).fit(frame, engine="exact_gaussian")
    )
    assert tied["region:a"] + tied["region:b"] == pytest.approx(4.0, abs=1e-9)


def test_nonzero_rhs_holds_under_laplace_engine():
    frame = pd.DataFrame(
        {
            "region": ["a", "b", "c", "a", "b", "c"],
            "time": [1, 1, 1, 2, 2, 2],
            "y": [12, 3, 6, 14, 2, 7],
        }
    )
    model = LGM(
        response="y",
        likelihood=Poisson(),
        predictor=Fixed("1") + IID("region", "region", 1.0),
        panel=("region",),
        time="time",
        constraints=[({"region:a": 1.0}, 0.3)],
    )
    mean = _labelled_mean(model.fit(frame, engine="laplace"))
    assert mean["region:a"] == pytest.approx(0.3, abs=1e-8)


def test_unknown_label_is_rejected_at_compile():
    model = _model([{"region:a": 1.0, "region:zzz": -1.0}])
    with pytest.raises(CompilationError, match="unknown latent label"):
        model.fit(_gaussian_frame(), engine="exact_gaussian")


@pytest.mark.parametrize(
    "bad",
    [
        [{}],  # empty row
        [{"region:a": 0.0}],  # all-zero coefficients
        [{"region:a": float("inf")}],  # non-finite
        [{"region:a": "x"}],  # non-numeric coefficient
        [["region:a", 1.0]],  # not a mapping
        [({"region:a": 1.0}, "x")],  # non-numeric rhs
        [({"region:a": 1.0}, float("inf"))],  # non-finite rhs
        [({"region:a": 1.0}, 1.0, 2.0)],  # over-long tuple
    ],
)
def test_malformed_constraint_is_rejected_at_construction(bad):
    with pytest.raises((ValueError, TypeError)):
        _model(bad)

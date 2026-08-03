import pandas as pd
import pytest

from pylgm.config.experiment import EvaluationConfig, ExperimentDataConfig, OriginConfig
from pylgm.evaluation import FoldDefinition, materialize_fold, persistence_predictions
from pylgm.exceptions import FoldConstructionError


DATA = ExperimentDataConfig(time="month", response="y", panel=("region",))


def evaluation(horizon: int) -> EvaluationConfig:
    return EvaluationConfig(horizons=(horizon,), origins=OriginConfig(last=1))


def test_persistence_uses_last_panel_response_and_exact_horizon_variance() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A"] * 5 + ["B"] * 5,
            "month": [0, 1, 2, 3, 4] * 2,
            "y": [1.0, 2.0, 4.0, 8.0, 99.0, 10.0, 10.0, 13.0, 13.0, 88.0],
        }
    )
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(2),
        FoldDefinition(origin=2, target=4, horizon=2),
    )

    predictions = persistence_predictions(fold)

    assert predictions[["region", "actual", "mean"]].to_dict("records") == [
        {"region": "A", "actual": 99.0, "mean": 4.0},
        {"region": "B", "actual": 88.0, "mean": 13.0},
    ]
    assert predictions["evaluation_mode"].eq("latest").all()
    # Exact h=2 pre-origin errors: (4 - 1)^2 and (13 - 10)^2.
    assert predictions["variance"].tolist() == [9.0, 9.0]


def test_persistence_mean_uses_last_available_response_not_origin_row() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A"] * 4,
            "month": [0, 1, 2, 3],
            "y": [1.0, 2.0, None, 9.0],
        }
    )
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(1),
        FoldDefinition(origin=2, target=3, horizon=1),
    )

    predictions = persistence_predictions(fold)

    assert predictions["mean"].tolist() == [2.0]
    assert predictions["variance"].tolist() == [1.0]


def test_zero_persistence_error_receives_positive_numerical_variance_floor() -> None:
    frame = pd.DataFrame({"region": ["A"] * 4, "month": range(4), "y": [2.0, 2.0, 2.0, 2.0]})
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(1),
        FoldDefinition(origin=2, target=3, horizon=1),
    )

    predictions = persistence_predictions(fold)

    assert predictions["variance"].tolist() == [1e-12]


def test_persistence_requires_history_for_each_target_panel() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A", "A", "A", "B"],
            "month": [0, 1, 2, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(1),
        FoldDefinition(origin=1, target=2, horizon=1),
    )

    with pytest.raises(FoldConstructionError, match="history"):
        persistence_predictions(fold)


def test_persistence_requires_pre_origin_residual_history_for_the_horizon() -> None:
    frame = pd.DataFrame({"region": ["A"] * 3, "month": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(2),
        FoldDefinition(origin=0, target=2, horizon=2),
    )

    with pytest.raises(FoldConstructionError, match="residual|variance"):
        persistence_predictions(fold)


def test_persistence_rejects_nonfinite_residual_history_instead_of_dropping_it() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A"] * 3 + ["B"] * 3,
            "month": [0, 1, 2] * 2,
            "y": [1.0, 2.0, 3.0, float("inf"), 1.0, 2.0],
        }
    )
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(1),
        FoldDefinition(origin=1, target=2, horizon=1),
    )

    with pytest.raises(FoldConstructionError, match="finite"):
        persistence_predictions(fold)


def test_persistence_result_is_isolated_from_fold_and_accessor_mutation() -> None:
    frame = pd.DataFrame({"region": ["A"] * 4, "month": range(4), "y": [1.0, 2.0, 4.0, 8.0]})
    fold = materialize_fold(
        frame,
        DATA,
        evaluation(1),
        FoldDefinition(origin=2, target=3, horizon=1),
    )
    predictions = persistence_predictions(fold)

    predictions.loc[:, "mean"] = -1.0
    exposed_target = fold.target_frame
    exposed_target.loc[:, "y"] = -2.0

    assert persistence_predictions(fold)["mean"].tolist() == [4.0]
    assert fold.target_frame["y"].tolist() == [8.0]

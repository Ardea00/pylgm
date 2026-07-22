from datetime import date

import pandas as pd
import pytest

from pylgm.config.experiment import (
    CovariateConfig,
    EvaluationConfig,
    ExperimentDataConfig,
    OriginConfig,
    WindowConfig,
)
from pylgm.evaluation import FoldDefinition, build_fold_definitions, materialize_fold
from pylgm.exceptions import CovariateAvailabilityError, FoldConstructionError


def data_config(**covariates: CovariateConfig) -> ExperimentDataConfig:
    return ExperimentDataConfig(
        time="month",
        response="y",
        panel=("region",),
        covariates=covariates,
    )


def evaluation(
    *,
    horizons: tuple[int, ...] = (1,),
    origins: OriginConfig | None = None,
    window: WindowConfig | None = None,
    mode: str = "latest",
) -> EvaluationConfig:
    return EvaluationConfig(
        mode=mode,
        horizons=horizons,
        origins=origins or OriginConfig(last=1),
        window=window or WindowConfig(),
    )


@pytest.fixture
def panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["A"] * 7,
            "month": list(range(7)),
            "y": [0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0],
        }
    )


def test_horizon_three_masks_every_response_after_origin(panel_frame: pd.DataFrame) -> None:
    definition = FoldDefinition(origin=3, target=6, horizon=3)

    fold = materialize_fold(panel_frame, data_config(), evaluation(horizons=(3,)), definition)

    assert fold.model_frame.loc[fold.model_frame.month > 3, "y"].isna().all()
    assert fold.target_frame["month"].eq(6).all()
    assert fold.target_frame["y"].tolist() == [60.0]
    assert fold.model_frame["month"].max() == 6
    assert fold.training_frame["month"].max() == 3


def test_builds_arbitrary_horizons_from_irregular_ordered_time_positions() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A"] * 5,
            "month": [date(2025, month, 1) for month in (1, 2, 4, 7, 11)],
            "y": range(5),
        }
    )
    config = evaluation(
        horizons=(1, 3),
        origins=OriginConfig(values=(date(2025, 1, 1), date(2025, 2, 1))),
    )

    definitions = build_fold_definitions(frame, data_config(), config)

    assert definitions == (
        FoldDefinition(date(2025, 1, 1), date(2025, 2, 1), 1),
        FoldDefinition(date(2025, 1, 1), date(2025, 7, 1), 3),
        FoldDefinition(date(2025, 2, 1), date(2025, 4, 1), 1),
        FoldDefinition(date(2025, 2, 1), date(2025, 11, 1), 3),
    )


def test_last_n_origins_are_selected_only_when_all_horizons_have_targets() -> None:
    frame = pd.DataFrame({"region": ["A"] * 6, "month": range(6), "y": range(6)})

    definitions = build_fold_definitions(
        frame,
        data_config(),
        evaluation(horizons=(1, 2), origins=OriginConfig(last=2)),
    )

    assert definitions == (
        FoldDefinition(2, 3, 1),
        FoldDefinition(2, 4, 2),
        FoldDefinition(3, 4, 1),
        FoldDefinition(3, 5, 2),
    )


@pytest.mark.parametrize(
    ("origins", "match"),
    [
        (OriginConfig(values=(4,)), "future"),
        (OriginConfig(values=(99,)), "origin"),
        (OriginConfig(last=5), "origins"),
    ],
)
def test_invalid_or_insufficient_origins_raise_typed_errors(
    origins: OriginConfig, match: str
) -> None:
    frame = pd.DataFrame({"region": ["A"] * 5, "month": range(5), "y": range(5)})

    with pytest.raises(FoldConstructionError, match=match):
        build_fold_definitions(
            frame, data_config(), evaluation(horizons=(2,), origins=origins)
        )


def test_incomparable_time_levels_raise_a_typed_error() -> None:
    frame = pd.DataFrame({"region": ["A", "A"], "month": [1, "2"], "y": [1, 2]})

    with pytest.raises(FoldConstructionError, match="order"):
        build_fold_definitions(frame, data_config(), evaluation())


def test_expanding_and_rolling_windows_keep_the_correct_training_levels(
    panel_frame: pd.DataFrame,
) -> None:
    definition = FoldDefinition(origin=4, target=6, horizon=2)

    expanding = materialize_fold(
        panel_frame, data_config(), evaluation(horizons=(2,)), definition
    )
    rolling = materialize_fold(
        panel_frame,
        data_config(),
        evaluation(horizons=(2,), window=WindowConfig(type="rolling", length=2)),
        definition,
    )

    assert expanding.training_frame["month"].tolist() == [0, 1, 2, 3, 4]
    assert rolling.training_frame["month"].tolist() == [3, 4]
    assert rolling.model_frame["month"].tolist() == [3, 4, 5, 6]


def test_materialize_requires_an_exact_positional_target(panel_frame: pd.DataFrame) -> None:
    with pytest.raises(FoldConstructionError, match="target|horizon"):
        materialize_fold(
            panel_frame,
            data_config(),
            evaluation(horizons=(2,)),
            FoldDefinition(origin=2, target=5, horizon=2),
        )


def test_latest_mode_rejects_duplicate_panel_time_keys() -> None:
    frame = pd.DataFrame(
        {"region": ["A", "A"], "month": [1, 1], "y": [1.0, 2.0]}
    )

    with pytest.raises(FoldConstructionError, match="duplicate"):
        materialize_fold(
            frame,
            data_config(),
            evaluation(),
            FoldDefinition(origin=0, target=1, horizon=1),
        )


def test_vintage_fold_uses_latest_release_available_at_origin() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A", "A", "A", "A", "A"],
            "month": [1, 2, 2, 3, 3],
            "release": [1, 1, 3, 1, 3],
            "y": [1.0, 2.1, 2.9, None, 3.9],
            "x": [10.0, 20.0, 29.0, 30.0, 39.0],
        }
    )
    data = ExperimentDataConfig(
        time="month",
        response="y",
        panel=("region",),
        vintage="release",
        covariates={"x": CovariateConfig(availability="known_future")},
    )
    config = evaluation(
        mode="vintage", horizons=(1,), origins=OriginConfig(values=(2,))
    )

    fold = materialize_fold(
        frame, data, config, FoldDefinition(origin=2, target=3, horizon=1)
    )

    assert fold.training_frame.loc[fold.training_frame.month.eq(2), "y"].item() == 2.1
    assert 2.9 not in fold.training_frame["y"].tolist()
    assert fold.model_frame.loc[fold.model_frame.month.eq(3), "x"].item() == 30.0
    assert fold.target_frame["y"].tolist() == [3.9]
    assert fold.target_frame["x"].tolist() == [39.0]


def test_vintage_truth_covariates_are_never_copied_into_the_origin_snapshot() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A", "A", "A"],
            "month": [1, 2, 2],
            "release": [1, 1, 2],
            "y": [1.0, None, 2.0],
            "x": [10.0, None, 999.0],
        }
    )
    data = ExperimentDataConfig(
        time="month",
        response="y",
        panel=("region",),
        vintage="release",
        covariates={"x": CovariateConfig(availability="known_future")},
    )

    with pytest.raises(CovariateAvailabilityError, match="x"):
        materialize_fold(
            frame,
            data,
            evaluation(mode="vintage"),
            FoldDefinition(origin=1, target=2, horizon=1),
        )


def test_every_truth_target_coordinate_must_exist_in_origin_snapshot() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A", "A"],
            "month": [1, 2],
            "release": [1, 2],
            "y": [1.0, 2.0],
        }
    )
    data = ExperimentDataConfig(
        time="month", response="y", panel=("region",), vintage="release"
    )

    with pytest.raises(FoldConstructionError, match="coordinate"):
        materialize_fold(
            frame,
            data,
            evaluation(mode="vintage"),
            FoldDefinition(origin=1, target=2, horizon=1),
        )


def test_duplicate_vintage_release_for_a_coordinate_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "region": ["A", "A", "A"],
            "month": [1, 1, 2],
            "release": [1, 1, 1],
            "y": [1.0, 1.1, 2.0],
        }
    )
    data = ExperimentDataConfig(
        time="month", response="y", panel=("region",), vintage="release"
    )

    with pytest.raises(FoldConstructionError, match="duplicate"):
        materialize_fold(
            frame,
            data,
            evaluation(mode="vintage"),
            FoldDefinition(origin=1, target=2, horizon=1),
        )


@pytest.mark.parametrize(
    ("availability", "lag", "horizon", "is_valid"),
    [
        ("known_future", None, 2, True),
        ("observed_with_lag", 2, 2, True),
        ("observed_with_lag", 1, 2, False),
        ("unavailable_future", None, 1, False),
    ],
)
def test_target_covariate_availability_is_enforced(
    availability: str, lag: int | None, horizon: int, is_valid: bool
) -> None:
    frame = pd.DataFrame(
        {
            "region": ["A"] * 4,
            "month": [0, 1, 2, 3],
            "y": [0.0, 1.0, 2.0, 3.0],
            "x": [10.0, 11.0, 12.0, 13.0],
        }
    )
    data = data_config(
        x=CovariateConfig(availability=availability, lag=lag)  # type: ignore[arg-type]
    )
    definition = FoldDefinition(origin=1, target=1 + horizon, horizon=horizon)

    if is_valid:
        fold = materialize_fold(frame, data, evaluation(horizons=(horizon,)), definition)
        assert fold.model_frame.loc[fold.model_frame.month.eq(1 + horizon), "x"].item()
    else:
        with pytest.raises(CovariateAvailabilityError, match="x"):
            materialize_fold(frame, data, evaluation(horizons=(horizon,)), definition)


@pytest.mark.parametrize("case", ["missing_column", "missing_target_value"])
def test_missing_target_covariates_raise_typed_errors(case: str) -> None:
    frame = pd.DataFrame(
        {"region": ["A"] * 3, "month": [0, 1, 2], "y": [0.0, 1.0, 2.0]}
    )
    if case == "missing_target_value":
        frame["x"] = [1.0, 2.0, None]

    with pytest.raises(CovariateAvailabilityError, match="x"):
        materialize_fold(
            frame,
            data_config(x=CovariateConfig(availability="known_future")),
            evaluation(),
            FoldDefinition(origin=1, target=2, horizon=1),
        )


def test_materialization_does_not_mutate_source_or_expose_retained_frames(
    panel_frame: pd.DataFrame,
) -> None:
    original = panel_frame.copy(deep=True)
    fold = materialize_fold(
        panel_frame,
        data_config(),
        evaluation(horizons=(2,)),
        FoldDefinition(origin=2, target=4, horizon=2),
    )

    panel_frame.loc[:, "y"] = -1.0
    exposed = fold.model_frame
    exposed.loc[:, "y"] = -2.0

    pd.testing.assert_frame_equal(original, pd.DataFrame({
        "region": ["A"] * 7,
        "month": list(range(7)),
        "y": [0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0],
    }))
    assert fold.model_frame.loc[fold.model_frame.month.le(2), "y"].tolist() == [0.0, 1.0, 2.0]

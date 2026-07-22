from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from pylgm.config.experiment import EvaluationConfig, ExperimentDataConfig
from pylgm.evaluation.availability import validate_future_covariates
from pylgm.exceptions import FoldConstructionError


def _copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for position, dtype in enumerate(result.dtypes):
        if is_object_dtype(dtype):
            result.iloc[:, position] = result.iloc[:, position].map(copy.deepcopy)
    result.attrs = copy.deepcopy(frame.attrs)
    return result


@dataclass(frozen=True)
class FoldDefinition:
    origin: object
    target: object
    horizon: int


@dataclass(frozen=True, init=False)
class FoldData:
    definition: FoldDefinition
    _training_frame: pd.DataFrame = field(repr=False)
    _model_frame: pd.DataFrame = field(repr=False)
    _target_frame: pd.DataFrame = field(repr=False)
    _history_frame: pd.DataFrame = field(repr=False)
    _panel: tuple[str, ...] = field(repr=False)
    _time: str = field(repr=False)
    _response: str = field(repr=False)
    _levels: tuple[object, ...] = field(repr=False)

    def __init__(
        self,
        definition: FoldDefinition,
        training_frame: pd.DataFrame,
        model_frame: pd.DataFrame,
        target_frame: pd.DataFrame,
        *,
        history_frame: pd.DataFrame | None = None,
        panel: tuple[str, ...] = (),
        time: str = "",
        response: str = "",
        levels: tuple[object, ...] = (),
    ) -> None:
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "_training_frame", _copy_frame(training_frame))
        object.__setattr__(self, "_model_frame", _copy_frame(model_frame))
        object.__setattr__(self, "_target_frame", _copy_frame(target_frame))
        history = training_frame if history_frame is None else history_frame
        object.__setattr__(self, "_history_frame", _copy_frame(history))
        object.__setattr__(self, "_panel", tuple(panel))
        object.__setattr__(self, "_time", time)
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "_levels", tuple(levels))

    @property
    def training_frame(self) -> pd.DataFrame:
        return _copy_frame(self._training_frame)

    @property
    def model_frame(self) -> pd.DataFrame:
        return _copy_frame(self._model_frame)

    @property
    def target_frame(self) -> pd.DataFrame:
        return _copy_frame(self._target_frame)


def _required_columns(data: ExperimentDataConfig) -> tuple[str, ...]:
    vintage = () if data.vintage is None else (data.vintage,)
    return (*data.panel, data.time, data.response, *vintage)


def _validate_source(
    frame: pd.DataFrame, data: ExperimentDataConfig, evaluation: EvaluationConfig
) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise FoldConstructionError("evaluation frame column labels must be unique")
    missing = sorted(set(_required_columns(data)).difference(frame.columns))
    if missing:
        raise FoldConstructionError(f"evaluation frame is missing columns: {missing}")
    keys = [*data.panel, data.time]
    if frame.loc[:, keys].isna().any(axis=None):
        raise FoldConstructionError("panel/time coordinates must not contain null values")
    if evaluation.mode == "latest":
        if data.vintage is not None:
            raise FoldConstructionError("latest mode cannot use a vintage column")
        if frame.duplicated(keys).any():
            raise FoldConstructionError("latest-mode input contains duplicate panel/time keys")
    else:
        if data.vintage is None:
            raise FoldConstructionError("vintage mode requires a vintage column")
        if frame[data.vintage].isna().any():
            raise FoldConstructionError("vintage releases must not contain null values")
        if frame.duplicated([*keys, data.vintage]).any():
            raise FoldConstructionError("duplicate vintage release for a panel/time coordinate")
    return _copy_frame(frame)


def _ordered_levels(frame: pd.DataFrame, time: str) -> tuple[object, ...]:
    try:
        levels = tuple(sorted(frame[time].drop_duplicates().tolist()))
    except (TypeError, ValueError) as error:
        raise FoldConstructionError("time levels must have a total order") from error
    if not levels:
        raise FoldConstructionError("evaluation frame contains no time levels")
    return levels


def build_fold_definitions(
    frame: pd.DataFrame,
    data: ExperimentDataConfig,
    evaluation: EvaluationConfig,
) -> tuple[FoldDefinition, ...]:
    source = _validate_source(frame, data, evaluation)
    levels = _ordered_levels(source, data.time)
    maximum_horizon = max(evaluation.horizons)
    if len(levels) <= maximum_horizon:
        raise FoldConstructionError("insufficient future time levels for configured horizons")
    eligible = levels[:-maximum_horizon]
    if evaluation.origins.last is not None:
        count = evaluation.origins.last
        if count > len(eligible):
            raise FoldConstructionError("insufficient eligible origins for requested last-n folds")
        origins = eligible[-count:]
    else:
        origins = tuple(evaluation.origins.values or ())
        missing = [origin for origin in origins if origin not in levels]
        if missing:
            raise FoldConstructionError(f"configured origin is not a time level: {missing}")
        ineligible = [origin for origin in origins if origin not in eligible]
        if ineligible:
            raise FoldConstructionError(
                f"configured origin has insufficient future levels: {ineligible}"
            )
    positions = {value: index for index, value in enumerate(levels)}
    return tuple(
        FoldDefinition(origin, levels[positions[origin] + horizon], horizon)
        for origin in origins
        for horizon in evaluation.horizons
    )


def _latest_available_vintage(
    frame: pd.DataFrame,
    data: ExperimentDataConfig,
    evaluation: EvaluationConfig,
    origin: object,
) -> pd.DataFrame:
    if evaluation.mode == "latest":
        return _copy_frame(frame)
    assert data.vintage is not None
    try:
        available = frame.loc[frame[data.vintage].le(origin)].copy()
        available = available.sort_values(
            [*data.panel, data.time, data.vintage], kind="stable"
        )
    except (TypeError, ValueError) as error:
        raise FoldConstructionError("vintage releases must be comparable with the origin") from error
    keys = [*data.panel, data.time]
    return available.drop_duplicates(keys, keep="last").reset_index(drop=True)


def _latest_vintage(
    frame: pd.DataFrame,
    data: ExperimentDataConfig,
    evaluation: EvaluationConfig,
) -> pd.DataFrame:
    if evaluation.mode == "latest":
        return _copy_frame(frame)
    assert data.vintage is not None
    try:
        ordered = frame.sort_values(
            [*data.panel, data.time, data.vintage], kind="stable"
        )
    except (TypeError, ValueError) as error:
        raise FoldConstructionError("vintage releases must have a total order") from error
    keys = [*data.panel, data.time]
    return ordered.drop_duplicates(keys, keep="last").reset_index(drop=True)


def _validate_definition(
    definition: FoldDefinition, levels: tuple[object, ...]
) -> tuple[int, int]:
    if type(definition.horizon) is not int or definition.horizon <= 0:
        raise FoldConstructionError("fold horizon must be a positive integer")
    try:
        origin_position = levels.index(definition.origin)
        target_position = levels.index(definition.target)
    except ValueError as error:
        raise FoldConstructionError("fold origin and target must be exact time levels") from error
    if target_position - origin_position != definition.horizon:
        raise FoldConstructionError("fold target does not match its positional horizon")
    return origin_position, target_position


def _apply_training_window(
    source: pd.DataFrame,
    data: ExperimentDataConfig,
    evaluation: EvaluationConfig,
    levels: tuple[object, ...],
    origin_position: int,
    target_position: int,
) -> pd.DataFrame:
    allowed = set(levels[: target_position + 1])
    result = source.loc[source[data.time].isin(allowed)].copy()
    if evaluation.window.type == "rolling":
        assert evaluation.window.length is not None
        length = evaluation.window.length
        if origin_position + 1 < length:
            raise FoldConstructionError("rolling window has insufficient history levels")
        retained_history = set(levels[origin_position - length + 1 : origin_position + 1])
        retained_future = set(levels[origin_position + 1 : target_position + 1])
        result = result.loc[result[data.time].isin(retained_history | retained_future)].copy()
    try:
        return result.sort_values([*data.panel, data.time], kind="stable").reset_index(drop=True)
    except (TypeError, ValueError) as error:
        raise FoldConstructionError("panel/time coordinates must be orderable") from error


def _target_rows(
    truth: pd.DataFrame, data: ExperimentDataConfig, target: object
) -> pd.DataFrame:
    result = truth.loc[
        truth[data.time].eq(target) & truth[data.response].notna()
    ].copy()
    if result.empty:
        raise FoldConstructionError("fold has no observed scoring targets")
    return result.sort_values([*data.panel, data.time], kind="stable").reset_index(drop=True)


def _require_target_coordinates(
    model_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    data: ExperimentDataConfig,
) -> None:
    keys = [*data.panel, data.time]
    model_coordinates = model_frame.loc[:, keys].drop_duplicates()
    target_coordinates = target_frame.loc[:, keys].drop_duplicates()
    joined = target_coordinates.merge(model_coordinates, on=keys, how="left", indicator=True)
    if joined["_merge"].ne("both").any():
        raise FoldConstructionError(
            "every truth target coordinate must exist in the origin snapshot"
        )


def materialize_fold(
    frame: pd.DataFrame,
    data: ExperimentDataConfig,
    evaluation: EvaluationConfig,
    definition: FoldDefinition,
) -> FoldData:
    raw = _validate_source(frame, data, evaluation)
    levels = _ordered_levels(raw, data.time)
    origin_position, target_position = _validate_definition(definition, levels)
    source = _latest_available_vintage(raw, data, evaluation, definition.origin)
    truth = _latest_vintage(raw, data, evaluation)
    history_levels = set(levels[: origin_position + 1])
    history_frame = source.loc[source[data.time].isin(history_levels)].copy()
    model_frame = _apply_training_window(
        source, data, evaluation, levels, origin_position, target_position
    )
    target_frame = _target_rows(truth, data, definition.target)
    _require_target_coordinates(model_frame, target_frame, data)
    future_levels = set(levels[origin_position + 1 : target_position + 1])
    future_mask = model_frame[data.time].isin(future_levels)
    validate_future_covariates(model_frame.loc[future_mask], data, definition.horizon)
    model_frame.loc[future_mask, data.response] = np.nan
    training_frame = model_frame.loc[
        model_frame[data.time].isin(set(levels[: origin_position + 1]))
    ].copy()
    return FoldData(
        definition,
        training_frame,
        model_frame,
        target_frame,
        history_frame=history_frame,
        panel=data.panel,
        time=data.time,
        response=data.response,
        levels=levels,
    )

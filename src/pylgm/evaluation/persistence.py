from __future__ import annotations

import numpy as np
import pandas as pd

from pylgm.evaluation.folds import FoldData, _copy_frame
from pylgm.exceptions import FoldConstructionError

_MINIMUM_VARIANCE = 1e-12


def _panel_key(row: pd.Series, panel: tuple[str, ...]) -> tuple[object, ...]:
    return tuple(row[name] for name in panel)


def _persistence_variance(fold: FoldData) -> float:
    history = fold._history_frame
    response = fold._response
    positions = {level: position for position, level in enumerate(fold._levels)}
    values: dict[tuple[tuple[object, ...], object], object] = {}
    for _, row in history.loc[history[response].notna()].iterrows():
        values[(_panel_key(row, fold._panel), row[fold._time])] = row[response]
    squared_errors: list[float] = []
    for (panel_key, level), value in values.items():
        position = positions[level]
        previous_position = position - fold.definition.horizon
        if previous_position < 0:
            continue
        previous = values.get((panel_key, fold._levels[previous_position]))
        if previous is None:
            continue
        try:
            error = float(value) - float(previous)
        except (TypeError, ValueError) as cause:
            raise FoldConstructionError(
                "persistence residual history must contain numeric responses"
            ) from cause
        if not np.isfinite(error):
            raise FoldConstructionError("persistence residual history must be finite")
        squared_errors.append(error * error)
    if not squared_errors:
        raise FoldConstructionError(
            "insufficient pre-origin residual history for persistence variance"
        )
    variance = float(np.mean(squared_errors))
    if not np.isfinite(variance):
        raise FoldConstructionError("persistence variance is not finite")
    return max(variance, _MINIMUM_VARIANCE)


def persistence_predictions(fold: FoldData) -> pd.DataFrame:
    """Return leakage-safe persistence means and pooled horizon-specific variance.

    A zero empirical mean squared error receives a numerical variance floor of 1e-12.
    """
    if not fold._time or not fold._response:
        raise FoldConstructionError("fold lacks data-role metadata for persistence")
    history = fold._history_frame
    observed = history.loc[history[fold._response].notna()].copy()
    if observed.empty:
        raise FoldConstructionError("persistence prediction requires response history")
    order = {level: position for position, level in enumerate(fold._levels)}
    observed["__time_position"] = observed[fold._time].map(order)
    observed = observed.sort_values([*fold._panel, "__time_position"], kind="stable")
    if fold._panel:
        last = observed.drop_duplicates(list(fold._panel), keep="last")
        means = {
            _panel_key(row, fold._panel): row[fold._response]
            for _, row in last.iterrows()
        }
    else:
        means = {(): observed.iloc[-1][fold._response]}
    target = _copy_frame(fold._target_frame)
    target_keys = [_panel_key(row, fold._panel) for _, row in target.iterrows()]
    missing = [key for key in target_keys if key not in means]
    if missing:
        raise FoldConstructionError(
            f"persistence prediction has no response history for target panels: {missing}"
        )
    try:
        predictive_means = [float(means[key]) for key in target_keys]
    except (TypeError, ValueError) as cause:
        raise FoldConstructionError(
            "persistence history must contain numeric responses"
        ) from cause
    if not np.isfinite(predictive_means).all():
        raise FoldConstructionError("persistence means must be finite")
    result = target.loc[:, [*fold._panel, fold._time, fold._response]].rename(
        columns={fold._response: "actual"}
    )
    result["mean"] = predictive_means
    result["variance"] = _persistence_variance(fold)
    return result.reset_index(drop=True)

"""Vectorized probabilistic forecast scoring and row-weighted aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import norm

from pylgm.exceptions import DataContractError

_REQUIRED_PREDICTION_COLUMNS = ("actual", "mean", "variance", "candidate", "horizon")


def _interval_suffix(level: float) -> str:
    return str(level).replace(".", "_")


def _validated_levels(interval_levels: tuple[float, ...]) -> tuple[float, ...]:
    try:
        raw_levels = tuple(interval_levels)
    except TypeError as cause:
        raise DataContractError("interval levels must be a sequence of real numbers") from cause
    if any(
        not isinstance(level, Real) or isinstance(level, (bool, np.bool_))
        for level in raw_levels
    ):
        raise DataContractError("interval levels must be real numbers, not booleans")
    levels = tuple(float(level) for level in raw_levels)
    if any(not np.isfinite(level) or not 0.0 < level < 1.0 for level in levels):
        raise DataContractError("interval levels must be finite and strictly inside (0, 1)")
    if len(levels) != len(set(levels)):
        raise DataContractError("interval levels must be unique")
    return levels


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataContractError(f"prediction rows lack required columns: {missing}")


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    try:
        values = frame[name].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as cause:
        raise DataContractError(f"prediction column {name!r} must be numeric") from cause
    if not np.isfinite(values).all():
        raise DataContractError(f"prediction column {name!r} must be finite")
    return values


def _benchmark_identity(frame: pd.DataFrame) -> pd.Series:
    if "is_benchmark" not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame["is_benchmark"]
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataContractError("is_benchmark must contain booleans")
    return values.astype(bool, copy=True)


def score_predictions(
    predictions: pd.DataFrame, interval_levels: tuple[float, ...]
) -> pd.DataFrame:
    """Return independent, coordinate-preserving Gaussian forecast score rows."""
    _require_columns(predictions, _REQUIRED_PREDICTION_COLUMNS)
    levels = _validated_levels(interval_levels)
    actual = _numeric_column(predictions, "actual")
    mean = _numeric_column(predictions, "mean")
    variance = _numeric_column(predictions, "variance")
    if np.any(variance <= 0.0):
        raise DataContractError("prediction variance must be positive and finite")

    standard_deviation = np.sqrt(variance)
    result = predictions.copy(deep=True)
    result["is_benchmark"] = _benchmark_identity(predictions)
    residual = actual - mean
    result["log_predictive_density"] = norm.logpdf(
        actual, loc=mean, scale=standard_deviation
    )
    result["squared_error"] = residual**2
    result["absolute_error"] = np.abs(residual)
    for level in levels:
        suffix = _interval_suffix(level)
        quantile = norm.ppf((1.0 + level) / 2.0)
        lower = mean - quantile * standard_deviation
        upper = mean + quantile * standard_deviation
        result[f"lower_{suffix}"] = lower
        result[f"upper_{suffix}"] = upper
        result[f"covered_{suffix}"] = (actual >= lower) & (actual <= upper)
    return result


def _interval_suffixes(frame: pd.DataFrame) -> tuple[str, ...]:
    suffixes = tuple(
        column.removeprefix("covered_")
        for column in frame.columns
        if column.startswith("covered_")
    )
    missing_bounds = [
        suffix
        for suffix in suffixes
        if f"lower_{suffix}" not in frame or f"upper_{suffix}" not in frame
    ]
    if missing_bounds:
        raise DataContractError(f"scored rows lack interval bounds for: {missing_bounds}")
    return suffixes


def _aggregate_group(
    group: pd.DataFrame, *, candidate: object, benchmark: bool, horizon: object
) -> dict[str, object]:
    count = len(group)
    residual = group["actual"].to_numpy(dtype=float) - group["mean"].to_numpy(dtype=float)
    squared_error_sum = float(group["squared_error"].sum())
    absolute_error_sum = float(group["absolute_error"].sum())
    bias_sum = float(residual.sum())
    log_density_sum = float(group["log_predictive_density"].sum())
    result: dict[str, object] = {
        "candidate": candidate,
        "horizon": horizon,
        "is_benchmark": benchmark,
        "prediction_count": count,
        "squared_error_sum": squared_error_sum,
        "absolute_error_sum": absolute_error_sum,
        "bias_sum": bias_sum,
        "log_predictive_density_sum": log_density_sum,
        "rmse": float(np.sqrt(squared_error_sum / count)),
        "mae": absolute_error_sum / count,
        "bias": bias_sum / count,
        "mean_log_predictive_density": log_density_sum / count,
    }
    for suffix in _interval_suffixes(group):
        coverage_count = int(group[f"covered_{suffix}"].sum())
        width_sum = float(
            (group[f"upper_{suffix}"] - group[f"lower_{suffix}"]).sum()
        )
        result[f"coverage_count_{suffix}"] = coverage_count
        result[f"interval_width_sum_{suffix}"] = width_sum
        result[f"coverage_{suffix}"] = coverage_count / count
        result[f"average_width_{suffix}"] = width_sum / count
    return result


def aggregate_metrics(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored rows by candidate/horizon and candidate overall.

    Metrics are derived only after summing row-level losses, so folds with more
    predictions carry their natural weight.
    """
    required = (*_REQUIRED_PREDICTION_COLUMNS, "is_benchmark", "log_predictive_density", "squared_error", "absolute_error")
    _require_columns(scored_predictions, required)
    if scored_predictions.empty:
        raise DataContractError("cannot aggregate empty scored prediction rows")
    for column in ("actual", "mean", "log_predictive_density", "squared_error", "absolute_error"):
        _numeric_column(scored_predictions, column)
    benchmark = _benchmark_identity(scored_predictions)
    working = scored_predictions.copy(deep=True)
    working["is_benchmark"] = benchmark
    if working["candidate"].isna().any() or working["horizon"].isna().any():
        raise DataContractError("scored prediction rows require candidate and horizon")

    records: list[dict[str, object]] = []
    groups = working.groupby(["candidate", "is_benchmark", "horizon"], sort=True)
    for (candidate, is_benchmark, horizon), group in groups:
        records.append(
            _aggregate_group(
                group, candidate=candidate, benchmark=bool(is_benchmark), horizon=horizon
            )
        )
    overall_groups = working.groupby(["candidate", "is_benchmark"], sort=True)
    for (candidate, is_benchmark), group in overall_groups:
        records.append(
            _aggregate_group(
                group, candidate=candidate, benchmark=bool(is_benchmark), horizon=None
            )
        )
    return pd.DataFrame.from_records(records)

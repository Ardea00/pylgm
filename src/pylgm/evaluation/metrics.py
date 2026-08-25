"""Vectorized probabilistic forecast scoring and row-weighted aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import norm

from pylgm.exceptions import DataContractError

_REQUIRED_PREDICTION_COLUMNS = (
    "actual",
    "mean",
    "variance",
    "candidate",
    "horizon",
    "evaluation_mode",
)
_SIGNED_INTEGER = np.iinfo(np.int64)
_UNSIGNED_INTEGER = np.iinfo(np.uint64)


def _interval_suffix(level: float) -> str:
    return str(level).replace(".", "_")


def _validated_levels(interval_levels: tuple[float, ...]) -> tuple[float, ...]:
    try:
        raw_levels = tuple(interval_levels)
    except TypeError as cause:
        raise DataContractError("interval levels must be a sequence of real numbers") from cause
    if any(
        not isinstance(level, Real) or isinstance(level, (bool, np.bool_)) for level in raw_levels
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
    return values.astype(bool)


def _evaluation_modes(frame: pd.DataFrame) -> tuple[str, ...]:
    values = frame["evaluation_mode"]
    if (
        values.isna().any()
        or not values.map(lambda value: type(value) is str and value in {"latest", "vintage"}).all()
    ):
        raise DataContractError("evaluation mode must be 'latest' or 'vintage'")
    return tuple(dict.fromkeys(values.tolist()))


def _is_exact_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _timezone_key(timezone: object) -> str | None:
    if timezone is None:
        return None
    return str(getattr(timezone, "key", None) or getattr(timezone, "zone", None) or timezone)


def _origin_family(value: object) -> object:
    if _is_exact_integer(value):
        return "integer"
    if isinstance(value, pd.Timestamp):
        return ("timestamp", _timezone_key(value.tz))
    if isinstance(value, datetime):
        return ("datetime", _timezone_key(value.tzinfo))
    if isinstance(value, np.datetime64):
        return ("numpy_datetime", value.dtype.str)
    if isinstance(value, date):
        return "date"
    return type(value)


def _validate_origin_order(origins: pd.Series) -> None:
    if isinstance(origins.dtype, pd.CategoricalDtype):
        if not origins.cat.ordered:
            raise DataContractError("scored prediction origins must have a coherent total order")
    values = origins.dropna().tolist()
    families = {_origin_family(value) for value in values}
    if len(families) != 1:
        raise DataContractError("scored prediction origins must have a coherent total order")
    try:
        for value in values:
            hash(value)
        sorted(values)
    except (TypeError, ValueError) as cause:
        raise DataContractError(
            "scored prediction origins must have a coherent total order"
        ) from cause


def _origin_array(records: list[dict[str, object]]) -> pd.api.extensions.ExtensionArray:
    values = [record["origin"] for record in records]
    integers = [int(value) for value in values if value is not None and _is_exact_integer(value)]
    nonnull_count = sum(value is not None for value in values)
    if integers and len(integers) == nonnull_count:
        if _SIGNED_INTEGER.min <= min(integers) and max(integers) <= _SIGNED_INTEGER.max:
            return pd.array(values, dtype="Int64")
        if min(integers) >= 0 and max(integers) <= _UNSIGNED_INTEGER.max:
            return pd.array(values, dtype="UInt64")
    return pd.array(values, dtype=object)


def score_predictions(
    predictions: pd.DataFrame, interval_levels: tuple[float, ...]
) -> pd.DataFrame:
    """Return independent, coordinate-preserving Gaussian forecast score rows."""
    _require_columns(predictions, _REQUIRED_PREDICTION_COLUMNS)
    _evaluation_modes(predictions)
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
    result["log_predictive_density"] = norm.logpdf(actual, loc=mean, scale=standard_deviation)
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
        column.removeprefix("covered_") for column in frame.columns if column.startswith("covered_")
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
    group: pd.DataFrame,
    *,
    candidate: object,
    benchmark: bool,
    origin: object,
    horizon: object,
    evaluation_mode: str,
) -> dict[str, object]:
    count = len(group)
    residual = group["actual"].to_numpy(dtype=float) - group["mean"].to_numpy(dtype=float)
    squared_error_sum = float(group["squared_error"].sum())
    absolute_error_sum = float(group["absolute_error"].sum())
    bias_sum = float(residual.sum())
    log_density_sum = float(group["log_predictive_density"].sum())
    mean_log_predictive_density = log_density_sum / count
    result: dict[str, object] = {
        "candidate": candidate,
        "origin": origin,
        "horizon": horizon,
        "is_benchmark": benchmark,
        "evaluation_mode": evaluation_mode,
        "prediction_count": count,
        "squared_error_sum": squared_error_sum,
        "absolute_error_sum": absolute_error_sum,
        "bias_sum": bias_sum,
        "log_predictive_density_sum": log_density_sum,
        "rmse": float(np.sqrt(squared_error_sum / count)),
        "mae": absolute_error_sum / count,
        "bias": bias_sum / count,
        "mean_log_predictive_density": mean_log_predictive_density,
        # Compatibility alias for aggregate consumers. Row-level scored predictions
        # use this name for individual log densities; aggregate values are means.
        "log_predictive_density": mean_log_predictive_density,
    }
    for suffix in _interval_suffixes(group):
        coverage_count = int(group[f"covered_{suffix}"].sum())
        width_sum = float((group[f"upper_{suffix}"] - group[f"lower_{suffix}"]).sum())
        result[f"coverage_count_{suffix}"] = coverage_count
        result[f"interval_width_sum_{suffix}"] = width_sum
        result[f"coverage_{suffix}"] = coverage_count / count
        result[f"average_width_{suffix}"] = width_sum / count
    return result


def aggregate_metrics(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored rows by fold, horizon, and candidate overall.

    Metrics are derived only after summing row-level losses, so folds with more
    predictions carry their natural weight. At the row level,
    ``log_predictive_density`` is an individual Gaussian log density. In the
    returned aggregate table, ``mean_log_predictive_density`` is the canonical
    row-weighted mean and ``log_predictive_density`` is its compatibility alias.
    """
    required = (
        *_REQUIRED_PREDICTION_COLUMNS,
        "origin",
        "is_benchmark",
        "log_predictive_density",
        "squared_error",
        "absolute_error",
    )
    _require_columns(scored_predictions, required)
    if scored_predictions.empty:
        raise DataContractError("cannot aggregate empty scored prediction rows")
    for column in ("actual", "mean", "log_predictive_density", "squared_error", "absolute_error"):
        _numeric_column(scored_predictions, column)
    benchmark = _benchmark_identity(scored_predictions)
    working = scored_predictions.copy(deep=True)
    working["is_benchmark"] = benchmark
    modes = _evaluation_modes(working)
    if len(modes) != 1:
        raise DataContractError("aggregation requires a single consistent evaluation mode")
    evaluation_mode = modes[0]
    if working[["candidate", "origin", "horizon"]].isna().any(axis=None):
        raise DataContractError("scored prediction rows require candidate, origin, and horizon")
    if not working["horizon"].map(_is_exact_integer).all():
        raise DataContractError("scored prediction horizons must be exact integers")
    _validate_origin_order(working["origin"])

    records: list[dict[str, object]] = []
    fold_groups = working.groupby(
        ["candidate", "is_benchmark", "origin", "horizon"], sort=True, observed=True
    )
    for (candidate, is_benchmark, origin, horizon), group in fold_groups:
        records.append(
            _aggregate_group(
                group,
                candidate=candidate,
                benchmark=bool(is_benchmark),
                origin=origin,
                horizon=horizon,
                evaluation_mode=evaluation_mode,
            )
        )
    groups = working.groupby(["candidate", "is_benchmark", "horizon"], sort=True, observed=True)
    for (candidate, is_benchmark, horizon), group in groups:
        records.append(
            _aggregate_group(
                group,
                candidate=candidate,
                benchmark=bool(is_benchmark),
                origin=None,
                horizon=horizon,
                evaluation_mode=evaluation_mode,
            )
        )
    overall_groups = working.groupby(["candidate", "is_benchmark"], sort=True, observed=True)
    for (candidate, is_benchmark), group in overall_groups:
        records.append(
            _aggregate_group(
                group,
                candidate=candidate,
                benchmark=bool(is_benchmark),
                origin=None,
                horizon=None,
                evaluation_mode=evaluation_mode,
            )
        )
    result = pd.DataFrame.from_records(records)
    result["origin"] = _origin_array(records)
    result["horizon"] = pd.array([record["horizon"] for record in records], dtype="Int64")
    return result

"""Supported deterministic scalar domain for canonical panels."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


_NUMPY_SCALAR_KINDS = frozenset("biufcMmSU")


def is_supported_numpy_scalar(value: np.generic) -> bool:
    dtype = value.dtype
    return dtype.fields is None and dtype.kind in _NUMPY_SCALAR_KINDS


def _has_supported_timezone(value: datetime) -> bool:
    timezone = value.tzinfo
    if timezone is None:
        return True
    if getattr(timezone, "key", None) or getattr(timezone, "zone", None):
        return True
    try:
        return timezone.utcoffset(None) is not None
    except (TypeError, ValueError):
        return False


def is_supported_object_value(value: Any) -> bool:
    """Return whether the fingerprint encoder has a total deterministic encoding."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, np.generic):
        return is_supported_numpy_scalar(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return _has_supported_timezone(value)
    if isinstance(value, (bool, int, float, str, bytes, bytearray, memoryview, date, Decimal)):
        return True
    if isinstance(value, pd.Timedelta):
        return True
    if isinstance(value, pd.Period):
        return bool(getattr(value, "freqstr", None))
    if isinstance(value, pd.Interval):
        return is_supported_object_value(value.left) and is_supported_object_value(value.right)
    return False


def ordered_observed_levels(values: pd.Series) -> tuple[object, ...]:
    """Return observed levels in their declared or natural total order.

    Ordered categoricals retain the declared relative category order while unused
    categories are omitted. An unordered categorical is not a valid ordered index.
    """
    if values.isna().any():
        raise ValueError("ordered index levels must not contain null values")
    if isinstance(values.dtype, pd.CategoricalDtype):
        if not values.cat.ordered:
            raise ValueError("categorical index levels must be ordered")
        levels = tuple(values.cat.remove_unused_categories().cat.categories.tolist())
    else:
        try:
            levels = tuple(sorted(values.drop_duplicates().tolist()))
        except (TypeError, ValueError) as error:
            raise ValueError("index levels must have a total order") from error
    if not levels:
        raise ValueError("ordered index contains no observed levels")
    return levels

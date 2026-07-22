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

"""Stable typed fingerprints for canonical panels."""

import hashlib
import struct
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype, is_object_dtype

from pylgm.data.panel import CanonicalPanel
from pylgm.data.scalars import is_supported_numpy_scalar


def _record(tag: str, payload: bytes = b"") -> bytes:
    """Return an unambiguous length-prefixed binary record."""
    encoded_tag = tag.encode("ascii")
    return (
        len(encoded_tag).to_bytes(2, "big")
        + encoded_tag
        + len(payload).to_bytes(8, "big")
        + payload
    )


def _sequence(tag: str, values: list[bytes]) -> bytes:
    return _record(tag, len(values).to_bytes(8, "big") + b"".join(values))


def _timezone_descriptor(timezone: Any) -> bytes:
    key = getattr(timezone, "key", None) or getattr(timezone, "zone", None)
    if key is not None:
        return _record("timezone-name", str(key).encode("utf-8"))
    offset = timezone.utcoffset(None)
    if offset is None:
        raise TypeError(f"unsupported timezone type: {type(timezone).__qualname__}")
    return _record("timezone-offset-seconds", str(int(offset.total_seconds())).encode())


def _dtype_descriptor(dtype: Any) -> bytes:
    """Encode dtype parameters that affect a frame's logical schema."""
    if isinstance(dtype, pd.CategoricalDtype):
        categories = [
            _scalar_descriptor(value, dtype.categories.dtype) for value in dtype.categories
        ]
        return _record(
            "categorical-dtype",
            _dtype_descriptor(dtype.categories.dtype)
            + _record("ordered", b"1" if dtype.ordered else b"0")
            + _sequence("categories", categories),
        )
    if isinstance(dtype, pd.DatetimeTZDtype):
        return _record(
            "datetime-tz-dtype",
            _record("unit", dtype.unit.encode()) + _timezone_descriptor(dtype.tz),
        )
    if isinstance(dtype, pd.PeriodDtype):
        return _record("period-dtype", _record("freq", _frequency_string(dtype).encode()))
    if isinstance(dtype, pd.IntervalDtype):
        return _record(
            "interval-dtype",
            _dtype_descriptor(dtype.subtype) + _record("closed", dtype.closed.encode()),
        )
    if isinstance(dtype, pd.SparseDtype):
        return _record(
            "sparse-dtype",
            _dtype_descriptor(dtype.subtype) + _scalar_descriptor(dtype.fill_value, dtype.subtype),
        )
    if isinstance(dtype, pd.StringDtype):
        return _record(
            "string-dtype",
            _record("storage", dtype.storage.encode())
            + _record("na-value", _scalar_descriptor(dtype.na_value, object)),
        )
    if is_extension_array_dtype(dtype):
        name = f"{type(dtype).__module__}.{type(dtype).__qualname__}".encode()
        return _record(
            "extension-dtype", _record("class", name) + _record("name", dtype.name.encode())
        )
    return _record("numpy-dtype", np.dtype(dtype).str.encode())


def _frequency_string(value: Any) -> str:
    frequency = getattr(value, "freqstr", None) or getattr(value, "_freqstr", None)
    if frequency is None:
        frequency = getattr(getattr(value, "freq", None), "rule_code", None)
    if not frequency:
        raise TypeError(f"unsupported period frequency: {type(value).__qualname__}")
    return str(frequency)


def _extended_float_value(value: Any, dtype: np.dtype[Any]) -> bytes:
    """Deterministic value bytes for extended-precision floats.

    ``np.longdouble`` is 80-bit extended padded to 16 bytes on x86, so
    ``tobytes()`` leaks the uninitialised padding and hashes equal values
    differently. Encode the exact shortest round-tripping decimal instead:
    equal values hash equally. Sub-float64 distinctness is neither promised nor
    portable, which is exactly the encoder contract the fingerprint tests assert.
    """
    parts = (value.real, value.imag) if dtype.kind == "c" else (value,)
    return b"|".join(
        np.format_float_scientific(np.longdouble(part), unique=True, trim="-").encode()
        for part in parts
    )


def _numpy_scalar_descriptor(value: Any, dtype: np.dtype[Any]) -> bytes:
    """Encode a NumPy scalar in canonical big-endian dtype and exact bytes."""
    canonical_dtype = dtype.newbyteorder(">")
    # Extended-precision floats (padded longdouble/clongdouble) only; every other
    # numeric dtype is fixed-width with no padding, so raw bytes stay exact.
    if (dtype.kind == "f" and dtype.itemsize > 8) or (dtype.kind == "c" and dtype.itemsize > 16):
        payload = _extended_float_value(value, dtype)
    else:
        payload = np.asarray(value, dtype=dtype).astype(canonical_dtype, copy=False).tobytes()
    return _record(
        "numpy-scalar",
        _record("dtype", canonical_dtype.str.encode()) + _record("value", payload),
    )


def _scalar_descriptor(value: Any, dtype: Any) -> bytes:
    """Encode supported scalar values without repr, pickle, or lossy hashing."""
    object_dtype = is_object_dtype(dtype)
    if value is None:
        return _record("object-none" if object_dtype else "missing")
    if value is pd.NA:
        return _record("object-pandas-na" if object_dtype else "missing")
    if value is pd.NaT:
        return _record("object-nat" if object_dtype else "missing")
    if not object_dtype and isinstance(value, (float, np.floating)) and np.isnan(value):
        return _record("object-float-nan" if object_dtype else "missing")
    if isinstance(dtype, pd.CategoricalDtype):
        return _scalar_descriptor(value, dtype.categories.dtype)
    if isinstance(value, pd.Timestamp):
        timezone = _record("naive") if value.tz is None else _timezone_descriptor(value.tz)
        return _record("timestamp", int(value.value).to_bytes(8, "big", signed=True) + timezone)
    if isinstance(value, datetime):
        timezone = _record("naive") if value.tzinfo is None else _timezone_descriptor(value.tzinfo)
        return _record(
            "datetime",
            int(pd.Timestamp(value).value).to_bytes(8, "big", signed=True) + timezone,
        )
    if isinstance(value, date):
        return _record("date", value.isoformat().encode())
    if isinstance(value, Decimal):
        decimal = value.as_tuple()
        return _record(
            "decimal",
            _record("sign", str(decimal.sign).encode())
            + _record("digits", bytes(decimal.digits))
            + _record("exponent", str(decimal.exponent).encode()),
        )
    if isinstance(value, np.generic):
        if not is_supported_numpy_scalar(value):
            raise TypeError(f"unsupported NumPy scalar type: {value.dtype}")
        return _numpy_scalar_descriptor(value, value.dtype)
    if not object_dtype:
        try:
            numeric_dtype = np.dtype(dtype)
        except TypeError:
            numeric_dtype = None
        if numeric_dtype is not None and numeric_dtype.kind in frozenset("biufc"):
            return _numpy_scalar_descriptor(value, numeric_dtype)
    if isinstance(value, float) and np.isnan(value):
        return _record("object-float-nan")
    if isinstance(value, (bool, np.bool_)):
        return _record("bool", b"1" if value else b"0")
    if isinstance(value, (int, np.integer)):
        return _record("integer", str(int(value)).encode())
    if isinstance(value, (float, np.floating)):
        return _record("float64", struct.pack(">d", float(value)))
    if isinstance(value, (str, np.str_)):
        return _record("string", str(value).encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _record("bytes", bytes(value))
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        nanoseconds = pd.Timedelta(value).value
        return _record("timedelta-nanoseconds", int(nanoseconds).to_bytes(8, "big", signed=True))
    if isinstance(value, pd.Period):
        return _record(
            "period",
            _record("ordinal", int(value.ordinal).to_bytes(8, "big", signed=True))
            + _record("freq", _frequency_string(value).encode()),
        )
    if isinstance(value, pd.Interval):
        return _record(
            "interval",
            _scalar_descriptor(value.left, object)
            + _scalar_descriptor(value.right, object)
            + _record("closed", value.closed.encode()),
        )
    raise TypeError(f"unsupported frame value type: {type(value).__qualname__}")


def panel_fingerprint(panel: CanonicalPanel) -> str:
    """Hash typed canonical panel contents and full ordered schema deterministically."""
    frame = panel.frame
    columns = list(frame.columns)
    schema = _sequence(
        "schema",
        [
            _record("column-string", column.encode("utf-8"))
            + _dtype_descriptor(frame.dtypes.iloc[index])
            for index, column in enumerate(columns)
        ],
    )
    rows = _sequence(
        "rows",
        [
            _sequence(
                "row",
                [
                    _scalar_descriptor(value, frame.dtypes.iloc[column_index])
                    for column_index, value in enumerate(row)
                ],
            )
            for row in frame.itertuples(index=False, name=None)
        ],
    )
    metadata = _sequence(
        "panel-metadata",
        [
            _sequence(
                "key-columns", [_record("column", key.encode("utf-8")) for key in panel.key_columns]
            ),
            _record("response", panel.response.encode("utf-8")),
        ],
    )
    return hashlib.sha256(_record("pylgm-canonical-panel-v1", schema + rows + metadata)).hexdigest()


__all__ = ["panel_fingerprint"]

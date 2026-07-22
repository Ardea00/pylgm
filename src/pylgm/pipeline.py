"""Local orchestration for exact Gaussian pyLGM fits."""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype

from pylgm.artifacts.run import write_run
from pylgm.compiler import compile_model
from pylgm.config import RunConfig, load_config
from pylgm.data import CanonicalPanel
from pylgm.inference import GaussianResult, fit_gaussian


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
        categories = [_scalar_descriptor(value, dtype.categories.dtype) for value in dtype.categories]
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
        return _record("period-dtype", _record("freq", dtype.freqstr.encode()))
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
        return _record("string-dtype", _record("storage", dtype.storage.encode()))
    if is_extension_array_dtype(dtype):
        name = f"{type(dtype).__module__}.{type(dtype).__qualname__}".encode()
        return _record("extension-dtype", _record("class", name) + _record("name", dtype.name.encode()))
    return _record("numpy-dtype", np.dtype(dtype).str.encode())


def _scalar_descriptor(value: Any, dtype: Any) -> bytes:
    """Encode supported scalar values without repr, pickle, or lossy hashing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return _record("missing")
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return _record("missing")
    if isinstance(dtype, pd.CategoricalDtype):
        return _scalar_descriptor(value, dtype.categories.dtype)
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
    if isinstance(value, pd.Timestamp):
        return _record("timestamp-nanoseconds", int(value.value).to_bytes(8, "big", signed=True))
    if isinstance(value, np.datetime64):
        return _record("datetime64-nanoseconds", int(value.astype("datetime64[ns]").astype("int64")).to_bytes(8, "big", signed=True))
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        nanoseconds = pd.Timedelta(value).value
        return _record("timedelta-nanoseconds", int(nanoseconds).to_bytes(8, "big", signed=True))
    if isinstance(value, pd.Period):
        return _record("period-ordinal", int(value.ordinal).to_bytes(8, "big", signed=True))
    if isinstance(value, pd.Interval):
        return _record(
            "interval",
            _scalar_descriptor(value.left, dtype.subtype)
            + _scalar_descriptor(value.right, dtype.subtype)
            + _record("closed", value.closed.encode()),
        )
    raise TypeError(f"unsupported frame value type: {type(value).__qualname__}")


def _panel_fingerprint(panel: CanonicalPanel) -> str:
    """Hash typed canonical panel contents and full ordered schema deterministically."""
    frame = panel.frame
    columns = list(frame.columns)
    schema = _sequence(
        "schema",
        [
            _record("column", str(column).encode("utf-8")) + _dtype_descriptor(frame.dtypes.iloc[index])
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
            _sequence("key-columns", [_record("column", key.encode("utf-8")) for key in panel.key_columns]),
            _record("response", panel.response.encode("utf-8")),
        ],
    )
    return hashlib.sha256(_record("pylgm-canonical-panel-v1", schema + rows + metadata)).hexdigest()


@dataclass(frozen=True)
class Pipeline:
    config: RunConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        return cls(load_config(Path(path)))

    def run(self, frame: pd.DataFrame, output: str | Path) -> GaussianResult:
        panel = CanonicalPanel.from_frame(frame, self.config.data)
        result = fit_gaussian(compile_model(self.config, panel))
        write_run(Path(output), self.config, result, _panel_fingerprint(panel))
        return result

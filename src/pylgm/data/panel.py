from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.exceptions import DataContractError


def _is_supported_object_value(value: object) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
            bytes,
            bytearray,
            memoryview,
            datetime,
            date,
            Decimal,
            np.generic,
            pd.Timedelta,
            pd.Period,
            pd.Interval,
        ),
    )


def _validate_frame_contract(frame: pd.DataFrame) -> None:
    if any(not isinstance(column, str) for column in frame.columns):
        raise DataContractError("column labels must be strings")
    for column in frame.columns:
        values = frame[column]
        if values.dtype != object:
            continue
        for value in values:
            if not _is_supported_object_value(value):
                raise DataContractError(
                    f"unsupported object value type in column {column!r}: "
                    f"{type(value).__qualname__}"
                )


@dataclass(frozen=True)
class CanonicalPanel:
    frame: pd.DataFrame
    observed: np.ndarray
    key_columns: tuple[str, ...]
    response: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, config: DataConfig) -> "CanonicalPanel":
        _validate_frame_contract(frame)
        keys = (*config.panel, config.time)
        required = {*keys, config.response}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataContractError(f"missing columns: {missing}")
        if frame.duplicated(list(keys)).any():
            raise DataContractError(f"duplicate panel keys: {keys}")
        ordered = frame.sort_values(list(keys), kind="stable").reset_index(drop=True).copy()
        observed = ordered[config.response].notna().to_numpy(dtype=bool)
        if not observed.any():
            raise DataContractError("panel contains no observed responses")
        return cls(ordered, observed, keys, config.response)

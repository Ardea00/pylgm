from dataclasses import dataclass
import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.data.scalars import is_supported_object_value
from pylgm.exceptions import DataContractError


def _validate_frame_contract(frame: pd.DataFrame) -> None:
    if any(not isinstance(column, str) for column in frame.columns):
        raise DataContractError("column labels must be strings")
    for column in frame.columns:
        values = frame[column]
        if values.dtype != object:
            continue
        for value in values:
            if not is_supported_object_value(value):
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

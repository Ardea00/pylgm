import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from pylgm.config.schema import DataConfig
from pylgm.data.scalars import is_supported_object_value
from pylgm.exceptions import DataContractError


def _validate_frame_contract(frame: pd.DataFrame) -> None:
    if any(not isinstance(column, str) for column in frame.columns):
        raise DataContractError("column labels must be strings")
    if not frame.columns.is_unique:
        raise DataContractError("column labels must be unique")
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


def _copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for position, dtype in enumerate(result.dtypes):
        if not is_object_dtype(dtype):
            continue
        column = result.iloc[:, position]
        result.iloc[:, position] = column.map(
            lambda value: memoryview(bytes(value))
            if isinstance(value, memoryview)
            else copy.deepcopy(value)
        )
    return result


@dataclass(frozen=True, init=False)
class CanonicalPanel:
    _frame: pd.DataFrame = field(repr=False)
    _observed: np.ndarray = field(repr=False)
    _source_positions: np.ndarray = field(repr=False)
    key_columns: tuple[str, ...]
    response: str

    def __init__(
        self,
        frame: pd.DataFrame,
        observed: np.ndarray,
        key_columns: tuple[str, ...],
        response: str,
        source_positions: np.ndarray | None = None,
    ) -> None:
        isolated_observed = np.array(observed, dtype=bool, copy=True)
        isolated_observed.setflags(write=False)
        if source_positions is None:
            isolated_source_positions = np.arange(len(frame), dtype=np.int64)
        else:
            positions = np.asarray(source_positions)
            expected = np.arange(len(frame), dtype=np.int64)
            if (
                positions.ndim != 1
                or not np.issubdtype(positions.dtype, np.integer)
                or positions.size != len(frame)
                or not np.array_equal(np.sort(positions), expected)
            ):
                raise DataContractError(
                    "source positions must be a positional permutation of panel rows"
                )
            isolated_source_positions = np.array(
                positions, dtype=np.int64, copy=True
            )
        isolated_source_positions.setflags(write=False)
        object.__setattr__(self, "_frame", _copy_frame(frame))
        object.__setattr__(self, "_observed", isolated_observed)
        object.__setattr__(self, "_source_positions", isolated_source_positions)
        object.__setattr__(self, "key_columns", tuple(key_columns))
        object.__setattr__(self, "response", response)

    @property
    def frame(self) -> pd.DataFrame:
        return _copy_frame(self._frame)

    @property
    def observed(self) -> np.ndarray:
        result = self._observed.copy()
        result.setflags(write=False)
        return result

    @property
    def source_positions(self) -> np.ndarray:
        """Caller row positions for each row in canonical panel order."""
        result = self._source_positions.copy()
        result.setflags(write=False)
        return result

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, config: DataConfig) -> "CanonicalPanel":
        _validate_frame_contract(frame)
        keys = (*config.panel, config.time)
        required = {*keys, config.response}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataContractError(f"missing columns: {missing}")
        if frame.loc[:, list(keys)].isna().any(axis=None):
            raise DataContractError(f"panel key columns contain null values: {keys}")
        if frame.duplicated(list(keys)).any():
            raise DataContractError(f"duplicate panel keys: {keys}")
        try:
            positioned = frame.copy(deep=False)
            positioned.index = pd.RangeIndex(len(positioned))
            ordered = positioned.sort_values(list(keys), kind="stable")
        except (TypeError, ValueError) as error:
            raise DataContractError(f"panel key columns must be sortable/orderable: {keys}") from error
        source_positions = ordered.index.to_numpy(dtype=np.int64, copy=True)
        ordered = ordered.reset_index(drop=True)
        observed = ordered[config.response].notna().to_numpy(dtype=bool)
        if not observed.any():
            raise DataContractError("panel contains no observed responses")
        return cls(ordered, observed, keys, config.response, source_positions)

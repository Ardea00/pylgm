from dataclasses import dataclass

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.exceptions import DataContractError


@dataclass(frozen=True)
class CanonicalPanel:
    frame: pd.DataFrame
    observed: np.ndarray
    key_columns: tuple[str, ...]
    response: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, config: DataConfig) -> "CanonicalPanel":
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

import pandas as pd
import pytest

from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError


def test_panel_sorts_keys_and_preserves_missing_targets() -> None:
    frame = pd.DataFrame({"month": [2, 1], "region": ["A", "A"], "y": [None, 1.2]})
    panel = CanonicalPanel.from_frame(
        frame,
        DataConfig(time="month", response="y", panel=("region",)),
    )
    assert panel.frame["month"].tolist() == [1, 2]
    assert panel.observed.tolist() == [True, False]


def test_duplicate_panel_key_fails() -> None:
    frame = pd.DataFrame({"month": [1, 1], "region": ["A", "A"], "y": [1.0, 2.0]})
    with pytest.raises(DataContractError, match="duplicate"):
        CanonicalPanel.from_frame(
            frame,
            DataConfig(time="month", response="y", panel=("region",)),
        )

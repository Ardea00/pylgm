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


def test_panel_is_value_isolated_from_source_and_accessor_mutations() -> None:
    source = pd.DataFrame({"month": [2, 1], "region": ["A", "A"], "y": [2.0, 1.0]})
    panel = CanonicalPanel.from_frame(
        source, DataConfig(time="month", response="y", panel=("region",))
    )

    source.loc[:, "y"] = 99.0
    exposed_frame = panel.frame
    exposed_frame.loc[:, "y"] = 0.0
    exposed_observed = panel.observed
    with pytest.raises(ValueError):
        exposed_observed[:] = False

    assert panel.frame["y"].tolist() == [1.0, 2.0]
    assert panel.observed.tolist() == [True, True]


def test_duplicate_column_names_fail_at_the_panel_boundary() -> None:
    frame = pd.DataFrame([[1, 1.0, 2.0]], columns=["month", "y", "y"])

    with pytest.raises(DataContractError, match="unique"):
        CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))


def test_null_key_fails_with_a_typed_data_error() -> None:
    frame = pd.DataFrame({"month": [1, None], "y": [1.0, 2.0]})

    with pytest.raises(DataContractError, match="null"):
        CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))


def test_unorderable_keys_fail_with_a_typed_data_error() -> None:
    frame = pd.DataFrame({"month": [1, "two"], "y": [1.0, 2.0]})

    with pytest.raises(DataContractError, match="sortable|orderable"):
        CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))

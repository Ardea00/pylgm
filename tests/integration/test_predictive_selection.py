from pathlib import Path

import pandas as pd

from pylgm import Experiment


def test_predictive_selection_example_runs(tmp_path: Path) -> None:
    result = Experiment.from_yaml(
        Path("examples/predictive_selection/config.yaml")
    ).compare(
        pd.read_csv("examples/predictive_selection/data.csv"),
        tmp_path / "comparison",
    )

    assert result.selected in result.candidates
    assert set(result.predictions["horizon"]) == {1, 2}

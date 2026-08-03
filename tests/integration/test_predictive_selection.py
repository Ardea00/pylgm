import json
from pathlib import Path

import pandas as pd

from pylgm import Experiment


def test_predictive_selection_example_runs(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    result = Experiment.from_yaml(
        Path("examples/predictive_selection/config.yaml")
    ).compare(
        pd.read_csv("examples/predictive_selection/data.csv"),
        output,
    )
    summary = json.loads((output / "summary.json").read_text())

    assert result.selected in result.candidates
    assert set(result.predictions["horizon"]) == {1, 2}
    assert summary["artifact_schema_version"] == 2
    assert summary["selected"]
    assert summary["selected"] == result.selected

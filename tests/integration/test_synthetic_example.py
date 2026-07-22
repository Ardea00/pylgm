from pathlib import Path

import pandas as pd

from pylgm import Pipeline


def test_synthetic_example_runs(tmp_path: Path) -> None:
    root = Path("examples/synthetic_panel")
    result = Pipeline.from_yaml(root / "config.yaml").run(
        pd.read_csv(root / "data.csv"), tmp_path / "run"
    )
    assert result.predictive_mean.shape == (8,)
    assert result.labels[:2] == ("fixed:Intercept", "fixed:x")
    assert all(value > 0 for value in result.predictive_variance)

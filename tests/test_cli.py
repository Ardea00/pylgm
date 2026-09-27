from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylgm.cli import app


def test_cli_fits_csv(tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.csv"
    output = tmp_path / "run"
    config.write_text(
        "schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n"
    )
    data.write_text("month,y\n1,1.0\n2,2.0\n")

    app(["fit", str(config), str(data), "--output", str(output)])

    assert "exact_gaussian" in capsys.readouterr().out


def test_cli_reports_existing_output_failure(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.csv"
    output = tmp_path / "run"
    config.write_text(
        "schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n"
    )
    data.write_text("month,y\n1,1.0\n2,2.0\n")
    output.mkdir()

    with pytest.raises(FileExistsError, match="File exists"):
        app(["fit", str(config), str(data), "--output", str(output)])


def _write_experiment_config(path: Path) -> Path:
    path.write_text(
        """schema_version: 2
data: {time: month, response: y, panel: [region]}
model: {fixed: '1', sigma: 1.0, effects: []}
candidates:
  - {name: base}
inference:
  engine: exact_gaussian
  hyperparameters:
    strategy: empirical_bayes
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
evaluation:
  horizons: [1]
  origins: {last: 1}
  interval_levels: [0.8]
  max_abs_coverage_error: 1.0
"""
    )
    return path


def _comparison_frame() -> pd.DataFrame:
    month = np.tile(np.arange(6), 2)
    return pd.DataFrame(
        {
            "region": np.repeat(["north", "south"], 6),
            "month": month,
            "y": 1.0 + month,
        }
    )


@pytest.mark.parametrize("suffix", [".CSV", ".parquet", ".PQ"])
def test_cli_compares_supported_local_frames(tmp_path: Path, suffix: str, capsys) -> None:
    config = _write_experiment_config(tmp_path / "experiment.yaml")
    data = tmp_path / f"data{suffix}"
    output = tmp_path / "comparison"
    frame = _comparison_frame()
    if suffix.lower() == ".csv":
        frame.to_csv(data, index=False)
    else:
        frame.to_parquet(data, index=False)

    app(["compare", str(config), str(data), "--output", str(output)])

    assert "selected=base candidates=1" in capsys.readouterr().out
    assert (output / "summary.json").exists()


def test_cli_compare_rejects_unsupported_data_suffix(tmp_path: Path) -> None:
    config = _write_experiment_config(tmp_path / "experiment.yaml")
    data = tmp_path / "data.json"
    data.write_text("[]")

    with pytest.raises(ValueError, match="data must be CSV or Parquet"):
        app(["compare", str(config), str(data), "--output", str(tmp_path / "comparison")])

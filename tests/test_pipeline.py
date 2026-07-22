import json
from pathlib import Path

import pandas as pd
import pytest

from pylgm import Pipeline


def _write_config(path: Path) -> None:
    path.write_text(
        "schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n"
    )


def test_pipeline_persists_resolved_run(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"

    result = Pipeline.from_yaml(config).run(
        pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output
    )

    summary = json.loads((output / "summary.json").read_text())
    resolved = json.loads((output / "resolved_config.json").read_text())
    assert result.mean.shape == (1,)
    assert summary["engine"] == "exact_gaussian"
    assert summary["conditional_on_fixed_hyperparameters"] is True
    assert len(summary["data_fingerprint"]) == 64
    assert resolved["model"]["fixed_prior_precision"] == 1e-6
    assert (output / "posterior.npz").exists()
    assert (output / "environment.json").exists()


def test_pipeline_rejects_existing_output_directory(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(FileExistsError):
        Pipeline.from_yaml(config).run(
            pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output
        )


def test_pipeline_fingerprint_is_stable_for_equivalent_input_order(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    pipeline = Pipeline.from_yaml(config)
    first = tmp_path / "first"
    second = tmp_path / "second"

    pipeline.run(pd.DataFrame({"month": [2, 1], "y": [2.0, 1.0]}), first)
    pipeline.run(pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), second)

    assert json.loads((first / "summary.json").read_text())["data_fingerprint"] == json.loads(
        (second / "summary.json").read_text()
    )["data_fingerprint"]

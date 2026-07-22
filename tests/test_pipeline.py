import json
from pathlib import Path

import pandas as pd
import pytest

from pylgm import Pipeline
from pylgm.artifacts import run as run_artifacts
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.pipeline import _panel_fingerprint


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


def _fingerprint(frame: pd.DataFrame) -> str:
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))
    return _panel_fingerprint(panel)


def test_panel_fingerprint_is_stable_for_equivalent_categorical_frames() -> None:
    category = pd.CategoricalDtype(categories=["low", "high"], ordered=True)
    first = pd.DataFrame(
        {"month": [2, 1], "y": [2.0, 1.0], "group": pd.Series(["high", "low"], dtype=category)}
    )
    second = pd.DataFrame(
        {"month": [1, 2], "y": [1.0, 2.0], "group": pd.Series(["low", "high"], dtype=category)}
    )

    assert _fingerprint(first) == _fingerprint(second)


def test_panel_fingerprint_distinguishes_dtype_and_categorical_metadata() -> None:
    ordered = pd.CategoricalDtype(categories=["low", "high"], ordered=True)
    reordered = pd.CategoricalDtype(categories=["high", "low"], ordered=True)
    categorical = pd.DataFrame(
        {"month": [1, 2], "y": [1.0, 2.0], "group": pd.Series(["low", "high"], dtype=ordered)}
    )
    different_categories = categorical.assign(
        group=pd.Series(["low", "high"], dtype=reordered)
    )
    nullable_integer = categorical.assign(month=pd.Series([1, 2], dtype="Int64"))

    assert _fingerprint(categorical) != _fingerprint(different_categories)
    assert _fingerprint(categorical) != _fingerprint(nullable_integer)


def test_panel_fingerprint_distinguishes_content_and_column_order() -> None:
    original = pd.DataFrame(
        {"month": [1, 2], "y": [1.0, 2.0], "note": ["a", None]}
    )
    changed_content = original.assign(note=["a", "b"])
    changed_column_order = original.loc[:, ["month", "note", "y"]]

    assert _fingerprint(original) != _fingerprint(changed_content)
    assert _fingerprint(original) != _fingerprint(changed_column_order)


def test_pipeline_removes_temporary_artifact_directory_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("injected artifact failure")

    monkeypatch.setattr(run_artifacts.np, "savez_compressed", fail_save)

    with pytest.raises(OSError, match="injected artifact failure"):
        Pipeline.from_yaml(config).run(
            pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".run.tmp-*"))

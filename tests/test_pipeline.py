import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from pylgm import Pipeline
from pylgm.artifacts import run as run_artifacts
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError
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


def test_panel_contract_rejects_non_string_column_labels() -> None:
    frame = pd.DataFrame({"month": [1], "y": [1.0], 1: ["value"]})

    with pytest.raises(DataContractError, match="column labels must be strings"):
        CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))


def test_object_fingerprint_distinguishes_missing_kinds_and_timestamp_timezone() -> None:
    base = pd.DataFrame({"month": [1], "y": [1.0], "value": [None]})
    pandas_missing = base.assign(value=pd.Series([pd.NA], dtype=object))
    datetime_missing = base.assign(value=pd.Series([pd.NaT], dtype=object))
    float_missing = base.assign(value=pd.Series([float("nan")], dtype=object))
    naive = base.assign(value=pd.Series([datetime(2024, 1, 1)], dtype=object))
    aware = base.assign(value=pd.Series([datetime(2024, 1, 1, tzinfo=timezone.utc)], dtype=object))

    assert len({_fingerprint(frame) for frame in (base, pandas_missing, datetime_missing, float_missing)}) == 4
    assert _fingerprint(naive) != _fingerprint(aware)


def test_object_fingerprint_supports_datetime_date_and_decimal() -> None:
    frame = pd.DataFrame(
        {
            "month": [1],
            "y": [1.0],
            "when": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "amount": [Decimal("1.20")],
        }
    )

    assert _fingerprint(frame) == _fingerprint(frame.copy())


def test_panel_contract_rejects_unsupported_object_values_before_inference() -> None:
    frame = pd.DataFrame({"month": [1], "y": [1.0], "value": [object()]})

    with pytest.raises(DataContractError, match="unsupported object value type"):
        CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))


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


def test_pipeline_recovers_stale_lock_file(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"
    (tmp_path / ".run.lock").write_text("stale")

    Pipeline.from_yaml(config).run(pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output)

    assert output.exists()


def test_pipeline_rejects_dangling_symlink_output(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"
    output.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError):
        Pipeline.from_yaml(config).run(pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output)


def test_pipeline_does_not_overwrite_destination_created_during_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"
    publish = run_artifacts._publish_no_replace

    def destination_race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("other writer")
        publish(source, destination)

    monkeypatch.setattr(run_artifacts, "_publish_no_replace", destination_race)

    with pytest.raises(FileExistsError):
        Pipeline.from_yaml(config).run(
            pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output
        )

    assert (output / "owner.txt").read_text() == "other writer"
    assert not list(tmp_path.glob(".run.tmp-*"))

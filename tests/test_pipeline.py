import json
import ctypes
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from pylgm import Pipeline
from pylgm.artifacts import run as run_artifacts
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.data.fingerprint import panel_fingerprint
from pylgm.exceptions import DataContractError


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
    # v2: predictive_variance became the linear-predictor variance and
    # observation_variance was added, so a v1 artifact is not comparable.
    assert summary["artifact_schema_version"] == 2
    assert summary["conditional_on_fixed_hyperparameters"] is True
    assert len(summary["data_fingerprint"]) == 64
    assert resolved["model"]["fixed_prior_precision"] == 1e-6
    assert (output / "posterior.npz").exists()
    environment = json.loads((output / "environment.json").read_text())
    assert {"PyYAML", "typer"}.issubset(environment["dependencies"])


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


def test_legacy_pipeline_predictions_remain_in_canonical_panel_order(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema_version: 1\n"
        "data: {time: month, response: y}\n"
        "model: {fixed: '0 + month', fixed_prior_precision: 1.0, sigma: 1.0}\n"
    )
    pipeline = Pipeline.from_yaml(config)
    unsorted = pd.DataFrame({"month": [2, 1, 3], "y": [4.0, 2.0, 6.0]})
    sorted_frame = unsorted.sort_values("month").reset_index(drop=True)

    from_unsorted = pipeline.run(unsorted, tmp_path / "unsorted")
    from_sorted = pipeline.run(sorted_frame, tmp_path / "sorted")

    np.testing.assert_allclose(
        from_unsorted.predictive_mean, from_sorted.predictive_mean
    )
    np.testing.assert_allclose(
        from_unsorted.predictive_variance, from_sorted.predictive_variance
    )


def _fingerprint(frame: pd.DataFrame) -> str:
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y"))
    return panel_fingerprint(panel)


def test_legacy_panel_fingerprint_digest_is_stable() -> None:
    frame = pd.DataFrame({"month": [2, 1], "y": [2.0, 1.0]})

    assert _fingerprint(frame) == "ce57a6f79e9e60cb6be739cd37b6b11e4cfe7bd12a8fffd45f628d72fdc17cd6"


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


def test_object_fingerprint_supports_exact_numpy_complex_and_longdouble_scalars() -> None:
    complex_first = pd.DataFrame({"month": [1], "y": [1.0], "value": [np.complex128(1 + 2j)]})
    complex_second = pd.DataFrame({"month": [1], "y": [1.0], "value": [np.complex128(1 + 3j)]})

    assert _fingerprint(complex_first) != _fingerprint(complex_second)

    # Build object columns by numpy item-assignment so pandas cannot re-infer and
    # downcast a wide (80-bit) longdouble to float64. Assert only when the frame
    # actually stored distinct values: where the platform/pandas collapses this
    # sub-ULP step, the two frames are genuinely equal and must hash equal.
    def _longdouble_frame(scalar: np.longdouble) -> pd.DataFrame:
        column = np.empty(1, dtype=object)
        column[0] = scalar
        return pd.DataFrame({"month": [1], "y": [1.0], "value": column})

    first_frame = _longdouble_frame(np.longdouble(1))
    second_frame = _longdouble_frame(np.nextafter(np.longdouble(1), np.longdouble(2)))
    stored_first = next(first_frame.itertuples(index=False, name=None))[-1]
    stored_second = next(second_frame.itertuples(index=False, name=None))[-1]
    if stored_first != stored_second:
        assert _fingerprint(first_frame) != _fingerprint(second_frame)


def test_object_fingerprint_distinguishes_period_frequency_and_interval_endpoints() -> None:
    monthly = pd.DataFrame({"month": [1], "y": [1.0], "value": [pd.Period(ordinal=1, freq="M")]})
    daily = pd.DataFrame({"month": [1], "y": [1.0], "value": [pd.Period(ordinal=1, freq="D")]})
    integer_interval = pd.DataFrame({"month": [1], "y": [1.0], "value": [pd.Interval(1, 2)]})
    float_interval = pd.DataFrame({"month": [1], "y": [1.0], "value": [pd.Interval(1.0, 2.0)]})

    assert _fingerprint(monthly) != _fingerprint(daily)
    assert _fingerprint(integer_interval) != _fingerprint(float_interval)


def test_panel_contract_rejects_unsupported_numpy_scalar_before_fingerprint() -> None:
    unsupported = np.array((1, 2), dtype=[("left", "i4"), ("right", "i4")])[()]
    frame = pd.DataFrame({"month": [1], "y": [1.0], "value": [unsupported]})

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
    try:
        output.symlink_to(tmp_path / "missing")
    except (OSError, NotImplementedError) as error:
        # Windows blocks symlink creation without Developer Mode / admin.
        pytest.skip(f"symlinks unavailable on this platform: {error}")

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


def test_windows_publisher_declares_movefileex_signature_and_maps_existing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MoveFileEx:
        def __init__(self, result: int) -> None:
            self.result = result
            self.argtypes: object = None
            self.restype: object = None
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return self.result

    class Kernel32:
        def __init__(self, move: MoveFileEx) -> None:
            self.MoveFileExW = move

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    move = MoveFileEx(1)
    monkeypatch.setattr(run_artifacts.sys, "platform", "win32")
    monkeypatch.setattr(run_artifacts.os, "name", "nt")
    monkeypatch.setattr(run_artifacts.ctypes, "WinDLL", lambda *args, **kwargs: Kernel32(move), raising=False)
    monkeypatch.setattr(run_artifacts, "_destination_exists", lambda path: False)

    run_artifacts._publish_no_replace(source, destination)

    assert move.argtypes == [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    assert move.restype is ctypes.c_int
    assert move.calls == [(str(source), str(destination), 0x8)]

    existing = MoveFileEx(0)
    monkeypatch.setattr(run_artifacts.ctypes, "WinDLL", lambda *args, **kwargs: Kernel32(existing), raising=False)
    monkeypatch.setattr(run_artifacts.ctypes, "get_last_error", lambda: 183, raising=False)
    with pytest.raises(FileExistsError):
        run_artifacts._publish_no_replace(source, destination)


def test_linux_publisher_prefers_exported_renameat2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RenameAt2:
        def __init__(self) -> None:
            self.argtypes: object = None
            self.restype: object = None
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return 0

    class LibC:
        def __init__(self, rename: RenameAt2) -> None:
            self.renameat2 = rename

        def syscall(self, *args: object) -> int:
            raise AssertionError("raw syscall must not be used when renameat2 is exported")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    rename = RenameAt2()
    monkeypatch.setattr(run_artifacts.sys, "platform", "linux")
    monkeypatch.setattr(run_artifacts.platform, "machine", lambda: "unknown")
    monkeypatch.setattr(run_artifacts.ctypes, "CDLL", lambda *args, **kwargs: LibC(rename))
    monkeypatch.setattr(run_artifacts, "_destination_exists", lambda path: False)

    run_artifacts._publish_no_replace(source, destination)

    assert rename.argtypes == [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    assert rename.restype is ctypes.c_int
    assert rename.calls == [
        (
            run_artifacts._AT_FDCWD,
            run_artifacts.os.fsencode(source),
            run_artifacts._AT_FDCWD,
            run_artifacts.os.fsencode(destination),
            run_artifacts._RENAME_NOREPLACE,
        )
    ]

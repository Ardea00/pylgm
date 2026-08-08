import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import pylgm.experiment as experiment_module
from pylgm import Experiment
from pylgm.artifacts import experiment as experiment_artifacts
from pylgm.exceptions import DenseReferenceLimitError, OptimizationError


def _panel(offset: float = 0.0) -> pd.DataFrame:
    month = np.tile(np.arange(10), 2)
    region = np.repeat(["north", "south"], 10)
    return pd.DataFrame(
        {
            "region": region,
            "month": month,
            "y": offset + 1.0 + 0.4 * month + (region == "south") * 0.2,
        }
    )


def _config(path: Path, *, horizons: str = "[1, 2]", trend: str = "1 + month") -> Path:
    path.write_text(
        f"""schema_version: 2
data:
  time: month
  response: y
  panel: [region]
model:
  fixed: "1"
  sigma: 1.0
  effects: []
candidates:
  - {{name: base}}
  - name: trend
    overrides: {{fixed: "{trend}"}}
inference:
  engine: exact_gaussian
  hyperparameters:
    strategy: empirical_bayes
    optimize:
      sigma: {{initial: 1.0, lower: 0.1, upper: 5.0}}
evaluation:
  horizons: {horizons}
  origins: {{last: 2}}
  interval_levels: [0.8]
  max_abs_coverage_error: 1.0
"""
    )
    return path


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _summary_fingerprint(summary: dict[str, object]) -> str:
    core = {name: value for name, value in summary.items() if name != "artifact_fingerprint"}
    return hashlib.sha256(_canonical(core)).hexdigest()


def test_artifact_json_normalization_is_deep_and_noncoercive() -> None:
    payload = MappingProxyType(
        {"nested": (np.int64(2), np.longdouble("1.234567890123456789"), b"\x00\xff")}
    )

    normalized = experiment_artifacts._json_ready(payload)
    encoded = _canonical(normalized)

    assert json.loads(encoded)["nested"][0] == 2
    assert normalized["nested"][1]["type"] == "numpy_scalar"
    assert normalized["nested"][2] == {"type": "bytes", "value_hex": "00ff"}


def test_experiment_artifact_is_complete_versioned_and_hashed(tmp_path: Path) -> None:
    output = tmp_path / "comparison"

    result = Experiment.from_yaml(_config(tmp_path / "experiment.yaml")).compare(_panel(), output)

    summary = json.loads((output / "summary.json").read_text())
    expected_payloads = {
        "decision.json",
        "environment.json",
        "failures.json",
        "folds.json",
        "metrics.parquet",
        "optimization.parquet",
        "predictions.parquet",
        "resolved_candidates.json",
        "resolved_config.json",
    }
    assert summary["artifact_schema_version"] == 2
    assert summary["selected"] == result.selected
    assert summary["decision"] == result.decision.to_dict()
    assert set(summary["payload_sha256"]) == expected_payloads
    assert len(summary["data_fingerprint"]) == 64
    assert len(summary["experiment_fingerprint"]) == 64
    assert len(summary["artifact_fingerprint"]) == 64
    assert summary["artifact_fingerprint"] == _summary_fingerprint(summary)
    for name, digest in summary["payload_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert json.loads((output / "resolved_candidates.json").read_text()) == [
        candidate for candidate in summary["resolved_candidates"]
    ]
    assert json.loads((output / "decision.json").read_text()) == result.decision.to_dict()
    assert json.loads((output / "failures.json").read_text()) == {}
    persisted_predictions = pd.read_parquet(output / "predictions.parquet")
    pd.testing.assert_frame_equal(persisted_predictions, result.predictions)
    assert persisted_predictions["evaluation_mode"].eq("latest").all()
    persisted_metrics = pd.read_parquet(output / "metrics.parquet")
    pd.testing.assert_frame_equal(persisted_metrics, result.metrics)
    assert persisted_metrics["evaluation_mode"].eq("latest").all()
    levels = set(zip(persisted_metrics["origin"].notna(), persisted_metrics["horizon"].notna()))
    assert levels == {(True, True), (False, True), (False, False)}
    diagnostics = pd.read_parquet(output / "optimization.parquet")
    diagnostics["numerical_failures"] = diagnostics["numerical_failures"].map(tuple)
    pd.testing.assert_frame_equal(diagnostics, result.diagnostics)
    for name in expected_payloads - {
        "metrics.parquet",
        "optimization.parquet",
        "predictions.parquet",
    }:
        value = json.loads((output / name).read_text())
        assert (output / name).read_bytes() == _canonical(value)
    assert (output / "summary.json").read_bytes() == _canonical(summary)
    assert (
        json.loads((output / "resolved_config.json").read_text())["inference"]["allow_large_dense"]
        is False
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("engine", "tampered"),
        ("selected", "tampered"),
        ("candidates", ["tampered"]),
        ("decision", {"selected": "tampered"}),
        ("data_fingerprint", "0" * 64),
        ("resolved_candidates", []),
    ],
)
def test_artifact_fingerprint_rejects_manifest_tampering(
    tmp_path: Path, field: str, replacement: object
) -> None:
    output = tmp_path / "comparison"
    Experiment.from_yaml(_config(tmp_path / "experiment.yaml")).compare(_panel(), output)
    summary = json.loads((output / "summary.json").read_text())
    stored = summary["artifact_fingerprint"]

    summary[field] = replacement

    assert _summary_fingerprint(summary) != stored


def test_experiment_environment_records_all_direct_dependencies(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    Experiment.from_yaml(_config(tmp_path / "experiment.yaml")).compare(_panel(), output)

    environment = json.loads((output / "environment.json").read_text())

    assert environment["python"]
    assert environment["platform"]
    assert set(environment["dependencies"]) == {
        "pylgm",
        "formulaic",
        "numpy",
        "pandas",
        "pydantic",
        "PyYAML",
        "scipy",
        "typer",
        "pyarrow",
    }
    assert all(environment["dependencies"].values())
    assert environment["dependencies"]["pylgm"] == "0.3.0"


@pytest.mark.parametrize(
    ("change", "expected_different"),
    [
        ({"horizons": "[1]"}, True),
        ({"trend": "1 + month + C(region)"}, True),
        ({}, False),
    ],
)
def test_experiment_fingerprint_covers_fold_and_candidate_semantics(
    tmp_path: Path, change: dict[str, str], expected_different: bool
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    Experiment.from_yaml(_config(tmp_path / "first.yaml")).compare(_panel(), first)
    Experiment.from_yaml(_config(tmp_path / "second.yaml", **change)).compare(_panel(), second)

    left = json.loads((first / "summary.json").read_text())["experiment_fingerprint"]
    right = json.loads((second / "summary.json").read_text())["experiment_fingerprint"]

    assert (left != right) is expected_different


def test_experiment_fingerprint_changes_with_data(tmp_path: Path) -> None:
    config = _config(tmp_path / "experiment.yaml")
    Experiment.from_yaml(config).compare(_panel(), tmp_path / "first")
    Experiment.from_yaml(config).compare(_panel(0.25), tmp_path / "second")

    first = json.loads((tmp_path / "first" / "summary.json").read_text())
    second = json.loads((tmp_path / "second" / "summary.json").read_text())

    assert first["data_fingerprint"] != second["data_fingerprint"]
    assert first["experiment_fingerprint"] != second["experiment_fingerprint"]


def test_experiment_metrics_preserve_large_integer_origins(tmp_path: Path) -> None:
    base = 2**60 + 1
    panel = _panel()
    panel["month"] = panel["month"].astype("int64") + base
    output = tmp_path / "comparison"

    Experiment.from_yaml(_config(tmp_path / "experiment.yaml", trend="1")).compare(panel, output)

    metrics = pd.read_parquet(output / "metrics.parquet")
    folds = metrics.loc[metrics["origin"].notna()]
    assert str(metrics["origin"].dtype) == "Int64"
    assert str(metrics["horizon"].dtype) == "Int64"
    assert set(folds["origin"].astype("int64")) == {base + 6, base + 7}


def test_experiment_artifact_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "comparison"

    def fail_parquet(*args: object, **kwargs: object) -> None:
        raise OSError("injected Parquet failure")

    monkeypatch.setattr(experiment_artifacts, "_write_parquet", fail_parquet)

    with pytest.raises(OSError, match="injected Parquet failure"):
        Experiment.from_yaml(_config(tmp_path / "experiment.yaml")).compare(_panel(), output)

    assert not output.exists()
    assert not list(tmp_path.glob(".comparison.tmp-*"))


def test_experiment_artifact_does_not_replace_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    output.mkdir()
    marker = output / "owner.txt"
    marker.write_text("existing")

    with pytest.raises(FileExistsError):
        Experiment.from_yaml(_config(tmp_path / "experiment.yaml")).compare(_panel(), output)

    assert marker.read_text() == "existing"
    assert not list(tmp_path.glob(".comparison.tmp-*"))


def test_failed_optimizer_diagnostics_are_structured_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "comparison"
    real_optimizer = experiment_module.optimize_empirical_bayes

    def selectively_failing_optimizer(family, bounds, *, initial=None, allow_large_dense=False):
        if family.blocks[0].block.labels == ("Intercept",):
            cause = DenseReferenceLimitError("dense root cause")
            error = OptimizationError(
                "no valid optimum",
                evaluations=7,
                cache_hits=2,
                numerical_failures=(
                    "log_parameters=(0.0,): DenseReferenceLimitError: dense root cause",
                ),
            )
            raise error from cause
        return real_optimizer(
            family,
            bounds,
            initial=initial,
            allow_large_dense=allow_large_dense,
        )

    monkeypatch.setattr(
        experiment_module,
        "optimize_empirical_bayes",
        selectively_failing_optimizer,
    )

    result = Experiment.from_yaml(_config(tmp_path / "experiment.yaml")).compare(_panel(), output)
    failure = result.failures["base"][0]
    persisted = json.loads((output / "failures.json").read_text())

    assert failure.evaluations == 7
    assert failure.cache_hits == 2
    assert failure.numerical_failures == (
        "log_parameters=(0.0,): DenseReferenceLimitError: dense root cause",
    )
    assert failure.root_cause is not None
    assert failure.root_cause.type == "DenseReferenceLimitError"
    assert failure.root_cause.message == "dense root cause"
    assert persisted == {"base": [failure.to_dict()]}
    assert persisted["base"][0]["candidate"] == "base"
    assert persisted["base"][0]["type"] == "OptimizationError"
    assert persisted["base"][0]["origin"] == 6
    assert persisted["base"][0]["root_cause"] == {
        "type": "DenseReferenceLimitError",
        "message": "dense root cause",
    }
    snapshot = failure.to_dict()
    snapshot["numerical_failures"].append("mutated")
    assert len(failure.numerical_failures) == 1

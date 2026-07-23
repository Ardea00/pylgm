from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pylgm.experiment as experiment_module
from pylgm import Experiment
from pylgm.exceptions import (
    ConfigurationError,
    DenseReferenceLimitError,
    OptimizationError,
    SelectionError,
)


def _panel() -> pd.DataFrame:
    month = np.tile(np.arange(10), 2)
    region = np.repeat(["north", "south"], 10)
    return pd.DataFrame(
        {
            "region": region,
            "month": month,
            "y": 1.0 + 0.4 * month + (region == "south") * 0.2,
        }
    )


def _write_config(path: Path, candidates: str | None = None) -> Path:
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
{
            candidates
            or '''  - {name: base}
  - name: trend
    overrides: {fixed: "1 + month"}'''
        }
inference:
  engine: exact_gaussian
  hyperparameters:
    strategy: empirical_bayes
    optimize:
      sigma: {{initial: 1.0, lower: 0.1, upper: 5.0}}
evaluation:
  horizons: [1, 2]
  origins: {{last: 2}}
  interval_levels: [0.8]
  max_abs_coverage_error: 1.0
"""
    )
    return path


def test_experiment_compares_candidates_and_persistence(tmp_path: Path) -> None:
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml"))

    result = experiment.compare(_panel(), output=None)
    predictions = result.predictions
    benchmark = predictions["candidate"].eq("persistence")

    assert set(result.candidates) == {"base", "trend"}
    assert set(predictions["candidate"]) == {"base", "trend", "persistence"}
    assert result.selected in {"base", "trend"}
    assert predictions["origin"].notna().all()
    assert predictions["target_time"].notna().all()
    assert predictions.loc[benchmark, "engine"].eq("persistence").all()
    assert predictions.loc[~benchmark, "engine"].eq("exact_gaussian").all()
    assert predictions.loc[benchmark, "is_benchmark"].all()
    assert not predictions.loc[~benchmark, "is_benchmark"].any()


def test_reversed_explicit_origins_optimize_chronologically_and_warm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, dict[str, float] | None]] = []
    optimize = experiment_module.optimize_empirical_bayes

    def recording_optimizer(family, bounds, *, initial=None):
        calls.append((family.y.size, None if initial is None else dict(initial)))
        return optimize(family, bounds, initial=initial)

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", recording_optimizer)
    path = _write_config(tmp_path / "experiment.yaml", "  - {name: base}")
    path.write_text(path.read_text().replace("origins: {last: 2}", "origins: {values: [7, 6]}"))

    Experiment.from_yaml(path).compare(_panel())

    assert [row_count for row_count, _ in calls] == [14, 16]
    assert calls[0][1] is None
    assert calls[1][1] is not None


def test_persistence_is_reserved_for_the_benchmark(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "experiment.yaml", "  - {name: persistence}")

    with pytest.raises(ConfigurationError, match="persistence.*reserved|reserved.*persistence"):
        Experiment.from_yaml(path)


def test_optimizer_runs_once_per_candidate_origin_and_warm_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, float, dict[str, float] | None]] = []
    optimize = experiment_module.optimize_empirical_bayes

    def recording_optimizer(family, bounds, *, initial=None):
        calls.append(
            (
                family.y.size,
                float(family.y.max()),
                None if initial is None else dict(initial),
            )
        )
        return optimize(family, bounds, initial=initial)

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", recording_optimizer)
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml"))
    panel = _panel()
    panel.loc[panel["month"].ge(8), "y"] = 1_000_000.0

    result = experiment.compare(panel)

    assert [row_count for row_count, _, _ in calls] == [14, 16, 14, 16]
    assert all(maximum < 1_000_000.0 for _, maximum, _ in calls)
    assert calls[0][2] is None
    assert calls[1][2] is not None
    assert calls[2][2] is None
    assert calls[3][2] is not None
    assert len(result.diagnostics) == 4
    assert set(result.diagnostics["candidate"]) == {"base", "trend"}
    assert set(result.diagnostics["parameter"]) == {"sigma"}


def test_dense_guard_failure_is_recorded_without_stopping_other_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = """  - name: too_dense
    overrides:
      effects: [{name: bad, type: iid, index: region, precision: 1.0}]
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      bad.precision: {initial: 1.0, lower: 0.1, upper: 5.0}
  - {name: good}"""
    optimize = experiment_module.optimize_empirical_bayes

    def guarded_optimizer(family, bounds, *, initial=None):
        if any(block.block.name == "bad" for block in family.blocks):
            raise DenseReferenceLimitError("synthetic dense limit")
        return optimize(family, bounds, initial=initial)

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", guarded_optimizer)
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml", candidates))

    result = experiment.compare(_panel())

    assert result.selected == "good"
    assert set(result.predictions["candidate"]) == {"good", "persistence"}
    assert "DenseReferenceLimitError" in result.failures["too_dense"][0]
    assert "origin=" in result.failures["too_dense"][0]


def test_all_candidate_failures_raise_selection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_optimizer(*args, **kwargs):
        raise OptimizationError("no valid optimum")

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", failing_optimizer)
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml"))

    with pytest.raises(SelectionError, match="no eligible"):
        experiment.compare(_panel())


def test_programming_errors_are_not_converted_to_candidate_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_optimizer(*args, **kwargs):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", broken_optimizer)
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml"))

    with pytest.raises(RuntimeError, match="programming bug"):
        experiment.compare(_panel())


def test_comparison_is_deterministic_and_tables_are_defensive(
    tmp_path: Path,
) -> None:
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml"))
    source = _panel()
    source_before = source.copy(deep=True)

    first = experiment.compare(source)
    second = experiment.compare(source)

    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    assert first.decision.to_dict() == second.decision.to_dict()
    pd.testing.assert_frame_equal(source, source_before)
    exposed = first.predictions
    exposed.loc[0, "mean"] = -999.0
    assert first.predictions.loc[0, "mean"] != -999.0
    with pytest.raises(TypeError):
        first.failures["new"] = ("failure",)  # type: ignore[index]

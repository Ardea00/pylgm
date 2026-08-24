from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pylgm.experiment as experiment_module
from pylgm import Experiment
from pylgm.optimization import empirical_bayes
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
    assert predictions["evaluation_mode"].eq("latest").all()
    assert result.metrics["evaluation_mode"].eq("latest").all()


def test_reversed_explicit_origins_optimize_chronologically_and_warm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, dict[str, float] | None]] = []
    optimize = experiment_module.optimize_empirical_bayes

    def recording_optimizer(family, bounds, *, initial=None, allow_large_dense=False):
        calls.append((family.y.size, None if initial is None else dict(initial)))
        return optimize(
            family,
            bounds,
            initial=initial,
            allow_large_dense=allow_large_dense,
        )

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

    def recording_optimizer(family, bounds, *, initial=None, allow_large_dense=False):
        calls.append(
            (
                family.y.size,
                float(family.y.max()),
                None if initial is None else dict(initial),
            )
        )
        return optimize(
            family,
            bounds,
            initial=initial,
            allow_large_dense=allow_large_dense,
        )

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

    def guarded_optimizer(family, bounds, *, initial=None, allow_large_dense=False):
        if any(block.block.name == "bad" for block in family.blocks):
            raise DenseReferenceLimitError("synthetic dense limit")
        return optimize(
            family,
            bounds,
            initial=initial,
            allow_large_dense=allow_large_dense,
        )

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", guarded_optimizer)
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml", candidates))

    result = experiment.compare(_panel())

    assert result.selected == "good"
    assert set(result.predictions["candidate"]) == {"good", "persistence"}
    failure = result.failures["too_dense"][0]
    assert failure.candidate == "too_dense"
    assert failure.origin is not None
    assert failure.type == "DenseReferenceLimitError"
    assert failure.message == "synthetic dense limit"
    assert failure.evaluations == 0
    assert failure.cache_hits == 0
    assert failure.numerical_failures == ()
    assert failure.root_cause is None
    assert "DenseReferenceLimitError" in str(failure)
    assert "origin=" in str(failure)
    assert any(
        "DenseReferenceLimitError" in reason for reason in result.decision.reasons["too_dense"]
    )


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


def test_all_candidate_families_are_preflighted_before_any_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = """  - name: oversized_later_fold
    overrides:
      effects: [{name: time_levels, type: iid, index: month, precision: 1.0}]
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      time_levels.precision: {initial: 1.0, lower: 0.1, upper: 5.0}
  - {name: good}"""
    optimizer_candidates: list[tuple[str, ...]] = []
    real_optimizer = experiment_module.optimize_empirical_bayes

    def recording_optimizer(family, bounds, *, initial=None, allow_large_dense=False):
        names = tuple(block.block.name for block in family.blocks)
        optimizer_candidates.append(names)
        assert "time_levels" not in names
        return real_optimizer(
            family,
            bounds,
            initial=initial,
            allow_large_dense=allow_large_dense,
        )

    monkeypatch.setattr(experiment_module, "optimize_empirical_bayes", recording_optimizer)
    path = _write_config(tmp_path / "experiment.yaml", candidates)
    path.write_text(
        path.read_text()
        .replace("horizons: [1, 2]", "horizons: [1]")
        .replace("origins: {last: 2}", "origins: {values: [2894]}")
    )
    experiment = Experiment.from_yaml(path)
    panel = pd.DataFrame(
        {
            "region": ["only"] * 2896,
            "month": np.arange(2896),
            "y": np.arange(2896, dtype=float),
        }
    )

    result = experiment.compare(panel)

    assert optimizer_candidates == [("fixed",)]
    assert result.selected == "good"
    failure = result.failures["oversized_later_fold"][0]
    assert failure.stage == "preflight_prediction"
    assert failure.origin == 2894
    assert failure.horizon == 1
    assert failure.type == "DenseReferenceLimitError"
    assert "estimated dense workspace" in failure.message


def test_dense_override_flows_to_preflight_optimizer_evaluations_and_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path / "experiment.yaml", "  - {name: base}")
    path.write_text(
        path.read_text().replace(
            "engine: exact_gaussian",
            "engine: exact_gaussian\n  allow_large_dense: true",
        )
    )
    preflight_flags: list[bool] = []
    optimizer_fit_flags: list[bool] = []
    prediction_fit_flags: list[bool] = []
    real_preflight = experiment_module.preflight_dense_reference
    real_optimizer_fit = empirical_bayes.fit_gaussian
    real_prediction_fit = experiment_module.fit_gaussian

    def recording_preflight(model, *, allow_large_dense=False):
        preflight_flags.append(allow_large_dense)
        return real_preflight(model, allow_large_dense=allow_large_dense)

    def recording_optimizer_fit(model, *, allow_large_dense=False):
        optimizer_fit_flags.append(allow_large_dense)
        return real_optimizer_fit(model, allow_large_dense=allow_large_dense)

    def recording_prediction_fit(model, *, allow_large_dense=False):
        prediction_fit_flags.append(allow_large_dense)
        return real_prediction_fit(model, allow_large_dense=allow_large_dense)

    monkeypatch.setattr(experiment_module, "preflight_dense_reference", recording_preflight)
    monkeypatch.setattr(empirical_bayes, "fit_gaussian", recording_optimizer_fit)
    monkeypatch.setattr(experiment_module, "fit_gaussian", recording_prediction_fit)

    Experiment.from_yaml(path).compare(_panel())

    assert preflight_flags and all(preflight_flags)
    assert optimizer_fit_flags and all(optimizer_fit_flags)
    assert prediction_fit_flags and all(prediction_fit_flags)


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


def test_backtest_variance_is_on_the_response_scale(tmp_path: Path) -> None:
    """The scored variance column must carry the observation noise.

    ``evaluation.metrics`` feeds it to ``norm.logpdf(actual, ...)`` and to the
    interval bounds, so it is a RESPONSE-scale variance -- whereas
    ``predictive_variance`` is the linear-predictor variance and deliberately
    excludes sigma^2. Passing it through unadjusted silently made every
    probabilistic score wrong (log predictive density ~11x worse, intervals ~4x
    too narrow) while the whole suite still passed, because nothing pinned this
    column. The model's variance must exceed the bare linear-predictor variance
    by exactly the fitted sigma^2.
    """
    experiment = Experiment.from_yaml(_write_config(tmp_path / "experiment.yaml"))
    result = experiment.compare(_panel(), output=None)
    predictions = result.predictions
    model_rows = predictions[predictions["candidate"].eq("base")]
    assert not model_rows.empty

    variances = model_rows["variance"].to_numpy()
    assert np.all(variances > 0.0)
    # Dropping the observation noise shrinks these from ~0.75/0.96 to
    # ~0.050/0.057 on this fixture -- more than an order of magnitude. 0.5
    # separates the two regimes with wide margin without pinning exact values.
    assert np.all(variances > 0.5), (
        "scored variance looks like Var(eta) without the observation noise: "
        f"{sorted(set(np.round(variances, 6)))}"
    )

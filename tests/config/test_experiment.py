from pathlib import Path

import pytest

from pylgm.config import load_experiment_config, resolve_candidates
from pylgm.exceptions import ConfigurationError


def write_experiment(path: Path, extra: str = "") -> None:
    path.write_text(
        """schema_version: 2
data:
  time: month
  response: y
  panel: [region]
  covariates:
    lag1: {availability: observed_with_lag, lag: 1}
model:
  fixed: "1 + lag1"
  sigma: 1.0
  effects: [{name: trend, type: rw1, index: month, precision: 2.0}]
candidates:
  - {name: base}
inference:
  engine: exact_gaussian
  hyperparameters:
    strategy: empirical_bayes
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      trend.precision: {initial: 2.0, lower: 0.01, upper: 100.0}
evaluation:
  horizons: [1, 3, 6]
  origins: {last: 8}
  window: {type: expanding}
"""
        + extra
    )


def test_experiment_config_resolves_complete_candidates(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """schema_version: 2
data:
  time: month
  response: y
  panel: [region]
  covariates:
    lag1: {availability: observed_with_lag, lag: 1}
model:
  fixed: "1 + lag1"
  sigma: 1.0
  effects: [{name: trend, type: rw1, index: month, precision: 2.0}]
candidates:
  - {name: base}
  - name: iid_only
    overrides:
      effects: [{name: region, type: iid, index: region, precision: 1.0}]
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      region.precision: {initial: 1.0, lower: 0.01, upper: 100.0}
inference:
  engine: exact_gaussian
  hyperparameters:
    strategy: empirical_bayes
    optimize:
      sigma: {initial: 1.0, lower: 0.1, upper: 5.0}
      trend.precision: {initial: 2.0, lower: 0.01, upper: 100.0}
evaluation:
  horizons: [1, 3, 6]
  origins: {last: 8}
  window: {type: expanding}
"""
    )

    config = load_experiment_config(path)
    candidates = resolve_candidates(config)

    assert [candidate.name for candidate in candidates] == ["base", "iid_only"]
    assert candidates[0].model.effects[0].name == "trend"
    assert candidates[1].model.effects[0].name == "region"
    assert "region.precision" in candidates[1].optimize


@pytest.mark.parametrize("version", [True, 2.0, "2", 1])
def test_schema_version_must_be_integer_two(tmp_path: Path, version: object) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace("schema_version: 2", f"schema_version: {version!r}"))

    with pytest.raises(ConfigurationError, match="schema_version"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("candidates: []", "candidates"),
        ("candidates:\n  - {name: base}\n  - {name: base}", "unique"),
        ("evaluation:\n  horizons: [1, 1]\n  origins: {last: 8}", "horizons"),
        ("evaluation:\n  horizons: [0]\n  origins: {last: 8}", "positive"),
    ],
)
def test_candidate_and_horizon_constraints(
    tmp_path: Path, replacement: str, message: str
) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    text = path.read_text()
    if replacement.startswith("candidates:"):
        path.write_text(text.replace("candidates:\n  - {name: base}", replacement))
    else:
        path.write_text(text.replace(text[text.index("evaluation:") :], replacement))

    with pytest.raises(ConfigurationError, match=message):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("window: {type: expanding}", "window: {type: rolling}", "length"),
        ("window: {type: expanding}", "window: {type: expanding, length: 4}", "rolling"),
        ("window: {type: expanding}", "mode: vintage\n  window: {type: expanding}", "vintage"),
        ("data:\n  time: month", "data:\n  vintage: release\n  time: month", "vintage"),
        ("strategy: empirical_bayes", "strategy: grid", "strategy"),
    ],
)
def test_experiment_mode_and_strategy_constraints(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace(old, new))

    with pytest.raises(ConfigurationError, match=message):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ("{initial: .inf, lower: 0.1, upper: 5}", "finite"),
        ("{initial: 6, lower: 0.1, upper: 5}", "lower <= initial <= upper"),
        ("{initial: 1, lower: 1, upper: 1}", "lower <= initial <= upper"),
    ],
)
def test_optimization_bounds_are_finite_and_ordered(
    tmp_path: Path, bounds: str, message: str
) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace("{initial: 1.0, lower: 0.1, upper: 5.0}", bounds))

    with pytest.raises(ConfigurationError, match=message):
        load_experiment_config(path)


def test_resolved_candidate_validates_optimized_effect_and_replaces_lists(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace("""candidates:
  - {name: base}""", """candidates:
  - name: bad
    overrides:
      effects: [{name: region, type: iid, index: region}]
    optimize:
      trend.precision: {initial: 1, lower: 0.1, upper: 2}
"""))

    with pytest.raises(ConfigurationError, match="unknown optimized effect"):
        load_experiment_config(path)


def test_all_resolved_formulas_require_covariate_availability(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace("""candidates:
  - {name: base}""", """candidates:
  - name: missing_availability
    overrides: {fixed: "1 + undeclared"}
"""))

    with pytest.raises(ConfigurationError, match="availability"):
        load_experiment_config(path)

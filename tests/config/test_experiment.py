from pathlib import Path
import json
import pytest

from pylgm.config import ExperimentConfig, load_experiment_config, resolve_candidates
from pylgm.config.experiment import ExperimentDataConfig
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
        ("evaluation:\n  horizons: [0]\n  origins: {last: 8}", "greater than 0"),
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


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("lag: 1", "lag: true"),
        ("lag: 1", "lag: '1'"),
        ("origins: {last: 8}", "origins: {last: 8.0}"),
        ("origins: {last: 8}", "origins: {last: true}"),
        ("window: {type: expanding}", "window: {type: rolling, length: '2'}"),
        ("window: {type: expanding}", "window: {type: rolling, length: 2.0}"),
        ("horizons: [1, 3, 6]", "horizons: [1, '3', 6]"),
        ("horizons: [1, 3, 6]", "horizons: [1, true, 6]"),
        ("horizons: [1, 3, 6]", "horizons: [1, 3.0, 6]"),
        ("origins: {last: 8}", "origins: {values: [1.0]}"),
        ("origins: {last: 8}", "origins: {values: [true]}"),
    ],
)
def test_integer_v2_fields_reject_coercion(tmp_path: Path, old: str, new: str) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace(old, new))

    with pytest.raises(ConfigurationError):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("window: {type: expanding}", "interval_levels: ['0.5']\n  window: {type: expanding}"),
        ("window: {type: expanding}", "interval_levels: [true]\n  window: {type: expanding}"),
        ("window: {type: expanding}", "max_abs_coverage_error: '0.1'\n  window: {type: expanding}"),
        ("window: {type: expanding}", "max_abs_coverage_error: true\n  window: {type: expanding}"),
        ("window: {type: expanding}", "rmse_tie_tolerance: '0.1'\n  window: {type: expanding}"),
        ("window: {type: expanding}", "rmse_tie_tolerance: false\n  window: {type: expanding}"),
    ],
)
def test_real_v2_fields_reject_coercion(tmp_path: Path, old: str, new: str) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace(old, new))

    with pytest.raises(ConfigurationError):
        load_experiment_config(path)


def test_mapping_surfaces_are_isolated_immutable_and_serializable(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    payload = __import__("yaml").safe_load(path.read_text())
    payload["candidates"].append(
        {
            "name": "candidate",
            "optimize": {
                "sigma": {"initial": 1.0, "lower": 0.1, "upper": 5.0},
            },
        }
    )
    config = ExperimentConfig.model_validate(payload)
    payload["data"]["covariates"]["lag1"] = {"availability": "known_future"}
    payload["inference"]["hyperparameters"]["optimize"].clear()
    payload["candidates"][1]["optimize"].clear()

    candidates = resolve_candidates(config)
    assert config.data.covariates["lag1"].availability == "observed_with_lag"
    assert "sigma" in config.inference.hyperparameters.optimize
    assert candidates[0].optimize is not config.inference.hyperparameters.optimize
    assert candidates[0].optimize == config.inference.hyperparameters.optimize
    assert candidates[1].optimize is not config.candidates[1].optimize
    assert candidates[1].optimize == config.candidates[1].optimize
    with pytest.raises(TypeError):
        config.data.covariates["other"] = config.data.covariates["lag1"]  # type: ignore[index]
    with pytest.raises(TypeError):
        config.inference.hyperparameters.optimize["other"] = next(iter(candidates[0].optimize.values()))  # type: ignore[index]
    with pytest.raises(TypeError):
        config.candidates[1].optimize["other"] = next(iter(candidates[1].optimize.values()))  # type: ignore[index,union-attr]
    with pytest.raises(TypeError):
        candidates[0].optimize["other"] = next(iter(candidates[0].optimize.values()))  # type: ignore[index]
    assert config.model_dump()["data"]["covariates"]["lag1"]["lag"] == 1
    assert '"lag1"' in config.model_dump_json()


@pytest.mark.parametrize("formula", ["1 + (", "1 + ["])
def test_formulaic_errors_become_configuration_errors(tmp_path: Path, formula: str) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace('fixed: "1 + lag1"', f'fixed: "{formula}"'))

    with pytest.raises(ConfigurationError):
        load_experiment_config(path)


def test_candidate_formulaic_errors_become_configuration_errors(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    path.write_text(path.read_text().replace("- {name: base}", "- name: base\n    overrides: {fixed: '1 + ('}"))

    with pytest.raises(ConfigurationError):
        load_experiment_config(path)


def test_resolved_candidate_serializes_to_an_isolated_json_ready_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment.yaml"
    write_experiment(path)
    config = load_experiment_config(path)
    candidate, other = resolve_candidates(config), resolve_candidates(config)

    first = candidate[0].to_dict()
    second = candidate[0].to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["name"] == "base"
    assert first["model"]["sigma"] == 1.0  # type: ignore[index]
    assert first["model"]["effects"][0]["name"] == "trend"  # type: ignore[index]
    assert first["optimize"]["trend.precision"]["upper"] == 100.0  # type: ignore[index]
    first["model"]["effects"][0]["name"] = "changed"  # type: ignore[index]
    first["optimize"]["sigma"]["upper"] = 99.0  # type: ignore[index]
    assert candidate[0].model.effects[0].name == "trend"
    assert candidate[0].optimize["sigma"].upper == 5.0
    assert config.inference.hyperparameters.optimize["sigma"].upper == 5.0
    assert other[0].model.effects[0].name == "trend"
    assert other[0].optimize["sigma"].upper == 5.0


def test_omitted_covariates_default_is_immutable_isolated_and_serializable() -> None:
    first = ExperimentDataConfig(time="month", response="y")
    second = ExperimentDataConfig(time="month", response="y")

    assert first.covariates is not second.covariates
    assert first.model_dump()["covariates"] == {}
    assert first.model_dump_json() == second.model_dump_json()
    with pytest.raises(TypeError):
        first.covariates["lag1"] = object()  # type: ignore[index]

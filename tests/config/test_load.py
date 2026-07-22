from pathlib import Path

import pytest

from pylgm.config import load_config
from pylgm.exceptions import ConfigurationError


def test_load_config_resolves_defaults(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 1
data: {time: month, response: y, panel: [region]}
model:
  fixed: "1 + x"
  sigma: 0.5
  effects:
    - {name: trend, type: rw1, index: month, precision: 2.0}
"""
    )
    config = load_config(path)
    assert config.model.likelihood == "gaussian"
    assert config.model.fixed_prior_precision == 1e-6
    assert config.model.effects[0].type == "rw1"


def test_unknown_key_is_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\ndata: {time: t, response: y}\nmodel: {fixed: '1', typo: 3}\n")
    with pytest.raises(ConfigurationError, match="typo"):
        load_config(path)


@pytest.mark.parametrize(
    "document",
    [
        "schema_version: 1\ndata: {time: t, time: month, response: y}\nmodel: {sigma: 1}\n",
        (
            "schema_version: 1\ndata: {time: t, response: y}\nmodel:\n"
            "  sigma: 1\n  effects:\n    - {name: trend, type: rw1, index: t, index: month}\n"
        ),
    ],
)
def test_duplicate_mapping_key_at_any_nesting_is_configuration_error(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(document)

    with pytest.raises(ConfigurationError, match="duplicate mapping key"):
        load_config(path)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_schema_version_requires_the_actual_integer_one(
    tmp_path: Path, schema_version: object
) -> None:
    path = tmp_path / "version.yaml"
    path.write_text(
        f"schema_version: {schema_version!r}\n"
        "data: {time: t, response: y}\nmodel: {sigma: 1}\n"
    )

    with pytest.raises(ConfigurationError, match="schema_version"):
        load_config(path)


@pytest.mark.parametrize(
    "model",
    [
        "{sigma: .inf}",
        "{sigma: 0}",
        "{sigma: 1, fixed_prior_precision: .nan}",
        "{sigma: 1, fixed_prior_precision: -1}",
        "{sigma: 1, effects: [{name: trend, type: rw1, index: t, precision: .inf}]}",
        "{sigma: 1, effects: [{name: trend, type: rw1, index: t, precision: 0}]}",
    ],
)
def test_configured_precisions_and_sigma_are_finite_and_positive(
    tmp_path: Path, model: str
) -> None:
    path = tmp_path / "precision.yaml"
    path.write_text(f"schema_version: 1\ndata: {{time: t, response: y}}\nmodel: {model}\n")

    with pytest.raises(ConfigurationError, match="finite|greater than 0"):
        load_config(path)


def test_fixed_effect_name_is_reserved(tmp_path: Path) -> None:
    path = tmp_path / "fixed-name.yaml"
    path.write_text(
        "schema_version: 1\ndata: {time: t, response: y}\nmodel:\n"
        "  sigma: 1\n  effects: [{name: fixed, type: iid, index: group}]\n"
    )

    with pytest.raises(ConfigurationError, match="reserved"):
        load_config(path)

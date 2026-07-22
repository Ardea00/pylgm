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

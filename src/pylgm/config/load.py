from pathlib import Path

import yaml
from pydantic import ValidationError

from pylgm.config.schema import RunConfig
from pylgm.exceptions import ConfigurationError


def load_config(path: Path) -> RunConfig:
    try:
        payload = yaml.safe_load(path.read_text())
        return RunConfig.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc

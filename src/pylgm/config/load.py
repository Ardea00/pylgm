from pathlib import Path

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from pylgm.config.schema import RunConfig
from pylgm.exceptions import ConfigurationError


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that treats duplicate mapping keys as invalid input."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate mapping key: {key!r}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_config(path: Path) -> RunConfig:
    try:
        payload = yaml.load(path.read_text(), Loader=_UniqueKeySafeLoader)
        return RunConfig.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc

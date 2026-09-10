from pathlib import Path

import yaml
from formulaic.errors import FormulaicError
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from pylgm.config.schema import RunConfig
from pylgm.config.experiment import ExperimentConfig, resolve_candidates
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


def _inline_graphs(payload: object, base_dir: Path) -> object:
    """Replace every ``graph_file``/``graph_files`` reference with the graph itself.

    Paths in a document are relative to the document, but the compiler builds
    effects without knowing where the YAML came from. Reading them here -- the
    last point where the directory is known -- keeps a config that references
    ``graph.json`` next to it working from any working directory.
    """
    from pylgm.effects import load_graph_file

    if isinstance(payload, list):
        return [_inline_graphs(item, base_dir) for item in payload]
    if not isinstance(payload, dict):
        return payload
    result = {key: _inline_graphs(value, base_dir) for key, value in payload.items()}
    if isinstance(result.get("graph_file"), str):
        result["graph"] = load_graph_file(base_dir / result.pop("graph_file"))
    files = result.get("graph_files")
    if isinstance(files, dict):
        result.pop("graph_files")
        result["graphs"] = {
            key: load_graph_file(base_dir / name) for key, name in files.items()
        }
    return result


def load_config(path: Path) -> RunConfig:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
        return RunConfig.model_validate(_inline_graphs(payload, path.parent))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


def load_experiment_config(path: Path) -> ExperimentConfig:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
        config = ExperimentConfig.model_validate(_inline_graphs(payload, path.parent))
        resolve_candidates(config)
        return config
    except (
        OSError,
        yaml.YAMLError,
        ValidationError,
        TypeError,
        ValueError,
        FormulaicError,
    ) as exc:
        raise ConfigurationError(str(exc)) from exc

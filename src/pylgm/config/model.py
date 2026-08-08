"""Standalone YAML frontend for declarative latent Gaussian models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import Field, ValidationError

from pylgm.exceptions import ConfigurationError

from pylgm.config.load import _UniqueKeySafeLoader
from pylgm.config.schema import FinitePositiveFloat, StrictModel

if TYPE_CHECKING:
    from pylgm.model import LGM


class _DataModelConfig(StrictModel):
    panel: tuple[str, ...] = ()
    time: str | None = None


class _LikelihoodModelConfig(StrictModel):
    family: Literal["gaussian"]
    sigma: FinitePositiveFloat


class _EffectModelConfig(StrictModel):
    name: str
    type: Literal["iid", "rw1", "rw2"]
    index: str
    precision: FinitePositiveFloat


class _PredictorModelConfig(StrictModel):
    fixed: str = "1"
    effects: tuple[_EffectModelConfig, ...] = ()


class _StandaloneModelConfig(StrictModel):
    response: str
    likelihood: _LikelihoodModelConfig
    data: _DataModelConfig = Field(default_factory=_DataModelConfig)
    predictor: _PredictorModelConfig


def _build_effect(config: _EffectModelConfig) -> object:
    from pylgm.effects import IID, RW1, RW2

    effect_types = {"iid": IID, "rw1": RW1, "rw2": RW2}
    return effect_types[config.type](config.name, config.index, config.precision)


def _build_model(config: _StandaloneModelConfig) -> LGM:
    from pylgm.effects import Fixed, Predictor
    from pylgm.likelihoods import Gaussian
    from pylgm.model import LGM

    effects = (Fixed(config.predictor.fixed),) + tuple(
        _build_effect(effect) for effect in config.predictor.effects
    )
    return LGM(
        response=config.response,
        likelihood=Gaussian(config.likelihood.sigma),
        predictor=Predictor(effects),
        panel=config.data.panel,
        time=config.data.time,
    )


def load_model(path: Path) -> LGM:
    """Load a standalone declarative :class:`LGM` YAML document."""
    try:
        payload = yaml.load(path.read_text(), Loader=_UniqueKeySafeLoader)
        return _build_model(_StandaloneModelConfig.model_validate(payload))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


__all__ = ["load_model"]

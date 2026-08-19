"""Standalone YAML frontend for declarative latent Gaussian models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from pylgm.exceptions import ConfigurationError

from pylgm.config.load import _UniqueKeySafeLoader
from pylgm.config.schema import FinitePositiveFloat, StrictModel

if TYPE_CHECKING:
    from pylgm.model import LGM


class _DataModelConfig(StrictModel):
    panel: tuple[str, ...] = ()
    time: str | None = None


class _LikelihoodModelConfig(StrictModel):
    family: Literal["gaussian", "poisson", "bernoulli"]
    sigma: FinitePositiveFloat | None = None

    @model_validator(mode="after")
    def _sigma_matches_family(self) -> "_LikelihoodModelConfig":
        if self.family == "gaussian" and self.sigma is None:
            raise ValueError("sigma is required for likelihood: {family: gaussian}")
        if self.family != "gaussian" and self.sigma is not None:
            raise ValueError(f"sigma is not a valid field for likelihood: {{family: {self.family}}}")
        return self


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
    offset: str | None = None
    data: _DataModelConfig = Field(default_factory=_DataModelConfig)
    predictor: _PredictorModelConfig


def _build_effect(config: _EffectModelConfig) -> object:
    from pylgm.effects import IID, RW1, RW2

    effect_types = {"iid": IID, "rw1": RW1, "rw2": RW2}
    return effect_types[config.type](config.name, config.index, config.precision)


def _build_likelihood(config: _LikelihoodModelConfig) -> object:
    from pylgm.likelihoods import Bernoulli, Gaussian, Poisson

    if config.family == "gaussian":
        return Gaussian(config.sigma)
    if config.family == "poisson":
        return Poisson()
    return Bernoulli()


def _build_model(config: _StandaloneModelConfig) -> LGM:
    from pylgm.effects import Fixed, Predictor
    from pylgm.model import LGM

    effects = (Fixed(config.predictor.fixed),) + tuple(
        _build_effect(effect) for effect in config.predictor.effects
    )
    return LGM(
        response=config.response,
        likelihood=_build_likelihood(config.likelihood),
        predictor=Predictor(effects),
        panel=config.data.panel,
        time=config.data.time,
        offset=config.offset,
    )


def load_model(path: Path) -> LGM:
    """Load a standalone declarative :class:`LGM` YAML document."""
    try:
        payload = yaml.load(path.read_text(), Loader=_UniqueKeySafeLoader)
        return _build_model(_StandaloneModelConfig.model_validate(payload))
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


__all__ = ["load_model"]

"""Standalone YAML frontend for declarative latent Gaussian models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from pylgm.exceptions import ConfigurationError

from pylgm.config.load import _UniqueKeySafeLoader
from pylgm.config.schema import (
    EffectConfig,
    FinitePositiveFloat,
    StrictModel,
    build_effect,
)

if TYPE_CHECKING:
    from pylgm.model import LGM


class _DataModelConfig(StrictModel):
    panel: tuple[str, ...] = ()
    time: str | None = None


_PHI_FAMILIES = ("nbinomial", "gamma", "beta")


class _LikelihoodModelConfig(StrictModel):
    family: Literal["gaussian", "poisson", "bernoulli", "binomial", "nbinomial",
                    "gamma", "beta", "weibullsurv", "exponentialsurv"]
    sigma: FinitePositiveFloat | None = None
    # NB/Gamma/Beta dispersion/precision; fixed only in YAML (a Hyperparameter phi
    # stays Python-API-only), optional because the families default it to 1.0.
    phi: FinitePositiveFloat | None = None
    # Binomial per-row trials column name (required for binomial, rejected otherwise).
    trials: str | None = None
    # Survival event/censoring indicator column (required for weibullsurv/exponentialsurv).
    event: str | None = None
    # Survival left-truncation entry-time column (weibullsurv/exponentialsurv only).
    entry: str | None = None
    # Weibull shape alpha; fixed only in YAML (a Hyperparameter shape stays
    # Python-API-only), optional because weibullsurv defaults it to 1.0.
    shape: FinitePositiveFloat | None = None

    @model_validator(mode="after")
    def _fields_match_family(self) -> "_LikelihoodModelConfig":
        if self.family == "gaussian" and self.sigma is None:
            raise ValueError("sigma is required for likelihood: {family: gaussian}")
        if self.family != "gaussian" and self.sigma is not None:
            raise ValueError(f"sigma is not a valid field for likelihood: {{family: {self.family}}}")
        if self.family not in _PHI_FAMILIES and self.phi is not None:
            raise ValueError(f"phi is not a valid field for likelihood: {{family: {self.family}}}")
        if self.family == "binomial" and not self.trials:
            raise ValueError("trials is required for likelihood: {family: binomial}")
        if self.family != "binomial" and self.trials is not None:
            raise ValueError(f"trials is not a valid field for likelihood: {{family: {self.family}}}")
        survival = self.family in ("weibullsurv", "exponentialsurv")
        if survival and not self.event:
            raise ValueError(f"event is required for likelihood: {{family: {self.family}}}")
        if not survival and self.event is not None:
            raise ValueError(f"event is not a valid field for likelihood: {{family: {self.family}}}")
        if not survival and self.entry is not None:
            raise ValueError(f"entry is not a valid field for likelihood: {{family: {self.family}}}")
        if self.family != "weibullsurv" and self.shape is not None:
            raise ValueError(f"shape is not a valid field for likelihood: {{family: {self.family}}}")
        return self


class _PredictorModelConfig(StrictModel):
    fixed: str = "1"
    effects: tuple[EffectConfig, ...] = ()

    @model_validator(mode="after")
    def named_effects(self) -> "_PredictorModelConfig":
        # Only the inner spec of a wrapper may omit its name.
        if not all(effect.name for effect in self.effects):
            raise ValueError("effect names must not be empty")
        return self


class _StandaloneModelConfig(StrictModel):
    response: str
    likelihood: _LikelihoodModelConfig
    offset: str | None = None
    data: _DataModelConfig = Field(default_factory=_DataModelConfig)
    predictor: _PredictorModelConfig


def _build_likelihood(config: _LikelihoodModelConfig) -> object:
    from pylgm.likelihoods import (
        Bernoulli,
        Beta,
        Binomial,
        ExponentialSurv,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        WeibullSurv,
    )

    if config.family == "gaussian":
        return Gaussian(config.sigma)
    if config.family == "poisson":
        return Poisson()
    if config.family == "bernoulli":
        return Bernoulli()
    if config.family == "binomial":
        return Binomial(config.trials)
    if config.family == "weibullsurv":
        shape = 1.0 if config.shape is None else config.shape
        return WeibullSurv(config.event, shape=shape, entry=config.entry)
    if config.family == "exponentialsurv":
        return ExponentialSurv(config.event, entry=config.entry)
    # phi-families: phi is optional in YAML, the family defaults it to 1.0.
    phi_families = {"nbinomial": NegativeBinomial, "gamma": Gamma, "beta": Beta}
    family = phi_families[config.family]
    return family() if config.phi is None else family(config.phi)


def _build_model(config: _StandaloneModelConfig, base_dir: Path) -> LGM:
    from pylgm.effects import Fixed, Predictor
    from pylgm.model import LGM

    effects = (Fixed(config.predictor.fixed),) + tuple(
        build_effect(effect, base_dir) for effect in config.predictor.effects
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
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
        return _build_model(_StandaloneModelConfig.model_validate(payload), path.parent)
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


__all__ = ["load_model"]

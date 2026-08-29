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


_SIMPLE_EFFECTS = ("iid", "rw1", "rw2")
_SPATIAL_EFFECTS = ("besag", "proper_car", "bym2")


class _EffectModelConfig(StrictModel):
    name: str
    type: Literal["iid", "rw1", "rw2", "besag", "proper_car", "bym2"]
    index: str
    # Fixed values only; estimating precision/rho/phi (a Hyperparameter) stays
    # Python-API-only. Spatial effects default precision in the builder, so it is
    # optional here and required for the simple effects via the validator below.
    precision: FinitePositiveFloat | None = None
    graph: dict | None = None
    graph_file: str | None = None
    rho: float | None = None
    phi: float | None = None
    scale: bool | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> "_EffectModelConfig":
        if self.type in _SIMPLE_EFFECTS:
            if self.precision is None:
                raise ValueError(f"precision is required for effect type {self.type!r}")
            present = [
                name
                for name in ("graph", "graph_file", "rho", "phi", "scale")
                if getattr(self, name) is not None
            ]
            if present:
                raise ValueError(f"fields {present} are not valid for effect type {self.type!r}")
            return self
        if (self.graph is None) == (self.graph_file is None):
            raise ValueError(
                f"spatial effect {self.name!r} requires exactly one of graph or graph_file"
            )
        if self.type == "proper_car" and self.rho is None:
            raise ValueError("rho is required for effect type 'proper_car'")
        if self.type != "proper_car" and self.rho is not None:
            raise ValueError(f"rho is not a valid field for effect type {self.type!r}")
        if self.type != "bym2" and self.phi is not None:
            raise ValueError(f"phi is not a valid field for effect type {self.type!r}")
        if self.type != "besag" and self.scale is not None:
            raise ValueError(f"scale is not a valid field for effect type {self.type!r}")
        return self


class _PredictorModelConfig(StrictModel):
    fixed: str = "1"
    effects: tuple[_EffectModelConfig, ...] = ()


class _StandaloneModelConfig(StrictModel):
    response: str
    likelihood: _LikelihoodModelConfig
    offset: str | None = None
    data: _DataModelConfig = Field(default_factory=_DataModelConfig)
    predictor: _PredictorModelConfig


def _build_effect(config: _EffectModelConfig, base_dir: Path) -> object:
    from pylgm.effects import IID, RW1, RW2, Besag, BYM2, ProperCAR, load_graph_file

    if config.type in _SIMPLE_EFFECTS:
        simple = {"iid": IID, "rw1": RW1, "rw2": RW2}
        return simple[config.type](config.name, config.index, config.precision)

    graph = load_graph_file(base_dir / config.graph_file) if config.graph_file else config.graph
    precision = 1.0 if config.precision is None else config.precision
    if config.type == "besag":
        scale = True if config.scale is None else config.scale
        return Besag(config.name, config.index, graph, precision, scale)
    if config.type == "proper_car":
        return ProperCAR(config.name, config.index, graph, config.rho, precision)
    phi = 0.5 if config.phi is None else config.phi
    return BYM2(config.name, config.index, graph, precision, phi)


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
        _build_effect(effect, base_dir) for effect in config.predictor.effects
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
        return _build_model(_StandaloneModelConfig.model_validate(payload), path.parent)
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


__all__ = ["load_model"]

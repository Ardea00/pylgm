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
_TEMPORAL_EFFECTS = ("ar1", "seasonal")
_MIDAS_EFFECTS = ("midas", "midas_parametric")
# Every optional effect field, and the set each type accepts. Anything set but
# not in a type's set is rejected -- one table instead of per-branch lists, so a
# new effect cannot silently swallow a field that belongs to another.
_OPTIONAL_FIELDS = (
    "index", "space", "time", "unit", "columns", "precision", "graph",
    "graph_file", "graphs", "graph_files", "rho", "phi", "scale", "group",
    "period", "order", "ridge", "kernel", "interaction", "gamma", "eta",
)
_ALLOWED_FIELDS = {
    "iid": {"index", "precision"},
    "rw1": {"index", "precision"},
    "rw2": {"index", "precision"},
    "ar1": {"index", "precision", "rho", "group"},
    "seasonal": {"index", "precision", "period", "ridge"},
    "besag": {"index", "precision", "graph", "graph_file", "scale"},
    "proper_car": {"index", "precision", "graph", "graph_file", "rho"},
    "bym2": {"index", "precision", "graph", "graph_file", "phi"},
    "sar": {"index", "precision", "graph", "graph_file", "rho"},
    "spacetime": {"space", "time", "precision", "graph", "graph_file",
                  "interaction", "order", "scale"},
    "dynamicspatialpanel": {"unit", "time", "graphs", "graph_files", "rho",
                            "gamma", "eta", "precision"},
    "midas": {"columns", "precision", "order", "ridge"},
    "midas_parametric": {"columns", "kernel"},
}


class _EffectModelConfig(StrictModel):
    name: str
    type: Literal["iid", "rw1", "rw2", "besag", "proper_car", "bym2", "sar",
                  "ar1", "seasonal", "spacetime", "dynamicspatialpanel",
                  "midas", "midas_parametric"]
    # Most effects are indexed by a single column; MIDAS uses a list of HF lag
    # columns instead, spacetime a (space, time) pair, and dynamicspatialpanel a
    # (unit, time) pair. Which fields each type accepts is _ALLOWED_FIELDS above.
    index: str | None = None
    space: str | None = None
    time: str | None = None
    unit: str | None = None
    # Fixed values only; estimating precision/rho/phi (a Hyperparameter) stays
    # Python-API-only. Spatial effects default precision in the builder, so it is
    # optional here and required for the simple effects via the validator below.
    precision: FinitePositiveFloat | None = None
    graph: dict | None = None
    graph_file: str | None = None
    # DynamicSpatialPanel per-period networks: inline {period: graph} or a
    # {period: filename} mapping (each file loaded via load_graph_file).
    graphs: dict | None = None
    graph_files: dict | None = None
    rho: float | None = None
    phi: float | None = None
    scale: bool | None = None
    # DynamicSpatialPanel temporal (gamma) and spatio-temporal-diffusion (eta)
    # coefficients; both default to 0.0 in the builder.
    gamma: float | None = None
    eta: float | None = None
    # Temporal family: group-wise AR1 panel column (ar1); cycle length (seasonal).
    group: str | None = None
    period: int | None = None
    # MIDAS family: lag columns; RW smoothness order and ridge (midas/seasonal);
    # the parametric lag-weight kernel (midas_parametric). spacetime interaction
    # type. Kernel/interaction shapes stay Python-API-only where estimated.
    columns: tuple[str, ...] | None = None
    order: int | None = None
    ridge: FinitePositiveFloat | None = None
    kernel: str | None = None
    interaction: str | None = None

    @model_validator(mode="after")
    def _fields_match_type(self) -> "_EffectModelConfig":
        present = {f for f in _OPTIONAL_FIELDS if getattr(self, f) is not None}
        bad = present - _ALLOWED_FIELDS[self.type]
        if bad:
            raise ValueError(
                f"fields {sorted(bad)} are not valid for effect type {self.type!r}"
            )
        if self.type in _MIDAS_EFFECTS:
            return self._validate_midas()
        if self.type == "spacetime":
            return self._validate_spacetime()
        if self.type == "dynamicspatialpanel":
            return self._validate_dsp()
        # Single-index effects (simple, temporal, spatial).
        if self.index is None:
            raise ValueError(f"index is required for effect type {self.type!r}")
        if self.type in _SIMPLE_EFFECTS:
            if self.precision is None:
                raise ValueError(f"precision is required for effect type {self.type!r}")
            return self
        if self.type in _TEMPORAL_EFFECTS:
            if self.type == "seasonal" and self.period is None:
                raise ValueError("period is required for effect type 'seasonal'")
            return self
        # spatial
        if (self.graph is None) == (self.graph_file is None):
            raise ValueError(
                f"spatial effect {self.name!r} requires exactly one of graph or graph_file"
            )
        if self.type in ("proper_car", "sar") and self.rho is None:
            raise ValueError(f"rho is required for effect type {self.type!r}")
        return self

    def _validate_midas(self) -> "_EffectModelConfig":
        if not self.columns:
            raise ValueError(f"columns is required for effect type {self.type!r}")
        if self.type == "midas":
            if self.order is not None and self.order not in (1, 2):
                raise ValueError("order must be 1 or 2")
            return self
        if self.kernel is not None and self.kernel not in ("beta", "exp_almon"):
            raise ValueError("kernel must be 'beta' or 'exp_almon'")
        return self

    def _validate_spacetime(self) -> "_EffectModelConfig":
        # Two index columns; interaction/order/graph-required checks are the
        # SpaceTime spec's job (raised at build time, wrapped as ConfigurationError).
        if self.space is None or self.time is None:
            raise ValueError("space and time are required for effect type 'spacetime'")
        if self.graph is not None and self.graph_file is not None:
            raise ValueError("spacetime effect requires at most one of graph or graph_file")
        return self

    def _validate_dsp(self) -> "_EffectModelConfig":
        # (unit, time) indexed with per-period graphs; rho fixed, gamma/eta/
        # precision optional. rho-range and graph shape are the spec's job.
        if self.unit is None or self.time is None:
            raise ValueError("unit and time are required for effect type 'dynamicspatialpanel'")
        if self.rho is None:
            raise ValueError("rho is required for effect type 'dynamicspatialpanel'")
        if (self.graphs is None) == (self.graph_files is None):
            raise ValueError(
                "dynamicspatialpanel requires exactly one of graphs or graph_files"
            )
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
    from pylgm.effects import (
        AR1, IID, RW1, RW2, SAR, Besag, BYM2, DynamicSpatialPanel, MIDAS,
        MIDASParametric, ProperCAR, Seasonal, SpaceTime, load_graph_file,
    )

    if config.type in _SIMPLE_EFFECTS:
        simple = {"iid": IID, "rw1": RW1, "rw2": RW2}
        return simple[config.type](config.name, config.index, config.precision)

    if config.type == "ar1":
        precision = 1.0 if config.precision is None else config.precision
        rho = 0.5 if config.rho is None else config.rho
        return AR1(config.name, config.index, precision, rho, config.group)
    if config.type == "seasonal":
        precision = 1.0 if config.precision is None else config.precision
        ridge = 1e-6 if config.ridge is None else config.ridge
        return Seasonal(config.name, config.index, config.period, precision, ridge)

    if config.type == "midas":
        precision = 1.0 if config.precision is None else config.precision
        order = 2 if config.order is None else config.order
        ridge = 1e-6 if config.ridge is None else config.ridge
        return MIDAS(config.name, config.columns, precision, order, ridge)
    if config.type == "midas_parametric":
        return MIDASParametric(config.name, config.columns, config.kernel or "beta")

    if config.type == "dynamicspatialpanel":
        precision = 1.0 if config.precision is None else config.precision
        if config.graph_files:
            graphs = {k: load_graph_file(base_dir / v) for k, v in config.graph_files.items()}
        else:
            graphs = config.graphs
        return DynamicSpatialPanel(
            config.name, config.unit, config.time, graphs, config.rho,
            config.gamma or 0.0, config.eta or 0.0, precision,
        )

    graph = load_graph_file(base_dir / config.graph_file) if config.graph_file else config.graph
    precision = 1.0 if config.precision is None else config.precision
    if config.type == "spacetime":
        return SpaceTime(
            config.name, config.space, config.time, graph,
            config.interaction or "IV", config.order or 1, precision,
            True if config.scale is None else config.scale,
        )
    if config.type == "besag":
        scale = True if config.scale is None else config.scale
        return Besag(config.name, config.index, graph, precision, scale)
    if config.type == "proper_car":
        return ProperCAR(config.name, config.index, graph, config.rho, precision)
    if config.type == "sar":
        return SAR(config.name, config.index, graph, config.rho, precision)
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
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
        return _build_model(_StandaloneModelConfig.model_validate(payload), path.parent)
    except (OSError, yaml.YAMLError, ValidationError, TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


__all__ = ["load_model"]

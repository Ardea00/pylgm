"""Declarative latent-effect specifications."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import TypeAlias

from pylgm.parameters import Hyperparameter


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


def _positive_precision(
    value: float | Hyperparameter, name: str
) -> float | Hyperparameter:
    return value if isinstance(value, Hyperparameter) else _positive_real(value, name)


def _finite_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    return float(value)


class _ComposableEffect:
    def __add__(self, other: object) -> "Predictor":
        return Predictor((self,)) + other


@dataclass(frozen=True)
class Fixed(_ComposableEffect):
    """A fixed-effect formula and its Gaussian prior precision."""

    formula: str
    prior_precision: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "formula", _non_empty_string(self.formula, "formula"))
        object.__setattr__(
            self,
            "prior_precision",
            _positive_real(self.prior_precision, "prior_precision"),
        )

    @property
    def name(self) -> str:
        return "fixed"


@dataclass(frozen=True)
class IID(_ComposableEffect):
    """An independent and identically distributed latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )


@dataclass(frozen=True)
class RW1(_ComposableEffect):
    """A first-order random-walk latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )


@dataclass(frozen=True)
class RW2(_ComposableEffect):
    """A second-order random-walk latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )


@dataclass(frozen=True)
class AR1(_ComposableEffect):
    """A stationary first-order autoregressive latent effect."""

    name: str
    index: str
    precision: float | Hyperparameter = 1.0
    rho: float | Hyperparameter = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.rho, Hyperparameter):
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)


@dataclass(frozen=True)
class Besag(_ComposableEffect):
    """A Besag / intrinsic CAR (ICAR) spatial latent effect."""

    name: str
    index: str
    graph: Mapping
    precision: float | Hyperparameter = 1.0
    scale: bool = True

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be a boolean")
        object.__setattr__(self, "graph", canonical_graph(self.graph))


@dataclass(frozen=True)
class ProperCAR(_ComposableEffect):
    """A proper conditional autoregressive (proper CAR) spatial latent effect."""

    name: str
    index: str
    graph: Mapping
    rho: float | Hyperparameter
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)  # interval resolved at compile time
        else:
            object.__setattr__(self, "rho", _finite_real(self.rho, "rho"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "graph", canonical_graph(self.graph))


@dataclass(frozen=True)
class SAR(_ComposableEffect):
    """A directed spatial-autoregressive latent effect: precision
    ``precision * (I - ρW)ᵀ(I - ρW)`` on a row-standardized directed graph."""

    name: str
    index: str
    graph: Mapping
    rho: float | Hyperparameter
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.directed_graph import canonical_directed_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)
        else:
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "graph", canonical_directed_graph(self.graph))


@dataclass(frozen=True)
class DynamicSpatialPanel(_ComposableEffect):
    """A dynamic spatial panel (SDPD): per-period directed networks ``W_t`` with
    contemporaneous (``rho``), temporal (``gamma``) and spatio-temporal-diffusion
    (``eta``) coefficients. Precision ``precision * MᵀM`` for the block-bidiagonal
    operator ``M`` (see the S-slice design spec)."""

    name: str
    unit: str
    time: str
    graphs: Mapping
    rho: float | Hyperparameter
    gamma: float | Hyperparameter = 0.0
    eta: float | Hyperparameter = 0.0
    precision: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        from pylgm.effects.directed_graph import (
            canonical_directed_graph,
            graphs_by_label,
            sorted_time_keys,
        )

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "unit", _non_empty_string(self.unit, "unit"))
        object.__setattr__(self, "time", _non_empty_string(self.time, "time"))
        if isinstance(self.rho, Hyperparameter):
            object.__setattr__(self, "rho", self.rho)
        else:
            rho = _finite_real(self.rho, "rho")
            if not -1.0 < rho < 1.0:
                raise ValueError("rho must lie strictly inside (-1, 1)")
            object.__setattr__(self, "rho", rho)
        for field_name in ("gamma", "eta"):
            value = getattr(self, field_name)
            if not isinstance(value, Hyperparameter):
                object.__setattr__(self, field_name, _finite_real(value, field_name))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.graphs, Mapping) or not self.graphs:
            raise ValueError("graphs must be a non-empty mapping of time -> graph")
        by_label = graphs_by_label(self.graphs)
        canonical = tuple(
            (t, canonical_directed_graph(by_label[t]))
            for t in sorted_time_keys(self.graphs)
        )
        object.__setattr__(self, "graphs", canonical)

    @property
    def index(self) -> str:
        return self.unit


@dataclass(frozen=True)
class BYM2(_ComposableEffect):
    """A BYM2 spatial latent effect: scaled ICAR + IID mixed by ``phi``."""

    name: str
    index: str
    graph: Mapping
    precision: float | Hyperparameter = 1.0
    phi: float | Hyperparameter = 0.5

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "index", _non_empty_string(self.index, "index"))
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.phi, Hyperparameter):
            phi = _finite_real(self.phi, "phi")
            if not 0.0 < phi < 1.0:
                raise ValueError("phi must lie strictly inside (0, 1)")
            object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "graph", canonical_graph(self.graph))


@dataclass(frozen=True)
class MIDAS(_ComposableEffect):
    """A mixed-frequency distributed-lag effect: the caller-supplied ``columns``
    are the HF covariate at each lag, penalised by a random-walk smoothness
    prior over the lag index. Level and slope of the lag curve stay free of the
    smoothing precision (see the S1 design spec)."""

    name: str
    columns: tuple[str, ...]
    precision: float | Hyperparameter = 1.0
    order: int = 2
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        columns = tuple(self.columns)
        for column in columns:
            _non_empty_string(column, "column")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if len(columns) <= self.order:
            raise ValueError(f"columns must have more than order ({self.order}) entries")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        object.__setattr__(self, "ridge", _positive_real(self.ridge, "ridge"))


@dataclass(frozen=True)
class MIDASParametric(_ComposableEffect):
    """Restricted MIDAS: HF lag ``columns`` collapsed into one regressor
    ``beta * sum_k w(k; theta) * x_{t,k}`` under a parametric lag-weight kernel
    (``"beta"`` or ``"exp_almon"``). The two shape parameters are estimated
    (EB) / integrated (INLA) when given as ``Hyperparameter``s, or fixed when
    given as floats; the loading ``beta`` carries a fixed vague Gaussian prior
    (``prior_precision``). See the S2 design spec."""

    name: str
    columns: tuple[str, ...]
    kernel: str = "beta"
    shape1: "float | Hyperparameter | None" = None
    shape2: "float | Hyperparameter | None" = None
    prior_precision: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        columns = tuple(self.columns)
        for column in columns:
            _non_empty_string(column, "column")
        if len(columns) < 2:
            raise ValueError("MIDASParametric requires at least 2 lag columns")
        object.__setattr__(self, "columns", columns)
        if self.kernel not in ("beta", "exp_almon"):
            raise ValueError("kernel must be 'beta' or 'exp_almon'")
        object.__setattr__(self, "prior_precision", _positive_real(self.prior_precision, "prior_precision"))
        object.__setattr__(self, "shape1", self._resolve_shape(self.shape1, 1))
        object.__setattr__(self, "shape2", self._resolve_shape(self.shape2, 2))

    def _resolve_shape(self, shape, index):
        if shape is None:
            if self.kernel == "beta":
                return Hyperparameter(f"{self.name}.shape{index}", initial=2.0, transform="log")
            initial = 0.0 if index == 1 else -0.1
            return Hyperparameter(f"{self.name}.shape{index}", initial=initial, transform="identity")
        if isinstance(shape, Hyperparameter):
            return shape
        if type(shape) in (int, float):
            return float(shape)
        raise ValueError("shape must be a float, a Hyperparameter, or None")


@dataclass(frozen=True)
class SpaceTime(_ComposableEffect):
    """A Knorr-Held space-time interaction effect (interaction types I-IV)."""

    name: str
    space: str
    time: str
    graph: Mapping | None = None
    interaction: str = "IV"
    order: int = 1
    precision: float | Hyperparameter = 1.0
    scale: bool = True

    def __post_init__(self) -> None:
        from pylgm.effects.graph import canonical_graph

        object.__setattr__(self, "name", _non_empty_string(self.name, "name"))
        object.__setattr__(self, "space", _non_empty_string(self.space, "space"))
        object.__setattr__(self, "time", _non_empty_string(self.time, "time"))
        if self.interaction not in ("I", "II", "III", "IV"):
            raise ValueError("interaction must be one of 'I', 'II', 'III', 'IV'")
        if type(self.order) is not int or self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        object.__setattr__(
            self, "precision", _positive_precision(self.precision, "precision")
        )
        if not isinstance(self.scale, bool):
            raise ValueError("scale must be a boolean")
        if self.interaction in ("III", "IV") and self.graph is None:
            raise ValueError(
                f"interaction {self.interaction!r} needs a spatial neighbour graph; "
                "pass graph=..."
            )
        if self.graph is not None:
            object.__setattr__(self, "graph", canonical_graph(self.graph))

    @property
    def index(self) -> str:
        return self.space


EffectSpec: TypeAlias = (
    Fixed
    | IID
    | RW1
    | RW2
    | AR1
    | Besag
    | ProperCAR
    | SAR
    | DynamicSpatialPanel
    | BYM2
    | MIDAS
    | MIDASParametric
    | SpaceTime
)


@dataclass(frozen=True)
class Predictor:
    """An ordered, immutable collection of declarative latent effects."""

    effects: tuple[EffectSpec, ...]

    def __post_init__(self) -> None:
        try:
            effects = tuple(self.effects)
        except TypeError as error:
            raise TypeError("effects must be an iterable of effect specifications") from error
        if any(
            not isinstance(
                effect,
                (
                    Fixed,
                    IID,
                    RW1,
                    RW2,
                    AR1,
                    Besag,
                    ProperCAR,
                    SAR,
                    DynamicSpatialPanel,
                    BYM2,
                    MIDAS,
                    MIDASParametric,
                    SpaceTime,
                ),
            )
            for effect in effects
        ):
            raise TypeError("effects must contain only effect specifications")
        names = [effect.name for effect in effects]
        if len(names) != len(set(names)):
            raise ValueError("effect names must be unique")
        object.__setattr__(self, "effects", effects)

    def __add__(self, other: object) -> "Predictor":
        if isinstance(other, Predictor):
            return Predictor(self.effects + other.effects)
        if isinstance(
            other,
            (
                Fixed,
                IID,
                RW1,
                RW2,
                AR1,
                Besag,
                ProperCAR,
                SAR,
                DynamicSpatialPanel,
                BYM2,
                MIDAS,
                MIDASParametric,
                SpaceTime,
            ),
        ):
            return Predictor(self.effects + (other,))
        return NotImplemented


__all__ = [
    "AR1",
    "Besag",
    "BYM2",
    "DynamicSpatialPanel",
    "Fixed",
    "IID",
    "MIDAS",
    "MIDASParametric",
    "Predictor",
    "ProperCAR",
    "RW1",
    "RW2",
    "SAR",
    "SpaceTime",
]

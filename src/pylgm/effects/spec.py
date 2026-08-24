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




EffectSpec: TypeAlias = Fixed | IID | RW1 | RW2 | AR1 | Besag | ProperCAR | BYM2


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
            not isinstance(effect, (Fixed, IID, RW1, RW2, AR1, Besag, ProperCAR, BYM2))
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
        if isinstance(other, (Fixed, IID, RW1, RW2, AR1, Besag, ProperCAR, BYM2)):
            return Predictor(self.effects + (other,))
        return NotImplemented


__all__ = ["AR1", "Besag", "BYM2", "Fixed", "IID", "Predictor", "ProperCAR", "RW1", "RW2"]

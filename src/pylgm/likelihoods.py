"""Likelihood definitions for latent Gaussian models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

from pylgm.links import IdentityLink
from pylgm.parameters import Hyperparameter


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


@dataclass(frozen=True)
class CompiledGaussian:
    """A Gaussian likelihood with a resolved standard deviation."""

    sigma: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_real(self.sigma, "sigma"))

    @property
    def variance(self) -> float:
        return self.sigma**2


@dataclass(frozen=True)
class Gaussian:
    """A Gaussian likelihood with a fixed or optimisable standard deviation."""

    sigma: float | Hyperparameter = 1.0
    link: IdentityLink = field(default_factory=IdentityLink, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sigma, Hyperparameter):
            return
        object.__setattr__(self, "sigma", _positive_real(self.sigma, "sigma"))

    def materialize(self, values: Mapping[str, float]) -> CompiledGaussian:
        if isinstance(self.sigma, Hyperparameter):
            if not isinstance(values, Mapping):
                raise ValueError("values must be a mapping")
            expected = {self.sigma.name}
            if set(values) != expected:
                raise ValueError(
                    f"values must contain exactly {self.sigma.name!r}"
                )
            return CompiledGaussian(values[self.sigma.name])
        return CompiledGaussian(self.sigma)

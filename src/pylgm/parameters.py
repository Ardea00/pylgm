"""Hyperparameter definitions for latent Gaussian models."""

from dataclasses import dataclass
from typing import Literal, Protocol

import math


class Prior(Protocol):
    """A prior density evaluated on its native parameter scale."""

    def logpdf(self, value: float) -> float: ...


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


@dataclass(frozen=True)
class Hyperparameter:
    """A named, positive model parameter and its optional prior.

    In pyLGM 0.3, declarative exact-Gaussian compilation uses ``initial`` as a
    fixed plug-in value. Names, transforms, and priors remain declaration
    metadata and are not yet preserved in the materialized IR or used by
    inference.
    """

    name: str
    initial: float
    prior: Prior | None = None
    transform: Literal["identity", "log"] = "log"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "initial", _positive_real(self.initial, "initial"))
        if self.prior is not None and not callable(getattr(self.prior, "logpdf", None)):
            raise ValueError("prior must provide a logpdf method")
        if self.transform not in ("identity", "log"):
            raise ValueError("transform must be 'identity' or 'log'")

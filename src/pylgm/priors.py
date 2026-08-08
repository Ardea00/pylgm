"""Prior distributions used by model hyperparameters."""

from dataclasses import dataclass
import math

import numpy as np


def _finite_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    return float(value)


def _positive_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class GaussianPrior:
    """A Gaussian prior parameterized by mean and precision."""

    mean: float
    precision: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _finite_real(self.mean, "mean"))
        object.__setattr__(self, "precision", _positive_real(self.precision, "precision"))

    def logpdf(self, value: float) -> float:
        value = _finite_real(value, "value")
        return 0.5 * (np.log(self.precision) - np.log(2.0 * np.pi)) - (
            0.5 * self.precision * (value - self.mean) ** 2
        )


@dataclass(frozen=True)
class PCPrecision:
    """Penalised-complexity prior for a Gaussian precision."""

    upper_sd: float
    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "upper_sd", _positive_real(self.upper_sd, "upper_sd"))
        alpha = _finite_real(self.alpha, "alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        object.__setattr__(self, "alpha", alpha)

    def logpdf(self, value: float) -> float:
        value = _positive_real(value, "value")
        rate = -np.log(self.alpha) / self.upper_sd
        return np.log(rate / 2.0) - 1.5 * np.log(value) - rate / np.sqrt(value)

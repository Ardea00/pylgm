"""Likelihood definitions for latent Gaussian models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.special import gammaln, erf, gammaincc

from pylgm.links import IdentityLink, LogLink, LogitLink
from pylgm.exceptions import DataContractError
from pylgm.parameters import Hyperparameter


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


@dataclass(frozen=True)
class CompiledGaussian:
    """A Gaussian likelihood with a resolved standard deviation."""

    sigma: float
    link: IdentityLink = field(default_factory=IdentityLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_real(self.sigma, "sigma"))

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        residual = np.asarray(y, dtype=float) - np.asarray(eta, dtype=float)
        n = residual.size
        return float(
            -0.5 * (n * np.log(2 * np.pi * self.variance) + residual @ residual / self.variance)
        )

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - np.asarray(eta, dtype=float)) / self.variance

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(eta).shape, 1.0 / self.variance, dtype=float)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(eta, dtype=float).shape)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta, dtype=float)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return np.asarray(eta_mean, dtype=float)

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return -0.5 * (np.log(2 * np.pi * self.variance) + (y - eta) ** 2 / self.variance)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return 0.5 * (1.0 + erf((y - eta) / (np.sqrt(self.variance) * np.sqrt(2.0))))

    def validate_response(self, y: np.ndarray) -> None:
        return None


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


@dataclass(frozen=True)
class CompiledPoisson:
    """A Poisson likelihood with a canonical log link."""

    link: LogLink = field(default_factory=LogLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)

        return float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1.0)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self.link.inverse(eta)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -np.exp(np.asarray(eta, dtype=float))

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float) + 0.5 * np.asarray(eta_variance, dtype=float))

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return y * eta - np.exp(eta) - gammaln(y + 1.0)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return gammaincc(np.floor(y) + 1.0, np.exp(eta))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.round(y)):
            raise DataContractError("Poisson responses must be non-negative integers")


@dataclass(frozen=True)
class CompiledBernoulli:
    """A Bernoulli likelihood with a canonical logit link."""

    link: LogitLink = field(default_factory=LogitLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        # log(1+exp(eta)) computed stably
        softplus = np.logaddexp(0.0, eta)
        return float(np.sum(y * eta - softplus))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self.link.inverse(eta)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(eta)
        return p * (1.0 - p)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(np.asarray(eta, dtype=float))
        return -p * (1.0 - p) * (1.0 - 2.0 * p)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float))  # point estimate; variance ignored

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return y * eta - np.logaddexp(0.0, eta)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.where(y >= 1.0, 1.0, 1.0 - self.link.inverse(eta))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise DataContractError("Bernoulli responses must be 0 or 1")


@dataclass(frozen=True)
class Poisson:
    """A Poisson likelihood with a canonical log link."""

    def materialize(self, values: Mapping[str, float]) -> CompiledPoisson:
        return CompiledPoisson()


@dataclass(frozen=True)
class Bernoulli:
    """A Bernoulli likelihood with a canonical logit link."""

    def materialize(self, values: Mapping[str, float]) -> CompiledBernoulli:
        return CompiledBernoulli()

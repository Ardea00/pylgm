"""Link functions for likelihoods."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class IdentityLink:
    """The identity link."""

    name: str = field(default="identity", init=False)


@dataclass(frozen=True)
class LogLink:
    """The log link; inverse is exp."""

    name: str = field(default="log", init=False)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(eta, dtype=float))


@dataclass(frozen=True)
class LogitLink:
    """The logit link; inverse is a numerically stable logistic sigmoid."""

    name: str = field(default="logit", init=False)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        result = np.empty_like(eta)
        positive = eta >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
        exp_eta = np.exp(eta[~positive])
        result[~positive] = exp_eta / (1.0 + exp_eta)
        return result

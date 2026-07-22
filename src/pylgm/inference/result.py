from dataclasses import dataclass, field

import numpy as np


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, init=False)
class GaussianResult:
    labels: tuple[str, ...]
    _mean: np.ndarray = field(repr=False)
    _covariance: np.ndarray = field(repr=False)
    log_marginal_likelihood: float
    _predictive_mean: np.ndarray = field(repr=False)
    _predictive_variance: np.ndarray = field(repr=False)

    def __init__(
        self,
        labels: tuple[str, ...],
        mean: np.ndarray,
        covariance: np.ndarray,
        log_marginal_likelihood: float,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
    ) -> None:
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "_mean", _readonly_array(mean))
        object.__setattr__(self, "_covariance", _readonly_array(covariance))
        object.__setattr__(self, "log_marginal_likelihood", float(log_marginal_likelihood))
        object.__setattr__(self, "_predictive_mean", _readonly_array(predictive_mean))
        object.__setattr__(self, "_predictive_variance", _readonly_array(predictive_variance))

    @property
    def mean(self) -> np.ndarray:
        return _readonly_array(self._mean)

    @property
    def covariance(self) -> np.ndarray:
        return _readonly_array(self._covariance)

    @property
    def predictive_mean(self) -> np.ndarray:
        return _readonly_array(self._predictive_mean)

    @property
    def predictive_variance(self) -> np.ndarray:
        return _readonly_array(self._predictive_variance)

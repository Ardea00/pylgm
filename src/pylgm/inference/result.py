from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _marginal_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(result.dtype, np.number) or not np.isrealobj(result):
        raise TypeError(f"{name} must have a real numeric dtype")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, init=False)
class GaussianMarginals:
    """Independent Gaussian summaries for one or more posterior quantities."""

    _mean: np.ndarray = field(repr=False)
    _variance: np.ndarray = field(repr=False)

    def __init__(self, mean: np.ndarray, variance: np.ndarray) -> None:
        mean = _marginal_array(mean, "mean")
        variance = _marginal_array(variance, "variance")
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have matching shapes")
        if np.any(variance < 0):
            raise ValueError("variance must be non-negative")
        object.__setattr__(self, "_mean", _readonly_array(mean))
        object.__setattr__(self, "_variance", _readonly_array(variance))

    @property
    def mean(self) -> np.ndarray:
        return _readonly_array(self._mean)

    @property
    def variance(self) -> np.ndarray:
        return _readonly_array(self._variance)

    @property
    def std(self) -> np.ndarray:
        return _readonly_array(np.sqrt(self._variance))

    def quantile(self, probability: float) -> np.ndarray:
        if not isinstance(probability, (int, float, np.number)) or not 0 < probability < 1:
            raise ValueError("probability must satisfy 0 < probability < 1")
        return _readonly_array(self._mean + norm.ppf(probability) * np.sqrt(self._variance))


@dataclass(frozen=True, init=False)
class GaussianResult:
    labels: tuple[str, ...]
    _mean: np.ndarray = field(repr=False)
    _covariance: np.ndarray = field(repr=False)
    log_marginal_likelihood: float
    _predictive_mean: np.ndarray = field(repr=False)
    _predictive_variance: np.ndarray = field(repr=False)
    block_slices: Mapping[str, slice]
    diagnostics: Mapping[str, object]

    def __init__(
        self,
        labels: tuple[str, ...],
        mean: np.ndarray,
        covariance: np.ndarray,
        log_marginal_likelihood: float,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
        block_slices: Mapping[str, slice] | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        if block_slices is not None and not isinstance(block_slices, Mapping):
            raise TypeError("block_slices must be a mapping")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "_mean", _readonly_array(mean))
        object.__setattr__(self, "_covariance", _readonly_array(covariance))
        object.__setattr__(self, "log_marginal_likelihood", float(log_marginal_likelihood))
        object.__setattr__(self, "_predictive_mean", _readonly_array(predictive_mean))
        object.__setattr__(self, "_predictive_variance", _readonly_array(predictive_variance))
        object.__setattr__(
            self, "block_slices", MappingProxyType(dict(block_slices or {}))
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(diagnostics or {})))

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

    @property
    def engine(self) -> str:
        return "exact_gaussian"

    @property
    def converged(self) -> bool:
        return True

    def latent_marginals(self, block: str | None = None) -> GaussianMarginals:
        if block is None:
            selection = slice(None)
        else:
            try:
                selection = self.block_slices[block]
            except (KeyError, TypeError) as error:
                raise KeyError(f"unknown latent block {block!r}") from error
        covariance = self._covariance[selection, selection]
        return GaussianMarginals(self._mean[selection], np.diag(covariance))

    def hyperparameter_marginals(self) -> Mapping[str, GaussianMarginals]:
        return MappingProxyType({})

    def linear_combinations(self, weights: csr_matrix | np.ndarray) -> GaussianMarginals:
        if isinstance(weights, csr_matrix):
            values = weights.data
            width = weights.shape[1]
        elif isinstance(weights, np.ndarray):
            if weights.ndim != 2:
                raise ValueError("weights must be a two-dimensional matrix")
            values = weights
            width = weights.shape[1]
        else:
            raise TypeError("weights must be a CSR sparse matrix or numpy array")
        if not np.issubdtype(weights.dtype, np.number) or not np.isrealobj(values):
            raise TypeError("weights must have a real numeric dtype")
        if not np.isfinite(values).all():
            raise ValueError("weights must be finite")
        if width != self._mean.size:
            raise ValueError("weights must have one column per latent dimension")
        mean = np.asarray(weights @ self._mean).reshape(-1)
        covariance = np.asarray(weights @ self._covariance @ weights.T)
        return GaussianMarginals(mean, np.diag(covariance))

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm


_IMMUTABLE_DIAGNOSTIC_TYPES = (
    str,
    bytes,
    bool,
    int,
    float,
    complex,
    type(None),
)


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _readonly_diagnostics(value: Mapping[str, object]) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for name, item in value.items():
        if not isinstance(name, str):
            raise TypeError("diagnostics keys must be strings")
        if isinstance(item, np.generic):
            item = _normalized_numpy_scalar(item)
        elif type(item) not in _IMMUTABLE_DIAGNOSTIC_TYPES:
            raise TypeError("diagnostics values must be immutable scalar values")
        result[name] = item
    return MappingProxyType(result)


def _normalized_numpy_scalar(value: np.generic) -> object:
    kind = value.dtype.kind
    if kind == "b":
        return bool(value)
    if kind in {"i", "u"}:
        return int(value)
    if kind == "f":
        return float(value)
    if kind == "c":
        return complex(value)
    if kind == "U":
        return str(value)
    if kind == "S":
        return bytes(value)
    raise TypeError("diagnostics values must be immutable scalar values")


def _log_absolute_row_sums(value: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        absolute = np.abs(np.asarray(value, dtype=np.float64))
    if not absolute.shape[1]:
        return np.full(absolute.shape[0], -np.inf)
    maximum = absolute.max(axis=1)
    result = np.full(absolute.shape[0], -np.inf)
    finite_nonzero = np.isfinite(maximum) & (maximum > 0)
    if np.any(finite_nonzero):
        normalized = absolute[finite_nonzero] / maximum[finite_nonzero, np.newaxis]
        result[finite_nonzero] = np.log(maximum[finite_nonzero]) + np.log(
            normalized.sum(axis=1)
        )
    return result


def _log_csr_absolute_row_sums(value: csr_matrix) -> np.ndarray:
    result = np.full(value.shape[0], -np.inf)
    for row in range(value.shape[0]):
        start, stop = value.indptr[row : row + 2]
        with np.errstate(over="ignore", invalid="ignore"):
            absolute = np.abs(np.asarray(value.data[start:stop], dtype=np.float64))
        if not absolute.size:
            continue
        maximum = absolute.max()
        if np.isfinite(maximum) and maximum > 0:
            result[row] = np.log(maximum) + np.log(np.sum(absolute / maximum))
    return result


def _roundoff_tolerances(weights: csr_matrix | np.ndarray, covariance: np.ndarray) -> np.ndarray:
    if isinstance(weights, csr_matrix):
        log_weight_scales = _log_csr_absolute_row_sums(weights)
    else:
        log_weight_scales = _log_absolute_row_sums(weights)
    result = np.zeros(log_weight_scales.shape)
    log_covariance_row_sums = _log_absolute_row_sums(covariance)
    if not log_covariance_row_sums.size:
        return result
    log_covariance_scale = np.max(log_covariance_row_sums)
    if not np.isfinite(log_covariance_scale):
        return result
    log_tolerance = np.log(np.finfo(float).eps) + np.maximum(
        0.0, 2 * log_weight_scales + log_covariance_scale
    )
    finite = np.isfinite(log_tolerance) & (
        log_tolerance <= np.log(np.finfo(float).max)
    )
    result[finite] = np.exp(log_tolerance[finite])
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
    _prediction_keys: pd.DataFrame | None = field(repr=False)
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
        prediction_keys: pd.DataFrame | None = None,
    ) -> None:
        if block_slices is not None and not isinstance(block_slices, Mapping):
            raise TypeError("block_slices must be a mapping")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        if prediction_keys is not None:
            if not isinstance(prediction_keys, pd.DataFrame):
                raise TypeError("prediction keys must be a Pandas DataFrame")
            if not all(isinstance(column, str) for column in prediction_keys.columns):
                raise TypeError("prediction key column labels must be strings")
            if not prediction_keys.columns.is_unique:
                raise ValueError("prediction key column labels must be unique")
            if prediction_keys.isna().to_numpy().any():
                raise ValueError("prediction keys must not contain null cells")
            if len(prediction_keys) != np.asarray(predictive_mean).size:
                raise ValueError("prediction keys row count must match predictive results")
        covariance = np.asarray(covariance)
        if not np.issubdtype(covariance.dtype, np.number) or not np.isrealobj(
            covariance
        ):
            raise TypeError("covariance must have a real numeric dtype")
        if not np.isfinite(covariance).all():
            raise ValueError("covariance must be finite")
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "_mean", _readonly_array(mean))
        object.__setattr__(self, "_covariance", _readonly_array(covariance))
        object.__setattr__(self, "log_marginal_likelihood", float(log_marginal_likelihood))
        object.__setattr__(self, "_predictive_mean", _readonly_array(predictive_mean))
        object.__setattr__(self, "_predictive_variance", _readonly_array(predictive_variance))
        object.__setattr__(
            self,
            "_prediction_keys",
            prediction_keys.copy(deep=True) if prediction_keys is not None else None,
        )
        object.__setattr__(
            self, "block_slices", MappingProxyType(dict(block_slices or {}))
        )
        object.__setattr__(
            self,
            "diagnostics",
            _readonly_diagnostics(diagnostics if diagnostics is not None else {}),
        )

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
    def prediction_keys(self) -> pd.DataFrame | None:
        if self._prediction_keys is None:
            return None
        return self._prediction_keys.copy(deep=True)

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
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.asarray(weights @ self._mean).reshape(-1)
            projected_covariance = np.asarray(weights @ self._covariance)
            if isinstance(weights, csr_matrix):
                variance = np.asarray(
                    weights.multiply(projected_covariance).sum(axis=1)
                ).reshape(-1)
            else:
                variance = np.einsum(
                    "ij,ij->i", projected_covariance, weights
                )
        if not np.isfinite(projected_covariance).all() or not np.isfinite(
            variance
        ).all():
            raise ValueError("propagated covariance must be finite")
        roundoff_tolerance = _roundoff_tolerances(weights, self._covariance)
        variance[(variance < 0) & (variance >= -roundoff_tolerance)] = 0.0
        return GaussianMarginals(mean, variance)

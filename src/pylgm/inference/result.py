from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import owens_t
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


def _readonly_hyperparameters(
    values: Mapping[str, float] | None,
) -> Mapping[str, float] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError("hyperparameters must be a mapping")
    resolved = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError("hyperparameter names must be non-empty strings")
        resolved[name] = float(value)
        if not np.isfinite(resolved[name]):
            raise ValueError("hyperparameter values must be finite")
    return MappingProxyType(resolved)


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


def latent_marginals_from(
    mean: np.ndarray,
    covariance: np.ndarray,
    block_slices: Mapping[str, slice],
    block: str | None = None,
) -> "GaussianMarginals":
    if block is None:
        selection = slice(None)
    else:
        try:
            selection = block_slices[block]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown latent block {block!r}") from error
    selected_covariance = covariance[selection, selection]
    return GaussianMarginals(mean[selection], np.diag(selected_covariance))


def linear_combinations_from(
    mean: np.ndarray,
    covariance: np.ndarray,
    weights: csr_matrix | np.ndarray,
) -> "GaussianMarginals":
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
    if width != mean.size:
        raise ValueError("weights must have one column per latent dimension")
    with np.errstate(over="ignore", invalid="ignore"):
        result_mean = np.asarray(weights @ mean).reshape(-1)
        projected_covariance = np.asarray(weights @ covariance)
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
    roundoff_tolerance = _roundoff_tolerances(weights, covariance)
    variance[(variance < 0) & (variance >= -roundoff_tolerance)] = 0.0
    return GaussianMarginals(result_mean, variance)


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


def _skew_normal_marginal_array(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional (n_components, n_grid)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, init=False)
class SkewNormalMarginals:
    """Grid-mixture-of-skew-normals summaries for one or more posterior quantities.

    For each component (row), the marginal is a weighted mixture, over the grid
    (columns), of skew-normal densities parameterized by ``location``/``scale``/
    ``shape``; ``weights`` sum to 1 across the grid for each component. Reduces
    to a Gaussian mixture when ``shape == 0`` everywhere.
    """

    _weights: np.ndarray = field(repr=False)
    _location: np.ndarray = field(repr=False)
    _scale: np.ndarray = field(repr=False)
    _shape: np.ndarray = field(repr=False)

    def __init__(
        self,
        weights: np.ndarray,
        location: np.ndarray,
        scale: np.ndarray,
        shape: np.ndarray,
    ) -> None:
        weights = _skew_normal_marginal_array(weights, "weights")
        location = _skew_normal_marginal_array(location, "location")
        scale = _skew_normal_marginal_array(scale, "scale")
        shape = _skew_normal_marginal_array(shape, "shape")
        if not (weights.shape == location.shape == scale.shape == shape.shape):
            raise ValueError("weights, location, scale, shape must have matching shapes")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if not np.allclose(weights.sum(axis=1), 1.0):
            raise ValueError("weights must sum to 1 across the grid for each component")
        if np.any(scale <= 0):
            raise ValueError("scale must be positive")
        object.__setattr__(self, "_weights", _readonly_array(weights))
        object.__setattr__(self, "_location", _readonly_array(location))
        object.__setattr__(self, "_scale", _readonly_array(scale))
        object.__setattr__(self, "_shape", _readonly_array(shape))

    def _component_moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta = self._shape / np.sqrt(1.0 + self._shape ** 2)
        mu = self._location + self._scale * delta * np.sqrt(2.0 / np.pi)
        v = self._scale ** 2 * (1.0 - 2.0 * delta ** 2 / np.pi)
        mu3 = self._scale ** 3 * (4.0 - np.pi) / 2.0 * (delta * np.sqrt(2.0 / np.pi)) ** 3
        return mu, v, mu3

    @property
    def mean(self) -> np.ndarray:
        mu, _, _ = self._component_moments()
        return _readonly_array(np.sum(self._weights * mu, axis=1))

    @property
    def variance(self) -> np.ndarray:
        mu, v, _ = self._component_moments()
        mean = np.sum(self._weights * mu, axis=1)
        return _readonly_array(np.sum(self._weights * (v + mu ** 2), axis=1) - mean ** 2)

    @property
    def std(self) -> np.ndarray:
        return _readonly_array(np.sqrt(self.variance))

    @property
    def skewness(self) -> np.ndarray:
        mu, v, mu3 = self._component_moments()
        mean = np.sum(self._weights * mu, axis=1)
        std = np.sqrt(np.sum(self._weights * (v + mu ** 2), axis=1) - mean ** 2)
        centered = mu - mean[:, np.newaxis]
        numerator = np.sum(
            self._weights * (mu3 + 3.0 * centered * v + centered ** 3), axis=1
        )
        return _readonly_array(numerator / std ** 3)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        z = (x - self._location) / self._scale
        density = (2.0 / self._scale) * norm.pdf(z) * norm.cdf(self._shape * z)
        return _readonly_array(np.sum(self._weights * density, axis=1))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        z = (x - self._location) / self._scale
        component_cdf = norm.cdf(z) - 2.0 * owens_t(z, self._shape)
        return _readonly_array(np.sum(self._weights * component_cdf, axis=1))

    def quantile(self, probability: float) -> np.ndarray:
        if not isinstance(probability, (int, float, np.number)) or not 0 < probability < 1:
            raise ValueError("probability must satisfy 0 < probability < 1")
        mean = self.mean
        std = self.std
        lo = mean - 50.0 * std - 1.0
        hi = mean + 50.0 * std + 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < probability
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return _readonly_array(0.5 * (lo + hi))

    def select(self, selection: slice) -> "SkewNormalMarginals":
        """Return the component-subset marginals for the given slice of the component axis."""
        return SkewNormalMarginals(
            self._weights[selection], self._location[selection],
            self._scale[selection], self._shape[selection],
        )


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
    _hyperparameters: Mapping[str, float] | None = field(repr=False)

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
        *,
        hyperparameters: Mapping[str, float] | None = None,
    ) -> None:
        if block_slices is not None and not isinstance(block_slices, Mapping):
            raise TypeError("block_slices must be a mapping")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        _validate_prediction_keys(prediction_keys, predictive_mean)
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
        object.__setattr__(
            self, "_hyperparameters", _readonly_hyperparameters(hyperparameters)
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
    def hyperparameters(self) -> Mapping[str, float] | None:
        return self._hyperparameters

    @property
    def engine(self) -> str:
        return "exact_gaussian"

    @property
    def converged(self) -> bool:
        return True

    def latent_marginals(self, block: str | None = None) -> GaussianMarginals:
        return latent_marginals_from(self._mean, self._covariance, self.block_slices, block)

    def hyperparameter_marginals(self) -> Mapping[str, GaussianMarginals]:
        return MappingProxyType({})

    def linear_combinations(self, weights: csr_matrix | np.ndarray) -> GaussianMarginals:
        return linear_combinations_from(self._mean, self._covariance, weights)


@dataclass(frozen=True, init=False)
class ModelCriteria:
    """DIC, WAIC, CPO, and PIT model-assessment criteria.

    For discrete-response likelihoods (e.g. Poisson, Bernoulli), the harmonic-mean
    CPO/PIT estimator integrates ``E[1/p]`` against a heavy/divergent tail under the
    Gaussian eta-approximation, so both the CPO/PIT values and ``cpo_failures`` are
    sensitive to the Gauss-Hermite quadrature node count; ``cpo_failures`` is a lower
    bound on unreliable observations, not a guarantee that the rest are trustworthy.
    """

    dic: float
    dic_effective_parameters: float
    waic: float
    waic_effective_parameters: float
    _cpo: np.ndarray = field(repr=False)
    _pit: np.ndarray = field(repr=False)
    cpo_failures: int
    log_cpo_sum: float

    def __init__(
        self,
        dic: float,
        dic_effective_parameters: float,
        waic: float,
        waic_effective_parameters: float,
        cpo: np.ndarray,
        pit: np.ndarray,
        cpo_failures: int,
        log_cpo_sum: float,
    ) -> None:
        object.__setattr__(self, "dic", float(dic))
        object.__setattr__(self, "dic_effective_parameters", float(dic_effective_parameters))
        object.__setattr__(self, "waic", float(waic))
        object.__setattr__(self, "waic_effective_parameters", float(waic_effective_parameters))
        object.__setattr__(self, "_cpo", _readonly_array(cpo))
        object.__setattr__(self, "_pit", _readonly_array(pit))
        object.__setattr__(self, "cpo_failures", int(cpo_failures))
        object.__setattr__(self, "log_cpo_sum", float(log_cpo_sum))

    @property
    def cpo(self) -> np.ndarray:
        return _readonly_array(self._cpo)

    @property
    def pit(self) -> np.ndarray:
        return _readonly_array(self._pit)

    def reordered(self, order: np.ndarray) -> "ModelCriteria":
        """Return a new ModelCriteria with cpo/pit permuted by ``order``; scalars unchanged."""
        return ModelCriteria(
            self.dic, self.dic_effective_parameters,
            self.waic, self.waic_effective_parameters,
            self._cpo[order], self._pit[order],
            self.cpo_failures, self.log_cpo_sum,
        )


def _validate_prediction_keys(
    prediction_keys: pd.DataFrame | None, predictive_mean: np.ndarray
) -> None:
    if prediction_keys is None:
        return
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


@dataclass(frozen=True, init=False)
class LaplaceResult:
    labels: tuple[str, ...]
    _mean: np.ndarray = field(repr=False)
    _covariance: np.ndarray = field(repr=False)
    log_marginal_likelihood: float
    _predictive_mean: np.ndarray = field(repr=False)
    _predictive_variance: np.ndarray = field(repr=False)
    _fitted_mean: np.ndarray = field(repr=False)
    link_name: str
    _prediction_keys: pd.DataFrame | None = field(repr=False)
    block_slices: Mapping[str, slice]
    diagnostics: Mapping[str, object]
    _hyperparameters: Mapping[str, float] | None = field(repr=False)

    def __init__(
        self,
        labels: tuple[str, ...],
        mean: np.ndarray,
        covariance: np.ndarray,
        log_marginal_likelihood: float,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
        fitted_mean: np.ndarray,
        link_name: str,
        block_slices: Mapping[str, slice] | None = None,
        diagnostics: Mapping[str, object] | None = None,
        prediction_keys: pd.DataFrame | None = None,
        *,
        hyperparameters: Mapping[str, float] | None = None,
    ) -> None:
        if block_slices is not None and not isinstance(block_slices, Mapping):
            raise TypeError("block_slices must be a mapping")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        _validate_prediction_keys(prediction_keys, predictive_mean)
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
        object.__setattr__(self, "_fitted_mean", _readonly_array(fitted_mean))
        object.__setattr__(self, "link_name", str(link_name))
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
        object.__setattr__(
            self, "_hyperparameters", _readonly_hyperparameters(hyperparameters)
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
        """Linear-predictor (eta) posterior variance; excludes response-scale observation noise."""
        return _readonly_array(self._predictive_variance)

    @property
    def fitted_mean(self) -> np.ndarray:
        return _readonly_array(self._fitted_mean)

    @property
    def prediction_keys(self) -> pd.DataFrame | None:
        if self._prediction_keys is None:
            return None
        return self._prediction_keys.copy(deep=True)

    @property
    def hyperparameters(self) -> Mapping[str, float] | None:
        return self._hyperparameters

    @property
    def engine(self) -> str:
        return "laplace"

    @property
    def converged(self) -> bool:
        return True

    def latent_marginals(self, block: str | None = None) -> GaussianMarginals:
        return latent_marginals_from(self._mean, self._covariance, self.block_slices, block)

    def hyperparameter_marginals(self) -> Mapping[str, GaussianMarginals]:
        return MappingProxyType({})

    def linear_combinations(self, weights: csr_matrix | np.ndarray) -> GaussianMarginals:
        return linear_combinations_from(self._mean, self._covariance, weights)


def _readonly_hyperparameter_marginals(
    values: Mapping[str, GaussianMarginals],
) -> Mapping[str, GaussianMarginals]:
    if not isinstance(values, Mapping):
        raise TypeError("hyperparameter_marginals must be a mapping")
    resolved = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError("hyperparameter_marginals keys must be non-empty strings")
        if not isinstance(value, GaussianMarginals):
            raise TypeError("hyperparameter_marginals values must be GaussianMarginals")
        resolved[name] = value
    return MappingProxyType(resolved)


@dataclass(frozen=True, init=False)
class INLAResult:
    labels: tuple[str, ...]
    _mean: np.ndarray = field(repr=False)
    _covariance: np.ndarray = field(repr=False)
    log_marginal_likelihood: float
    _predictive_mean: np.ndarray = field(repr=False)
    _predictive_variance: np.ndarray = field(repr=False)
    _hyperparameter_marginals: Mapping[str, GaussianMarginals] = field(repr=False)
    _criteria: ModelCriteria = field(repr=False)
    _fitted_mean: np.ndarray | None = field(repr=False)
    link_name: str | None
    _prediction_keys: pd.DataFrame | None = field(repr=False)
    block_slices: Mapping[str, slice]
    diagnostics: Mapping[str, object]
    _hyperparameters: Mapping[str, float] | None = field(repr=False)
    _latent_marginal_table: "SkewNormalMarginals | None" = field(repr=False)

    def __init__(
        self,
        labels: tuple[str, ...],
        mean: np.ndarray,
        covariance: np.ndarray,
        log_marginal_likelihood: float,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray,
        hyperparameter_marginals: Mapping[str, GaussianMarginals],
        *,
        criteria: ModelCriteria,
        fitted_mean: np.ndarray | None = None,
        link_name: str | None = None,
        block_slices: Mapping[str, slice] | None = None,
        diagnostics: Mapping[str, object] | None = None,
        prediction_keys: pd.DataFrame | None = None,
        hyperparameters: Mapping[str, float] | None = None,
        latent_marginal_table: "SkewNormalMarginals | None" = None,
    ) -> None:
        if block_slices is not None and not isinstance(block_slices, Mapping):
            raise TypeError("block_slices must be a mapping")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        if not isinstance(criteria, ModelCriteria):
            raise TypeError("criteria must be a ModelCriteria")
        if latent_marginal_table is not None and not isinstance(
            latent_marginal_table, SkewNormalMarginals
        ):
            raise TypeError("latent_marginal_table must be a SkewNormalMarginals")
        _validate_prediction_keys(prediction_keys, predictive_mean)
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
            "_hyperparameter_marginals",
            _readonly_hyperparameter_marginals(hyperparameter_marginals),
        )
        object.__setattr__(self, "_criteria", criteria)
        object.__setattr__(
            self,
            "_fitted_mean",
            _readonly_array(fitted_mean) if fitted_mean is not None else None,
        )
        object.__setattr__(self, "link_name", str(link_name) if link_name is not None else None)
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
        object.__setattr__(
            self, "_hyperparameters", _readonly_hyperparameters(hyperparameters)
        )
        object.__setattr__(self, "_latent_marginal_table", latent_marginal_table)

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
    def fitted_mean(self) -> np.ndarray | None:
        if self._fitted_mean is None:
            return None
        return _readonly_array(self._fitted_mean)

    @property
    def prediction_keys(self) -> pd.DataFrame | None:
        if self._prediction_keys is None:
            return None
        return self._prediction_keys.copy(deep=True)

    @property
    def hyperparameters(self) -> Mapping[str, float] | None:
        return self._hyperparameters

    @property
    def latent_marginal_table(self) -> "SkewNormalMarginals | None":
        return self._latent_marginal_table

    @property
    def engine(self) -> str:
        return "inla"

    @property
    def converged(self) -> bool:
        return True

    def latent_marginals(self, block: str | None = None) -> "GaussianMarginals | SkewNormalMarginals":
        if self._latent_marginal_table is not None:
            if block is None:
                selection = slice(None)
            else:
                try:
                    selection = self.block_slices[block]
                except KeyError as error:
                    raise KeyError(f"unknown latent block {block!r}") from error
            return self._latent_marginal_table.select(selection)
        return latent_marginals_from(self._mean, self._covariance, self.block_slices, block)

    def hyperparameter_marginals(self) -> Mapping[str, GaussianMarginals]:
        return self._hyperparameter_marginals

    @property
    def criteria(self) -> ModelCriteria:
        return self._criteria

    def linear_combinations(self, weights: csr_matrix | np.ndarray) -> GaussianMarginals:
        return linear_combinations_from(self._mean, self._covariance, weights)

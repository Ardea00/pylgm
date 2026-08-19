from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.optimize

from pylgm.exceptions import InferenceError, NumericalError, OptimizationError
from pylgm.inference import GaussianResult, fit_gaussian
from pylgm.ir import CompiledGaussianFamily
from pylgm.optimization.result import EmpiricalBayesResult, OptimizationDiagnostics


def _transform_objective(raw_objective: float) -> float:
    magnitude = float(np.log1p(abs(raw_objective)))
    return -magnitude if raw_objective < 0.0 else magnitude


_INVALID_OBJECTIVE = _transform_objective(float(np.finfo(float).max)) + 1.0
# Absolute distance in log space used both to snap and report active bounds.
_LOG_BOUND_ATOL = 1e-8


def _ordinary_positive(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be an ordinary finite positive number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be an ordinary finite positive number") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be an ordinary finite positive number")
    return result


@dataclass(frozen=True)
class OptimizationBounds:
    initial: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        initial = _ordinary_positive(self.initial, "initial")
        lower = _ordinary_positive(self.lower, "lower")
        upper = _ordinary_positive(self.upper, "upper")
        if lower >= upper or not lower <= initial <= upper:
            raise ValueError(
                "parameter bounds must satisfy lower <= initial <= upper with lower < upper"
            )
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class _Evaluation:
    objective: float
    raw_objective: float | None
    fit: GaussianResult | None
    parameters: Mapping[str, float] | None


def _validate_problem(
    family: CompiledGaussianFamily,
    bounds: Mapping[str, OptimizationBounds],
    initial: Mapping[str, float] | None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    if not isinstance(family, CompiledGaussianFamily):
        raise TypeError("family must be a compiled Gaussian family")
    if not isinstance(bounds, Mapping):
        raise TypeError("bounds must be a mapping")
    names = tuple(bounds)
    if not names:
        raise OptimizationError("bounds must contain at least one parameter")
    if set(names) != set(family.parameter_names):
        raise ValueError("bounds must contain exactly the compiled family parameter names")
    for name in names:
        if not isinstance(bounds[name], OptimizationBounds):
            raise TypeError(f"bounds for {name!r} must be OptimizationBounds")

    if initial is None:
        return names, {name: bounds[name].initial for name in names}
    if not isinstance(initial, Mapping):
        raise TypeError("initial must be a mapping")
    if set(initial) != set(names):
        raise ValueError("warm-start initial must contain exactly the bounded parameters")
    start: dict[str, float] = {}
    for name in names:
        value = _ordinary_positive(initial[name], f"initial value for {name!r}")
        parameter_bounds = bounds[name]
        if not parameter_bounds.lower <= value <= parameter_bounds.upper:
            raise ValueError(f"initial value for {name!r} must lie within its bounds")
        start[name] = value
    return names, start


def _parameter_values(
    names: tuple[str, ...],
    log_parameters: tuple[float, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    log_lower: np.ndarray,
    log_upper: np.ndarray,
) -> dict[str, float]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        transformed = np.exp(np.asarray(log_parameters, dtype=float))
    transformed = np.clip(transformed, lower, upper)
    for index, value in enumerate(log_parameters):
        low = log_lower[index]
        high = log_upper[index]
        if value == low:
            transformed[index] = lower[index]
            continue
        if value == high:
            transformed[index] = upper[index]
            continue
        tolerance = _log_bound_tolerance(low, high)
        low_distance = abs(value - low)
        high_distance = abs(value - high)
        if low_distance < high_distance and low_distance <= tolerance:
            transformed[index] = lower[index]
        elif high_distance < low_distance and high_distance <= tolerance:
            transformed[index] = upper[index]
    if not np.isfinite(transformed).all() or np.any(transformed <= 0):
        raise NumericalError("log-transformed parameters must be finite and positive")
    return {name: float(value) for name, value in zip(names, transformed, strict=True)}


def _log_bound_tolerance(lower: float, upper: float) -> float:
    half_gap = (upper - lower) / 2.0
    return min(_LOG_BOUND_ATOL, float(np.nextafter(half_gap, 0.0)))


def optimize_empirical_bayes(
    family: CompiledGaussianFamily,
    bounds: Mapping[str, OptimizationBounds],
    *,
    initial: Mapping[str, float] | None = None,
    allow_large_dense: bool = False,
    fit: Callable[..., object] | None = None,
) -> EmpiricalBayesResult:
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if fit is None:
        fit = fit_gaussian
    names, initial_values = _validate_problem(family, bounds, initial)
    lower = np.log([bounds[name].lower for name in names])
    upper = np.log([bounds[name].upper for name in names])
    collapsed = tuple(
        name for name, low, high in zip(names, lower, upper, strict=True) if not low < high
    )
    if collapsed:
        raise OptimizationError(
            f"natural bounds for parameters {collapsed!r} collapse in log space"
        )
    natural_lower = np.asarray([bounds[name].lower for name in names])
    natural_upper = np.asarray([bounds[name].upper for name in names])
    start = np.log([initial_values[name] for name in names])
    scipy_bounds = list(zip(lower, upper, strict=True))

    cache: dict[tuple[float, ...], _Evaluation] = {}
    failures: list[str] = []
    latest_failure: InferenceError | None = None
    evaluations = 0
    cache_hits = 0

    def objective(log_parameters: np.ndarray) -> float:
        nonlocal evaluations, cache_hits, latest_failure
        key = tuple(float(value) for value in log_parameters)
        if key in cache:
            cache_hits += 1
            return cache[key].objective
        evaluations += 1
        try:
            parameters = _parameter_values(
                names,
                key,
                natural_lower,
                natural_upper,
                lower,
                upper,
            )
            model = family.materialize(parameters)
            result = (
                fit(model, allow_large_dense=True)
                if allow_large_dense
                else fit(model)
            )
            raw_objective = -float(result.log_marginal_likelihood)
            if not np.isfinite(raw_objective):
                raise NumericalError("optimization objective must be finite")
            evaluation = _Evaluation(
                _transform_objective(raw_objective),
                raw_objective,
                result,
                parameters,
            )
        except InferenceError as error:
            latest_failure = error
            failures.append(f"log_parameters={key!r}: {type(error).__name__}: {error}")
            evaluation = _Evaluation(_INVALID_OBJECTIVE, None, None, None)
        cache[key] = evaluation
        return evaluation.objective

    def fail(message: str) -> None:
        error = OptimizationError(
            message,
            evaluations=evaluations,
            cache_hits=cache_hits,
            numerical_failures=tuple(failures),
        )
        if latest_failure is not None:
            raise error from latest_failure
        raise error

    started = perf_counter()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        solution = scipy.optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"ftol": 1e-12, "gtol": 1e-10},
        )
    elapsed_seconds = perf_counter() - started

    if not bool(solution.success):
        message = str(getattr(solution, "message", "unknown optimizer failure"))
        fail(f"empirical-Bayes optimization did not converge: {message}")

    final_vector = np.asarray(solution.x, dtype=float)
    if (
        final_vector.shape != start.shape
        or not np.isfinite(final_vector).all()
        or np.any(final_vector < lower)
        or np.any(final_vector > upper)
    ):
        fail("empirical-Bayes optimizer returned an invalid final point")
    objective(final_vector)
    final_key = tuple(float(value) for value in final_vector)
    final_evaluation = cache.get(final_key)
    if final_evaluation is None or final_evaluation.fit is None:
        fail("empirical-Bayes optimization did not produce a valid final point")
    assert final_evaluation.parameters is not None
    assert final_evaluation.raw_objective is not None

    active_bounds = tuple(
        name
        for name, value, low, high in zip(names, final_vector, lower, upper, strict=True)
        if min(abs(value - low), abs(value - high)) <= _log_bound_tolerance(low, high)
    )
    diagnostics = OptimizationDiagnostics(
        converged=True,
        objective=final_evaluation.raw_objective,
        evaluations=evaluations,
        cache_hits=cache_hits,
        elapsed_seconds=elapsed_seconds,
        initial=initial_values,
        optimum=final_evaluation.parameters,
        active_bounds=active_bounds,
        numerical_failures=tuple(failures),
    )
    return EmpiricalBayesResult(
        parameters=final_evaluation.parameters,
        fit=final_evaluation.fit,
        diagnostics=diagnostics,
    )

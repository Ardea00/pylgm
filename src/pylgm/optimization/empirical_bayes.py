from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import scipy.optimize

from pylgm.exceptions import InferenceError, NumericalError, OptimizationError
from pylgm.inference import GaussianResult, LaplaceResult, fit_gaussian
from pylgm.optimization.result import EmpiricalBayesResult, OptimizationDiagnostics
from pylgm.optimization.transforms import LogTransform, Transform


def _transform_objective(raw_objective: float) -> float:
    magnitude = float(np.log1p(abs(raw_objective)))
    return -magnitude if raw_objective < 0.0 else magnitude


_INVALID_OBJECTIVE = _transform_objective(float(np.finfo(float).max)) + 1.0
# Absolute distance in log space used both to snap and report active bounds.
_LOG_BOUND_ATOL = 1e-8


def _ordinary_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be an ordinary finite positive number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be an ordinary finite positive number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be an ordinary finite positive number")
    return result


@dataclass(frozen=True)
class OptimizationBounds:
    initial: float
    lower: float
    upper: float
    transform: Transform = field(default_factory=LogTransform)

    def __post_init__(self) -> None:
        initial = _ordinary_number(self.initial, "initial")
        lower = _ordinary_number(self.lower, "lower")
        upper = _ordinary_number(self.upper, "upper")
        if not self.transform.contains(lower) or not self.transform.contains(upper):
            raise ValueError(
                "lower and upper must be finite numbers within "
                f"{self.transform.domain_description()}"
            )
        if lower >= upper or not lower <= initial <= upper:
            raise ValueError(
                "parameter bounds must satisfy lower <= initial <= upper with lower < upper"
            )
        # NOTE: whether lower/upper collapse to the same *internal* value (e.g. two
        # distinct naturals that round-trip to one float in log-space) is checked by
        # optimize_empirical_bayes, which raises a typed OptimizationError for it —
        # not here, so bounds that are merely constructed (not yet optimized) don't
        # raise a bare ValueError for that case.
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class _Evaluation:
    objective: float
    raw_objective: float | None
    fit: GaussianResult | LaplaceResult | None
    parameters: Mapping[str, float] | None


def _validate_problem(
    family: object,
    bounds: Mapping[str, OptimizationBounds],
    initial: Mapping[str, float] | None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    parameter_names = getattr(family, "parameter_names", None)
    if not isinstance(parameter_names, tuple) or not callable(
        getattr(family, "materialize", None)
    ):
        raise TypeError("family must expose parameter_names and a materialize method")
    if not isinstance(bounds, Mapping):
        raise TypeError("bounds must be a mapping")
    names = tuple(bounds)
    if not names:
        raise OptimizationError("bounds must contain at least one parameter")
    if set(names) != set(parameter_names):
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
        parameter_bounds = bounds[name]
        value = float(initial[name])
        if not np.isfinite(value):
            raise ValueError(f"initial value for {name!r} must be finite")
        if not parameter_bounds.lower <= value <= parameter_bounds.upper:
            raise ValueError(f"initial value for {name!r} must lie within its bounds")
        start[name] = value
    return names, start


def _parameter_values(
    names: tuple[str, ...],
    internal_parameters: tuple[float, ...],
    transforms: list[Transform],
    lower: np.ndarray,
    upper: np.ndarray,
    internal_lower: np.ndarray,
    internal_upper: np.ndarray,
) -> dict[str, float]:
    values = {}
    for index, name in enumerate(names):
        u = float(internal_parameters[index])
        low, high = internal_lower[index], internal_upper[index]
        if u <= low:
            theta = lower[index]
        elif u >= high:
            theta = upper[index]
        else:
            tolerance = _log_bound_tolerance(low, high)
            if abs(u - low) <= tolerance and abs(u - low) < abs(u - high):
                theta = lower[index]
            elif abs(u - high) <= tolerance and abs(u - high) < abs(u - low):
                theta = upper[index]
            else:
                theta = transforms[index].from_internal(u)
        if not np.isfinite(theta) or not transforms[index].contains(theta):
            # allow the exact closed bounds (contains is open); clip to them
            theta = float(np.clip(theta, lower[index], upper[index]))
            if not np.isfinite(theta):
                raise NumericalError("transformed parameters must be finite")
        values[name] = float(theta)
    return values


def _log_bound_tolerance(lower: float, upper: float) -> float:
    half_gap = (upper - lower) / 2.0
    return min(_LOG_BOUND_ATOL, float(np.nextafter(half_gap, 0.0)))


def optimize_empirical_bayes(
    family: object,
    bounds: Mapping[str, OptimizationBounds],
    *,
    initial: Mapping[str, float] | None = None,
    allow_large_dense: bool = False,
    fit: Callable[..., object] | None = None,
    penalty: Callable[[Mapping[str, float]], float] | None = None,
) -> EmpiricalBayesResult:
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if fit is None:
        fit = fit_gaussian
    names, initial_values = _validate_problem(family, bounds, initial)
    transforms = [bounds[name].transform for name in names]
    natural_lower = np.asarray([bounds[name].lower for name in names])
    natural_upper = np.asarray([bounds[name].upper for name in names])
    lower = np.asarray(
        [t.to_internal(b) for t, b in zip(transforms, natural_lower, strict=True)]
    )
    upper = np.asarray(
        [t.to_internal(b) for t, b in zip(transforms, natural_upper, strict=True)]
    )
    collapsed = tuple(
        name for name, low, high in zip(names, lower, upper, strict=True) if not low < high
    )
    if collapsed:
        raise OptimizationError(
            f"natural bounds for parameters {collapsed!r} collapse in the transform's "
            "internal space"
        )
    start = np.asarray(
        [t.to_internal(initial_values[name]) for t, name in zip(transforms, names, strict=True)]
    )
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
                transforms,
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
            if penalty is not None:
                raw_objective -= float(penalty(parameters))
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

    # L-BFGS-B reports status 2 (ABNORMAL_TERMINATION_IN_LNSRCH) when the line
    # search can no longer make progress. On an ill-conditioned direction — a
    # bounded (logit) hyperparameter, say — that routinely happens *at* the
    # optimum, because `_transform_objective`'s log1p compression pushes the
    # remaining change below ftol/gtol. Treat it as converged when the returned
    # point is usable and no worse than the start, and record it; anything else
    # is still a hard failure.
    line_search_stalled = not bool(solution.success) and int(getattr(solution, "status", -1)) == 2
    if line_search_stalled:
        candidate = np.asarray(solution.x, dtype=float)
        usable = (
            candidate.shape == start.shape
            and np.isfinite(candidate).all()
            and not np.any(candidate < lower)
            and not np.any(candidate > upper)
            and objective(candidate) <= objective(start)
        )
        if not usable:
            line_search_stalled = False
    if not bool(solution.success) and not line_search_stalled:
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
    if line_search_stalled:
        failures.append(
            "line search stalled at the optimum "
            f"({str(getattr(solution, 'message', 'ABNORMAL')).strip()}); "
            "accepted the returned point"
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

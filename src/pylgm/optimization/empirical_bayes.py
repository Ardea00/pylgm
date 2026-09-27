import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import scipy.optimize

from pylgm.exceptions import InferenceError, NumericalError, OptimizationError
from pylgm.inference import GaussianResult, LaplaceResult, fit_gaussian
from pylgm.optimization.result import EmpiricalBayesResult, OptimizationDiagnostics
from pylgm.optimization.transforms import LogTransform, Transform
from pylgm.parallel import blas_limit, map_ordered, validate_blas_threads, validate_workers


def _transform_objective(raw_objective: float) -> float:
    magnitude = float(np.log1p(abs(raw_objective)))
    return -magnitude if raw_objective < 0.0 else magnitude


_INVALID_OBJECTIVE = _transform_objective(float(np.finfo(float).max)) + 1.0
# Absolute distance in log space used both to snap and report active bounds.
_LOG_BOUND_ATOL = 1e-8
# Absolute (not relative) finite-difference step on the internal (log/logit)
# scale. At scale the objective's noise floor is ~1e-6 on the transformed scale
# (~1e-3 in LML), so a central quotient carries ~noise/step of error: at 1e-4
# that swamps weak gradient components near the optimum, while 1e-3 and 1e-2
# agree to ~10%, i.e. truncation error is still negligible at 1e-3.
_FINITE_DIFFERENCE_STEP = 1e-3
# Max displacement of the parameter point (not just the objective) on the
# internal (log/logit) scale over the plateau window -- i.e. about a 1%
# relative change of a log-scale hyperparameter. A plateau means both the
# objective AND the point have stopped moving; an objective that is merely
# flat while the point still creeps towards a bound (e.g. an LML that is
# almost flat in one hyperparameter) is not a plateau.
_PLATEAU_STEP_TOLERANCE = 1e-2


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
    objective_tolerance: float = 1e-5,
    stall_iterations: int = 3,
    num_workers: int = 1,
    blas_threads: int | None = None,
) -> EmpiricalBayesResult:
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if type(objective_tolerance) not in (int, float) or not np.isfinite(
        objective_tolerance
    ) or objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be a finite non-negative number")
    if type(stall_iterations) is not int or stall_iterations < 1:
        raise ValueError("stall_iterations must be a positive integer")
    num_workers = validate_workers(num_workers, "num_workers")
    blas_threads = validate_blas_threads(blas_threads)
    # Plateau stop: relative improvement of the raw objective (-LML, minus the
    # penalty if any -- i.e. `_Evaluation.raw_objective`) required over the
    # last `stall_iterations` iterations to keep going -- the same spirit as
    # scipy's `ftol`, but measured over a window rather than a single step.
    # `objective_tolerance=0` disables the plateau stop.
    if fit is None:
        fit = fit_gaussian
    # Predictive variances are never read from an intermediate objective
    # evaluation -- only `log_marginal_likelihood` is -- yet computing them
    # dominates the per-evaluation cost of a full Gaussian fit at scale. Skip
    # them during the search and refit once, with variances, at the final
    # point. Gated on `fit` actually accepting `predictive_variances` (true
    # for the real `fit_gaussian`, false for `fit_laplace` -- INLA's/Joint's
    # conditional fit needs every intermediate fit's variances -- and false
    # for a test double that doesn't declare the parameter), not on identity:
    # a test that monkeypatches the module's `fit_gaussian` global gets a
    # stand-in with a different signature, and this must follow the
    # signature, not the name.
    try:
        fit_parameters = inspect.signature(fit).parameters
    except (TypeError, ValueError):
        fit_parameters = {}
    skip_intermediate_variances = "predictive_variances" in fit_parameters
    # A Laplace fit warm-starts Newton from ONE fixed mode -- the first
    # evaluation's, at the initial point -- never from the previous evaluation:
    # finite-difference gradients turn a history-dependent change in the
    # objective into a gradient error that moves the optimum.
    warm_start = "initial_mode" in fit_parameters
    last_mode: np.ndarray | None = None
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
    # At most one fit (the most recent successful, non-cached evaluation) is
    # ever kept alive besides the one currently being computed -- the cache
    # itself never retains a fit (see _Evaluation entries below), which is
    # what keeps memory bounded across hundreds of evaluations at scale.
    latest_fit: tuple[tuple[float, ...], object] | None = None

    def _prepare(point: np.ndarray) -> tuple[tuple[float, ...], object, object]:
        """Cache lookup, parameter conversion and materialize -- serial, warm-start
        order preserved. Returns ``(key, model_or_None, error_or_None)``; a cache
        hit or a conversion/materialize failure both come back with
        ``model_or_None is None`` (the caller distinguishes them by ``key in cache``)."""
        key = tuple(float(value) for value in point)
        if key in cache:
            return key, None, None
        try:
            parameters = _parameter_values(
                names, key, transforms, natural_lower, natural_upper, lower, upper,
            )
            model = family.materialize(parameters)
            return key, (model, parameters), None
        except InferenceError as error:
            return key, None, error

    def _run_fit(model: object, snapshot_mode: np.ndarray | None) -> object:
        """Fit ``model`` with today's kwargs -- thread-safe, touches no shared state."""
        fit_kwargs: dict[str, object] = {}
        if allow_large_dense:
            fit_kwargs["allow_large_dense"] = True
        if skip_intermediate_variances:
            fit_kwargs["predictive_variances"] = False
        if warm_start and snapshot_mode is not None:
            fit_kwargs["initial_mode"] = snapshot_mode
        try:
            return fit(model, **fit_kwargs)
        except InferenceError as error:
            return error

    def _record(key: tuple[float, ...], parameters: object, outcome: object) -> float:
        """Today's post-fit bookkeeping: warm start, latest_fit, penalty, cache."""
        nonlocal evaluations, cache_hits, latest_failure, latest_fit, last_mode
        if key in cache:
            cache_hits += 1
            return cache[key].objective
        evaluations += 1
        if isinstance(outcome, InferenceError):
            latest_failure = outcome
            failures.append(f"log_parameters={key!r}: {type(outcome).__name__}: {outcome}")
            evaluation = _Evaluation(_INVALID_OBJECTIVE, None, None, None)
            cache[key] = evaluation
            return evaluation.objective
        result = outcome
        try:
            if warm_start and last_mode is None:
                last_mode = np.asarray(result.mean)
            raw_objective = -float(result.log_marginal_likelihood)
            if penalty is not None:
                raw_objective -= float(penalty(parameters))
            if not np.isfinite(raw_objective):
                raise NumericalError("optimization objective must be finite")
            latest_fit = (key, result)
            evaluation = _Evaluation(
                _transform_objective(raw_objective), raw_objective, None, parameters,
            )
        except InferenceError as error:
            latest_failure = error
            failures.append(f"log_parameters={key!r}: {type(error).__name__}: {error}")
            evaluation = _Evaluation(_INVALID_OBJECTIVE, None, None, None)
        cache[key] = evaluation
        return evaluation.objective

    def objective(log_parameters: np.ndarray) -> float:
        key, prepared, error = _prepare(np.asarray(log_parameters, dtype=float))
        if key in cache:
            return _record(key, None, None)  # cache hit: _record returns immediately
        if prepared is None:
            return _record(key, None, error)
        model, parameters = prepared
        outcome = _run_fit(model, last_mode)
        return _record(key, parameters, outcome)

    def evaluate_batch(points: list[np.ndarray]) -> list[float]:
        """Objective at each point, in order -- dedupe within the batch AND against
        the cache (each repeat counts as a cache hit, as serial calls would),
        materialize serially (warm-start order preserved), fit via `map_ordered`,
        record serially in order."""
        if num_workers == 1:
            # Today's path: one model alive at a time, same results and counters.
            return [objective(point) for point in points]
        prepared: dict[tuple[float, ...], object] = {}
        errors: dict[tuple[float, ...], object] = {}
        keys: list[tuple[float, ...]] = []
        order: list[tuple[float, ...]] = []  # first-seen order of new keys, for the fit batch
        snapshot_mode = last_mode
        for point in points:
            key, entry, error = _prepare(np.asarray(point, dtype=float))
            keys.append(key)
            if key in cache or key in prepared or key in errors:
                continue
            if entry is None:
                errors[key] = error
            else:
                prepared[key] = entry
                order.append(key)
        models = [prepared[key][0] for key in order]
        outcomes = map_ordered(lambda model: _run_fit(model, snapshot_mode), models, num_workers)
        outcome_by_key = dict(zip(order, outcomes, strict=True))
        results = []
        for key in keys:
            if key in outcome_by_key:
                results.append(_record(key, prepared[key][1], outcome_by_key[key]))
            elif key in errors:
                results.append(_record(key, None, errors[key]))
            else:
                results.append(_record(key, None, None))
        return results

    def gradient(log_parameters: np.ndarray) -> np.ndarray:
        # Central differences with an ABSOLUTE step on the internal
        # (log/logit) scale: a step relative to |x| (as scipy's built-in
        # "3-point" jac uses) would depend on the arbitrary position on that
        # scale -- e.g. sigma ~ 1e153 puts x ~ 353, where a relative step is
        # ~100x coarser than `_FINITE_DIFFERENCE_STEP`. One-sided at a bound;
        # note that with an asymmetric clip `span` still gives a correct
        # central quotient.
        point = np.asarray(log_parameters, dtype=float)
        # `center` is normally cached: L-BFGS-B calls objective(x) before jac(x),
        # so this is a cache hit -- and it must be a plain `objective` call, not
        # part of the batch, so `last_mode` (the warm start) is set from it before
        # any batch runs.
        center = objective(point)
        result = np.empty_like(point)
        # Build the forward/backward points first, one-sided at a bound, in
        # today's per-index interleaved order (f0 already computed above, then
        # fwd0, bwd0, fwd1, bwd1, ...), then evaluate the whole batch at once so
        # num_workers > 1 can fan the independent fits out across threads.
        specs: list[tuple[int, float, float, bool, bool]] = []
        batch_points: list[np.ndarray] = []
        for index in range(point.size):
            step = _FINITE_DIFFERENCE_STEP
            forward = point.copy()
            backward = point.copy()
            forward[index] = min(point[index] + step, upper[index])
            backward[index] = max(point[index] - step, lower[index])
            span = forward[index] - backward[index]
            if span <= 0.0:
                specs.append((index, 0.0, 0.0, False, False))
                continue
            has_forward = forward[index] > point[index]
            has_backward = backward[index] < point[index]
            if has_forward:
                batch_points.append(forward)
            if has_backward:
                batch_points.append(backward)
            specs.append((index, forward[index] - point[index], point[index] - backward[index], has_forward, has_backward))
        batch_values = iter(evaluate_batch(batch_points))
        for index, forward_span, backward_span, has_forward, has_backward in specs:
            if not has_forward and not has_backward:
                result[index] = 0.0
                continue
            forward_value = next(batch_values) if has_forward else None
            backward_value = next(batch_values) if has_backward else None
            if has_forward and has_backward:
                result[index] = (forward_value - backward_value) / (forward_span + backward_span)
            elif has_forward:
                result[index] = (forward_value - center) / forward_span
            else:
                result[index] = (center - backward_value) / backward_span
        return result

    def fail(message: str) -> None:
        # Name the root cause in the message: a failure that is the same at every
        # theta (a dense size limit, say) otherwise reads as a convergence problem.
        if latest_failure is not None:
            message = f"{message} (last failure: {type(latest_failure).__name__}: {latest_failure})"
        error = OptimizationError(
            message,
            evaluations=evaluations,
            cache_hits=cache_hits,
            numerical_failures=tuple(failures),
        )
        if latest_failure is not None:
            raise error from latest_failure
        raise error

    history: list[float] = []
    points: list[np.ndarray] = []
    plateau_reached = False

    def plateau_callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        nonlocal plateau_reached
        if objective_tolerance <= 0.0:
            return
        point = np.asarray(intermediate_result.x, dtype=float).copy()
        entry = cache.get(tuple(float(value) for value in point))
        if entry is None or entry.raw_objective is None:
            return
        history.append(entry.raw_objective)
        points.append(point)
        # A plateau requires both the objective AND the point to have
        # stopped moving over the window: an objective that is merely flat
        # while the point still creeps towards a bound (e.g. an LML that is
        # almost flat in one hyperparameter) is not a plateau.
        if (
            len(history) > stall_iterations
            and history[-stall_iterations - 1] - history[-1]
            < objective_tolerance * max(1.0, abs(history[-1]))
            and np.max(np.abs(points[-1] - points[-stall_iterations - 1]))
            < _PLATEAU_STEP_TOLERANCE
        ):
            plateau_reached = True
            raise StopIteration

    with blas_limit(num_workers, blas_threads):
        started = perf_counter()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            solution = scipy.optimize.minimize(
                objective,
                start,
                method="L-BFGS-B",
                jac=gradient,
                bounds=scipy_bounds,
                callback=plateau_callback,
                options={
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    # `objective` has no scipy-computed jac -- `gradient` above supplies
                    # one -- because the posterior log-determinant it evaluates is
                    # reproducible at a fixed x but carries ~1e-7..1e-6 of noise between
                    # nearby x's -- generic ill-conditioning of the posterior precision
                    # (condition number ~1e9-1e14) re-formed and re-factorized at each
                    # point, seen on both the sparse splu path and the dense Cholesky
                    # path. Forward differences at a default-size step (or even
                    # `eps=1e-6`) measure that noise rather than the gradient and
                    # L-BFGS-B thrashes or gets stuck on a false plateau. Central
                    # differences cancel the noise to first order and keep making
                    # progress instead. `gradient` uses an ABSOLUTE step
                    # (`_FINITE_DIFFERENCE_STEP`) rather than scipy's built-in
                    # "3-point" jac's step relative to |x|, because a position on the
                    # internal log/logit scale is arbitrary -- e.g. sigma ~ 1e153 puts
                    # x ~ 353, where a relative step would be ~100x coarser than
                    # intended.
                    # ftol/gtol stay far below that noise floor on purpose: they are
                    # not what stops the search. Stopping is instead done by the
                    # plateau window below (`plateau_callback`), since L-BFGS-B would
                    # otherwise creep for hundreds of evaluations with improvements of
                    # ~1e-3 in LML that ftol/gtol never catch.
                },
            )
        elapsed_seconds = perf_counter() - started

        # L-BFGS-B reports status 2 (ABNORMAL_TERMINATION_IN_LNSRCH) when the line
        # search can no longer make progress. On an ill-conditioned direction — a
        # bounded (logit) hyperparameter, say — that routinely happens *at* the
        # optimum, because `_transform_objective`'s log1p compression pushes the
        # remaining change below ftol/gtol. Treat it as converged when the returned
        # point is usable and no worse than the start, and record it; anything else
        # is still a hard failure.
        # When the plateau callback (below) raises StopIteration, L-BFGS-B returns
        # success=False, status=2 -- the same status as a genuine line-search
        # stall -- so a plateau stop must be recognized and excluded from the
        # line-search-stall handling before it runs. `solution.x` after a callback
        # stop is the current iterate, which the callback just looked up in the
        # cache, so no extra validity check is needed here for that case.
        stopped_on_plateau = plateau_reached and not bool(solution.success)
        line_search_stalled = (
            not stopped_on_plateau
            and not bool(solution.success)
            and int(getattr(solution, "status", -1)) == 2
        )
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
        if not bool(solution.success) and not line_search_stalled and not stopped_on_plateau:
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
        if final_evaluation is None or final_evaluation.raw_objective is None:
            fail("empirical-Bayes optimization did not produce a valid final point")
        assert final_evaluation.parameters is not None
        assert final_evaluation.raw_objective is not None

        if skip_intermediate_variances:
            # The search itself never needed predictive variances, only
            # `log_marginal_likelihood` -- refit once, with variances, at the
            # accepted optimum so the returned fit is complete. This does not
            # count as an extra objective evaluation/cache entry: it recomputes
            # the same point the search already scored, just with the flag
            # flipped, so evaluations/cache_hits diagnostics stay exactly what
            # they would have been without this optimization.
            # Drop the single-slot cached fit first so two big fits are never
            # alive at once during this refit.
            latest_fit = None
            final_model = family.materialize(final_evaluation.parameters)
            final_fit_kwargs: dict[str, object] = {}
            if allow_large_dense:
                final_fit_kwargs["allow_large_dense"] = True
            if warm_start and last_mode is not None:
                final_fit_kwargs["initial_mode"] = last_mode
            final_fit = fit(final_model, **final_fit_kwargs)
            final_evaluation = _Evaluation(
                final_evaluation.objective,
                final_evaluation.raw_objective,
                final_fit,
                final_evaluation.parameters,
            )
        elif latest_fit is not None and latest_fit[0] == final_key:
            final_evaluation = _Evaluation(
                final_evaluation.objective,
                final_evaluation.raw_objective,
                latest_fit[1],
                final_evaluation.parameters,
            )
        else:
            # The final point was not the most recently computed evaluation (a
            # cache hit, or line-search bookkeeping moved past it) -- refit once
            # at the accepted parameters, with the same kwargs the search used
            # (no `predictive_variances` flag: this branch only runs when `fit`
            # does not accept it). Not counted in evaluations/cache_hits, same as
            # the skip_intermediate_variances refit above.
            final_model = family.materialize(final_evaluation.parameters)
            final_fit_kwargs: dict[str, object] = {}
            if allow_large_dense:
                final_fit_kwargs["allow_large_dense"] = True
            if warm_start and last_mode is not None:
                final_fit_kwargs["initial_mode"] = last_mode
            final_fit = fit(final_model, **final_fit_kwargs)
            final_evaluation = _Evaluation(
                final_evaluation.objective,
                final_evaluation.raw_objective,
                final_fit,
                final_evaluation.parameters,
            )

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

from collections.abc import Callable
import gc
import json
import weakref

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix, eye

from pylgm.exceptions import DenseReferenceLimitError, NumericalError, OptimizationError
from pylgm.inference import GaussianResult, fit_gaussian
from pylgm.inference.laplace import fit_laplace
from pylgm.ir import (
    CompiledFamily,
    CompiledGaussianFamily,
    CompiledLGM,
    Hyperparameters,
    LatentBlock,
    ScalableBlock,
)
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.optimization import (
    OptimizationBounds,
    optimize_empirical_bayes,
)
from pylgm.optimization import empirical_bayes
from pylgm.optimization.transforms import LogitTransform, LogTransform


def zero_latent_family(y: np.ndarray) -> CompiledGaussianFamily:
    return CompiledGaussianFamily(
        y=y,
        observed=np.ones(y.size, dtype=bool),
        offset=np.zeros(y.size),
        blocks=(),
        parameter_names=("sigma",),
        initial=Hyperparameters(sigma=1.0, precisions={}),
    )


def scalar_conjugate_family(y: float) -> CompiledGaussianFamily:
    block = LatentBlock(
        "latent",
        ("x",),
        csr_matrix([[1.0]]),
        csr_matrix([[1.0]]),
        np.empty((0, 1)),
    )
    return CompiledGaussianFamily(
        y=np.array([y]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(ScalableBlock(block, "latent.precision", 1.0),),
        parameter_names=("latent.precision",),
        initial=Hyperparameters(sigma=1.0, precisions={"latent": 1.0}),
    )


def scalable_precision_family(base_precision: float = 1.0) -> CompiledGaussianFamily:
    block = LatentBlock(
        "latent",
        ("x",),
        csr_matrix([[1.0]]),
        csr_matrix([[base_precision]]),
        np.empty((0, 1)),
    )
    return CompiledGaussianFamily(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(ScalableBlock(block, "latent.precision", 1.0),),
        parameter_names=("latent.precision",),
        initial=Hyperparameters(sigma=1.0, precisions={"latent": 1.0}),
    )


def constant_fit(
    model: CompiledLGM,
    *,
    log_marginal_likelihood: float = -1.0,
) -> GaussianResult:
    latent_size = len(model.labels)
    return GaussianResult(
        labels=model.labels,
        mean=np.zeros(latent_size),
        covariance=np.zeros((latent_size, latent_size)),
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=np.zeros(model.y.size),
        predictive_variance=np.ones(model.y.size),
    )


def test_sigma_optimization_matches_zero_latent_gaussian_mle() -> None:
    family = zero_latent_family(y=np.array([-2.0, 1.0, 3.0]))
    expected = np.sqrt(np.mean(family.y**2))

    result = optimize_empirical_bayes(
        family,
        {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
    )

    assert result.parameters["sigma"] == pytest.approx(expected, rel=1e-5)
    assert result.diagnostics.converged
    assert result.diagnostics.objective == pytest.approx(
        -result.fit.log_marginal_likelihood
    )


def test_precision_optimization_matches_scalar_conjugate_mle() -> None:
    family = scalar_conjugate_family(y=2.0)

    result = optimize_empirical_bayes(
        family,
        {
            "latent.precision": OptimizationBounds(
                initial=1.0,
                lower=0.05,
                upper=10.0,
            )
        },
    )

    # Marginally y ~ N(0, 1 + 1 / precision), whose one-point MLE is
    # 1 + 1 / precision = y**2.
    assert result.parameters["latent.precision"] == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_optimizer_uses_log_coordinates_and_caches_exact_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        *,
        method: str,
        bounds: list[tuple[float, float]],
        **_: object,
    ) -> OptimizeResult:
        captured.update(start=start.copy(), method=method, bounds=bounds)
        first = objective(start.copy())
        second = objective(start.copy())
        assert second == first
        return OptimizeResult(x=start.copy(), fun=first, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([2.0])),
        {"sigma": OptimizationBounds(initial=2.0, lower=0.25, upper=16.0)},
    )

    np.testing.assert_allclose(captured["start"], np.log([2.0]))
    assert captured["bounds"] == [(np.log(0.25), np.log(16.0))]
    assert captured["method"] == "L-BFGS-B"
    assert result.diagnostics.evaluations == 1
    assert result.diagnostics.cache_hits >= 1


@pytest.mark.parametrize(
    ("lower", "upper", "endpoint"),
    [
        (0.1, 10.0, "lower"),
        (0.1, 10.0, "upper"),
        (0.1, 100.0, "upper"),
        (float(np.nextafter(0.0, 1.0)), float(np.finfo(float).max), "lower"),
        (float(np.nextafter(0.0, 1.0)), float(np.finfo(float).max), "upper"),
    ],
)
def test_log_bound_endpoints_decode_exactly_and_round_trip_as_warm_starts(
    lower: float,
    upper: float,
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = lower if endpoint == "lower" else upper

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        *,
        bounds: list[tuple[float, float]],
        **_: object,
    ) -> OptimizeResult:
        point = np.array([bounds[0][0 if endpoint == "lower" else 1]])
        value = objective(point)
        return OptimizeResult(x=point, fun=value, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", constant_fit)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)
    bounds = {
        "sigma": OptimizationBounds(initial=1.0, lower=lower, upper=upper)
    }

    first = optimize_empirical_bayes(zero_latent_family(np.array([1.0])), bounds)
    second = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        bounds,
        initial=first.parameters,
    )

    assert first.parameters["sigma"] == expected
    assert second.diagnostics.initial["sigma"] == expected


@pytest.mark.parametrize("endpoint", [1.0, 1.000000005])
def test_real_narrow_bound_endpoint_is_decoded_exactly(endpoint: float) -> None:
    result = optimize_empirical_bayes(
        zero_latent_family(np.array([endpoint])),
        {
            "sigma": OptimizationBounds(
                initial=endpoint,
                lower=1.0,
                upper=1.000000005,
            )
        },
    )

    assert result.parameters["sigma"] == endpoint
    assert result.diagnostics.active_bounds == ("sigma",)


def test_nearest_narrow_endpoint_wins_when_global_tolerances_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = 1.0
    upper = 1.000000005
    log_lower = np.log(lower)
    log_upper = np.log(upper)
    nearer_upper = log_lower + 0.75 * (log_upper - log_lower)

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        point = np.array([nearer_upper])
        value = objective(point)
        return OptimizeResult(x=point, fun=value, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", constant_fit)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=lower, lower=lower, upper=upper)},
    )

    assert result.parameters["sigma"] == upper


def test_distinct_natural_bounds_that_collapse_in_log_space_are_rejected() -> None:
    lower = 1e100
    upper = float(np.nextafter(lower, np.inf))
    assert lower < upper
    assert np.log(lower) == np.log(upper)

    with pytest.raises(OptimizationError, match="collapse.*internal space"):
        optimize_empirical_bayes(
            zero_latent_family(np.array([lower])),
            {
                "sigma": OptimizationBounds(
                    initial=lower,
                    lower=lower,
                    upper=upper,
                )
            },
        )


def test_invalid_penalty_dominates_every_finite_valid_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extreme_fit(model: CompiledLGM) -> GaussianResult:
        if model.sigma == 2.0:
            raise NumericalError("invalid endpoint")
        return constant_fit(model, log_marginal_likelihood=-1e200)

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        *,
        bounds: list[tuple[float, float]],
        **_: object,
    ) -> OptimizeResult:
        valid = objective(start)
        invalid = objective(np.array([bounds[0][1]]))
        assert valid == pytest.approx(empirical_bayes._transform_objective(1e200))
        assert invalid > valid
        return OptimizeResult(x=start, fun=valid, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", extreme_fit)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=0.5, upper=2.0)},
    )

    assert result.diagnostics.objective == 1e200
    assert "invalid endpoint" in result.diagnostics.numerical_failures[0]


def test_optimizer_objective_transform_is_strictly_ordered_and_bounded() -> None:
    maximum = float(np.finfo(float).max)
    raw_values = (-maximum, -1e200, -1.0, 0.0, 1.0, 1e200, maximum)

    transformed = [
        empirical_bayes._transform_objective(value) for value in raw_values
    ]

    assert all(np.isfinite(value) for value in transformed)
    assert all(
        left < right for left, right in zip(transformed, transformed[1:])
    )
    assert transformed[-1] < empirical_bayes._INVALID_OBJECTIVE < 1_000.0


def test_extreme_scale_sigma_optimization_uses_real_lbfgsb_stably() -> None:
    family = zero_latent_family(np.array([4e153]))

    result = optimize_empirical_bayes(
        family,
        {
            "sigma": OptimizationBounds(
                initial=3e153,
                lower=1e153,
                upper=1e154,
            )
        },
    )

    assert result.parameters["sigma"] == pytest.approx(4e153, rel=1e-5)
    assert result.diagnostics.objective == pytest.approx(
        -result.fit.log_marginal_likelihood
    )
    assert result.diagnostics.converged


def test_repeated_fits_are_deterministic() -> None:
    family = zero_latent_family(np.array([-2.0, 1.0, 3.0]))
    bounds = {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)}

    first = optimize_empirical_bayes(family, bounds)
    second = optimize_empirical_bayes(family, bounds)

    assert dict(first.parameters) == dict(second.parameters)
    assert first.diagnostics.objective == second.diagnostics.objective
    assert first.diagnostics.evaluations == second.diagnostics.evaluations
    assert first.diagnostics.cache_hits == second.diagnostics.cache_hits
    assert first.diagnostics.active_bounds == second.diagnostics.active_bounds
    assert first.diagnostics.numerical_failures == second.diagnostics.numerical_failures


def test_warm_start_overrides_the_recorded_cold_start() -> None:
    warm = {"sigma": 3.0}

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([2.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
        initial=warm,
    )
    warm["sigma"] = 9.0

    assert result.diagnostics.initial["sigma"] == 3.0


def test_optimum_at_a_bound_is_reported() -> None:
    result = optimize_empirical_bayes(
        zero_latent_family(np.array([0.01])),
        {"sigma": OptimizationBounds(initial=1.0, lower=0.5, upper=10.0)},
    )

    assert result.parameters["sigma"] == pytest.approx(0.5)
    assert result.diagnostics.active_bounds == ("sigma",)


def test_invalid_numerical_evaluations_are_recorded_and_do_not_prevent_a_valid_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = np.log(np.array([1e-200]))

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        invalid_objective = objective(invalid)
        assert np.isfinite(invalid_objective)
        valid_objective = objective(start)
        return OptimizeResult(x=start, fun=valid_objective, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=1e-200, upper=10.0)},
    )

    assert len(result.diagnostics.numerical_failures) == 1
    assert "sigma squared must be finite and positive" in (
        result.diagnostics.numerical_failures[0]
    )


def test_dense_reference_safety_errors_are_recorded_as_typed_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fit = fit_gaussian
    calls = 0

    def fail_once(model: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DenseReferenceLimitError("dense safety stop")
        return real_fit(model)  # type: ignore[arg-type]

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        objective(np.log(np.array([2.0])))
        valid_objective = objective(start)
        return OptimizeResult(x=start, fun=valid_objective, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", fail_once)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
    )

    assert len(result.diagnostics.numerical_failures) == 1
    assert "dense safety stop" in result.diagnostics.numerical_failures[0]


def test_parameter_driven_precision_overflow_is_recorded_as_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = scalable_precision_family(base_precision=2.0)

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        *,
        bounds: list[tuple[float, float]],
        **_: object,
    ) -> OptimizeResult:
        invalid = objective(np.array([bounds[0][1]]))
        assert np.isfinite(invalid)
        valid = objective(start)
        return OptimizeResult(x=start, fun=valid, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        family,
        {
            "latent.precision": OptimizationBounds(
                initial=1.0,
                lower=0.1,
                upper=1e308,
            )
        },
    )

    assert len(result.diagnostics.numerical_failures) == 1
    assert "precision scaling" in result.diagnostics.numerical_failures[0]


def test_unsuccessful_scipy_solution_raises_even_with_a_valid_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        value = objective(start)
        return OptimizeResult(x=start, fun=value, success=False, message="line search failed")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    with pytest.raises(OptimizationError, match="line search failed"):
        optimize_empirical_bayes(
            zero_latent_family(np.array([1.0])),
            {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
        )


def test_invalid_final_point_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = np.log(np.array([1e-200]))

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        invalid_objective = objective(invalid)
        return OptimizeResult(
            x=invalid,
            fun=invalid_objective,
            success=True,
            message="reported success",
        )

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    with pytest.raises(OptimizationError, match="valid final point"):
        optimize_empirical_bayes(
            zero_latent_family(np.array([1.0])),
            {"sigma": OptimizationBounds(initial=1.0, lower=1e-200, upper=10.0)},
        )


def test_all_invalid_failure_preserves_counts_history_and_dense_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dense_failure(_: CompiledLGM) -> GaussianResult:
        raise DenseReferenceLimitError("dense root cause")

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        objective(start)
        objective(start.copy())
        return OptimizeResult(
            x=start,
            fun=np.finfo(float).max,
            success=False,
            message="all invalid",
        )

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", dense_failure)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    with pytest.raises(OptimizationError, match="all invalid") as captured:
        optimize_empirical_bayes(
            zero_latent_family(np.array([1.0])),
            {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
        )

    error = captured.value
    assert error.evaluations == 1
    assert error.cache_hits == 1
    assert error.numerical_failures == (
        "log_parameters=(0.0,): DenseReferenceLimitError: dense root cause",
    )
    assert isinstance(error.__cause__, DenseReferenceLimitError)
    snapshot = error.to_dict()
    assert json.loads(json.dumps(snapshot)) == snapshot
    snapshot["numerical_failures"].append("changed")
    assert len(error.numerical_failures) == 1


def test_multiple_failures_retain_only_latest_exception_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references: list[weakref.ReferenceType[NumericalError]] = []

    def numerical_failure(model: CompiledLGM) -> GaussianResult:
        error = NumericalError(f"failure at sigma={model.sigma}")
        references.append(weakref.ref(error))
        raise error

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        *,
        bounds: list[tuple[float, float]],
        **_: object,
    ) -> OptimizeResult:
        objective(np.array([bounds[0][0]]))
        objective(np.array([bounds[0][1]]))
        return OptimizeResult(
            x=start,
            fun=empirical_bayes._INVALID_OBJECTIVE,
            success=False,
            message="multiple failures",
        )

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", numerical_failure)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    with pytest.raises(OptimizationError, match="multiple failures") as captured:
        optimize_empirical_bayes(
            zero_latent_family(np.array([1.0])),
            {"sigma": OptimizationBounds(initial=1.0, lower=0.5, upper=2.0)},
        )

    gc.collect()
    error = captured.value
    assert error.evaluations == 2
    assert len(error.numerical_failures) == 2
    assert "sigma=0.5" in error.numerical_failures[0]
    assert "sigma=2.0" in error.numerical_failures[1]
    assert references[0]() is None
    assert references[1]() is error.__cause__


def test_programming_errors_from_inference_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_fit(_: object) -> object:
        raise RuntimeError("programming defect")

    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        objective(start)
        raise AssertionError("unreachable")

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", broken_fit)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    with pytest.raises(RuntimeError, match="programming defect"):
        optimize_empirical_bayes(
            zero_latent_family(np.array([1.0])),
            {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
        )


def test_returned_fit_equals_a_rematerialized_optimum() -> None:
    family = scalar_conjugate_family(y=2.0)
    result = optimize_empirical_bayes(
        family,
        {
            "latent.precision": OptimizationBounds(
                initial=1.0,
                lower=0.05,
                upper=10.0,
            )
        },
    )

    expected = fit_gaussian(family.materialize(result.parameters))

    assert result.fit.log_marginal_likelihood == expected.log_marginal_likelihood
    np.testing.assert_array_equal(result.fit.mean, expected.mean)
    np.testing.assert_array_equal(result.fit.covariance, expected.covariance)
    np.testing.assert_array_equal(result.fit.predictive_mean, expected.predictive_mean)
    np.testing.assert_array_equal(
        result.fit.predictive_variance,
        expected.predictive_variance,
    )


@pytest.mark.parametrize(
    "values",
    [
        (True, 0.1, 2.0),
        (np.float64(1.0), 0.1, 2.0),
        (1.0, 0.1, np.inf),
        (1.0, 0.1, np.nan),
        (10**1000, 0.1, 2.0),
    ],
)
def test_bounds_require_ordinary_finite_positive_numbers(
    values: tuple[object, object, object],
) -> None:
    with pytest.raises(ValueError, match="ordinary finite positive"):
        OptimizationBounds(*values)  # type: ignore[arg-type]


def test_bounds_outside_the_transform_domain_name_that_domain() -> None:
    # A well-formed number outside the transform's natural domain is a domain
    # error, and the message names the domain rather than assuming positivity.
    with pytest.raises(ValueError, match="the positive reals"):
        OptimizationBounds(1.0, 0.0, 2.0)
    with pytest.raises(ValueError, match=r"the open interval \(-1, 1\)"):
        OptimizationBounds(0.0, -2.0, 0.5, transform=LogitTransform(-1.0, 1.0))


@pytest.mark.parametrize("values", [(0.5, 1.0, 2.0), (3.0, 1.0, 2.0), (1.0, 1.0, 1.0)])
def test_bounds_require_a_nonempty_ordered_interval_containing_initial(
    values: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="lower.*initial.*upper"):
        OptimizationBounds(*values)


def test_empty_direct_optimization_problem_raises_typed_error() -> None:
    family = CompiledGaussianFamily(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(),
        parameter_names=(),
        initial=Hyperparameters(sigma=1.0, precisions={}),
    )

    with pytest.raises(OptimizationError, match="at least one parameter"):
        optimize_empirical_bayes(family, {})


@pytest.mark.parametrize(
    ("lower", "upper", "point", "expected"),
    [
        (1.0, 2.0, np.log(1.0) + 5e-9, ("sigma",)),
        (1.0, 1e200, np.log(1e200) - 1e-5, ()),
    ],
)
def test_active_bound_detection_uses_absolute_log_space_tolerance(
    lower: float,
    upper: float,
    point: float,
    expected: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_minimize(
        objective: Callable[[np.ndarray], float],
        start: np.ndarray,
        **_: object,
    ) -> OptimizeResult:
        final = np.array([point])
        value = objective(final)
        return OptimizeResult(x=final, fun=value, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes, "fit_gaussian", constant_fit)
    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)

    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=lower, upper=upper)},
    )

    assert result.diagnostics.active_bounds == expected


def test_returned_mappings_are_immutable_and_value_isolated() -> None:
    warm = {"sigma": 1.5}
    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
        initial=warm,
    )
    warm["sigma"] = 8.0

    assert result.diagnostics.initial["sigma"] == 1.5
    with pytest.raises(TypeError):
        result.parameters["sigma"] = 3.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.diagnostics.initial["sigma"] = 3.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.diagnostics.optimum["sigma"] = 3.0  # type: ignore[index]


def test_result_and_diagnostics_expose_isolated_json_ready_snapshots() -> None:
    result = optimize_empirical_bayes(
        zero_latent_family(np.array([1.0])),
        {"sigma": OptimizationBounds(initial=1.0, lower=0.1, upper=10.0)},
    )

    diagnostics = result.diagnostics.to_dict()
    snapshot = result.to_dict()

    assert json.loads(json.dumps(diagnostics)) == diagnostics
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["parameters"] == dict(result.parameters)
    assert snapshot["diagnostics"] == diagnostics
    assert snapshot["fit"] == {
        "log_marginal_likelihood": result.fit.log_marginal_likelihood,
        "latent_dimension": 0,
        "prediction_count": 1,
        "arrays_omitted": [
            "mean",
            "covariance",
            "predictive_mean",
            "predictive_variance",
        ],
    }
    snapshot["parameters"]["sigma"] = 99.0
    snapshot["diagnostics"]["initial"]["sigma"] = 99.0
    snapshot["fit"]["arrays_omitted"].append("changed")
    assert result.parameters["sigma"] != 99.0
    assert result.diagnostics.initial["sigma"] != 99.0
    assert len(result.to_dict()["fit"]["arrays_omitted"]) == 4


def _unit_block(name: str) -> LatentBlock:
    return LatentBlock(
        name,
        ("a", "b"),
        csr_matrix(np.eye(2)),
        eye(2, format="csr"),
        np.empty((0, 2), dtype=float),
    )


def test_optimize_over_compiled_family_gaussian() -> None:
    blocks = (ScalableBlock(_unit_block("g"), "g_prec", 1.0),)
    family = CompiledFamily(
        y=np.array([1.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        blocks=blocks,
        parameter_names=("g_prec",),
        likelihood_factory=lambda r: CompiledGaussian(1.0),
    )
    result = optimize_empirical_bayes(family, {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)})
    assert result.diagnostics.converged
    assert "g_prec" in result.parameters


def test_optimize_over_laplace_family_converges() -> None:
    blocks = (ScalableBlock(_unit_block("g"), "g_prec", 1.0),)
    family = CompiledFamily(
        y=np.array([2.0, 4.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        blocks=blocks,
        parameter_names=("g_prec",),
        likelihood_factory=lambda r: CompiledPoisson(),
    )
    result = optimize_empirical_bayes(
        family, {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)}, fit=fit_laplace
    )
    assert result.diagnostics.converged


def test_optimize_rejects_family_without_materialize() -> None:
    class _NoMaterialize:
        parameter_names = ("g_prec",)

    with pytest.raises(TypeError, match="materialize"):
        optimize_empirical_bayes(_NoMaterialize(), {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)})


def _unit_family():
    block = LatentBlock("g", ("a", "b"), csr_matrix(np.eye(2)), eye(2, format="csr"),
                        np.empty((0, 2), dtype=float))
    return CompiledFamily(
        y=np.array([1.0, -1.0]), observed=np.array([True, True]), offset=np.zeros(2),
        blocks=(ScalableBlock(block, "g_prec", 1.0),), parameter_names=("g_prec",),
        likelihood_factory=lambda r: CompiledGaussian(1.0),
    )


def test_penalty_none_matches_unpenalized():
    family = _unit_family()
    bounds = {"g_prec": OptimizationBounds(1.0, 1e-3, 1e3)}
    a = optimize_empirical_bayes(family, bounds)
    b = optimize_empirical_bayes(family, bounds, penalty=None)
    assert a.parameters["g_prec"] == b.parameters["g_prec"]


def test_penalty_shifts_the_optimum():
    family = _unit_family()
    bounds = {"g_prec": OptimizationBounds(1.0, 1e-3, 1e6)}
    plain = optimize_empirical_bayes(family, bounds)
    # a penalty that strongly rewards larger g_prec must move the optimum up
    penalized = optimize_empirical_bayes(
        family, bounds, penalty=lambda values: 5.0 * np.log(values["g_prec"]),
    )
    assert penalized.parameters["g_prec"] > plain.parameters["g_prec"]


class _ScalarFamily:
    """Family whose 'log marginal likelihood' peaks at a known natural value."""

    def __init__(self, name, target):
        self.parameter_names = (name,)
        self._name = name
        self._target = target

    def materialize(self, values):
        return values[self._name]


class _QuadraticFit:
    """Injected fit whose log marginal likelihood peaks at ``target``."""

    def __init__(self, value, target):
        self.log_marginal_likelihood = -((value - target) ** 2)


def test_optimizer_recovers_bounded_optimum():
    name = "rho"
    target = 0.6

    class _Fit:
        def __init__(self, value):
            # concave in value, peak at target, within (-1, 1)
            self.log_marginal_likelihood = -((value - target) ** 2)

    fam = _ScalarFamily(name, target)
    bounds = {name: OptimizationBounds(0.0, -0.99, 0.99, transform=LogitTransform(-1.0, 1.0))}
    result = optimize_empirical_bayes(fam, bounds, fit=lambda m, **k: _Fit(m))
    assert np.isclose(result.parameters[name], target, atol=1e-3)


def test_optimization_bounds_defaults_to_log_and_validates():
    b = OptimizationBounds(1.0, 0.1, 10.0)
    assert isinstance(b.transform, LogTransform)
    with pytest.raises(ValueError):
        OptimizationBounds(1.0, -0.1, 10.0)  # negative lower invalid under Log
    # Logit bounds must sit inside the interval:
    with pytest.raises(ValueError):
        OptimizationBounds(0.0, -1.0, 0.9, transform=LogitTransform(-1.0, 1.0))


def test_line_search_stall_at_the_optimum_is_accepted(monkeypatch) -> None:
    # L-BFGS-B reports status 2 (ABNORMAL_TERMINATION_IN_LNSRCH) when the line
    # search cannot progress, which on an ill-conditioned direction happens AT
    # the optimum. A usable in-box point that is no worse than the start must be
    # accepted and recorded, not turned into a hard OptimizationError.
    import scipy.optimize

    real_minimize = scipy.optimize.minimize

    def stalling_minimize(*args, **kwargs):
        solution = real_minimize(*args, **kwargs)
        solution.success = False
        solution.status = 2
        solution.message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        return solution

    monkeypatch.setattr(scipy.optimize, "minimize", stalling_minimize)

    family = _ScalarFamily("tau", 1.5)
    bounds = {"tau": OptimizationBounds(1.0, 0.1, 10.0)}
    result = optimize_empirical_bayes(
        family, bounds, fit=lambda model, **kwargs: _QuadraticFit(model, 1.5)
    )
    assert result.diagnostics.converged
    assert any("line search stalled" in note for note in result.diagnostics.numerical_failures)


def test_line_search_stall_at_an_unusable_point_still_fails(monkeypatch) -> None:
    import scipy.optimize

    real_minimize = scipy.optimize.minimize

    def bad_stalling_minimize(*args, **kwargs):
        solution = real_minimize(*args, **kwargs)
        solution.success = False
        solution.status = 2
        solution.message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        solution.x = np.array([np.nan])
        return solution

    monkeypatch.setattr(scipy.optimize, "minimize", bad_stalling_minimize)

    family = _ScalarFamily("tau", 1.5)
    bounds = {"tau": OptimizationBounds(1.0, 0.1, 10.0)}
    with pytest.raises(OptimizationError, match="did not converge"):
        optimize_empirical_bayes(
            family, bounds, fit=lambda model, **kwargs: _QuadraticFit(model, 1.5)
        )

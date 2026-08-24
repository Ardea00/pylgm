"""Golden baseline of the public result surface.

Captures a wide but shallow slice of everything a caller can observe on a
fitted result -- every public attribute, every method output, ``repr``, and
array immutability -- across a small matrix of deterministic fits spanning
all three likelihoods, all three hyperparameter-resolution modes, and four
effect types. The refactor that gives ``GaussianResult``/``LaplaceResult``/
``INLAResult`` a shared base must leave this file's assertions untouched:
``test_result_surface_matches_baseline`` is the acceptance gate.

Regenerate the baseline (only ever on code you trust to be correct) with:

    PYTHONPATH=src python tests/inference/test_result_surface.py
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Besag, Bernoulli, Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson, RW1
from pylgm.inference.result import GaussianResult, INLAResult, LaplaceResult
from pylgm.priors import PCPrecision

BASELINE_PATH = Path(__file__).parent / "result_surface_baseline.json"

# A tiny connected chain graph over four regions, reused by every Besag fit.
GRAPH = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 3] for i in range(4)}


# ---------------------------------------------------------------------------
# Deterministic small frames
# ---------------------------------------------------------------------------


def _respond(rng: np.random.Generator, eta: np.ndarray, response: str, **index_columns):
    if response == "gaussian":
        y = eta + rng.normal(scale=0.15, size=eta.shape)
    elif response == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif response == "bernoulli":
        p = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(eta.shape) < p).astype(float)
    else:  # pragma: no cover - matrix only ever passes the three above
        raise ValueError(f"unknown response family: {response!r}")
    data = dict(index_columns)
    data["y"] = y
    return pd.DataFrame(data)


def _iid_frame(seed: int, response: str, n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.array((["a", "b", "c", "d"] * ((n + 3) // 4))[:n])
    effect = {"a": 0.4, "b": -0.3, "c": 0.2, "d": -0.1}
    eta = 0.5 + np.array([effect[g] for g in groups])
    return _respond(rng, eta, response, grp=groups)


def _rw1_frame(seed: int, response: str, n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    eta = 0.3 + 0.05 * t
    return _respond(rng, eta, response, t=t)


def _ar1_frame(seed: int, response: str, rho: float, n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(scale=0.3)
    eta = 0.2 + x
    return _respond(rng, eta, response, t=np.arange(n))


def _besag_frame(seed: int, response: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = np.array(["0", "1", "2", "3"])
    eta = 0.4 + np.array([0.3, -0.2, 0.1, -0.4])
    return _respond(rng, eta, response, region=regions)


def _precision_hyperparameter(name: str) -> Hyperparameter:
    return Hyperparameter(name, initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.05))


# ---------------------------------------------------------------------------
# The fit matrix: Gaussian/Poisson/Bernoulli x plug-in/optimize/integrate x a
# handful of effects (IID, RW1, AR1 with a fixed rho, Besag).
# ---------------------------------------------------------------------------


def _gaussian_iid_plugin():
    frame = _iid_frame(1, "gaussian")
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + IID("grp", index="grp", precision=2.0),
    )
    return model, frame, dict(engine="exact_gaussian")


def _gaussian_iid_optimize():
    frame = _iid_frame(1, "gaussian")
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + IID("grp", index="grp", precision=_precision_hyperparameter("grp.precision")),
    )
    return model, frame, dict(engine="exact_gaussian", hyperparameters="optimize")


def _gaussian_iid_integrate():
    frame = _iid_frame(1, "gaussian")
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + IID("grp", index="grp", precision=_precision_hyperparameter("grp.precision")),
    )
    return model, frame, dict(engine="exact_gaussian", hyperparameters="integrate")


def _gaussian_rw1_plugin():
    frame = _rw1_frame(2, "gaussian")
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + RW1("trend", index="t", precision=1.5),
    )
    return model, frame, dict(engine="exact_gaussian")


def _gaussian_rw1_optimize():
    frame = _rw1_frame(2, "gaussian")
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + RW1("trend", index="t", precision=_precision_hyperparameter("trend.precision")),
    )
    return model, frame, dict(engine="exact_gaussian", hyperparameters="optimize")


def _gaussian_ar1_integrate_fixed_rho():
    frame = _ar1_frame(3, "gaussian", rho=0.6)
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + AR1(
            "trend", index="t", precision=_precision_hyperparameter("trend.precision"), rho=0.6
        ),
    )
    return model, frame, dict(engine="exact_gaussian", hyperparameters="integrate")


def _gaussian_besag_plugin():
    frame = _besag_frame(4, "gaussian")
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.2),
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=1.0),
    )
    return model, frame, dict(engine="exact_gaussian")


def _poisson_iid_laplace_plugin():
    frame = _iid_frame(5, "poisson")
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("grp", index="grp", precision=1.0),
    )
    return model, frame, dict(engine="laplace")


def _poisson_rw1_laplace_optimize():
    frame = _rw1_frame(6, "poisson")
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + RW1("trend", index="t", precision=_precision_hyperparameter("trend.precision")),
    )
    return model, frame, dict(engine="laplace", hyperparameters="optimize")


def _poisson_besag_laplace_integrate():
    frame = _besag_frame(7, "poisson")
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Besag(
            "region", index="region", graph=GRAPH, precision=_precision_hyperparameter("region.precision")
        ),
    )
    return model, frame, dict(engine="laplace", hyperparameters="integrate")


def _bernoulli_iid_laplace_plugin():
    frame = _iid_frame(8, "bernoulli")
    model = LGM(
        response="y", likelihood=Bernoulli(),
        predictor=Fixed("1") + IID("grp", index="grp", precision=1.0),
    )
    return model, frame, dict(engine="laplace")


def _bernoulli_ar1_laplace_optimize_fixed_rho():
    frame = _ar1_frame(9, "bernoulli", rho=0.5)
    model = LGM(
        response="y", likelihood=Bernoulli(),
        predictor=Fixed("1") + AR1(
            "trend", index="t", precision=_precision_hyperparameter("trend.precision"), rho=0.5
        ),
    )
    return model, frame, dict(engine="laplace", hyperparameters="optimize")


MATRIX: list[tuple[str, "callable"]] = [
    ("gaussian_iid_plugin", _gaussian_iid_plugin),
    ("gaussian_iid_optimize", _gaussian_iid_optimize),
    ("gaussian_iid_integrate", _gaussian_iid_integrate),
    ("gaussian_rw1_plugin", _gaussian_rw1_plugin),
    ("gaussian_rw1_optimize", _gaussian_rw1_optimize),
    ("gaussian_ar1_integrate_fixed_rho", _gaussian_ar1_integrate_fixed_rho),
    ("gaussian_besag_plugin", _gaussian_besag_plugin),
    ("poisson_iid_laplace_plugin", _poisson_iid_laplace_plugin),
    ("poisson_rw1_laplace_optimize", _poisson_rw1_laplace_optimize),
    ("poisson_besag_laplace_integrate", _poisson_besag_laplace_integrate),
    ("bernoulli_iid_laplace_plugin", _bernoulli_iid_laplace_plugin),
    ("bernoulli_ar1_laplace_optimize_fixed_rho", _bernoulli_ar1_laplace_optimize_fixed_rho),
]


def _build(name: str):
    for candidate, builder in MATRIX:
        if candidate == name:
            model, frame, fit_kwargs = builder()
            return model.fit(frame, **fit_kwargs)
    raise KeyError(name)  # pragma: no cover - defensive


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_RESULT_TYPES = (GaussianResult, LaplaceResult, INLAResult)


def _round_float(value: float, sig: int = 12) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    digits = sig - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, digits)


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _round_float(float(value))
    return value


def _raises_on_write(array: np.ndarray) -> bool | None:
    """True if writing to ``array`` raises; None if it has no elements to probe."""
    array = np.asarray(array)
    if array.size == 0:
        return None
    index = (0,) * array.ndim
    try:
        array[index] = array[index]
    except Exception:
        return True
    return False


def _linear_combination_weights(dimension: int) -> np.ndarray:
    """A fixed, small contrast matrix: the overall average and a first-vs-last spread."""
    average = np.full(dimension, 1.0 / dimension)
    spread = np.zeros(dimension)
    spread[0] = 1.0
    spread[-1] -= 1.0
    return np.vstack([average, spread])


def _marginals_surface(marginals) -> dict:
    return {
        "mean": _to_jsonable(marginals.mean),
        "variance": _to_jsonable(marginals.variance),
        "std": _to_jsonable(marginals.std),
        "mean_raises_on_write": _raises_on_write(marginals.mean),
        "variance_raises_on_write": _raises_on_write(marginals.variance),
        "std_raises_on_write": _raises_on_write(marginals.std),
    }


def _surface(result) -> dict:
    """Capture the full public surface of a fitted result as a JSON-able dict."""
    attributes: dict[str, object] = {
        "labels": _to_jsonable(result.labels),
        "mean": _to_jsonable(result.mean),
        "covariance": _to_jsonable(result.covariance),
        "log_marginal_likelihood": _to_jsonable(result.log_marginal_likelihood),
        "predictive_mean": _to_jsonable(result.predictive_mean),
        "predictive_variance": _to_jsonable(result.predictive_variance),
        "engine": result.engine,
        "converged": bool(result.converged),
        "hyperparameters": _to_jsonable(result.hyperparameters) if result.hyperparameters is not None else None,
    }
    if hasattr(result, "fitted_mean"):
        attributes["fitted_mean"] = (
            _to_jsonable(result.fitted_mean) if result.fitted_mean is not None else None
        )
    if hasattr(result, "link_name"):
        attributes["link_name"] = result.link_name

    immutability: dict[str, object] = {
        "mean": _raises_on_write(result.mean),
        "covariance": _raises_on_write(result.covariance),
        "predictive_mean": _raises_on_write(result.predictive_mean),
        "predictive_variance": _raises_on_write(result.predictive_variance),
    }
    if hasattr(result, "fitted_mean") and result.fitted_mean is not None:
        immutability["fitted_mean"] = _raises_on_write(result.fitted_mean)

    latent = result.latent_marginals()
    hyperparameter_marginals = {
        name: _marginals_surface(value) for name, value in sorted(result.hyperparameter_marginals().items())
    }
    weights = _linear_combination_weights(len(result.mean))
    combined = result.linear_combinations(weights)

    methods: dict[str, object] = {
        "latent_marginals": _marginals_surface(latent),
        "hyperparameter_marginals": hyperparameter_marginals,
        "linear_combinations": _marginals_surface(combined),
    }
    if hasattr(result, "criteria"):
        criteria = result.criteria
        methods["criteria"] = {
            "dic": _to_jsonable(criteria.dic),
            "dic_effective_parameters": _to_jsonable(criteria.dic_effective_parameters),
            "waic": _to_jsonable(criteria.waic),
            "waic_effective_parameters": _to_jsonable(criteria.waic_effective_parameters),
            "cpo": _to_jsonable(criteria.cpo),
            "pit": _to_jsonable(criteria.pit),
            "cpo_failures": _to_jsonable(criteria.cpo_failures),
            "log_cpo_sum": _to_jsonable(criteria.log_cpo_sum),
            "cpo_raises_on_write": _raises_on_write(criteria.cpo),
            "pit_raises_on_write": _raises_on_write(criteria.pit),
        }

    return {
        "type": type(result).__name__,
        "isinstance": {kind.__name__: isinstance(result, kind) for kind in _RESULT_TYPES},
        "attributes": attributes,
        "immutability": immutability,
        "methods": methods,
        "repr": repr(result),
    }


def _compute_surface_matrix() -> dict:
    return {name: _surface(_build(name)) for name, _ in MATRIX}


# ---------------------------------------------------------------------------
# First-differing-key-path diffing
# ---------------------------------------------------------------------------


def _first_difference(expected: object, actual: object, path: str = "$") -> str | None:
    if type(expected) is not type(actual) and not (
        isinstance(expected, (int, float)) and isinstance(actual, (int, float))
    ):
        return f"{path}: type mismatch, baseline={type(expected).__name__} actual={type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            return f"{path}: key set mismatch, missing={sorted(missing)} extra={sorted(extra)}"
        for key in sorted(expected_keys):
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length mismatch, baseline={len(expected)} actual={len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: value mismatch, baseline={expected!r} actual={actual!r}"
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_result_surface_matches_baseline():
    baseline = json.loads(BASELINE_PATH.read_text())
    actual = _to_jsonable(_compute_surface_matrix())
    difference = _first_difference(baseline, actual)
    assert difference is None, f"result surface diverged from baseline at {difference}"


def test_result_types_are_siblings():
    """Pins the invariant that ``LGM._rebuild_result``'s isinstance dispatch depends on."""
    instances = {
        GaussianResult: _build("gaussian_iid_plugin"),
        LaplaceResult: _build("poisson_iid_laplace_plugin"),
        INLAResult: _build("gaussian_iid_integrate"),
    }
    for left_type, left_instance in instances.items():
        for right_type, right_instance in instances.items():
            if left_type is right_type:
                continue
            assert not isinstance(left_instance, right_type), (
                f"{left_type.__name__} instance must not be an instance of {right_type.__name__}"
            )
            assert not isinstance(right_instance, left_type), (
                f"{right_type.__name__} instance must not be an instance of {left_type.__name__}"
            )


def _common_kwargs(**overrides):
    kwargs = dict(
        labels=("a", "b"),
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([0.0, 0.0]),
        predictive_variance=np.array([1.0, 1.0]),
    )
    kwargs.update(overrides)
    return kwargs


def _construct(result_type, **overrides):
    if result_type is GaussianResult:
        return GaussianResult(**_common_kwargs(**overrides))
    if result_type is LaplaceResult:
        kwargs = _common_kwargs(fitted_mean=np.array([0.0, 0.0]), link_name="identity")
        kwargs.update(overrides)
        return LaplaceResult(**kwargs)
    from pylgm.inference.result import ModelCriteria

    criteria = ModelCriteria(
        dic=1.0, dic_effective_parameters=1.0, waic=1.0, waic_effective_parameters=1.0,
        cpo=np.array([0.5, 0.5]), pit=np.array([0.5, 0.5]), cpo_failures=0, log_cpo_sum=0.0,
    )
    kwargs = _common_kwargs(hyperparameter_marginals={}, criteria=criteria)
    kwargs.update(overrides)
    return INLAResult(**kwargs)


@pytest.mark.parametrize("result_type", [GaussianResult, LaplaceResult, INLAResult])
def test_constructor_error_messages_are_stable(result_type):
    # Wrong ndim: a diagnostics value that is not an immutable scalar (an array).
    with pytest.raises(TypeError) as wrong_shape:
        _construct(result_type, diagnostics={"payload": np.array([1, 2])})
    assert str(wrong_shape.value) == "diagnostics values must be immutable scalar values"

    # Non-finite: a covariance entry that is NaN.
    with pytest.raises(ValueError) as non_finite:
        _construct(result_type, covariance=np.array([[np.nan, 0.0], [0.0, 1.0]]))
    assert str(non_finite.value) == "covariance must be finite"

    # Mismatched lengths: prediction_keys row count does not match predictive_mean.
    with pytest.raises(ValueError) as mismatched:
        _construct(result_type, prediction_keys=pd.DataFrame({"row": [1]}))
    assert str(mismatched.value) == "prediction keys row count must match predictive results"


def test_inla_constructor_error_messages_are_stable():
    with pytest.raises(TypeError) as bad_criteria:
        _construct(INLAResult, criteria=object())
    assert str(bad_criteria.value) == "criteria must be a ModelCriteria"

    with pytest.raises(TypeError) as bad_table:
        _construct(INLAResult, latent_marginal_table=object())
    assert str(bad_table.value) == (
        "latent_marginal_table must be a SkewNormalMarginals or TabulatedMarginals"
    )


if __name__ == "__main__":
    # One-off baseline generation: run on trusted, unmodified code only.
    matrix = _to_jsonable(_compute_surface_matrix())
    BASELINE_PATH.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(f"wrote {BASELINE_PATH} ({BASELINE_PATH.stat().st_size} bytes)")

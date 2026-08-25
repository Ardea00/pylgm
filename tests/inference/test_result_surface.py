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

import dataclasses
import json
import sys
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Besag, Bernoulli, Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson, RW1
from pylgm.inference.result import (
    GaussianResult,
    INLAResult,
    LaplaceResult,
    ModelCriteria,
    _BaseResult,
)
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


def _poisson_iid_laplace_integrate_simplified_laplace():
    # Exercises the SkewNormalMarginals branch of INLAResult.latent_marginal_table
    # (latent_strategy="simplified_laplace"): every other integrate fit in this
    # matrix leaves the table None, so that branch is otherwise dead code here.
    frame = _iid_frame(5, "poisson")
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("grp", index="grp", precision=_precision_hyperparameter("grp.precision")),
    )
    return model, frame, dict(
        engine="laplace", hyperparameters="integrate", latent_strategy="simplified_laplace"
    )


def _bernoulli_ar1_laplace_integrate_full_laplace():
    # Exercises the TabulatedMarginals branch (latent_strategy="laplace"). Full
    # Laplace rejects constrained (RW/Besag) effects, so this uses AR1, which is
    # unconstrained.
    frame = _ar1_frame(9, "bernoulli", rho=0.5)
    model = LGM(
        response="y", likelihood=Bernoulli(),
        predictor=Fixed("1") + AR1(
            "trend", index="t", precision=_precision_hyperparameter("trend.precision"), rho=0.5
        ),
    )
    return model, frame, dict(
        engine="laplace", hyperparameters="integrate", latent_strategy="laplace"
    )


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
    (
        "poisson_iid_laplace_integrate_simplified_laplace",
        _poisson_iid_laplace_integrate_simplified_laplace,
    ),
    (
        "bernoulli_ar1_laplace_integrate_full_laplace",
        _bernoulli_ar1_laplace_integrate_full_laplace,
    ),
]


def _build(name: str) -> tuple[object, pd.DataFrame]:
    """Return ``(result, frame)`` -- the fitted result and the frame it was fit on.

    The frame is needed alongside the result so ``_surface`` can exercise
    ``result.predict()`` against the same rows the model saw at fit time.
    """
    for candidate, builder in MATRIX:
        if candidate == name:
            model, frame, fit_kwargs = builder()
            return model.fit(frame, **fit_kwargs), frame
    raise KeyError(name)  # pragma: no cover - defensive


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_RESULT_TYPES = (GaussianResult, LaplaceResult, INLAResult)

# Baseline comparison tolerances (see ``_first_difference``).
_TIGHT_REL_TOL = 1e-5  # integrated hyperparameter posteriors drift ~2e-6 across platforms
_INTRINSIC_REL_TOL = 0.15  # RW1/Besag ill-determined tails; observed worst ~8.4%
_INTRINSIC_MARKERS = ("rw1", "besag")


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


def _prediction_context_summary(context: object | None) -> dict | None:
    """A stable summary of a ``PredictionContext`` -- never the object itself.

    Records only the number of design-block entries and their names, which is
    enough to catch a dropped/rebuilt ``prediction_context`` without pinning
    internals (``ModelSpec``, compiled likelihood objects) that aren't part of
    the public result surface.
    """
    if context is None:
        return None
    block_names = [
        payload[0] if kind == "structured" else kind for kind, payload in context.entries
    ]
    return {"entry_count": len(context.entries), "block_names": block_names}


def _marginals_surface(marginals) -> dict:
    return {
        "mean": _to_jsonable(marginals.mean),
        "variance": _to_jsonable(marginals.variance),
        "std": _to_jsonable(marginals.std),
        "mean_raises_on_write": _raises_on_write(marginals.mean),
        "variance_raises_on_write": _raises_on_write(marginals.variance),
        "std_raises_on_write": _raises_on_write(marginals.std),
    }


def _surface(result, frame: pd.DataFrame) -> dict:
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
        "prediction_context": _prediction_context_summary(result.prediction_context),
    }
    if hasattr(result, "fitted_mean"):
        attributes["fitted_mean"] = (
            _to_jsonable(result.fitted_mean) if result.fitted_mean is not None else None
        )
    if hasattr(result, "link_name"):
        attributes["link_name"] = result.link_name
    if hasattr(result, "observation_variance"):
        attributes["observation_variance"] = _to_jsonable(result.observation_variance)

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

    latent_by_block = {
        name: _marginals_surface(result.latent_marginals(block=name))
        for name in sorted(result.block_slices)
    }
    try:
        result.latent_marginals(block="__unknown_block__")
    except Exception as error:  # noqa: BLE001 - the error identity is exactly what's pinned
        unknown_block_error = {"type": type(error).__name__, "message": str(error)}
    else:  # pragma: no cover - would mean an unknown block silently resolves
        unknown_block_error = None

    prediction = result.predict(frame)

    methods: dict[str, object] = {
        "latent_marginals": _marginals_surface(latent),
        "latent_marginals_by_block": latent_by_block,
        "latent_marginals_unknown_block_error": unknown_block_error,
        "hyperparameter_marginals": hyperparameter_marginals,
        "linear_combinations": _marginals_surface(combined),
        "predict": {
            "predictive_mean": _to_jsonable(prediction.predictive_mean),
            "predictive_variance": _to_jsonable(prediction.predictive_variance),
            "fitted_mean": _to_jsonable(prediction.fitted_mean),
            "to_frame_columns": list(prediction.to_frame().columns),
        },
    }
    if hasattr(result, "latent_marginal_table"):
        table = result.latent_marginal_table
        if table is None:
            methods["latent_marginal_table"] = None
        else:
            table_surface: dict[str, object] = {
                "type": type(table).__name__,
                "mean": _to_jsonable(table.mean),
                "variance": _to_jsonable(table.variance),
                "std": _to_jsonable(table.std),
            }
            if hasattr(table, "skewness"):
                table_surface["skewness"] = _to_jsonable(table.skewness)
            methods["latent_marginal_table"] = table_surface
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
    return {name: _surface(*_build(name)) for name, _ in MATRIX}


# ---------------------------------------------------------------------------
# First-differing-key-path diffing
# ---------------------------------------------------------------------------


def _first_difference(expected: object, actual: object, path: str = "$") -> str | None:
    # Exact type match -- no int/float/bool blurring. ``_to_jsonable`` already
    # tags every numeric leaf with its real Python type (numpy ints -> int,
    # numpy floats -> float, numpy bools -> bool), so a float that quietly
    # became an int (a dtype regression) is a real divergence, not noise.
    if type(expected) is not type(actual):
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
    if isinstance(expected, float):
        # A 12-sig-fig exact baseline only reproduces on its author's machine;
        # cross-platform BLAS/scipy differences perturb the low-order bits. So:
        #  * abs_tol=1e-12 absorbs "numerical zero" dust (a mathematically-zero
        #    covariance entry reading 0.0 on one platform, ~1e-19 on another);
        #  * rel_tol handles genuine values. Intrinsic GMRFs (RW1, Besag) have
        #    singular precision, so their flat hyperparameter posteriors and
        #    small covariance-tail entries are ill-determined and drift a few
        #    percent (observed worst ~8.4%); everything else keeps the tight guard.
        rel_tol = _INTRINSIC_REL_TOL if any(m in path for m in _INTRINSIC_MARKERS) else _TIGHT_REL_TOL
        if not math.isclose(expected, actual, rel_tol=rel_tol, abs_tol=1e-12):
            return f"{path}: value mismatch, baseline={expected!r} actual={actual!r}"
        return None
    if path.endswith(".repr"):
        # ``repr`` bundles volatile optimizer telemetry -- empirical-Bayes/Newton
        # evaluation counts and near-zero convergence gradient norms -- that
        # legitimately varies across scipy/BLAS versions and admits no tolerance
        # (10 vs 12 iterations). Every correctness quantity in the repr (labels,
        # log-marginal, type, link_name) is already a structured leaf compared
        # above, and block structure is guarded via latent_marginals_by_block, so
        # the raw repr string is not compared across platforms.
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

    criteria = ModelCriteria(
        dic=1.0, dic_effective_parameters=1.0, waic=1.0, waic_effective_parameters=1.0,
        cpo=np.array([0.5, 0.5]), pit=np.array([0.5, 0.5]), cpo_failures=0, log_cpo_sum=0.0,
    )
    kwargs = _common_kwargs(hyperparameter_marginals={}, criteria=criteria)
    kwargs.update(overrides)
    return INLAResult(**kwargs)


def test_result_types_are_siblings():
    """Pins the invariant that ``LGM._rebuild_result``'s isinstance dispatch depends on.

    Built by direct, minimal-argument construction (``_construct``) rather than
    through ``LGM.fit`` / the matrix above: if a reviewer makes ``INLAResult``
    inherit ``LaplaceResult`` (or any such cross-inheritance), that alone
    should fail *this* test's assertions. Going through the fit matrix instead
    would let a dispatch or fixture crash elsewhere abort the test before it
    ever reaches them, silently shadowing the real failure.
    """
    instances = {
        GaussianResult: _construct(GaussianResult),
        LaplaceResult: _construct(LaplaceResult),
        INLAResult: _construct(INLAResult),
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


def test_backing_fields_pairs_every_private_field_with_its_public_property():
    """``_BACKING_FIELDS`` is hand-maintained -- nothing enforces that a newly
    added shared field is listed under BOTH its private (dataclass) name and,
    when a property exposes it, the public property name too. A field added
    under only one of the two names would silently escape the ``__getattr__``
    guard (see its docstring: an ``AttributeError`` inside a property getter
    re-enters ``__getattr__`` under the OUTER/public name). Derive the expected
    set from ``_BaseResult`` itself so the hand-list and the class cannot drift.
    """
    expected: set[str] = set()
    for declared_field in dataclasses.fields(_BaseResult):
        expected.add(declared_field.name)
        if declared_field.name.startswith("_"):
            public_name = declared_field.name[1:]
            if isinstance(getattr(_BaseResult, public_name, None), property):
                expected.add(public_name)
    assert _BaseResult._BACKING_FIELDS == frozenset(expected)


@pytest.mark.parametrize("result_type", [GaussianResult, LaplaceResult, INLAResult])
def test_constructor_error_messages_are_stable(result_type):
    # Diagnostics-value check: not a shape/ndim check at all -- a diagnostics
    # value must be an immutable scalar, and an ndarray (of any shape) fails
    # this the same way. It's exercised here as the one "malformed payload"
    # precondition shared verbatim by all three constructors today.
    with pytest.raises(TypeError) as bad_diagnostics:
        _construct(result_type, diagnostics={"payload": np.array([1, 2])})
    assert str(bad_diagnostics.value) == "diagnostics values must be immutable scalar values"

    # Non-finite: a covariance entry that is NaN.
    with pytest.raises(ValueError) as non_finite:
        _construct(result_type, covariance=np.array([[np.nan, 0.0], [0.0, 1.0]]))
    assert str(non_finite.value) == "covariance must be finite"

    # Mismatched lengths: prediction_keys row count does not match predictive_mean.
    with pytest.raises(ValueError) as mismatched:
        _construct(result_type, prediction_keys=pd.DataFrame({"row": [1]}))
    assert str(mismatched.value) == "prediction keys row count must match predictive results"


@pytest.mark.parametrize("result_type", [GaussianResult, LaplaceResult, INLAResult])
def test_constructor_validation_order_is_stable(result_type):
    """Pins which precondition wins when two are violated at once.

    Today, every constructor validates ``prediction_keys`` *before* checking
    covariance finiteness, so a non-finite covariance combined with a
    mismatched ``prediction_keys`` row count surfaces the prediction-keys
    message, not the finiteness one. A shared ``_init_common`` extracted by
    the refactor could easily reorder these checks -- e.g. validate the
    array-shaped arguments first -- and silently change which error a caller
    sees for a doubly-malformed construction. This test only holds if that
    order is preserved; it is not a claim about which order is "correct".
    """
    with pytest.raises(ValueError) as excinfo:
        _construct(
            result_type,
            covariance=np.array([[np.nan, 0.0], [0.0, 1.0]]),
            prediction_keys=pd.DataFrame({"row": [1]}),
        )
    assert str(excinfo.value) == "prediction keys row count must match predictive results"


@pytest.mark.parametrize("result_type", [GaussianResult, LaplaceResult, INLAResult])
def test_malformed_shapes_are_currently_accepted_without_validation(result_type):
    """Pins today's permissive (and arguably wrong) construction-time behaviour.

    None of the three constructors validates that ``mean`` is one-dimensional,
    nor that ``covariance``'s size actually matches ``mean``'s. A 2-D ``mean``
    paired with a ``covariance`` of an unrelated size constructs without
    complaint; the mismatch would only surface later, inside
    ``latent_marginals`` or ``linear_combinations``. If the refactor's shared
    ``_init_common`` "helpfully" adds shape validation, that is a real,
    user-visible behaviour change (a construction that used to succeed would
    now raise) and must be called out explicitly rather than slipping in
    unannounced -- which is exactly what a failure here would flag.
    """
    result = _construct(
        result_type,
        mean=np.zeros((2, 2)),
        covariance=np.eye(5),
    )
    assert result.mean.shape == (2, 2)
    assert result.covariance.shape == (5, 5)


def test_inla_constructor_error_messages_are_stable():
    with pytest.raises(TypeError) as bad_criteria:
        _construct(INLAResult, criteria=object())
    assert str(bad_criteria.value) == "criteria must be a ModelCriteria"

    with pytest.raises(TypeError) as bad_table:
        _construct(INLAResult, latent_marginal_table=object())
    assert str(bad_table.value) == (
        "latent_marginal_table must be a SkewNormalMarginals or TabulatedMarginals"
    )


if __name__ == "__main__":  # pragma: no cover
    # Gated: this overwrites a golden file whose whole value is being hard to
    # change by accident. Run with --regenerate to mean it.
    if "--regenerate" not in sys.argv:
        raise SystemExit(
            "refusing to overwrite the baseline; re-run with --regenerate"
        )
    # One-off baseline generation: run on trusted, unmodified code only.
    matrix = _to_jsonable(_compute_surface_matrix())
    BASELINE_PATH.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(f"wrote {BASELINE_PATH} ({BASELINE_PATH.stat().st_size} bytes)")


def test_validation_order_across_the_shared_per_type_boundary_is_stable():
    """Pin which error wins when a SHARED and a PER-TYPE precondition both fail.

    ``test_constructor_validation_order_is_stable`` only pins pairs whose checks
    both live in the shared prologue, so their relative order survives any
    refactor trivially. The boundary a shared ``_init_common`` actually moves is
    the one between shared and per-type work -- and the shared *storage* is not
    inert (``_readonly_diagnostics`` and ``_readonly_hyperparameters`` validate),
    so hoisting it ahead of the subclass's own field handling silently changes
    which exception a doubly-malformed construction raises. These cases pin that
    boundary; they are the ones that caught it.
    """
    base = dict(
        labels=("a",),
        mean=np.zeros(1),
        covariance=np.zeros((1, 1)),
        log_marginal_likelihood=0.0,
        predictive_mean=np.zeros(1),
        predictive_variance=np.zeros(1),
    )
    criteria = ModelCriteria(
        dic=0.0,
        dic_effective_parameters=0.0,
        waic=0.0,
        waic_effective_parameters=0.0,
        cpo=np.ones(1),
        pit=np.zeros(1),
        cpo_failures=0,
        log_cpo_sum=0.0,
    )
    ragged = [[1.0], [1.0, 2.0]]

    cases = [
        # per-type field bad + shared diagnostics bad -> the PER-TYPE error wins
        (
            "laplace fitted_mean vs diagnostics",
            lambda: LaplaceResult(
                **base, fitted_mean=ragged, link_name="log", diagnostics={"k": [1, 2]}
            ),
            ValueError,
            "setting an array element",
        ),
        # per-type field bad + shared hyperparameters bad -> PER-TYPE wins
        (
            "laplace fitted_mean vs hyperparameters",
            lambda: LaplaceResult(
                **base, fitted_mean=ragged, link_name="log",
                hyperparameters={"s": float("inf")},
            ),
            ValueError,
            "setting an array element",
        ),
        (
            "inla hyperparameter_marginals vs hyperparameters",
            lambda: INLAResult(
                **base, criteria=criteria, hyperparameter_marginals="nope",
                hyperparameters={"s": float("inf")},
            ),
            TypeError,
            "hyperparameter_marginals must be a mapping",
        ),
        (
            "inla hyperparameter_marginals vs diagnostics",
            lambda: INLAResult(
                **base, criteria=criteria, hyperparameter_marginals={"a": "x"},
                diagnostics={"k": [1, 2]},
            ),
            TypeError,
            "hyperparameter_marginals values must be",
        ),
    ]
    for name, build, error_type, fragment in cases:
        with pytest.raises(error_type) as excinfo:
            build()
        assert fragment in str(excinfo.value), name

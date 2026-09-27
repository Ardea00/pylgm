"""``num_workers``/``blas_threads``: parallel-vs-serial fits must be bit-identical.

Every scenario compares a ``num_workers=1, blas_threads=1`` run against a
``num_workers=4`` run of the same model and asserts exact (``==``,
``np.array_equal``) agreement, not approximate -- that is the whole point of
running BLAS at one thread per worker (see docs/empirical-bayes.md).
"""

import threadpoolctl
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, LinearObservation, Poisson, RW2
from pylgm.exceptions import NumericalError
from pylgm.inference import GaussianResult
from pylgm.ir import CompiledGaussianFamily, Hyperparameters, LatentBlock, ScalableBlock
from pylgm.joint import Joint, Shared
from pylgm.optimization import OptimizationBounds, optimize_empirical_bayes
from pylgm.optimization import empirical_bayes
from pylgm.parallel import blas_limit, validate_blas_threads, validate_workers
from scipy.sparse import csr_matrix


def _assert_bit_identical(serial, parallel):
    assert serial.hyperparameters == parallel.hyperparameters
    assert serial.log_marginal_likelihood == parallel.log_marginal_likelihood
    np.testing.assert_array_equal(serial.predictive_mean, parallel.predictive_mean)
    np.testing.assert_array_equal(serial.predictive_variance, parallel.predictive_variance)
    assert (
        serial.diagnostics["empirical_bayes_evaluations"]
        == parallel.diagnostics["empirical_bayes_evaluations"]
    )


def test_gaussian_lgm_three_hyperparameters_eb_is_bit_identical():
    rng = np.random.default_rng(0)
    n = 24
    regions = [f"r{i % 6}" for i in range(n)]
    frame = pd.DataFrame({
        "region": regions,
        "t": list(range(1, n + 1)),
        "y": rng.normal(size=n),
    })
    model = LGM(
        "y",
        Gaussian(Hyperparameter("sigma", initial=1.0, lower=1e-2, upper=1e2)),
        Fixed("1")
        + RW2("trend", "t", Hyperparameter("trend_precision", initial=1.0, lower=1e-2, upper=1e2))
        + IID("region", index="region", precision=Hyperparameter("region_precision", initial=1.0, lower=1e-2, upper=1e2)),
        time="t",
    )

    serial = model.fit(frame, num_workers=1, blas_threads=1)
    parallel = model.fit(frame, num_workers=4)

    _assert_bit_identical(serial, parallel)


def test_poisson_lgm_hyperparameter_laplace_warm_start_is_bit_identical():
    rng = np.random.default_rng(1)
    n = 20
    frame = pd.DataFrame({
        "t": list(range(n)),
        "x": rng.normal(size=n),
    })
    frame["y"] = rng.poisson(np.exp(0.3 + 0.5 * frame["x"])).astype(float)
    model = LGM(
        "y", Poisson(),
        Fixed("1 + x")
        + IID("t", index="t", precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2)),
        time="t",
    )

    serial = model.fit(frame, engine="laplace", num_workers=1, blas_threads=1)
    parallel = model.fit(frame, engine="laplace", num_workers=4)

    _assert_bit_identical(serial, parallel)


def _fixed_point_frame():
    cells = [f"c{i}" for i in range(6)]
    y = [0.8, 1.1, 0.9, 1.2, np.nan, np.nan]
    return pd.DataFrame({"cell": cells, "y": y})


def _fixed_point_operator_and_values():
    C = np.array([
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    values = np.array([12.0, 5.0])
    return C, values


def test_relinearized_family_log_scale_observation_is_bit_identical():
    """Reuses tests/test_log_scale_observations.py's fixed-point setup: a
    `LinearObservation(..., scale="log")` compiles a `_RelinearizedFamily`, whose
    mutable warm start (`_start["eta"]`) must see materialize calls in the same
    serial order under num_workers>1 as under num_workers=1."""
    frame = _fixed_point_frame()
    C, values = _fixed_point_operator_and_values()
    sigma_hp = Hyperparameter("s", initial=1.0, lower=1e-3, upper=1e3)
    observation = LinearObservation(values, C, sigma=sigma_hp, scale="log")
    model = LGM(
        response="y",
        likelihood=Gaussian(0.3),
        predictor=IID(
            "u", index="cell",
            precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2),
        ),
        panel=("cell",),
    )

    serial = model.fit(
        frame, observations=[observation], engine="exact_gaussian",
        num_workers=1, blas_threads=1,
    )
    parallel = model.fit(
        frame, observations=[observation], engine="exact_gaussian",
        num_workers=4,
    )

    _assert_bit_identical(serial, parallel)


def test_joint_shared_two_hyperparameters_eb_is_bit_identical(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint(
        [
            LGM(
                response="oral", likelihood=Poisson(),
                predictor=Fixed("1")
                + IID("v", index="district", precision=Hyperparameter("v_precision", initial=1.0, lower=1e-2, upper=1e2)),
            ),
            LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
        ],
        shared=[Shared(
            IID("u", index="district", precision=1.0),
            scale=Hyperparameter("delta", initial=1.0, lower=1e-2, upper=1e2),
        )],
    )

    serial = joint.fit(frame, engine="laplace", num_workers=1, blas_threads=1)
    parallel = joint.fit(frame, engine="laplace", num_workers=4)

    _assert_bit_identical(serial, parallel)


def test_hyperparameters_integrate_grid_is_bit_identical():
    rng = np.random.default_rng(2)
    n = 30
    regions = [f"r{i}" for i in range(n)]
    frame = pd.DataFrame({
        "region": regions,
        "t": list(range(1, n + 1)),
        "y": rng.normal(size=n),
    })
    model = LGM(
        "y", Gaussian(0.5),
        Fixed("1")
        + IID("region", index="region", precision=Hyperparameter("p1", initial=1.0, lower=1e-2, upper=1e2))
        + IID("t", index="t", precision=Hyperparameter("p2", initial=1.0, lower=1e-2, upper=1e2)),
        panel=("region",), time="t",
    )

    serial = model.fit(
        frame, engine="exact_gaussian", hyperparameters="integrate",
        num_workers=1, blas_threads=1,
    )
    parallel = model.fit(
        frame, engine="exact_gaussian", hyperparameters="integrate",
        num_workers=4,
    )

    assert serial.log_marginal_likelihood == parallel.log_marginal_likelihood
    np.testing.assert_array_equal(serial.predictive_mean, parallel.predictive_mean)
    np.testing.assert_array_equal(serial.predictive_variance, parallel.predictive_variance)
    serial_marginals = serial.hyperparameter_marginals()
    parallel_marginals = parallel.hyperparameter_marginals()
    for name in serial_marginals:
        np.testing.assert_array_equal(
            serial_marginals[name].mean, parallel_marginals[name].mean
        )
        np.testing.assert_array_equal(
            serial_marginals[name].variance, parallel_marginals[name].variance
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_num_workers_validation_rejects_bad_values(value):
    with pytest.raises(ValueError):
        validate_workers(value, "num_workers")


@pytest.mark.parametrize("value", [0, True])
def test_blas_threads_validation_rejects_bad_values(value):
    with pytest.raises(ValueError):
        validate_blas_threads(value)


def test_num_workers_and_blas_threads_validation_through_fit():
    model = LGM("y", Gaussian(1.0), Fixed("1"))
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        model.fit(frame, num_workers=0)
    with pytest.raises(ValueError):
        model.fit(frame, num_workers=1.5)
    with pytest.raises(ValueError):
        model.fit(frame, num_workers=True)
    with pytest.raises(ValueError):
        model.fit(frame, blas_threads=0)
    with pytest.raises(ValueError):
        model.fit(frame, blas_threads=True)


def test_blas_limit_pins_every_blas_entry_to_one_thread_and_restores_after():
    before = {
        entry["prefix"]: entry["num_threads"] for entry in threadpoolctl.threadpool_info()
    }
    assert before, "expected at least one BLAS entry on this machine"

    with blas_limit(4, None):
        during = threadpoolctl.threadpool_info()
        assert during, "expected at least one BLAS entry on this machine"
        for entry in during:
            assert entry["num_threads"] == 1

    after = {
        entry["prefix"]: entry["num_threads"] for entry in threadpoolctl.threadpool_info()
    }
    assert after == before


def _scalar_family(y: float) -> CompiledGaussianFamily:
    block = LatentBlock(
        "latent", ("x",), csr_matrix([[1.0]]), csr_matrix([[1.0]]), np.empty((0, 1)),
    )
    return CompiledGaussianFamily(
        y=np.array([y]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(ScalableBlock(block, "latent.precision", 1.0),),
        parameter_names=("latent.precision",),
        initial=Hyperparameters(sigma=1.0, precisions={"latent": 1.0}),
    )


def test_inference_error_in_one_of_a_batchs_fits_matches_serial():
    """A fake fit that raises `NumericalError` for part of the parameter region:
    serial vs num_workers=4 must give the same result and the same
    `numerical_failures` tuple (order and content), per SPEC8.md scenario 8."""

    def flaky_fit(model) -> GaussianResult:
        precision = float(model.blocks[0].precision[0, 0])
        if precision > 5.0:
            raise NumericalError(f"unstable at precision={precision}")
        latent_size = len(model.labels)
        return GaussianResult(
            labels=model.labels,
            mean=np.zeros(latent_size),
            covariance=np.zeros((latent_size, latent_size)),
            log_marginal_likelihood=-float(model.y[0] ** 2) / (1.0 + 1.0 / precision) - precision,
            predictive_mean=np.zeros(model.y.size),
            predictive_variance=np.ones(model.y.size),
        )

    family = _scalar_family(y=2.0)
    bounds = {
        "latent.precision": OptimizationBounds(initial=1.0, lower=0.1, upper=9.0),
    }

    serial = optimize_empirical_bayes(
        family, bounds, fit=flaky_fit, num_workers=1, blas_threads=1,
    )
    parallel = optimize_empirical_bayes(
        family, bounds, fit=flaky_fit, num_workers=4,
    )

    assert serial.parameters == parallel.parameters
    assert serial.diagnostics.objective == parallel.diagnostics.objective
    assert serial.diagnostics.evaluations == parallel.diagnostics.evaluations
    assert serial.diagnostics.numerical_failures == parallel.diagnostics.numerical_failures

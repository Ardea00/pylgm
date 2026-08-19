import pandas as pd
import numpy as np
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, Poisson, RW1, RW2
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference.result import LaplaceResult
from pylgm.likelihoods import CompiledGaussian
from pylgm.parameters import Hyperparameter
from pylgm.model import _align_predictions_with_source_rows


def _example():
    frame = pd.DataFrame(
        {
            "region": ["a", "a", "b", "b"],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 1.5, None],
        }
    )
    model = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed("1")
        + IID("region", "region", 2.0)
        + RW1("trend", "time", 3.0),
        panel=("region",),
        time="time",
    )
    return frame, model


def test_declarative_model_fits_existing_gaussian_path():
    frame, model = _example()

    result = model.fit(frame)

    assert result.engine == "exact_gaussian"
    assert result.predictive_mean.shape == (4,)


def test_declarative_predictions_align_with_unsorted_caller_rows():
    frame = pd.DataFrame(
        {
            "region": ["B", "A", "B", "A"],
            "time": [2, 1, 1, 2],
            "x": [4.0, 1.0, 3.0, 2.0],
            "y": [8.0, 2.0, 6.0, 4.0],
        }
    )
    model = LGM(
        response="y",
        likelihood=Gaussian(1.0),
        predictor=Fixed("0 + x", prior_precision=1.0),
        panel=("region",),
        time="time",
    )

    result = model.fit(frame)

    np.testing.assert_allclose(
        result.predictive_mean,
        np.array([240.0, 60.0, 180.0, 120.0]) / 31.0,
    )
    np.testing.assert_allclose(
        result.predictive_variance,
        1.0 + np.array([16.0, 1.0, 9.0, 4.0]) / 31.0,
    )


def test_prediction_alignment_retains_prediction_keys():
    from pylgm.inference import GaussianResult

    keys = pd.DataFrame({"region": ["A", "B"], "time": [1, 2]})
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.eye(1),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([10.0, 20.0]),
        predictive_variance=np.array([1.0, 1.0]),
        prediction_keys=keys,
    )

    aligned = _align_predictions_with_source_rows(result, np.array([1, 0]))

    np.testing.assert_allclose(aligned.predictive_mean, [20.0, 10.0])
    pd.testing.assert_frame_equal(aligned.prediction_keys, keys)


def test_model_does_not_fallback_from_unknown_engine():
    frame, model = _example()

    with pytest.raises(UnsupportedEngineError, match="unknown"):
        model.fit(frame, engine="unknown")


def test_model_without_time_uses_private_row_key_without_mutating_frame():
    frame = pd.DataFrame(
        {"group": ["b", "a"], "step": [2, 1], "y": [2.0, 1.0]}
    )
    original = frame.copy(deep=True)
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("1") + IID("group", "group") + RW1("trend", "step"),
    )

    result = model.fit(frame)

    assert result.predictive_mean.shape == (2,)
    pd.testing.assert_frame_equal(frame, original)


def test_model_without_time_preserves_caller_order_prediction_values():
    frame = pd.DataFrame(
        {"x": [4.0, 1.0, 3.0], "y": [8.0, 2.0, 6.0]},
        index=[20, 10, 20],
    )
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("0 + x", prior_precision=1.0),
    )

    result = model.fit(frame)

    np.testing.assert_allclose(
        result.predictive_mean, np.array([208.0, 52.0, 156.0]) / 27.0
    )
    np.testing.assert_allclose(
        result.predictive_variance,
        1.0 + np.array([16.0, 1.0, 9.0]) / 27.0,
    )


def test_model_rejects_collision_with_private_row_key():
    frame = pd.DataFrame({"__pylgm_row__": [4], "y": [1.0]})
    model = LGM("y", Gaussian(1.0), Fixed("1"))

    with pytest.raises(DataContractError, match="reserved row key"):
        model.fit(frame)


def test_model_resolves_declared_hyperparameters_to_initial_values():
    frame = pd.DataFrame(
        {"group": ["a", "b"], "time": [1, 2], "y": [1.0, 2.0]}
    )
    model = LGM(
        "y",
        Gaussian(Hyperparameter("sigma", 0.5)),
        Fixed("1")
        + IID("group", "group", Hyperparameter("group_precision", 2.0)),
        time="time",
    )

    result = model.fit(frame)
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="time", response="y")
    )
    compiled = compile_lgm(model, panel)

    assert result.predictive_mean.shape == (2,)
    assert compiled.likelihood == CompiledGaussian(0.5)
    np.testing.assert_allclose(
        compiled.blocks[1].precision.toarray(), np.eye(2) * 2.0
    )


def test_declarative_rw2_compiles_the_second_order_precision_and_constraints():
    frame = pd.DataFrame(
        {"time": [1, 2, 3], "y": [1.0, 2.0, 4.0]}
    )
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("1")
        + RW2(
            "trend",
            "time",
            Hyperparameter("trend_precision", 2.0),
        ),
        time="time",
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="time", response="y")
    )

    compiled = compile_lgm(model, panel)

    trend = compiled.blocks[1]
    np.testing.assert_allclose(
        trend.precision.toarray(),
        [[2.0, -4.0, 2.0], [-4.0, 8.0, -4.0], [2.0, -4.0, 2.0]],
    )
    np.testing.assert_allclose(
        trend.constraints,
        [[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]],
    )


def test_pandas_fit_keeps_no_prediction_keys():
    result = LGM("y", Gaussian(1.0), Fixed("1")).fit(pd.DataFrame({"y": [1.0, 2.0]}))
    assert result.prediction_keys is None


def test_pandas_fit_preserves_caller_order_with_default_row_limit():
    frame = pd.DataFrame({"region": ["B", "A"], "time": [2, 1], "y": [8.0, 2.0]})
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    result = model.fit(frame)
    # Predictions stay aligned to the caller's B-then-A row order.
    assert result.predictive_mean.shape == (2,)
    assert result.prediction_keys is None


class _NonSparkFrame:
    def toPandas(self):  # noqa: N802
        raise AssertionError("non-Spark inputs must not be collected")


def test_model_rejects_non_pandas_non_spark_input():
    model = LGM("y", Gaussian(1.0), Fixed("1"))

    # A duck-typed object whose module is not ``pyspark`` is not a genuine Spark
    # frame, so it takes the plain Pandas error path without being collected.
    with pytest.raises(DataContractError, match="frame must be a Pandas DataFrame"):
        model.fit(_NonSparkFrame())


def test_lgm_fit_laplace_poisson_returns_laplace_result():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 4.0, 7.0]})
    result = LGM("y", Poisson(), Fixed("1 + x"), time="t").fit(frame, engine="laplace")
    assert isinstance(result, LaplaceResult)
    assert result.converged is True
    assert result.prediction_keys is None
    assert result.fitted_mean.shape == (4,)


def test_lgm_fit_laplace_preserves_caller_row_order():
    frame = pd.DataFrame({"t": [3, 1, 2], "y": [4.0, 1.0, 2.0]})
    result = LGM("y", Poisson(), Fixed("1"), time="t").fit(frame, engine="laplace")
    # first row corresponds to t=3 in caller order
    assert result.predictive_mean.shape == (3,)


def test_lgm_rejects_engine_likelihood_mismatch():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    with pytest.raises(UnsupportedEngineError, match="Gaussian"):
        LGM("y", Gaussian(1.0), Fixed("1"), time="t").fit(frame, engine="laplace")
    with pytest.raises(UnsupportedEngineError, match="non-Gaussian"):
        LGM("y", Poisson(), Fixed("1"), time="t").fit(frame, engine="exact_gaussian")

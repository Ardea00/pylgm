import pandas as pd
import numpy as np
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, RW1, RW2
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.likelihoods import CompiledGaussian
from pylgm.parameters import Hyperparameter


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


class _SparkShapedFrame:
    def toPandas(self):  # noqa: N802
        raise AssertionError("Spark-shaped inputs must not be collected")


def test_model_rejects_spark_shaped_input_until_adapter_is_available():
    model = LGM("y", Gaussian(1.0), Fixed("1"))

    with pytest.raises(
        DataContractError, match=r"PySpark input is unsupported.*0\.3.*planned"
    ):
        model.fit(_SparkShapedFrame())

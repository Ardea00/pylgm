import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, RW1
from pylgm.exceptions import DataContractError, UnsupportedEngineError
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

    assert result.predictive_mean.shape == (2,)


class _SparkShapedFrame:
    def toPandas(self):  # noqa: N802
        raise AssertionError("Spark-shaped inputs must not be collected")


def test_model_rejects_spark_shaped_input_until_adapter_is_available():
    model = LGM("y", Gaussian(1.0), Fixed("1"))

    with pytest.raises(
        DataContractError, match="PySpark support requires the spark extra"
    ):
        model.fit(_SparkShapedFrame())

"""``Joint.fit`` on a Spark DataFrame (skips without PySpark/JVM, like test_model_spark)."""
import numpy as np
import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.joint import Joint, Shared


def _python() -> Joint:
    return Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=2.0), scale=(1.6, 0.625))],
    )


def test_joint_fit_on_spark_equals_pandas(spark, shared_component_frame):
    frame, _ = shared_component_frame()
    joint = _python()
    # An unused column must be projected away, not collected.
    sdf = spark.createDataFrame(frame.assign(unused="x"))
    a, b = joint.fit(sdf), joint.fit(frame)
    assert np.isclose(a.log_marginal_likelihood, b.log_marginal_likelihood)
    assert a.labels == b.labels


def test_joint_fit_spark_guards(spark, shared_component_frame):
    frame, _ = shared_component_frame()
    sdf = spark.createDataFrame(frame)
    joint = _python()
    with pytest.raises(DataContractError, match="max_driver_rows"):
        joint.fit(sdf, max_driver_rows=3)
    with pytest.raises(DataContractError, match="missing columns"):
        joint.fit(sdf.drop("district"))
    with pytest.raises(UnsupportedEngineError, match="pandas input"):
        joint.fit(sdf, hold_out={"oral": np.zeros(len(frame), dtype=bool)})

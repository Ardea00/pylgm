import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pylgm import Fixed, Gaussian, LGM
from pylgm.exceptions import DataContractError


def test_spark_fit_matches_sorted_pandas_and_exposes_join_keys(spark):
    rows = [("B", 2, 4.0, 8.0), ("A", 1, 1.0, 2.0), ("B", 1, 3.0, 6.0), ("A", 2, 2.0, 4.0)]
    columns = ["region", "time", "x", "y"]
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("0 + x", prior_precision=1.0),
        panel=("region",),
        time="time",
    )

    spark_result = model.fit(spark.createDataFrame(rows, columns))
    pandas_result = model.fit(
        pd.DataFrame(rows, columns=columns).sort_values(["region", "time"])
    )

    np.testing.assert_allclose(
        spark_result.predictive_mean, pandas_result.predictive_mean
    )
    np.testing.assert_allclose(
        spark_result.predictive_variance, pandas_result.predictive_variance
    )
    np.testing.assert_allclose(
        spark_result.log_marginal_likelihood,
        pandas_result.log_marginal_likelihood,
    )
    assert spark_result.prediction_keys.to_dict("list") == {
        "region": ["A", "A", "B", "B"],
        "time": [1, 2, 1, 2],
    }


def test_spark_fit_rejects_missing_time(spark):
    model = LGM("y", Gaussian(1.0), Fixed("1"))
    with pytest.raises(DataContractError, match="explicit time"):
        model.fit(spark.createDataFrame([("A", 1, 1.0)], ["region", "time", "y"]))


def test_spark_fit_row_limit_raises_before_fitting(spark):
    rows = [("A", 1, 1.0), ("A", 2, 2.0)]
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    with pytest.raises(DataContractError, match="max_driver_rows"):
        model.fit(
            spark.createDataFrame(rows, ["region", "time", "y"]), max_driver_rows=1
        )

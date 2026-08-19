import pytest

pyspark = pytest.importorskip("pyspark")

from pylgm import Fixed, Gaussian, LGM
from pylgm.data.spark import (
    SparkCanonicalized,
    canonicalize_spark_frame,
    is_spark_dataframe,
)
from pylgm.exceptions import DataContractError


def test_is_spark_dataframe_without_pyspark_module():
    class _Fake:
        def toPandas(self):  # noqa: N802
            return None

    assert is_spark_dataframe(_Fake()) is False
    assert is_spark_dataframe(object()) is False


def test_is_spark_dataframe_detects_real_frame(spark):
    frame = spark.createDataFrame([(1.0,)], ["y"])
    assert is_spark_dataframe(frame) is True


def test_spark_adapter_projects_only_model_columns_and_canonicalizes(spark):
    frame = spark.createDataFrame(
        [("B", 2, 4.0, 8.0, "discard"), ("A", 1, 1.0, 2.0, "discard")],
        ["region", "time", "x", "y", "raw_only"],
    )
    model = LGM("y", Gaussian(1.0), Fixed("0 + x"), panel=("region",), time="time")

    canonical = canonicalize_spark_frame(frame, model, max_driver_rows=10)

    assert isinstance(canonical, SparkCanonicalized)
    assert canonical.panel.frame.columns.tolist() == ["region", "time", "x", "y"]
    assert canonical.prediction_keys.to_dict("list") == {
        "region": ["A", "B"],
        "time": [1, 2],
    }


def test_spark_adapter_projects_formula_and_categorical_variables(spark):
    frame = spark.createDataFrame(
        [("A", 1, 1.0, 5.0, "junk"), ("B", 2, 2.0, 6.0, "junk")],
        ["region", "time", "x", "y", "raw_only"],
    )
    model = LGM(
        "y", Gaussian(1.0), Fixed("1 + x + C(region)"), panel=("region",), time="time"
    )

    canonical = canonicalize_spark_frame(frame, model, max_driver_rows=10)

    assert "raw_only" not in canonical.panel.frame.columns.tolist()
    assert set(canonical.panel.frame.columns) == {"region", "time", "x", "y"}


def test_spark_adapter_projects_effect_index(spark):
    frame = spark.createDataFrame(
        [("A", 1, 2.0, "junk"), ("A", 2, 4.0, "junk")],
        ["region", "period", "y", "raw_only"],
    )
    from pylgm import RW1

    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("1") + RW1("trend", index="period"),
        panel=("region",),
        time="period",
    )
    canonical = canonicalize_spark_frame(frame, model, max_driver_rows=10)
    assert "raw_only" not in canonical.panel.frame.columns.tolist()


def test_spark_adapter_only_collects_projected_ordered_frame(spark, monkeypatch):
    frame = spark.createDataFrame(
        [("B", 2, 4.0, 8.0, "raw"), ("A", 1, 1.0, 2.0, "raw")],
        ["region", "time", "x", "y", "raw_only"],
    )
    model = LGM("y", Gaussian(1.0), Fixed("0 + x"), panel=("region",), time="time")

    seen = {}
    frame_class = type(frame)
    original = frame_class.toPandas

    def spy(self):
        seen["columns"] = list(self.columns)
        return original(self)

    monkeypatch.setattr(frame_class, "toPandas", spy)
    canonicalize_spark_frame(frame, model, max_driver_rows=10)

    assert seen["columns"] == ["region", "time", "x", "y"]
    assert "raw_only" not in seen["columns"]


def test_spark_adapter_rejects_missing_time(spark):
    without_time = spark.createDataFrame([(1.0,)], ["y"])
    model = LGM("y", Gaussian(1.0), Fixed("1"))
    with pytest.raises(DataContractError, match="explicit time"):
        canonicalize_spark_frame(without_time, model, max_driver_rows=10)


def test_spark_adapter_rejects_missing_columns(spark):
    frame = spark.createDataFrame([("A", 1, 1.0)], ["region", "time", "y"])
    model = LGM(
        "y", Gaussian(1.0), Fixed("0 + missing"), panel=("region",), time="time"
    )
    with pytest.raises(DataContractError, match="missing columns"):
        canonicalize_spark_frame(frame, model, max_driver_rows=10)


def test_spark_adapter_rejects_null_keys(spark):
    from pyspark.sql import Row

    frame = spark.createDataFrame(
        [Row(region="A", time=1, y=1.0), Row(region=None, time=2, y=2.0)]
    )
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    with pytest.raises(DataContractError, match="null"):
        canonicalize_spark_frame(frame, model, max_driver_rows=10)


def test_spark_adapter_rejects_duplicate_keys(spark):
    duplicate = spark.createDataFrame(
        [("A", 1, 1.0), ("A", 1, 2.0)], ["region", "time", "y"]
    )
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    with pytest.raises(DataContractError, match="duplicate"):
        canonicalize_spark_frame(duplicate, model, max_driver_rows=10)


def test_spark_adapter_enforces_row_limit(spark):
    frame = spark.createDataFrame(
        [("A", 1, 1.0), ("A", 2, 2.0)], ["region", "time", "y"]
    )
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    with pytest.raises(DataContractError, match="max_driver_rows"):
        canonicalize_spark_frame(frame, model, max_driver_rows=1)


def test_spark_adapter_allows_disabled_row_limit(spark):
    frame = spark.createDataFrame(
        [("A", 1, 1.0), ("A", 2, 2.0)], ["region", "time", "y"]
    )
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    canonical = canonicalize_spark_frame(frame, model, max_driver_rows=None)
    assert len(canonical.prediction_keys) == 2


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "10"])
def test_spark_adapter_rejects_invalid_row_limit(spark, bad):
    frame = spark.createDataFrame([("A", 1, 1.0)], ["region", "time", "y"])
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    with pytest.raises(DataContractError, match="max_driver_rows"):
        canonicalize_spark_frame(frame, model, max_driver_rows=bad)


def test_spark_and_pandas_compile_to_equal_panels(spark):
    import pandas as pd

    from pylgm.compiler import compile_lgm

    rows = [("B", 2, 4.0, 8.0), ("A", 1, 1.0, 2.0), ("B", 1, 3.0, 6.0), ("A", 2, 2.0, 4.0)]
    columns = ["region", "time", "x", "y"]
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("0 + x", prior_precision=1.0),
        panel=("region",),
        time="time",
    )

    canonical = canonicalize_spark_frame(
        spark.createDataFrame(rows, columns), model, max_driver_rows=10
    )
    from pylgm.config.schema import DataConfig
    from pylgm.data import CanonicalPanel

    pandas_panel = CanonicalPanel.from_frame(
        pd.DataFrame(rows, columns=columns),
        DataConfig(time="time", response="y", panel=("region",)),
    )

    spark_compiled = compile_lgm(model, canonical.panel)
    pandas_compiled = compile_lgm(model, pandas_panel)

    assert spark_compiled.labels == pandas_compiled.labels
    assert (spark_compiled.design.toarray() == pandas_compiled.design.toarray()).all()
    assert (
        spark_compiled.precision.toarray() == pandas_compiled.precision.toarray()
    ).all()

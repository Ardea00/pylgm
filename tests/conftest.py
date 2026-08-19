import pytest


@pytest.fixture(scope="session")
def spark():
    """A local Spark session; skips the requesting test when PySpark is absent."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("pylgm-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()

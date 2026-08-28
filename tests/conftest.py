import pytest


@pytest.fixture(scope="session")
def spark():
    """A local Spark session; skips the requesting test when Spark is unavailable.

    ``importorskip`` covers a missing PySpark, but PySpark can be installed while
    the JVM it drives is not (no Java on the machine): then ``getOrCreate`` raises
    while launching the py4j gateway. Both mean Spark can't run here, so skip.
    """
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.master("local[2]")
            .appName("pylgm-tests")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
    except Exception as exc:  # no JVM/Java available to back PySpark
        pytest.skip(f"Spark session unavailable (is a JVM installed?): {exc}")
    yield session
    session.stop()

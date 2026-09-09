import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def shared_component_frame():
    """Simulated two-outcome shared-component data with a known delta.

    Returns ``(frame, true_delta)``. Both outcomes are Poisson over the same
    districts; `oral` carries `delta * u` and `larynx` carries `u / delta`.
    """
    def build(seed=0, n_districts=40, delta=1.6):
        rng = np.random.default_rng(seed)
        districts = [f"d{i}" for i in range(n_districts)]
        u = rng.normal(0.0, 0.7, size=n_districts)
        eta_oral = -0.3 + delta * u
        eta_larynx = 0.2 + u / delta
        frame = pd.DataFrame({
            "district": districts * 2,
            "outcome": ["oral"] * n_districts + ["larynx"] * n_districts,
            "oral": list(rng.poisson(np.exp(eta_oral)).astype(float)) + [np.nan] * n_districts,
            "larynx": [np.nan] * n_districts + list(rng.poisson(np.exp(eta_larynx)).astype(float)),
            "row": range(2 * n_districts),
        })
        return frame, delta

    return build


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

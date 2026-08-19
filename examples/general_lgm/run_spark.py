"""Fit the general LGM example from a Spark DataFrame.

Spark performs data preparation and canonical ordering only; the exact-Gaussian
fit runs on the driver. Requires ``pip install "pylgm[spark]"`` and Java.
"""

from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from pylgm.config import load_model


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    spark = SparkSession.builder.master("local[2]").appName("pylgm-example").getOrCreate()
    try:
        frame = spark.createDataFrame(pd.read_csv(EXAMPLE_DIRECTORY / "data.csv"))
        model = load_model(EXAMPLE_DIRECTORY / "config.yaml")
        result = model.fit(frame)
        print("prediction_keys:")
        print(result.prediction_keys.to_string(index=False))
        print("predictive_mean:", result.predictive_mean.round(4).tolist())
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

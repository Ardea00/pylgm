"""Optional PySpark data adapter.

Spark is a data boundary only: it validates, projects, and canonically orders a
Spark DataFrame, then collects the compact model table to the driver. Inference
still runs through the existing :class:`CanonicalPanel` compiler and the
exact-Gaussian reference engine. This module never imports PySpark at import
time, so Pandas-only installs never pull in Spark.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import pandas as pd
from formulaic import Formula

from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.effects import Fixed
from pylgm.exceptions import DataContractError


def is_spark_dataframe(frame: object) -> bool:
    """Detect a Spark DataFrame without importing PySpark when it is absent."""
    if type(frame).__module__.split(".", 1)[0] != "pyspark":
        return False
    return callable(getattr(frame, "toPandas", None))


@dataclass(frozen=True)
class SparkCanonicalized:
    """A collected canonical panel plus its immutable canonical prediction keys."""

    panel: CanonicalPanel
    prediction_keys: pd.DataFrame


def _validate_max_driver_rows(max_driver_rows: int | None) -> int | None:
    if max_driver_rows is None:
        return None
    if type(max_driver_rows) is not int or max_driver_rows <= 0:
        raise DataContractError(
            "max_driver_rows must be a positive integer or None"
        )
    return max_driver_rows


def _required_columns(model) -> set[str]:
    required = {model.response, *model.panel, model.time}
    if model.offset is not None:
        required.add(model.offset)
    for effect in model.predictor.effects:
        if isinstance(effect, Fixed):
            required.update(Formula(effect.formula).required_variables)
        else:
            required.add(effect.index)
    return required


def canonicalize_spark_frame(
    frame: object, model, *, max_driver_rows: int | None
) -> SparkCanonicalized:
    """Validate, project, order, and collect a Spark frame into a canonical panel."""
    max_driver_rows = _validate_max_driver_rows(max_driver_rows)

    if model.time is None:
        raise DataContractError(
            "Spark input requires an explicit time key; LGM.time must be set"
        )

    from pyspark.sql import functions as F

    try:  # AnalysisException moved to pyspark.errors in Spark 3.4+.
        from pyspark.errors import AnalysisException
    except ImportError:  # pragma: no cover - older PySpark
        from pyspark.sql.utils import AnalysisException

    keys = (*model.panel, model.time)
    required = _required_columns(model)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataContractError(f"missing columns: {missing}")

    null_key = reduce(lambda left, right: left | right, (F.col(key).isNull() for key in keys))
    if frame.filter(null_key).limit(1).count() > 0:
        raise DataContractError(f"panel key columns contain null values: {keys}")

    if frame.groupBy(*keys).count().filter("count > 1").limit(1).count() > 0:
        raise DataContractError(f"duplicate panel keys: {keys}")

    columns = tuple(column for column in frame.columns if column in required)
    projected = frame.select(*columns)

    if max_driver_rows is not None and projected.limit(max_driver_rows + 1).count() > max_driver_rows:
        raise DataContractError(
            f"Spark input exceeds max_driver_rows={max_driver_rows}; "
            "raise or disable the limit with max_driver_rows=None"
        )

    try:
        collected = projected.orderBy(*keys).toPandas()
    except AnalysisException as error:
        raise DataContractError(
            f"panel key columns must be sortable/orderable: {keys}"
        ) from error

    data = DataConfig(time=model.time, response=model.response, panel=model.panel)
    panel = CanonicalPanel.from_frame(collected, data)
    prediction_keys = panel.frame.loc[:, list(keys)]
    return SparkCanonicalized(panel=panel, prediction_keys=prediction_keys)


__all__ = ["SparkCanonicalized", "canonicalize_spark_frame", "is_spark_dataframe"]

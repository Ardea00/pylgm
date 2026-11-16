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


# Every attribute an effect or likelihood uses to name a data column. Wrappers
# (Weighted, Replicated, Grouped) nest the wrapped effect under ``.effect``.
_COLUMN_FIELDS = ("index", "unit", "space", "time", "over", "by", "replicate", "group",
                  "trials", "event", "entry")


def _declared_columns(item) -> set[str]:
    columns = set(getattr(item, "columns", None) or ())
    for field in _COLUMN_FIELDS:
        value = getattr(item, field, None)
        # A Correlated effect's index is a tuple of columns, one per component.
        columns.update(value if isinstance(value, tuple) else (value,))
    columns = {column for column in columns if isinstance(column, str)}
    for nested in (getattr(item, "effect", None), getattr(item, "base", None)):
        if nested is not None:
            columns |= _declared_columns(nested)
    return columns


def _required_columns(model) -> set[str]:
    required = {model.response, *model.panel}
    if model.time is not None:
        required.add(model.time)
    if model.offset is not None:
        required.add(model.offset)
    required |= _declared_columns(model.likelihood)
    for effect in model.predictor.effects:
        if isinstance(effect, Fixed):
            required.update(Formula(effect.formula).required_variables)
        else:
            required |= _declared_columns(effect)
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


def collect_joint_frame(frame: object, joint, *, max_driver_rows: int | None) -> pd.DataFrame:
    """Project a Spark frame to the columns a ``Joint`` declares and collect it.

    Rows come back in Spark's order: a ``Joint`` has no mandatory time key to
    sort on, and its row-aligned inputs (operators, masks) are pandas-only.
    """
    max_driver_rows = _validate_max_driver_rows(max_driver_rows)
    required = set().union(*(_required_columns(model) for model in joint.submodels))
    for shared in joint.shared:
        required |= _declared_columns(shared.effect)
    hurdle = joint.censoring
    if hurdle is not None:
        required |= {hurdle.censored, *([hurdle.threshold] if isinstance(hurdle.threshold, str) else [])}
    # A response column may be absent from the frame (see Joint.fit).
    missing = sorted(required - set(frame.columns) - {m.response for m in joint.submodels})
    if missing:
        raise DataContractError(f"missing columns: {missing}")
    projected = frame.select(*(c for c in frame.columns if c in required))
    if max_driver_rows is not None and projected.limit(max_driver_rows + 1).count() > max_driver_rows:
        raise DataContractError(
            f"Spark input exceeds max_driver_rows={max_driver_rows}; "
            "raise or disable the limit with max_driver_rows=None"
        )
    return projected.toPandas()


__all__ = [
    "SparkCanonicalized", "canonicalize_spark_frame", "collect_joint_frame", "is_spark_dataframe",
]

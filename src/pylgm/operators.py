"""Linear operator builders for ``LinearObservation``/``LinearConstraint``.

Each helper turns a pandas ``DataFrame`` into a sparse operator whose columns
are the rows of that frame, in the caller's (positional) order, so the result
can be passed directly as the ``operator`` of a linear observation or
constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _by_columns(by: str | Sequence[str]) -> list[str]:
    return [by] if isinstance(by, str) else list(by)


def _check_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"columns not found in frame: {missing}")


def _row_mask(rows: object, n: int) -> np.ndarray:
    if rows is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(rows)
    if mask.shape != (n,):
        raise ValueError(f"rows must have length {n}, got {mask.shape}")
    return mask.astype(bool)


def _weight_values(weights: object, frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    if weights is None:
        values = np.ones(n)
    elif isinstance(weights, str):
        _check_columns(frame, [weights])
        values = frame[weights].to_numpy(dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape != (n,):
            raise ValueError(f"weights must have length {n}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")
    return values


def aggregation_operator(
    frame: pd.DataFrame,
    by: str | Sequence[str],
    *,
    weights: object = None,
    rows: object = None,
) -> tuple[csr_matrix, pd.DataFrame]:
    """Sum (or weighted-sum) ``frame`` rows into one operator row per distinct key.

    One row per distinct value of ``by`` (sorted ascending, pandas ``groupby``
    semantics); ``weights`` defaults to 1.0 per row and ``rows`` selects a
    subset of ``frame`` to include. Returns ``(operator, keys)`` where ``keys``
    holds the ``by`` columns for each operator row, indexed ``0..k-1``.
    """
    columns = _by_columns(by)
    _check_columns(frame, columns)
    n = len(frame)
    mask = _row_mask(rows, n)
    weight_values = _weight_values(weights, frame)
    key_frame = frame[columns]
    if key_frame.loc[mask].isna().any(axis=None):
        raise ValueError(f"columns {columns} contain NaN among selected rows")
    if not mask.any():
        raise ValueError("rows selects zero rows")

    selected = np.flatnonzero(mask)
    grouped = key_frame.iloc[selected].groupby(columns, sort=True)
    group_indices = grouped.indices  # keyed by group value(s), positions into `selected`

    keys = pd.DataFrame(
        list(group_indices.keys()) if len(columns) > 1 else [(key,) for key in group_indices],
        columns=columns,
    )
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    for r, positions in enumerate(group_indices.values()):
        cols = selected[positions]
        row_idx.extend([r] * len(cols))
        col_idx.extend(cols.tolist())
        data.extend(weight_values[cols].tolist())

    operator = csr_matrix(
        (data, (row_idx, col_idx)), shape=(len(keys), n), dtype=float
    )
    return operator, keys


def _panel_columns(panel: str | Sequence[str] | None) -> list[str]:
    if panel is None:
        return []
    return _by_columns(panel)


def _validate_time_panel(frame: pd.DataFrame, time: str, panel_columns: Sequence[str]) -> None:
    columns = [time, *panel_columns]
    _check_columns(frame, columns)
    if frame[columns].isna().any(axis=None):
        raise ValueError(f"columns {columns} must not contain NaN")


def _units(frame: pd.DataFrame, time: str, panel_columns: Sequence[str]):
    """Yield ``(unit_key, positions)`` for each unit, units and within-unit time sorted.

    ``unit_key`` is ``()`` when there is no panel. ``positions`` are positional
    indices into ``frame``, sorted by ``time`` (stable) within the unit.
    """
    n = len(frame)
    time_values = frame[time].to_numpy()
    if panel_columns:
        grouped = frame[panel_columns].groupby(panel_columns, sort=True)
        groups = grouped.indices
        keys = list(groups.keys())
        if len(panel_columns) == 1:
            keys = [(key,) for key in keys]
        positions_by_unit = list(groups.values())
    else:
        keys = [()]
        positions_by_unit = [np.arange(n)]

    for key, positions in zip(keys, positions_by_unit):
        order = np.argsort(time_values[positions], kind="stable")
        ordered_positions = np.asarray(positions)[order]
        if pd.Series(time_values[ordered_positions]).duplicated().any():
            raise ValueError(f"duplicate time values within unit {key}")
        yield key, ordered_positions


def difference_operator(
    frame: pd.DataFrame,
    time: str,
    panel: str | Sequence[str] | None = None,
    *,
    lag: int = 1,
    order: int = 1,
) -> tuple[csr_matrix, pd.DataFrame]:
    """Build the finite-difference operator ``(1 - L^lag)^order`` within each unit.

    Within each unit (distinct values of ``panel``; ``None`` treats the whole
    frame as one series), rows are sorted by ``time`` (stable). For each
    position ``i >= lag * order`` in a unit, one operator row applies
    ``(1 - L^lag)^order`` to that unit's time-ordered values, producing a
    coefficient ``(-1)^k * C(order, k)`` on the value ``lag * k`` steps back.
    Rows are ordered by unit (sorted keys), then time; ``keys`` holds the
    panel columns (if any) and ``time`` for each operator row's target position.
    """
    if not isinstance(lag, int) or lag < 1:
        raise ValueError(f"lag must be a positive int, got {lag!r}")
    if not isinstance(order, int) or order < 1:
        raise ValueError(f"order must be a positive int, got {order!r}")
    panel_columns = _panel_columns(panel)
    _validate_time_panel(frame, time, panel_columns)

    n = len(frame)
    time_values = frame[time].to_numpy()
    coefficients = [
        ((-1) ** k) * comb(order, k) for k in range(order + 1)
    ]

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    key_rows: list[tuple] = []
    r = 0
    for key, positions in _units(frame, time, panel_columns):
        m = len(positions)
        for i in range(lag * order, m):
            target = positions[i]
            for k, coefficient in enumerate(coefficients):
                source = positions[i - k * lag]
                row_idx.append(r)
                col_idx.append(int(source))
                data.append(float(coefficient))
            key_rows.append((*key, time_values[target]))
            r += 1

    operator = csr_matrix((data, (row_idx, col_idx)), shape=(r, n), dtype=float)
    keys = pd.DataFrame(key_rows, columns=[*panel_columns, time])
    return operator, keys


def cumulation_operator(
    frame: pd.DataFrame,
    time: str,
    panel: str | Sequence[str] | None = None,
    *,
    lag: int = 1,
) -> csr_matrix:
    """Build the strided cumulative-sum operator, the right inverse of ``difference_operator``.

    Square, ``len(frame) x len(frame)``. Row ``r`` (caller order) sums the rows
    ``s`` in the same unit with ``time_s <= time_r`` and
    ``(position_r - position_s) % lag == 0`` (positions are rank in the unit's
    time order). ``lag=1`` is the running sum within each unit. Same
    validation as ``difference_operator``.
    """
    if not isinstance(lag, int) or lag < 1:
        raise ValueError(f"lag must be a positive int, got {lag!r}")
    panel_columns = _panel_columns(panel)
    _validate_time_panel(frame, time, panel_columns)

    n = len(frame)
    row_idx: list[int] = []
    col_idx: list[int] = []
    for _key, positions in _units(frame, time, panel_columns):
        m = len(positions)
        for i in range(m):
            for j in range(i, -1, -lag):
                row_idx.append(int(positions[i]))
                col_idx.append(int(positions[j]))

    data = np.ones(len(row_idx))
    return csr_matrix((data, (row_idx, col_idx)), shape=(n, n), dtype=float)


def compose(*operators: object) -> csr_matrix:
    """Left-to-right sparse matrix product ``operators[0] @ operators[1] @ ...``."""
    if not operators:
        raise ValueError("compose requires at least one operator")
    converted = [
        operator if isinstance(operator, csr_matrix) else csr_matrix(np.asarray(operator))
        for operator in operators
    ]
    result = converted[0]
    for operator in converted[1:]:
        if result.shape[1] != operator.shape[0]:
            raise ValueError(
                f"cannot compose operators of shape {result.shape} and {operator.shape}"
            )
        result = result @ operator
    return csr_matrix(result, dtype=float)


__all__ = [
    "aggregation_operator",
    "compose",
    "cumulation_operator",
    "difference_operator",
]

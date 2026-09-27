"""Index-number utilities for annual-overlap chain linking.

Under annual overlap, each sub-period of period ``p`` is valued at the
prices of period ``p-1`` and then linked to the chain by one factor per
group and period. Conversely, a chain-linked volume becomes a value at
previous-period prices when multiplied by ``k[p] = current_total[p-1] /
volume_total[p-1]``, computed from the previous period's totals. Values at
previous-period prices are additive across groups within a period, while
chain-linked volumes are not, since each group carries its own chain of
factors. This module builds the per-(group, period) factors from published
totals and converts between the two scales.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pylgm.operators import _by_columns, _check_columns

__all__ = ["align_factors", "chain", "overlap_factors", "unchain"]


def _group_columns(group: str | Sequence[str] | None) -> list[str]:
    return [] if group is None else _by_columns(group)


def _next_period(value: object) -> object:
    try:
        return value + 1
    except TypeError as error:
        raise TypeError(
            "period values must support + 1 (integers or pandas Period)"
        ) from error


def overlap_factors(
    totals: pd.DataFrame,
    *,
    volume: str,
    current: str,
    period: str,
    group: str | Sequence[str] | None = None,
    through: object = None,
) -> pd.DataFrame:
    """Build annual-overlap rescaling factors from per-(group, period) totals.

    ``totals`` has one row per ``(group, period)`` with columns ``volume``
    (chain-linked total) and ``current`` (current-price total), both either
    sums or means of the sub-periods. Every row whose ``volume`` and
    ``current`` are both finite and positive yields a factor for the next
    period, ``factor = current / volume``, with ``carried=False``; rows with
    missing or non-positive totals are silently skipped (unpublished
    periods). If ``through`` is given, each group's latest factor is carried
    forward, unchanged and flagged ``carried=True``, to every period up to
    and including ``through`` that would otherwise have no factor.
    """
    group_columns = _group_columns(group)
    key_columns = [*group_columns, period]
    _check_columns(totals, [*key_columns, volume, current])

    if totals[key_columns].duplicated().any():
        raise ValueError(f"duplicate keys in totals: {key_columns}")

    volume_values = totals[volume].to_numpy(dtype=float)
    current_values = totals[current].to_numpy(dtype=float)
    valid = (
        np.isfinite(volume_values)
        & np.isfinite(current_values)
        & (volume_values > 0)
        & (current_values > 0)
    )

    columns = [*group_columns, period, "factor", "carried"]
    rows: list[tuple] = []
    for position in np.flatnonzero(valid):
        row = totals.iloc[position]
        next_period = _next_period(row[period])
        key = tuple(row[column] for column in group_columns)
        factor = float(current_values[position] / volume_values[position])
        rows.append((*key, next_period, factor, False))

    result = pd.DataFrame(rows, columns=columns)

    if through is not None and not result.empty:
        if group_columns:
            groups = list(result.groupby(group_columns, sort=False).indices.items())
        else:
            groups = [((), np.arange(len(result)))]
        extra: list[tuple] = []
        for key, positions in groups:
            key = key if isinstance(key, tuple) else (key,)
            sub = result.iloc[positions]
            latest_position = sub[period].idxmax()
            latest_period = sub.loc[latest_position, period]
            latest_factor = sub.loc[latest_position, "factor"]
            current_period = latest_period
            while current_period < through:
                current_period = _next_period(current_period)
                extra.append((*key, current_period, latest_factor, True))
        if extra:
            extra_frame = pd.DataFrame(extra, columns=columns)
            result = pd.concat([result, extra_frame], ignore_index=True)

    result["carried"] = result["carried"].astype(bool)
    result = result.sort_values(key_columns, kind="stable").reset_index(drop=True)
    return result[columns]


def align_factors(
    frame: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    period: str,
    group: str | Sequence[str] | None = None,
) -> np.ndarray:
    """Look up, for every row of ``frame``, the matching factor in ``factors``.

    ``factors`` is shaped like the output of :func:`overlap_factors` (only
    its key columns and ``factor`` are read). Rows of ``frame`` with no
    matching ``(group, period)`` key get ``NaN``; duplicate keys in
    ``factors`` raise ``ValueError``.
    """
    group_columns = _group_columns(group)
    key_columns = [*group_columns, period]
    _check_columns(frame, key_columns)
    _check_columns(factors, [*key_columns, "factor"])

    if factors[key_columns].duplicated().any():
        raise ValueError(f"duplicate keys in factors: {key_columns}")

    left = frame[key_columns].reset_index(drop=True)
    left["__pylgm_row__"] = np.arange(len(left))
    merged = left.merge(
        factors[[*key_columns, "factor"]],
        on=key_columns,
        how="left",
        validate="many_to_one",
    )
    merged = merged.sort_values("__pylgm_row__", kind="stable")
    return merged["factor"].to_numpy(dtype=float)


def _values_array(frame: pd.DataFrame, values: object) -> np.ndarray:
    if isinstance(values, str):
        _check_columns(frame, [values])
        return frame[values].to_numpy(dtype=float)
    array = np.asarray(values, dtype=float)
    n = len(frame)
    if array.shape[-1] != n:
        raise ValueError(
            f"values last axis must have length {n}, got {array.shape[-1]}"
        )
    return array


def unchain(
    frame: pd.DataFrame,
    values: object,
    factors: pd.DataFrame,
    *,
    period: str,
    group: str | Sequence[str] | None = None,
) -> np.ndarray:
    """Convert chain-linked ``values`` to previous-period prices.

    ``values`` is a column name of ``frame``, or an array whose last axis has
    length ``len(frame)``. Returns ``values * align_factors(...)``; ``NaN``
    where the factor is missing.
    """
    array = _values_array(frame, values)
    factor = align_factors(frame, factors, period=period, group=group)
    return array * factor


def chain(
    frame: pd.DataFrame,
    values: object,
    factors: pd.DataFrame,
    *,
    period: str,
    group: str | Sequence[str] | None = None,
) -> np.ndarray:
    """Convert previous-period-price ``values`` back to chain-linked volumes.

    The inverse of :func:`unchain`: returns ``values / align_factors(...)``;
    ``NaN`` where the factor is missing.
    """
    array = _values_array(frame, values)
    factor = align_factors(frame, factors, period=period, group=group)
    return array / factor

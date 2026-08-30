"""Seasonal latent effect: a slowly-drifting periodic pattern.

The seasonal penalty sums every ``period`` consecutive levels and shrinks that
sum toward zero, so a pattern that repeats with the declared period is
*unpenalized* while drift away from it is penalized. That null space -- the
fixed seasonal patterns -- is exactly the signal, so unlike ``RW1``/``RW2``
it must not be constrained away. It is instead given a fixed, ``precision``-
independent ridge on its orthogonal projector, the same device
``midas_penalty`` uses for the lag curve's level and slope: the pattern stays
estimable and weakly shrunk, and the block stays positive definite with no
constraints.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from pylgm.data.scalars import ordered_observed_levels
from pylgm.ir.model import LatentBlock


def seasonal_operator(n: int, period: int) -> csr_matrix:
    """The ``(n - period + 1) x n`` sliding sum over ``period`` consecutive levels."""
    return diags(
        [np.ones(n - period + 1)] * period,
        list(range(period)),
        shape=(n - period + 1, n),
        format="csr",
    )


def seasonal_null_basis(n: int, period: int) -> np.ndarray:
    """Basis of ``null(S)``: the fixed seasonal patterns, ``n x (period - 1)``.

    ``S x = 0`` forces every ``period``-window to sum to zero; subtracting
    consecutive windows gives ``x_{i+period} = x_i``, so the null space is
    exactly the patterns that repeat with the declared period and sum to zero
    over one period. Column ``k`` puts ``+1`` on residue 0 and ``-1`` on residue
    ``k``, which any window meets exactly once each.
    """
    residue = np.arange(n) % period
    return np.column_stack([
        (residue == 0).astype(float) - (residue == k).astype(float)
        for k in range(1, period)
    ])


def seasonal_penalty(n: int, period: int) -> tuple[csr_matrix, csr_matrix]:
    """Return ``(SᵀS, P₀)``: the drift penalty and the projector onto ``null(S)``.

    ``precision * SᵀS + ridge * P₀`` is positive definite, so the effect needs
    no constraints.
    """
    s = seasonal_operator(n, period)
    sts = csr_matrix(s.T @ s)
    basis = seasonal_null_basis(n, period)
    projector = csr_matrix(basis @ np.linalg.pinv(basis))
    return sts, projector


def build_seasonal(
    frame: pd.DataFrame,
    name: str,
    index: str,
    precision: float,
    period: int,
    ridge: float,
) -> LatentBlock:
    levels = ordered_observed_levels(frame[index])
    if period < 2:
        raise ValueError(f"{name} period must be at least 2; got {period}")
    if len(levels) <= period:
        raise ValueError(
            f"{name} requires more than {period} ordered levels; got {len(levels)}"
        )
    positions = {level: column for column, level in enumerate(levels)}
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), columns)),
        shape=(len(frame), len(levels)),
    )
    sts, projector = seasonal_penalty(len(levels), period)
    precision_matrix = csr_matrix(precision * sts + ridge * projector)
    constraints = np.empty((0, len(levels)), dtype=float)
    return LatentBlock(name, tuple(map(str, levels)), design, precision_matrix, constraints)

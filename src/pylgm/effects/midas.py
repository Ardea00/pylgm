from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pylgm.effects.random_walk import difference_operator
from pylgm.ir.model import LatentBlock


def midas_penalty(n: int, order: Literal[1, 2]) -> tuple[csr_matrix, csr_matrix]:
    """The MIDAS penalty pieces over ``n`` lags: curvature ``DᵀD`` and the
    orthogonal projector ``P₀`` onto the penalty's null space (level, or
    level+slope), so ``τ·DᵀD + δ·P₀`` smooths curvature while leaving the lag
    curve's level/slope on a fixed, ``τ``-independent precision ``δ``.
    """
    difference = difference_operator(n, order)
    dtd = csr_matrix(difference.T @ difference)
    basis = [np.ones(n)] if order == 1 else [np.ones(n), np.arange(n, dtype=float)]
    t = np.column_stack(basis)
    projector = csr_matrix(t @ np.linalg.pinv(t))
    return dtd, projector


def build_midas(
    frame: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
    precision: float,
    order: Literal[1, 2],
    ridge: float,
) -> LatentBlock:
    if len(columns) <= order:
        raise ValueError(f"{name} requires more than {order} lag columns")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} lag columns not found: {missing}")
    values = frame[list(columns)].to_numpy(dtype=float)
    for offset, column in enumerate(columns):
        if not np.isfinite(values[:, offset]).all():
            raise ValueError(f"{name} lag column {column!r} contains non-finite values")
    design = csr_matrix(values)
    dtd, projector = midas_penalty(len(columns), order)
    precision_matrix = csr_matrix(precision * dtd + ridge * projector)
    constraints = np.empty((0, len(columns)), dtype=float)
    return LatentBlock(name, tuple(columns), design, precision_matrix, constraints)

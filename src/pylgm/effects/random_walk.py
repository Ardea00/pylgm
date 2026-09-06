from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from pylgm.data.scalars import ordered_observed_levels
from pylgm.effects.scaling import sorbye_rue_scale
from pylgm.ir.model import LatentBlock


def difference_operator(n: int, order: Literal[1, 2]) -> csr_matrix:
    """The order-1 or order-2 finite-difference operator over ``n`` points."""
    if order == 1:
        return diags(
            [-np.ones(n - 1), np.ones(n - 1)],
            [0, 1],
            shape=(n - 1, n),
            format="csr",
        )
    return diags(
        [np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csr",
    )


def rw_structure(level_count: int, order: Literal[1, 2], scale: bool = True) -> np.ndarray:
    """Sørbye-Rue-scaled RW structure ``DᵀD`` over ``level_count`` points.

    Lives here rather than in spacetime.py, which is where it started: both the
    space-time interaction and the between-group structures need it, and both
    already depend on this module for ``difference_operator``.
    """
    difference = difference_operator(level_count, order)
    r = (difference.T @ difference).toarray()
    return sorbye_rue_scale(r, null_dim=order) if scale else r


def build_random_walk(
    frame: pd.DataFrame,
    name: str,
    index: str,
    precision: float,
    order: Literal[1, 2],
) -> LatentBlock:
    levels = ordered_observed_levels(frame[index])
    if len(levels) <= order:
        raise ValueError(f"{name} requires more than {order} ordered levels")
    positions = {level: column for column, level in enumerate(levels)}
    rows = np.arange(len(frame))
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix((np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels)))
    difference = difference_operator(len(levels), order)
    precision_matrix = csr_matrix(precision * (difference.T @ difference))
    coordinate = np.arange(len(levels), dtype=float)
    constraints = np.ones((1, len(levels)))
    if order == 2:
        constraints = np.vstack([constraints, coordinate - coordinate.mean()])
    return LatentBlock(
        name,
        tuple(map(str, levels)),
        design,
        precision_matrix,
        constraints,
    )

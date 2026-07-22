from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pylgm.ir.model import LatentBlock


def build_random_walk(
    frame: pd.DataFrame,
    name: str,
    index: str,
    precision: float,
    order: Literal[1, 2],
) -> LatentBlock:
    levels = tuple(sorted(frame[index].drop_duplicates().tolist()))
    if len(levels) <= order:
        raise ValueError(f"{name} requires more than {order} ordered levels")
    positions = {level: column for column, level in enumerate(levels)}
    rows = np.arange(len(frame))
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix(
        (np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels))
    )
    difference = np.diff(np.eye(len(levels)), n=order, axis=0)
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

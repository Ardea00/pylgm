import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye

from pylgm.ir.model import LatentBlock


def build_iid(frame: pd.DataFrame, name: str, index: str, precision: float) -> LatentBlock:
    levels = tuple(sorted(frame[index].drop_duplicates().tolist()))
    positions = {level: column for column, level in enumerate(levels)}
    rows = np.arange(len(frame))
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix(
        (np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels))
    )
    return LatentBlock(
        name,
        tuple(map(str, levels)),
        design,
        eye(len(levels), format="csr") * precision,
        np.empty((0, len(levels))),
    )

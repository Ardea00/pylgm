import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy.sparse import csr_matrix, eye

from pylgm.ir.model import LatentBlock


def build_fixed(frame: pd.DataFrame, formula: str, prior_precision: float) -> LatentBlock:
    matrix = model_matrix(formula, frame)
    design = csr_matrix(np.asarray(matrix, dtype=float))
    labels = tuple(matrix.model_spec.column_names)
    return LatentBlock(
        name="fixed",
        labels=labels,
        design=design,
        precision=eye(design.shape[1], format="csr") * prior_precision,
        constraints=np.empty((0, design.shape[1]), dtype=float),
    )

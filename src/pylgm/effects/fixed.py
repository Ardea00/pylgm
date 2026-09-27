import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy.sparse import csr_matrix, diags, eye

from pylgm.ir.model import LatentBlock

_DIFFUSE_PRECISION = 1e-6


def build_fixed(
    frame: pd.DataFrame,
    formula: str,
    prior_precision: float,
    exempt_intercept: bool = False,
) -> LatentBlock:
    matrix = model_matrix(formula, frame)
    design = csr_matrix(np.asarray(matrix, dtype=float))
    labels = tuple(matrix.model_spec.column_names)
    if exempt_intercept:
        vector = np.full(design.shape[1], prior_precision, dtype=float)
        for index, label in enumerate(labels):
            if label == "Intercept":
                vector[index] = _DIFFUSE_PRECISION
        precision = diags(vector, format="csr")
    else:
        precision = eye(design.shape[1], format="csr") * prior_precision
    return LatentBlock(
        name="fixed",
        labels=labels,
        design=design,
        precision=precision,
        constraints=np.empty((0, design.shape[1]), dtype=float),
    )

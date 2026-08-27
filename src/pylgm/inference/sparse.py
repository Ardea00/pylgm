import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from pylgm.exceptions import NumericalError
from pylgm.ir.model import CompiledLGM


class SparseSpdFactor:
    """SuperLU factor of a sparse SPD matrix, with an exact log-determinant.

    Uses ``splu`` (not a sparse Cholesky, which SciPy does not ship). For an
    SPD matrix ``logdet = sum(log(diag(U)))`` from the LU factor; the row/column
    permutations contribute a determinant of +/-1, which cancels for the SPD
    magnitude. A non-positive product of diagonal entries means the matrix was
    not positive definite -- surfaced as NumericalError to match the dense path.
    """

    def __init__(self, matrix: csr_matrix, name: str) -> None:
        self._name = name
        try:
            # permc_spec chosen for fill-in reduction on GMRF precisions.
            self._lu = splu(matrix.tocsc(), permc_spec="COLAMD")
        except (RuntimeError, ValueError) as error:
            raise NumericalError(f"{name} must be positive definite") from error
        diag_u = self._lu.U.diagonal()
        if not np.all(diag_u > 0) or not np.isfinite(diag_u).all():
            raise NumericalError(f"{name} must be positive definite")
        self._logdet = float(np.sum(np.log(diag_u)))

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self._lu.solve(np.asarray(b, dtype=float))

    @property
    def logdet(self) -> float:
        return self._logdet


def _partition_blocks(model: CompiledLGM) -> tuple[np.ndarray, np.ndarray]:
    """Split latent columns into the sparse field block and the dense fixed block.

    A block with an all-zero prior precision is an improper/flat effect (a
    ``Fixed`` covariate or intercept). Its design column couples to many
    observations, so ``Z^T Z`` would densify a joint sparse factor -- it is
    quarantined into the small dense block. Every structured GMRF field
    (IID / RW / Besag / proper_car / AR1 / space-time) carries a nonzero prior
    precision and an incidence-like design, so it stays sparse.

    ponytail: criterion is precision.nnz==0, which is exact for the in-scope
    spatial models. A structured block with a *dense* design (e.g. MIDAS lag
    columns) would be misrouted into the sparse block and densify A_s. MIDAS at
    scale is out of scope (E-sparse targets spatial models); such a model still
    fits via the dense path under allow_large_dense. Upgrade path: also test
    design column density here and raise a clear unsupported error.
    """
    sparse_cols: list[int] = []
    dense_cols: list[int] = []
    start = 0
    for block in model.blocks:
        width = block.design.shape[1]
        columns = range(start, start + width)
        if block.precision.nnz == 0:
            dense_cols.extend(columns)
        else:
            sparse_cols.extend(columns)
        start += width
    return np.asarray(sparse_cols, dtype=int), np.asarray(dense_cols, dtype=int)

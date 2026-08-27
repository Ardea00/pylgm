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

    Route a block to the small dense (Schur) block when any of its design
    columns is nonzero on more than half the observations. That is exactly the
    intercept / continuous-covariate pattern Approach A must quarantine: a
    globally-coupling column turns ``Z^T Z`` into a dense row and densifies the
    sparse factor. Every structured GMRF field (IID / RW / Besag / proper_car /
    AR1 / space-time) has an incidence-like design whose columns each touch only
    their own cell's observations, so it stays sparse.

    The criterion tracks the actual fill-in cause, not the effect's nominal
    type, so it is self-correcting: a tiny field with a dense column routes
    dense harmlessly (the Schur block stays small), and a high-cardinality
    categorical ``Fixed`` effect with sparse dummy columns routes sparse and
    factors fine.

    ponytail: threshold is 0.5 of observations. A structured block with a
    genuinely dense design (e.g. MIDAS lag columns) would route dense and
    enlarge the Schur block; MIDAS at scale is out of scope (E-sparse targets
    spatial models) and still fits via the dense path under allow_large_dense.
    """
    n_obs = model.blocks[0].design.shape[0] if model.blocks else 0
    sparse_cols: list[int] = []
    dense_cols: list[int] = []
    start = 0
    for block in model.blocks:
        width = block.design.shape[1]
        columns = range(start, start + width)
        col_nnz = block.design.getnnz(axis=0)
        if n_obs and col_nnz.max() > 0.5 * n_obs:
            dense_cols.extend(columns)
        else:
            sparse_cols.extend(columns)
        start += width
    return np.asarray(sparse_cols, dtype=int), np.asarray(dense_cols, dtype=int)

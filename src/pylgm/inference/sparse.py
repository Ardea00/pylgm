from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from pylgm.exceptions import NumericalError
from pylgm.inference.gaussian import _block_slices, _factor_positive_definite
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


@dataclass(frozen=True)
class SparseFit:
    mean: np.ndarray
    log_marginal_likelihood: float
    predictive_mean: np.ndarray
    block_slices: Mapping[str, slice]
    diagnostics: dict[str, object]


def sparse_constrained_gaussian(model: CompiledLGM) -> SparseFit:
    """Unconstrained partitioned Schur solve, matching ``gaussian._fit_dense``.

    Partitions the latent columns into a sparse GMRF field block ``s`` and a
    small dense fixed block ``d`` (``_partition_blocks``), then solves the
    posterior precision ``Q + Z^T Z / sigma^2`` by a Schur complement on the
    dense block -- densifying only ``B`` (n_s x m), ``D`` and ``S`` (m x m),
    never the field. Constraints arrive in Task 4b.

    ponytail: assumes ``Q_sd == 0`` (block-diagonal prior + block-granular
    partition), so ``B = Z_s^T Z_d / sigma^2``. True for every LGM this path
    handles; a cross-block prior term would need adding here.
    """
    if model.constraints.shape[0]:
        raise NotImplementedError("constrained sparse solve arrives in Task 4b")

    variance = float(model.likelihood.variance)
    if not np.isfinite(variance) or variance <= 0:
        raise NumericalError("sigma squared must be finite and positive")

    sparse_index, dense_index = _partition_blocks(model)
    latent_size = model.precision.shape[0]
    observed = model.observed
    design = model.design
    observed_design = design[observed]
    residual = model.y[observed] - model.offset[observed]
    q = model.precision

    z_s = observed_design[:, sparse_index]
    z_d = observed_design[:, dense_index]
    n_s = sparse_index.size
    m = dense_index.size

    # Score g = Z^T r / sigma^2, split per block.
    g_s = np.asarray(z_s.T @ residual).reshape(-1) / variance
    g_d = np.asarray(z_d.T @ residual).reshape(-1) / variance

    mean = np.zeros(latent_size)
    logdet_prior = 0.0
    logdet_posterior = 0.0

    if n_s:
        q_ss = q[sparse_index][:, sparse_index]
        a_s = SparseSpdFactor(
            (q_ss + (z_s.T @ z_s) / variance).tocsr(), "sparse posterior precision"
        )
        logdet_prior += SparseSpdFactor(q_ss.tocsr(), "sparse prior precision").logdet
        logdet_posterior += a_s.logdet

    if m:
        q_dd = q[dense_index][:, dense_index].toarray()
        d = q_dd + (z_d.T @ z_d).toarray() / variance
        _, logdet_qdd = _factor_positive_definite(q_dd, "dense prior precision")
        logdet_prior += logdet_qdd

    if n_s and m:
        b = (z_s.T @ z_d).toarray() / variance  # n_s x m
        schur = d - b.T @ a_s.solve(b)  # m x m
        schur_factor, logdet_schur = _factor_positive_definite(schur, "dense Schur complement")
        logdet_posterior += logdet_schur
        x_d = cho_solve(schur_factor, g_d - b.T @ a_s.solve(g_s))
        x_s = a_s.solve(g_s - b @ x_d)
        mean[sparse_index] = x_s
        mean[dense_index] = x_d
    elif n_s:
        mean[sparse_index] = a_s.solve(g_s)
    else:
        d_factor, logdet_d = _factor_positive_definite(d, "dense posterior precision")
        logdet_posterior += logdet_d
        mean[dense_index] = cho_solve(d_factor, g_d)

    n_observed = int(np.count_nonzero(observed))
    g_full = np.zeros(latent_size)
    g_full[sparse_index] = g_s
    g_full[dense_index] = g_d
    quadratic = float(residual @ residual / variance - mean @ g_full)
    log_marginal_likelihood = -0.5 * (
        n_observed * np.log(2 * np.pi * variance)
        - logdet_prior
        + logdet_posterior
        + quadratic
    )
    predictive_mean = np.asarray(model.offset + design @ mean).reshape(-1)

    return SparseFit(
        mean=mean,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": n_observed,
            "constraint_count": 0,
            "sparse_dimension": int(n_s),
            "dense_dimension": int(m),
        },
    )

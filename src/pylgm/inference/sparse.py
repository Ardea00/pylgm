from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, null_space
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
class SparsePosterior:
    """Posterior precision factors from the Schur solve, exposed as an inverse
    operator and (Tasks 3-4) as diagonal-variance composers. Never forms the full
    n×n covariance above the dense guard.
    """

    latent_size: int
    sparse_index: np.ndarray
    dense_index: np.ndarray
    a_ss: "SparseSpdFactor | None"
    a_ss_matrix: "csr_matrix | None"   # A_ss = Q_ss + Z_s^T Z_s / sigma^2 (for Takahashi)
    b: "np.ndarray | None"             # B = Z_s^T Z_d / sigma^2  (n_s x m)
    schur_factor: object | None        # cho_factor of S = D - B^T A_ss^-1 B
    d_factor: object | None            # cho_factor of D (dense-only case)
    w_constraint: "np.ndarray | None"  # Sigma A_c^T  (latent x c), kriging basis
    cap_factor: object | None          # cho_factor of A_c Sigma A_c^T

    def apply_inverse(self, rhs: np.ndarray) -> np.ndarray:
        """``P_post^-1 @ rhs`` via the same Schur solve as the mean; 1-D or 2-D."""
        rhs = np.asarray(rhs, dtype=float)
        out = np.zeros_like(rhs)
        s, d = self.sparse_index, self.dense_index
        if s.size and d.size:
            v_s, v_d = rhs[s], rhs[d]
            x_d = cho_solve(self.schur_factor, v_d - self.b.T @ self.a_ss.solve(v_s))
            out[s] = self.a_ss.solve(v_s - self.b @ x_d)
            out[d] = x_d
        elif s.size:
            out[s] = self.a_ss.solve(rhs[s])
        else:
            out[d] = cho_solve(self.d_factor, rhs[d])
        return out

    def _unconstrained_marginal(self) -> np.ndarray:
        """diag(P_post^-1) -- the unconstrained posterior variance, full length."""
        diag = np.zeros(self.latent_size)
        s, d = self.sparse_index, self.dense_index
        if s.size:
            diag_ss = selected_inverse_diagonal(self.a_ss_matrix)   # diag(A_ss^-1)
            if d.size:
                w = self.a_ss.solve(self.b)                          # A_ss^-1 B  (n_s x m)
                sinv_wt = cho_solve(self.schur_factor, w.T)          # S^-1 W^T   (m x n_s)
                diag_ss = diag_ss + np.einsum("ij,ji->i", w, sinv_wt)
            diag[s] = diag_ss
        if d.size:
            m = d.size
            factor = self.schur_factor if s.size else self.d_factor
            diag[d] = np.diag(cho_solve(factor, np.eye(m)))
        return diag

    def marginal_variances(self) -> np.ndarray:
        """diag(Sigma_c) -- constrained posterior marginal variances, full length."""
        diag = self._unconstrained_marginal()
        if self.w_constraint is not None:
            w = self.w_constraint                                    # latent x c
            cw = cho_solve(self.cap_factor, w.T)                     # c x latent
            diag = diag - np.einsum("ij,ji->i", w, cw)
        return np.clip(diag, 0.0, None)

    def predictive_variances(self, design) -> np.ndarray:
        """diag(design Σ_c designᵀ) — one variance per row of ``design``."""
        dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
        # ponytail: apply_inverse densifies designᵀ to (latent x n_rows). At very
        # large latent × request width this is the memory ceiling — batch the
        # columns of designᵀ if it ever bites; fine at current network scale.
        cov_dt = self.apply_inverse(dense.T)                        # Σ designᵀ (latent x n_rows)
        var = np.einsum("ij,ji->i", dense, cov_dt)
        if self.w_constraint is not None:
            mw = dense @ self.w_constraint                          # n_rows x c
            cw = cho_solve(self.cap_factor, mw.T)                   # c x n_rows
            var = var - np.einsum("ij,ji->i", mw, cw)
        return np.clip(var, 0.0, None)

    def linear_combination_variances(self, weights) -> np.ndarray:
        """diag(M Σ_c Mᵀ) — same quadratic form as predictive_variances."""
        return self.predictive_variances(weights)

    def covariance_dense(self) -> np.ndarray:
        """Full constrained covariance Σ_c (O(n²) — opt-in escape hatch)."""
        sigma = self.apply_inverse(np.eye(self.latent_size))
        if self.w_constraint is not None:
            w = self.w_constraint
            sigma = sigma - w @ cho_solve(self.cap_factor, w.T)
        return 0.5 * (sigma + sigma.T)


@dataclass(frozen=True)
class SparseFit:
    mean: np.ndarray
    log_marginal_likelihood: float
    predictive_mean: np.ndarray
    block_slices: Mapping[str, slice]
    diagnostics: dict[str, object]
    posterior: "SparsePosterior"


def _block_column_confinement(model: CompiledLGM) -> list[tuple[int, int, np.ndarray]]:
    """For each block, its column span and the constraint rows confined to it.

    A row is *confined* to a block when all its nonzeros lie inside that block's
    columns. Returns ``(start, stop, confined_row_mask)`` per block. Raises when a
    row straddles two blocks -- the sparse prior/quadratic separability assumes
    each constraint touches a single block (block sum-to-zero rows and the
    single-block extra constraints these models carry). A cross-block row would
    couple two reduced priors; that is E-sparse-C territory.
    """
    a = np.asarray(model.constraints, dtype=float)
    row_mass = np.abs(a).sum(axis=1) if a.shape[0] else np.zeros(0)
    spans: list[tuple[int, int, np.ndarray]] = []
    confined_any = np.zeros(a.shape[0], dtype=bool)
    start = 0
    for block in model.blocks:
        stop = start + block.design.shape[1]
        if a.shape[0]:
            here = np.abs(a[:, start:stop]).sum(axis=1)
            confined = (here > 0) & (row_mass - here <= 1e-12)
            confined_any |= confined
        else:
            confined = np.zeros(0, dtype=bool)
        spans.append((start, stop, confined))
        start = stop
    if a.shape[0] and not confined_any.all():
        raise NotImplementedError(
            "sparse constrained solve requires each constraint row to touch a "
            "single latent block; cross-block constraints are not yet supported"
        )
    return spans


def _is_connected_intrinsic(rows: np.ndarray, precision) -> bool:
    """True when a block is a connected intrinsic field pinned by one
    unweighted sum-to-zero row: exactly one constraint row proportional to the
    all-ones vector, and ``precision @ 1 == 0`` (so null(precision) = span(1)).

    Only then does ``logdet(basis^T Q basis)`` equal the pseudo-determinant
    ``log det*(Q) = log(n) + logdet(Q_{-0,-0})`` (matrix-tree identity). Excludes
    full-rank fields (IID: Q @ 1 = tau*1 != 0) and multi-constraint / weighted
    blocks (RW2 null = span(1,t); label extra-constraints), which fall back to
    the dense reduction.
    """
    if rows.shape[0] != 1:
        return False
    r = np.asarray(rows, dtype=float).ravel()
    norm = np.linalg.norm(r)
    if norm == 0:
        return False
    ones = np.ones_like(r)
    unit, ref = r / norm, ones / np.linalg.norm(ones)
    if not (np.allclose(unit, ref) or np.allclose(unit, -ref)):
        return False
    q1 = np.asarray(precision @ ones, dtype=float).ravel()
    scale = abs(precision).max() if precision.nnz else 1.0
    return np.allclose(q1, 0.0, atol=1e-8 * max(1.0, scale))


def _prior_logdet(model: CompiledLGM, spans: list[tuple[int, int, np.ndarray]]) -> float:
    """``logdet(basis^T Q_prior basis)`` as a per-block sum.

    Block-separable because every constraint row is confined to one block. A
    block with confined rows is intrinsic: its prior is rank-deficient, so reduce
    it onto ``null_space(rows)`` (SPD once the constraint kills the null mode).
    For the dominant case -- a connected intrinsic field (Besag, RW1, ...) pinned
    by a single unweighted sum-to-zero row -- that reduced logdet equals the
    pseudo-determinant, computed sparsely in near-linear time via the
    matrix-tree cofactor identity (``_is_connected_intrinsic``). Everything else
    (RW2, weighted or multi-row extra-constraints) keeps the dense reduction as
    a fallback. A block with no confined rows is full rank -- its sparse logdet
    is used directly.
    """
    a = np.asarray(model.constraints, dtype=float)
    total = 0.0
    for block, (start, stop, confined) in zip(model.blocks, spans, strict=True):
        q_b = block.precision
        if confined.any():
            rows = a[confined, start:stop]
            if _is_connected_intrinsic(rows, q_b):
                # Cofactor / matrix-tree: log det*(Q) = log(n) + logdet(Q_{-0,-0}).
                # Sparse + near-linear; the dense reduction below is the fallback.
                n_b = stop - start
                reduced = q_b.tocsr()[1:, 1:]
                logdet = np.log(n_b) + SparseSpdFactor(
                    reduced, f"intrinsic prior [{block.name}]"
                ).logdet
            else:
                # ponytail: dense reduction, O(n^3), for the residual cases only
                # (RW2, weighted or multi-row extra-constraints) whose blocks are
                # small in practice. A large field carrying an extra label
                # constraint would hit this ceiling; generalize via the k-index
                # cofactor (det*(Q) det(N_S^T N_S)/det(N^T N)) if that ever bites.
                basis_b = null_space(rows)
                reduced = basis_b.T @ q_b.toarray() @ basis_b
                _, logdet = _factor_positive_definite(reduced, f"reduced prior [{block.name}]")
        else:
            logdet = SparseSpdFactor(q_b.tocsr(), f"prior [{block.name}]").logdet
        total += logdet
    return total


def selected_inverse_diagonal(matrix) -> np.ndarray:
    """Diagonal of ``matrix⁻¹`` for a sparse SPD matrix — the exact selected
    inverse via Takahashi recursion on the SuperLU fill pattern.

    Symmetric-mode ``splu`` (``MMD_AT_PLUS_A`` + ``SymmetricMode`` +
    ``diag_pivot_thresh=0.0``) yields ``perm_r == perm_c`` and a unit-lower ``L``
    with ``U == D·Lᵀ``. A reverse column sweep on the below-diagonal fill pattern
    reconstructs exactly the selected-inverse entries needed for the diagonal.
    Off-pattern sub-block entries are true zeros (SuperLU drops only numerical
    zeros), so ``Sig.get(key, 0.0)`` is exact. Result is in the original ordering.
    """
    q_csc = matrix.tocsc()
    n = q_csc.shape[0]
    lu = splu(
        q_csc,
        permc_spec="MMD_AT_PLUS_A",
        options=dict(SymmetricMode=True),
        diag_pivot_thresh=0.0,
    )
    pc = lu.perm_c
    if not np.array_equal(lu.perm_r, pc):
        raise NumericalError("selected inversion expected a symmetric factorization")
    lower = lu.L.tocsc()
    diag_u = lu.U.diagonal().astype(float)
    indptr, indices, data = lower.indptr, lower.indices, lower.data.astype(float)
    below_rows: list = [None] * n
    below_l: list = [None] * n
    sig: dict = {}
    for i in range(n):
        seg = indices[indptr[i]:indptr[i + 1]]
        val = data[indptr[i]:indptr[i + 1]]
        mask = seg > i
        below_rows[i] = seg[mask]
        below_l[i] = val[mask]
    for i in range(n - 1, -1, -1):
        below = below_rows[i]
        below_vals = below_l[i]
        if len(below):
            k = len(below)
            sub = np.empty((k, k))
            for a in range(k):
                for b in range(k):
                    ra, rb = below[a], below[b]
                    lo, hi = (ra, rb) if ra <= rb else (rb, ra)
                    sub[a, b] = sig.get((hi, lo), 0.0)
            sig_below = -sub @ below_vals
            for a in range(k):
                sig[(below[a], i)] = sig_below[a]
            sig[(i, i)] = 1.0 / diag_u[i] - below_vals @ sig_below
        else:
            sig[(i, i)] = 1.0 / diag_u[i]
    return np.array([sig[(pc[i], pc[i])] for i in range(n)])


def sparse_constrained_gaussian(model: CompiledLGM) -> SparseFit:
    """Partitioned Schur solve, matching ``gaussian._fit_dense``.

    Partitions the latent columns into a sparse GMRF field block ``s`` and a
    small dense fixed block ``d`` (``_partition_blocks``), then solves the
    posterior precision ``Q + Z^T Z / sigma^2`` by a Schur complement on the
    dense block -- densifying only ``B`` (n_s x m), ``D`` and ``S`` (m x m),
    never the field. The same solve is exposed as ``apply_inverse`` and reused
    for the unconstrained mean and, when ``model.constraints`` is nonempty, the
    conditioning-by-kriging correction (Rue & Held 2005 sec 2.3.3).

    ponytail: assumes ``Q_sd == 0`` (block-diagonal prior + block-granular
    partition), so ``B = Z_s^T Z_d / sigma^2``. True for every LGM this path
    handles; a cross-block prior term would need adding here.
    """
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

    # Score g = Z^T r / sigma^2 (full-length, split per block).
    g_full = np.zeros(latent_size)
    g_full[sparse_index] = np.asarray(z_s.T @ residual).reshape(-1) / variance
    g_full[dense_index] = np.asarray(z_d.T @ residual).reshape(-1) / variance

    logdet_posterior = 0.0
    a_s = schur_factor = d_factor = b = None
    a_ss_matrix = None

    if n_s:
        q_ss = q[sparse_index][:, sparse_index]
        a_ss_matrix = (q_ss + (z_s.T @ z_s) / variance).tocsr()
        a_s = SparseSpdFactor(a_ss_matrix, "sparse posterior precision")
        logdet_posterior += a_s.logdet
    if m:
        d = q[dense_index][:, dense_index].toarray() + (z_d.T @ z_d).toarray() / variance
    if n_s and m:
        b = (z_s.T @ z_d).toarray() / variance  # n_s x m
        schur = d - b.T @ a_s.solve(b)  # m x m
        schur_factor, logdet_schur = _factor_positive_definite(schur, "dense Schur complement")
        logdet_posterior += logdet_schur
    elif m:
        d_factor, logdet_d = _factor_positive_definite(d, "dense posterior precision")
        logdet_posterior += logdet_d

    def apply_inverse(rhs: np.ndarray) -> np.ndarray:
        """``Q_post^-1 @ rhs`` via the Schur solve; ``rhs`` is 1-D or 2-D."""
        rhs = np.asarray(rhs, dtype=float)
        out = np.zeros_like(rhs)
        if n_s and m:
            v_s, v_d = rhs[sparse_index], rhs[dense_index]
            x_d = cho_solve(schur_factor, v_d - b.T @ a_s.solve(v_s))
            out[sparse_index] = a_s.solve(v_s - b @ x_d)
            out[dense_index] = x_d
        elif n_s:
            out[sparse_index] = a_s.solve(rhs[sparse_index])
        else:
            out[dense_index] = cho_solve(d_factor, rhs[dense_index])
        return out

    mean = apply_inverse(g_full)  # unconstrained posterior mean

    spans = _block_column_confinement(model)
    logdet_prior = _prior_logdet(model, spans)

    constraint_count = model.constraints.shape[0]
    w_constraint = None
    cap_factor_ref = None
    if constraint_count:
        # Conditioning by kriging: correct the mean and the two SPD-identity
        # determinant terms, logdet(basis^T Q_post basis) = logdet(Q_post)
        # + logdet(A Q_post^-1 A^T) - logdet(A A^T).
        a = np.asarray(model.constraints, dtype=float)
        e = np.asarray(model.constraint_rhs, dtype=float)
        w = apply_inverse(a.T)  # latent x c
        capacitance = a @ w  # A Q_post^-1 A^T
        cap_factor, logdet_cap = _factor_positive_definite(capacitance, "kriging capacitance")
        _, logdet_gram = _factor_positive_definite(a @ a.T, "constraint gram")
        logdet_posterior += logdet_cap - logdet_gram
        mean = mean - w @ cho_solve(cap_factor, a @ mean - e)
        w_constraint = w
        cap_factor_ref = cap_factor

        # Two-term quadratic (robust to the confounded near-singular Q_post):
        # (r0 - Z mu*)^T (r0 - Z mu*) / var + (mu* - nu)^T Q (mu* - nu), where
        # nu = argmin_{A x = e} x^T Q x (zero for homogeneous constraints).
        # ponytail: when e is nonzero AND a connected-intrinsic sum-to-zero row
        # is present, a_sp.T @ a_sp is a dense rank-1 n x n block, so this
        # augmented factor densifies. Correct, but defeats sparsity; nonzero-rhs
        # extra-constraints at scale are E-sparse-C.
        if np.any(e):
            a_sp = csr_matrix(a)
            aug = SparseSpdFactor((q + a_sp.T @ a_sp).tocsr(), "augmented prior precision")
            w_aug = aug.solve(a.T)
            nu = w_aug @ np.linalg.solve(a @ w_aug, e)
        else:
            nu = np.zeros(latent_size)
        prior_residual = mean - nu
        model_residual = residual - np.asarray(observed_design @ mean).reshape(-1)
        quadratic = float(
            model_residual @ model_residual / variance
            + prior_residual @ np.asarray(q @ prior_residual).reshape(-1)
        )
    else:
        quadratic = float(residual @ residual / variance - mean @ g_full)

    n_observed = int(np.count_nonzero(observed))
    log_marginal_likelihood = -0.5 * (
        n_observed * np.log(2 * np.pi * variance)
        - logdet_prior
        + logdet_posterior
        + quadratic
    )
    predictive_mean = np.asarray(model.offset + design @ mean).reshape(-1)

    posterior = SparsePosterior(
        latent_size=latent_size,
        sparse_index=sparse_index,
        dense_index=dense_index,
        a_ss=a_s,
        a_ss_matrix=a_ss_matrix,
        b=b,
        schur_factor=schur_factor,
        d_factor=d_factor,
        w_constraint=w_constraint,
        cap_factor=cap_factor_ref,
    )
    return SparseFit(
        mean=mean,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": n_observed,
            "constraint_count": int(constraint_count),
            "sparse_dimension": int(n_s),
            "dense_dimension": int(m),
        },
        posterior=posterior,
    )

"""Sørbye–Rue variance scaling shared by intrinsic GMRF structures."""

import numpy as np


def _sorbye_sparse_diag(structure) -> np.ndarray:
    """diag(structure⁺) for a connected intrinsic sparse structure (null =
    span(1)), via double-centering of the grounded reduced inverse. All
    sparse, near-linear (Takahashi selected inverse + one sparse solve).

    ponytail: single-connected-component ceiling -- correct only when
    null(structure) == span(1) exactly (checked by the caller via null_dim==1
    plus the SPD-after-deleting-node-0 assumption); a multi-component ICAR
    field would need per-component reduction instead (handled upstream by
    Besag, which still routes dense per-component blocks here).
    """
    from pylgm.inference.sparse import SparseSpdFactor, selected_inverse_diagonal

    q = structure.tocsc()
    n = q.shape[0]
    q_r = q[1:, 1:].tocsc()  # delete node 0 -> SPD
    diag_k = selected_inverse_diagonal(q_r)  # diag(K), size n-1
    s = SparseSpdFactor(q_r.tocsr(), "sorbye reduced").solve(np.ones(n - 1))
    t = float(s.sum())
    diag = np.empty(n)
    diag[0] = t / n**2
    diag[1:] = diag_k - (2.0 / n) * s + t / n**2
    return diag


def sorbye_rue_scale(structure: np.ndarray, null_dim: int) -> np.ndarray:
    """Rescale an intrinsic structure to geometric-mean marginal variance 1.

    ``structure`` is a symmetric PSD matrix with ``null_dim`` zero eigenvalues
    (RW1: 1, RW2: 2, one connected ICAR component: 1). The generalized marginal
    variances are the diagonal of the pseudo-inverse formed by dropping those
    ``null_dim`` smallest eigenvalues explicitly -- ``pinv``'s default tolerance
    leaks the near-zero eigenvalue back in as a huge spurious variance once the
    structure grows beyond a few nodes.

    When ``structure`` is sparse and ``null_dim == 1`` (a single connected
    intrinsic field), a sparse fast path replaces the dense ``eigh`` with a
    Takahashi selected inverse + one sparse solve, returning a sparse result.
    """
    from scipy.sparse import issparse

    if null_dim == 1 and issparse(structure):
        variances = _sorbye_sparse_diag(structure)
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError("Sørbye-Rue scaling produced a non-positive marginal variance")
        scale = float(np.exp(np.mean(np.log(variances))))
        return scale * structure  # sparse in -> sparse out

    eigenvalues, eigenvectors = np.linalg.eigh(structure)
    inverse = np.zeros_like(eigenvalues)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse[null_dim:] = 1.0 / eigenvalues[null_dim:]
    variances = np.einsum("ij,j,ij->i", eigenvectors, inverse, eigenvectors)
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("Sørbye-Rue scaling produced a non-positive marginal variance")
    factor = float(np.exp(np.mean(np.log(variances))))
    return factor * structure

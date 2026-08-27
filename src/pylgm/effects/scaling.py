"""Sørbye–Rue variance scaling shared by intrinsic GMRF structures."""

import numpy as np


def sorbye_rue_scale(structure: np.ndarray, null_dim: int) -> np.ndarray:
    """Rescale an intrinsic structure to geometric-mean marginal variance 1.

    ``structure`` is a symmetric PSD matrix with ``null_dim`` zero eigenvalues
    (RW1: 1, RW2: 2, one connected ICAR component: 1). The generalized marginal
    variances are the diagonal of the pseudo-inverse formed by dropping those
    ``null_dim`` smallest eigenvalues explicitly -- ``pinv``'s default tolerance
    leaks the near-zero eigenvalue back in as a huge spurious variance once the
    structure grows beyond a few nodes.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(structure)
    inverse = np.zeros_like(eigenvalues)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse[null_dim:] = 1.0 / eigenvalues[null_dim:]
    variances = np.einsum("ij,j,ij->i", eigenvectors, inverse, eigenvectors)
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("Sørbye-Rue scaling produced a non-positive marginal variance")
    factor = float(np.exp(np.mean(np.log(variances))))
    return factor * structure

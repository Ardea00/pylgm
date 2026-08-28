"""BYM2 spatial latent effect (Riebler et al. 2016)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix, hstack, identity
from scipy.sparse.csgraph import connected_components

from pylgm.effects.besag import _scaled_structure, _sparse_scaled_structure
from pylgm.effects.graph import design_from_graph, normalize_graph
from pylgm.ir.model import LatentBlock

_BYM2_AUGMENT_NODES = 1024


def bym2_spectrum(graph: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Return the eigenvectors and eigenvalues of ``P = pinv(R*)``.

    ``R*`` is the Sorbye-Rue scaled ICAR structure, so ``P`` is its constrained
    generalized inverse. Computed once; the BYM2 precision at any ``(tau, phi)``
    is then an O(n^2) reassembly.
    """
    nodes, w = normalize_graph(graph)
    # ponytail: dense pinv + eigh, O(n^3) once at compile time; fine for the
    # dense reference regime (hundreds of regions).
    structure = _scaled_structure(w, nodes, True)
    values, vectors = np.linalg.eigh(np.linalg.pinv(structure))
    return vectors, values


def bym2_precision(
    vectors: np.ndarray, values: np.ndarray, precision: float, phi: float
) -> csr_matrix:
    """Assemble ``tau * U diag(1 / (1 - phi + phi * gamma)) U^T``."""
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must lie in [0, 1); got {phi}")
    denominator = 1.0 - phi + phi * values
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise ValueError("BYM2 mixing produced a non-positive marginal variance")
    scaled = (vectors / denominator) @ vectors.T
    dense = precision * 0.5 * (scaled + scaled.T)  # symmetrize against round-off
    if not np.all(np.isfinite(dense)):
        raise ValueError("BYM2 precision must be finite")
    return csr_matrix(dense)


def _build_bym2_augmented(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    precision: float,
    phi: float,
) -> LatentBlock:
    """Augmented (x, u*) BYM2 block: 2n latent, sparse joint precision, one
    sum-to-zero per connected component on u* only. Reproduces the dense BYM2
    x-marginal exactly and scales (no eigendecomposition). Selected for large
    graphs; multi-component graphs (islands) are supported — each component
    carries its own u* sum-to-zero, and an isolated node contributes an
    unconstrained IID u* (its R* diagonal is 1, matching the dense path).
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must lie in [0, 1); got {phi}")
    nodes, w = normalize_graph(graph)
    n = len(nodes)
    rstar = _sparse_scaled_structure(w)  # per-component sparse R*, isolated -> 1
    a = 1.0 / (1.0 - phi)
    b = -np.sqrt(phi) / (1.0 - phi)
    d = phi / (1.0 - phi)
    ident = identity(n, format="csr")
    joint = (precision * bmat([[a * ident, b * ident],
                               [b * ident, rstar + d * ident]], format="csr")).tocsr()
    x_design = design_from_graph(nodes, frame, index)  # n_obs x n
    design = hstack([x_design, csr_matrix((x_design.shape[0], n))], format="csr")
    labels = tuple(nodes) + tuple(f"{node}__u" for node in nodes)
    _, membership = connected_components(w, directed=False)
    rows = []
    for component in range(membership.max() + 1):
        mask = membership == component
        if np.count_nonzero(mask) == 1:
            continue  # isolated IID u*: proper on its own, no sum-to-zero
        row = np.zeros(2 * n)
        row[n:][mask] = 1.0
        rows.append(row)
    constraints = np.array(rows) if rows else np.empty((0, 2 * n))
    return LatentBlock(name, labels, design, joint, constraints)


def build_bym2(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    precision: float,
    phi: float,
) -> LatentBlock:
    nodes, _ = normalize_graph(graph)
    if len(nodes) > _BYM2_AUGMENT_NODES:
        return _build_bym2_augmented(frame, name, index, graph, precision, phi)
    design = design_from_graph(nodes, frame, index)
    vectors, values = bym2_spectrum(graph)
    precision_matrix = bym2_precision(vectors, values, precision, phi)
    constraints = np.empty((0, len(nodes)))
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

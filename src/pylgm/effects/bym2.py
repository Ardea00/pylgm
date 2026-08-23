"""BYM2 spatial latent effect (Riebler et al. 2016)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pylgm.effects.besag import _scaled_structure
from pylgm.effects.graph import design_from_graph, normalize_graph
from pylgm.ir.model import LatentBlock


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


def build_bym2(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    precision: float,
    phi: float,
) -> LatentBlock:
    nodes, _ = normalize_graph(graph)
    design = design_from_graph(nodes, frame, index)
    vectors, values = bym2_spectrum(graph)
    precision_matrix = bym2_precision(vectors, values, precision, phi)
    constraints = np.empty((0, len(nodes)))
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

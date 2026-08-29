"""Spatial-autoregressive (SAR) and dynamic spatial panel (SDPD) latent effects.

Both are the same construction: a sparse operator ``M`` whose Gram matrix
``Q = precision * MᵀM`` is a symmetric, positive-definite (proper) latent
precision. Static SAR is the ``T = 1`` special case (``M = I - ρW``); the
dynamic panel stacks per-period blocks into a block-bidiagonal ``M`` (Task 5).
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.graph import design_from_graph
from pylgm.ir.model import LatentBlock


def _gram_precision(m: csr_matrix, precision: float) -> csr_matrix:
    """The proper latent precision ``precision * MᵀM`` (symmetric, PD)."""
    return csr_matrix(precision * (m.T @ m))


def _sar_operator(w: csr_matrix, rho: float) -> csr_matrix:
    """``I - ρW`` on a row-standardized directed ``W``."""
    n = w.shape[0]
    return (identity(n, format="csr") - rho * w).tocsr()


def build_sar(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    rho: float,
    precision: float,
) -> LatentBlock:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"SAR rho must lie strictly inside (-1, 1); got {rho}")
    nodes, w = normalize_directed_graph(graph)
    w = row_standardize(w)
    design = design_from_graph(nodes, frame, index)
    m = _sar_operator(w, rho)
    precision_matrix = _gram_precision(m, precision)
    constraints = np.empty((0, len(nodes)))
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

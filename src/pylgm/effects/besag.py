"""Besag / intrinsic CAR (ICAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from pylgm.effects.graph import design_from_graph, normalize_graph
from pylgm.ir.model import LatentBlock


def _scaled_structure(w: csr_matrix, nodes: tuple[str, ...], scale: bool) -> np.ndarray:
    """Return the (optionally Sørbye-Rue scaled) ICAR structure matrix ``R*``."""
    degree = np.asarray(w.sum(axis=1)).ravel()
    r = (diags(degree) - w).toarray()
    n_components, membership = connected_components(w, directed=False)
    isolated = [nodes[i] for i in range(len(degree)) if degree[i] == 0]
    if isolated:
        raise ValueError(
            "regions have no neighbours (isolated ICAR nodes are undefined; "
            f"model them as an IID effect instead): {isolated!r}"
        )
    if not scale:
        return r
    # ponytail: dense pinv per component, O(n_c^3); fine for the dense reference
    # regime (hundreds of regions). Use a sparse eigensolver if graphs grow to
    # thousands of nodes per component.
    scaled = r.copy()
    for component in range(n_components):
        index = np.where(membership == component)[0]
        block = r[np.ix_(index, index)]
        variances = np.diag(np.linalg.pinv(block))
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError("ICAR scaling produced a non-positive marginal variance")
        factor = float(np.exp(np.mean(np.log(variances))))
        scaled[np.ix_(index, index)] = factor * block
    return scaled


def _component_constraints(w: csr_matrix) -> np.ndarray:
    n_components, membership = connected_components(w, directed=False)
    rows = np.zeros((n_components, w.shape[0]))
    for component in range(n_components):
        rows[component, membership == component] = 1.0
    return rows


def build_besag(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    precision: float,
    scale: bool = True,
) -> LatentBlock:
    nodes, w = normalize_graph(graph)
    design = design_from_graph(nodes, frame, index)
    structure = _scaled_structure(w, nodes, scale)
    precision_matrix = csr_matrix(precision * structure)
    constraints = _component_constraints(w)
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

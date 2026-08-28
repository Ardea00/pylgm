"""Besag / intrinsic CAR (ICAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from pylgm.effects.graph import design_from_graph, normalize_graph
from pylgm.effects.scaling import sorbye_rue_scale
from pylgm.ir.model import LatentBlock


def _scaled_structure(w: csr_matrix, nodes: tuple[str, ...], scale: bool) -> np.ndarray:
    """Return the (optionally Sørbye-Rue scaled) ICAR structure matrix ``R*``.

    Isolated nodes (no neighbours = singleton connected components) have an
    undefined ICAR null, so each is treated as an independent unit-variance
    (IID) node: its structure diagonal is 1 and it is excluded from per-
    component scaling. An island then keeps a proper region effect with no
    spatial borrowing (R-INLA's ``adjust.for.con.comp`` default) instead of
    aborting the fit.
    """
    degree = np.asarray(w.sum(axis=1)).ravel()
    r = (diags(degree) - w).toarray()
    n_components, membership = connected_components(w, directed=False)
    isolated = np.flatnonzero(degree == 0)
    r[isolated, isolated] = 1.0  # IID unit variance; proper on its own
    if not scale:
        return r
    # ponytail: dense pinv per component, O(n_c^3); fine for the dense reference
    # regime (hundreds of regions). Use a sparse eigensolver if graphs grow to
    # thousands of nodes per component.
    scaled = r.copy()
    for component in range(n_components):
        index = np.where(membership == component)[0]
        if index.size == 1:
            continue  # isolated IID node: unit variance already, no null to drop
        block = r[np.ix_(index, index)]
        # A connected component's ICAR Laplacian has exactly one null eigenvalue
        # (the constant vector); drop it explicitly (see sorbye_rue_scale).
        scaled[np.ix_(index, index)] = sorbye_rue_scale(block, null_dim=1)
    return scaled


def _sparse_scaled_structure(w: csr_matrix) -> csr_matrix:
    """Sparse Sørbye-Rue scaled ICAR structure ``R*`` for the augmented path.

    Yields the same ``R*`` as ``_scaled_structure(..., scale=True)`` but never
    densifies the full ``n x n``: each connected component is scaled on its own
    sub-Laplacian via the sparse (Takahashi) Sørbye-Rue path, so it holds at
    network scale. Isolated nodes become an independent unit-variance (IID)
    diagonal entry, matching the dense builder.
    """
    from scipy.sparse import coo_matrix

    degree = np.asarray(w.sum(axis=1)).ravel()
    laplacian = (diags(degree) - w).tocsr()
    n_components, membership = connected_components(w, directed=False)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for component in range(n_components):
        index = np.flatnonzero(membership == component)
        if index.size == 1:
            rows.append(int(index[0]))
            cols.append(int(index[0]))
            data.append(1.0)
            continue
        block = laplacian[index][:, index].tocsc()  # connected -> SPD after grounding
        scaled = sorbye_rue_scale(block, null_dim=1).tocoo()
        rows.extend(index[scaled.row].tolist())
        cols.extend(index[scaled.col].tolist())
        data.extend(scaled.data.tolist())
    return coo_matrix((data, (rows, cols)), shape=w.shape).tocsr()


def _component_constraints(w: csr_matrix) -> np.ndarray:
    """One sum-to-zero row per connected component of size >= 2.

    Isolated nodes are IID (proper on their own), so they get no constraint --
    a singleton sum-to-zero would pin the region's effect to exactly 0.
    """
    n_components, membership = connected_components(w, directed=False)
    rows = [
        np.where(membership == component, 1.0, 0.0)
        for component in range(n_components)
        if np.count_nonzero(membership == component) > 1
    ]
    return np.array(rows) if rows else np.empty((0, w.shape[0]))


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

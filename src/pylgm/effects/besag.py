"""Besag / intrinsic CAR (ICAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from pylgm.effects.graph import normalize_graph
from pylgm.ir.model import LatentBlock


def _scaled_structure(w: csr_matrix, scale: bool) -> np.ndarray:
    """Return the (optionally Sørbye-Rue scaled) ICAR structure matrix ``R*``."""
    degree = np.asarray(w.sum(axis=1)).ravel()
    r = (diags(degree) - w).toarray()
    n_components, membership = connected_components(w, directed=False)
    singletons = [i for i in range(len(degree)) if degree[i] == 0]
    if singletons:
        raise ValueError(
            "regions have no neighbours (isolated ICAR nodes are undefined; "
            f"model them as an IID effect instead): index positions {singletons}"
        )
    if not scale:
        return r
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
    position = {node: column for column, node in enumerate(nodes)}
    observed = [str(value) for value in frame[index]]
    missing = sorted({value for value in observed if value not in position})
    if missing:
        raise ValueError(f"observed region(s) {missing!r} are not in the graph")
    structure = _scaled_structure(w, scale)
    precision_matrix = csr_matrix(precision * structure)
    constraints = _component_constraints(w)
    rows = np.arange(len(observed))
    columns = np.array([position[value] for value in observed])
    design = csr_matrix(
        (np.ones(len(observed)), (rows, columns)), shape=(len(observed), len(nodes))
    )
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

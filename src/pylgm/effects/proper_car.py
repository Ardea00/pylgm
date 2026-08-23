"""Proper conditional autoregressive (proper CAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from pylgm.effects.graph import normalize_graph
from pylgm.ir.model import LatentBlock


def _validity_interval(w: csr_matrix, degree: np.ndarray) -> tuple[float, float]:
    """Return the open interval (1/mu_min, 1/mu_max) of valid rho for D - rho W."""
    inv_sqrt = 1.0 / np.sqrt(degree)
    dense = w.toarray()
    normalized = (dense * inv_sqrt[:, None]) * inv_sqrt[None, :]
    normalized = 0.5 * (normalized + normalized.T)  # symmetrize against round-off
    mu = np.linalg.eigvalsh(normalized)
    mu_min = float(mu[0])
    mu_max = float(mu[-1])
    lower = 1.0 / mu_min if mu_min < 0 else float("-inf")
    upper = 1.0 / mu_max if mu_max > 0 else float("inf")
    return lower, upper


def build_proper_car(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    rho: float,
    precision: float,
) -> LatentBlock:
    nodes, w = normalize_graph(graph)
    degree = np.asarray(w.sum(axis=1)).ravel()
    isolated = [nodes[i] for i in range(len(degree)) if degree[i] == 0]
    if isolated:
        raise ValueError(
            "regions have no neighbours (proper CAR is singular at isolated nodes; "
            f"model them as an IID effect instead): {isolated!r}"
        )
    position = {node: column for column, node in enumerate(nodes)}
    observed = [str(value) for value in frame[index]]
    missing = sorted({value for value in observed if value not in position})
    if missing:
        raise ValueError(f"observed region(s) {missing!r} are not in the graph")
    lower, upper = _validity_interval(w, degree)
    if not lower < rho < upper:
        raise ValueError(
            f"rho must lie in the open interval ({lower:.6g}, {upper:.6g}) for "
            f"D - rho*W to be positive definite; got {rho}"
        )
    structure = (diags(degree) - rho * w).tocsr()
    precision_matrix = csr_matrix(precision * structure)
    constraints = np.empty((0, len(nodes)))
    rows = np.arange(len(observed))
    columns = np.array([position[value] for value in observed])
    design = csr_matrix(
        (np.ones(len(observed)), (rows, columns)), shape=(len(observed), len(nodes))
    )
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

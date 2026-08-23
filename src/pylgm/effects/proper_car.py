"""Proper conditional autoregressive (proper CAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from pylgm.effects.graph import design_from_graph, normalize_graph
from pylgm.ir.model import LatentBlock


def _validity_interval(
    w: csr_matrix, degree: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (lower, upper, mu_min, mu_max) for the valid-rho interval.

    ``D - rho W`` is positive definite iff ``rho in (1/mu_min, 1/mu_max)`` where
    ``mu`` are the eigenvalues of the normalized adjacency ``D^-1/2 W D^-1/2``.
    The extreme eigenvalues are returned too so the caller can gate on the
    positive-definiteness margin directly (robust to the round-off that makes
    ``1/mu_max`` land just above the true boundary rho=1).
    """
    # ponytail: dense eigvalsh of the normalized adjacency, O(n^3); fine for the
    # dense reference regime (hundreds of regions). Use scipy.sparse.linalg.eigsh
    # for the two extreme eigenvalues if graphs grow to thousands of nodes.
    inv_sqrt = 1.0 / np.sqrt(degree)
    dense = w.toarray()
    normalized = (dense * inv_sqrt[:, None]) * inv_sqrt[None, :]
    normalized = 0.5 * (normalized + normalized.T)  # symmetrize against round-off
    mu = np.linalg.eigvalsh(normalized)
    mu_min = float(mu[0])
    mu_max = float(mu[-1])
    lower = 1.0 / mu_min if mu_min < 0 else float("-inf")
    upper = 1.0 / mu_max if mu_max > 0 else float("inf")
    return lower, upper, mu_min, mu_max


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
    design = design_from_graph(nodes, frame, index)
    lower, upper, mu_min, mu_max = _validity_interval(w, degree)
    # Gate on the smallest eigenvalue of (I - rho N) = 1 - rho * mu_extreme
    # directly, with a small margin — robust to the float round-off that makes
    # the raw open-interval bound 1/mu land on the singular side of rho = +/-1.
    margin = 1.0 - rho * (mu_max if rho >= 0 else mu_min)
    if margin <= 1e-8:
        raise ValueError(
            f"rho must lie in the open interval ({lower:.6g}, {upper:.6g}) for "
            f"D - rho*W to be positive definite; got {rho}"
        )
    structure = (diags(degree) - rho * w).tocsr()
    precision_matrix = csr_matrix(precision * structure)
    constraints = np.empty((0, len(nodes)))
    return LatentBlock(name, nodes, design, precision_matrix, constraints)

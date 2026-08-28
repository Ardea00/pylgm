"""Proper conditional autoregressive (proper CAR) spatial latent effect."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

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
    n = len(degree)
    isolated = np.flatnonzero(degree == 0)
    if isolated.size:
        raise ValueError(
            "proper CAR rho interval is undefined for isolated nodes "
            f"(regions have no neighbours): node indices {isolated.tolist()!r}"
        )
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = diags(inv_sqrt) @ w @ diags(inv_sqrt)
    normalized = 0.5 * (normalized + normalized.T)  # symmetrize against round-off
    if n <= 2:
        # ponytail: eigsh needs k<n-1; tiny graphs go dense — trivially cheap.
        mu = np.linalg.eigvalsh(normalized.toarray())
        mu_min = float(mu[0])
        mu_max = float(mu[-1])
    else:
        mu_min = float(eigsh(normalized, k=1, which="SA", return_eigenvectors=False)[0])
        mu_max = float(eigsh(normalized, k=1, which="LA", return_eigenvectors=False)[0])
    lower = 1.0 / mu_min if mu_min < 0 else float("-inf")
    upper = 1.0 / mu_max if mu_max > 0 else float("inf")
    return lower, upper, mu_min, mu_max


def car_rho_interval(graph: Mapping) -> tuple[float, float]:
    """Return the open ``(lower, upper)`` interval of valid rho for a graph.

    ``D - rho W`` is positive definite exactly for rho in this interval; shared
    by the compiler so it can build rho's transform/bounds without duplicating
    the eigenvalue math in ``_validity_interval``.
    """
    nodes, w = normalize_graph(graph)
    degree = np.asarray(w.sum(axis=1)).ravel()
    isolated = [nodes[i] for i in range(len(degree)) if degree[i] == 0]
    if isolated:
        raise ValueError(
            "proper CAR rho interval is undefined for isolated nodes "
            f"(regions have no neighbours): {isolated!r}"
        )
    lower, upper, _, _ = _validity_interval(w, degree)
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
            "use Besag or BYM2, which treat an isolated region as an independent "
            f"IID unit): {isolated!r}"
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

"""Spatial-autoregressive (SAR) and dynamic spatial panel (SDPD) latent effects.

Both are the same construction: a sparse operator ``M`` whose Gram matrix
``Q = precision * MᵀM`` is a symmetric, positive-definite (proper) latent
precision. Static SAR is the ``T = 1`` special case (``M = I - ρW``); the
dynamic panel stacks per-period blocks into a block-bidiagonal ``M`` (Task 5).
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix, identity

from pylgm.effects.directed_graph import (
    graphs_by_label,
    normalize_directed_graph,
    reindex_onto,
    row_standardize,
    sorted_time_keys,
)
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


def _panel_networks(
    graphs: Mapping,
) -> tuple[tuple[str, ...], tuple[str, ...], list[csr_matrix]]:
    """Align every period's directed network onto one balanced unit universe.

    ``units`` is the sorted union of all periods' nodes; ``times`` the sorted
    string keys of ``graphs``. Each returned ``W_t`` is ``n x n`` on the shared
    ``units`` ordering and row-standardized (a unit absent from period ``t`` is
    an all-zero row that period).
    """
    if not isinstance(graphs, Mapping) or not graphs:
        raise ValueError("graphs must be a non-empty mapping of time -> graph")
    by_label = graphs_by_label(graphs)
    times = sorted_time_keys(graphs)
    per_period: dict[str, tuple[tuple[str, ...], csr_matrix]] = {}
    union: set[str] = set()
    for t in times:
        nodes_t, w_t = normalize_directed_graph(dict(by_label[t]))
        per_period[t] = (nodes_t, w_t)
        union |= set(nodes_t)
    units = tuple(sorted(union))
    matrices = [
        row_standardize(reindex_onto(*per_period[t], units)) for t in times
    ]
    return units, times, matrices


def _sdpd_operator(
    ws: list[csr_matrix], rho: float, gamma: float, eta: float
) -> csr_matrix:
    """Block-bidiagonal SDPD operator ``M`` over the stacked field ``(x_1..x_T)``.

    Diagonal block ``A_t = I - ρW_t``; sub-diagonal block ``-B_t = -(γI + ηW_t)``
    for ``t >= 1``. The first block row is ``A_0`` only (conditional-on-initial).
    """
    t_count = len(ws)
    n = ws[0].shape[0]
    ident = identity(n, format="csr")
    blocks: list[list[object]] = [[None] * t_count for _ in range(t_count)]
    for t in range(t_count):
        blocks[t][t] = (ident - rho * ws[t]).tocsr()
        if t >= 1:
            blocks[t][t - 1] = -(gamma * ident + eta * ws[t]).tocsr()
    return bmat(blocks, format="csr")


def _panel_design(
    frame: pd.DataFrame,
    unit: str,
    time: str,
    units: tuple[str, ...],
    times: tuple[str, ...],
) -> csr_matrix:
    """Time-major one-hot: cell = ``time_pos * n_units + unit_pos``."""
    if unit not in frame.columns:
        raise ValueError(f"unit column {unit!r} not found")
    if time not in frame.columns:
        raise ValueError(f"time column {time!r} not found")
    upos = {u: i for i, u in enumerate(units)}
    tpos = {t: j for j, t in enumerate(times)}
    n = len(units)
    obs_u = [str(v) for v in frame[unit]]
    obs_t = [str(v) for v in frame[time]]
    missing_u = sorted({v for v in obs_u if v not in upos})
    if missing_u:
        raise ValueError(f"observed unit(s) {missing_u!r} are not in the network")
    missing_t = sorted({v for v in obs_t if v not in tpos})
    if missing_t:
        raise ValueError(f"observed time(s) {missing_t!r} have no network graph")
    cells = np.array([tpos[t] * n + upos[u] for u, t in zip(obs_u, obs_t)])
    return csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), cells)),
        shape=(len(frame), n * len(times)),
    )


def build_dynamic_spatial_panel(
    frame: pd.DataFrame,
    name: str,
    unit: str,
    time: str,
    graphs: Mapping,
    rho: float,
    gamma: float,
    eta: float,
    precision: float,
) -> LatentBlock:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"SDPD rho must lie strictly inside (-1, 1); got {rho}")
    units, times, ws = _panel_networks(graphs)
    n = len(units)
    m = _sdpd_operator(ws, rho, gamma, eta)
    precision_matrix = _gram_precision(m, precision)
    design = _panel_design(frame, unit, time, units, times)
    labels = tuple(f"{u}@{t}" for t in times for u in units)
    constraints = np.empty((0, n * len(times)))
    return LatentBlock(name, labels, design, precision_matrix, constraints)

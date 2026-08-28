# src/pylgm/effects/spacetime.py
"""Knorr-Held space-time interaction effect (Types I-IV)."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.csgraph import connected_components

from pylgm.data.scalars import ordered_observed_levels
from pylgm.effects.besag import _scaled_structure
from pylgm.effects.graph import normalize_graph
from pylgm.effects.random_walk import difference_operator
from pylgm.effects.scaling import sorbye_rue_scale
from pylgm.ir.model import LatentBlock

_SPACE_STRUCTURED = {"III", "IV"}
_TIME_STRUCTURED = {"II", "IV"}


def _rw_structure(level_count: int, order: int, scale: bool) -> np.ndarray:
    """Sørbye-Rue-scaled temporal RW structure ``DᵀD`` (order 1 or 2)."""
    # Reuse the shared difference operator (MIDAS/S1 exposed it in random_walk).
    difference = difference_operator(level_count, order)
    r = (difference.T @ difference).toarray()
    return sorbye_rue_scale(r, null_dim=order) if scale else r


def _space_null_basis(interaction: str, w, area_count: int) -> np.ndarray:
    """Columns spanning null(K_s): empty for I/II, component indicators for III/IV."""
    if interaction not in _SPACE_STRUCTURED:
        return np.zeros((area_count, 0))
    n_components, membership = connected_components(w, directed=False)
    basis = np.zeros((area_count, n_components))
    for component in range(n_components):
        basis[membership == component, component] = 1.0
    return basis


def _time_null_basis(interaction: str, order: int, time_count: int) -> np.ndarray:
    """Columns spanning null(K_t): empty for I/III, [1] for RW1, [1, k] for RW2."""
    if interaction not in _TIME_STRUCTURED:
        return np.zeros((time_count, 0))
    columns = [np.ones(time_count)]
    if order == 2:
        coordinate = np.arange(time_count, dtype=float)
        columns.append(coordinate - coordinate.mean())
    return np.column_stack(columns)


def _interaction_constraints(
    space_null: np.ndarray, time_null: np.ndarray, area_count: int, time_count: int
) -> np.ndarray:
    """Orthonormal basis of null(K_s (x) K_t), as constraint rows (r, S*T).

    null(K_s (x) K_t) = null(K_s) (x) R^T  +  R^S (x) null(K_t). Assemble the
    two Kronecker spans, take an orthonormal basis of their combined column span
    by SVD (dropping the 1_s (x) 1_t overlap counted twice), and transpose.
    """
    parts = []
    if space_null.shape[1]:
        parts.append(np.kron(space_null, np.eye(time_count)))
    if time_null.shape[1]:
        parts.append(np.kron(np.eye(area_count), time_null))
    if not parts:
        return np.zeros((0, area_count * time_count))
    b = np.hstack(parts)
    u, singular, _ = np.linalg.svd(b, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10)) if singular.size else 0
    return u[:, :rank].T


def build_spacetime(
    frame: pd.DataFrame,
    name: str,
    space: str,
    time: str,
    graph: Mapping | None,
    interaction: str,
    order: int,
    precision: float,
    scale: bool = True,
) -> LatentBlock:
    if space not in frame.columns:
        raise ValueError(f"space column {space!r} not found")
    if time not in frame.columns:
        raise ValueError(f"time column {time!r} not found")

    # Area universe: graph nodes when a graph is given, else observed levels.
    if graph is not None:
        areas, w = normalize_graph(dict(graph))
    else:
        areas = tuple(map(str, ordered_observed_levels(frame[space])))
        w = None
    times = tuple(map(str, ordered_observed_levels(frame[time])))
    S, T = len(areas), len(times)
    if interaction in _TIME_STRUCTURED and T <= order:
        raise ValueError(f"{name} type {interaction} requires more than {order} time levels")

    # Kronecker precision factors.
    if interaction in _SPACE_STRUCTURED:
        # ponytail: reject isolated areas here rather than inherit Besag's
        # graceful IID singleton. The space null-basis counts one indicator per
        # component, so an IID (nulless) singleton would be over-constrained.
        # Graceful isolated space-time areas are a separate slice.
        isolated = [areas[i] for i in np.flatnonzero(np.asarray(w.sum(axis=1)).ravel() == 0)]
        if isolated:
            raise ValueError(
                f"{name} type {interaction} has isolated area(s) with no neighbours "
                f"(unsupported in space-structured interactions): {isolated!r}"
            )
        k_s = csr_matrix(_scaled_structure(w, areas, scale))
    else:
        k_s = identity(S, format="csr")
    if interaction in _TIME_STRUCTURED:
        k_t = csr_matrix(_rw_structure(T, order, scale))
    else:
        k_t = identity(T, format="csr")
    precision_matrix = csr_matrix(precision * kron(k_s, k_t, format="csr"))

    # Area-major one-hot design; unseen area/time levels are a hard error.
    area_pos = {area: i for i, area in enumerate(areas)}
    time_pos = {t: j for j, t in enumerate(times)}
    observed_area = [str(v) for v in frame[space]]
    observed_time = [str(v) for v in frame[time]]
    missing_area = sorted({v for v in observed_area if v not in area_pos})
    if missing_area:
        raise ValueError(f"observed {space!r} level(s) {missing_area!r} not in the area universe")
    cells = np.array(
        [area_pos[a] * T + time_pos[t] for a, t in zip(observed_area, observed_time)]
    )
    design = csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), cells)), shape=(len(frame), S * T)
    )

    space_null = _space_null_basis(interaction, w, S)
    time_null = _time_null_basis(interaction, order, T)
    constraints = _interaction_constraints(space_null, time_null, S, T)

    labels = tuple(f"{area}|{t}" for area in areas for t in times)
    return LatentBlock(name, labels, design, precision_matrix, constraints)

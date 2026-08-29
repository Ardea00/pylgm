"""Directed-network adjacency for spatial-autoregressive (SAR/SDPD) effects.

Unlike ``pylgm.effects.graph.normalize_graph`` these helpers do NOT require a
symmetric graph: ``W[i, j]`` is the influence of node ``j`` on node ``i`` (row
``i`` lists the influencers of ``i``), which is exactly the directed economic
relationship a symmetrized CAR discards.
"""

from collections.abc import Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags

from pylgm.effects.graph import _parse_neighbours


def normalize_directed_graph(
    graph: Mapping,
) -> tuple[tuple[str, ...], csr_matrix]:
    """Return sorted string node labels and a directed adjacency ``W``.

    The node set is the union of keys and referenced neighbours, so a pure sink
    (only ever a neighbour) still gets a row/column. Reuses the shared weight
    parser, so non-finite / non-positive weights are rejected identically to the
    undirected path. Self-loops are rejected.
    """
    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("graph must be a non-empty mapping of node -> neighbours")
    adjacency: dict[str, dict[str, float]] = {}
    referenced: set[str] = set()
    for node, neighbours in graph.items():
        key = str(node)
        adjacency.setdefault(key, {})
        for neighbour, weight in _parse_neighbours(key, neighbours):
            if neighbour == key:
                raise ValueError(f"graph has a self-loop at node {key!r}")
            adjacency[key][neighbour] = weight
            referenced.add(neighbour)
    nodes = tuple(sorted(set(adjacency) | referenced))
    position = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for node, neighbours in adjacency.items():
        for neighbour, weight in neighbours.items():
            rows.append(position[node])
            cols.append(position[neighbour])
            data.append(weight)
    w = csr_matrix(
        (np.array(data, dtype=float), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    return nodes, w


def row_standardize(w: csr_matrix) -> csr_matrix:
    """Scale each row of ``w`` to sum to 1; all-zero rows (sinks) stay zero."""
    w = w.tocsr().astype(float)
    sums = np.asarray(w.sum(axis=1)).ravel()
    scale = np.divide(1.0, sums, out=np.zeros_like(sums), where=sums != 0.0)
    return (diags(scale) @ w).tocsr()


def canonical_directed_graph(graph: Mapping) -> tuple[tuple[str, tuple], ...]:
    """Immutable, canonically ordered directed graph (frozen-dataclass safe).

    ``dict()`` on the result reconstructs a mapping ``normalize_directed_graph``
    accepts again. Unweighted graphs store sorted bare labels; weighted graphs
    store sorted ``(label, weight)`` pairs so weights survive the freeze. A pure
    sink is stored with an empty neighbour tuple so it is preserved.
    """
    nodes, w = normalize_directed_graph(graph)
    weighted = bool(w.data.size) and not bool(np.all(w.data == 1.0))
    result = []
    for i, node in enumerate(nodes):
        columns = w.indices[w.indptr[i] : w.indptr[i + 1]]
        weights = w.data[w.indptr[i] : w.indptr[i + 1]]
        if weighted:
            neighbours: tuple = tuple(
                sorted((nodes[j], float(v)) for j, v in zip(columns, weights))
            )
        else:
            neighbours = tuple(sorted(nodes[j] for j in columns))
        result.append((node, neighbours))
    return tuple(result)

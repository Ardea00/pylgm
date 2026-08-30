"""Directed-network adjacency for spatial-autoregressive (SAR/SDPD) effects.

Unlike ``pylgm.effects.graph.normalize_graph`` these helpers do NOT require a
symmetric graph: ``W[i, j]`` is the influence of node ``j`` on node ``i`` (row
``i`` lists the influencers of ``i``), which is exactly the directed economic
relationship a symmetrized CAR discards.
"""

from collections.abc import Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags

from pylgm.effects.graph import _parse_neighbours, freeze_adjacency


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
    return freeze_adjacency(*normalize_directed_graph(graph))


def reindex_onto(
    nodes: tuple[str, ...], w: csr_matrix, units: tuple[str, ...]
) -> csr_matrix:
    """Re-express ``W`` (indexed by ``nodes``) on the wider ``units`` ordering.

    Units absent from ``nodes`` become all-zero rows/columns. Shared by the
    panel builder and the forecaster, which both align a period's own network
    onto the panel's balanced unit universe.
    """
    position = {u: i for i, u in enumerate(units)}
    coo = w.tocoo()
    rows = [position[nodes[i]] for i in coo.row]
    cols = [position[nodes[j]] for j in coo.col]
    return csr_matrix((coo.data, (rows, cols)), shape=(len(units), len(units)))


def graphs_by_label(graphs: Mapping) -> dict[str, object]:
    """Key ``graphs`` by ``str(period)``, rejecting labels that collide.

    Period keys reach the panel builders as ints, strings, or timestamps, but
    the latent labels and the design lookup are string-keyed. Two keys with the
    same ``str()`` would silently drop a period's network, so that is an error.
    """
    by_label: dict[str, object] = {}
    for key, graph in graphs.items():
        label = str(key)
        if label in by_label:
            raise ValueError(f"graphs has two period keys that stringify to {label!r}")
        by_label[label] = graph
    return by_label


def sorted_time_keys(keys) -> tuple[str, ...]:
    """Period labels sorted numerically when all parse as numbers, else lexically.

    Guards against lexicographic mis-ordering of integer period labels
    (``"10" < "2"`` as strings), which would silently mis-specify the SDPD
    temporal adjacency for panels keyed 1..12 or any period >= 10.
    """
    labels = [str(k) for k in keys]
    try:
        return tuple(sorted(labels, key=lambda s: (float(s), s)))
    except ValueError:
        return tuple(sorted(labels))

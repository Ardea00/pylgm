"""Neighbourhood-graph adjacency for spatial (CAR) effects."""

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def design_from_graph(
    nodes: tuple[str, ...], frame: pd.DataFrame, index: str
) -> csr_matrix:
    """One-hot design mapping each observed row to its graph-node column.

    The latent domain is the graph ``nodes`` (canonical order); every observed
    value must be one of them, else a ``ValueError`` names the missing regions.
    Shared by the spatial-effect builders (Besag, proper CAR).
    """
    position = {node: column for column, node in enumerate(nodes)}
    observed = [str(value) for value in frame[index]]
    missing = sorted({value for value in observed if value not in position})
    if missing:
        raise ValueError(f"observed region(s) {missing!r} are not in the graph")
    rows = np.arange(len(observed))
    columns = np.array([position[value] for value in observed])
    return csr_matrix(
        (np.ones(len(observed)), (rows, columns)), shape=(len(observed), len(nodes))
    )


def canonical_graph(
    graph: Mapping[object, Sequence[object]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate a graph and return an immutable, canonically ordered form.

    Spatial effect specs are frozen dataclasses, so they cannot hold a mutable
    mapping and stay hashable. ``dict()`` on the result reconstructs a neighbour
    mapping that ``normalize_graph`` accepts again.
    """
    nodes, w = normalize_graph(graph)
    return tuple(
        (node, tuple(sorted(nodes[j] for j in w.indices[w.indptr[i] : w.indptr[i + 1]])))
        for i, node in enumerate(nodes)
    )


def normalize_graph(
    graph: Mapping[object, Sequence[object]],
) -> tuple[tuple[str, ...], csr_matrix]:
    """Return canonically sorted string node labels and a symmetric 0/1 adjacency.

    The node set is the mapping's keys (stringified). Validates that the graph is
    non-empty, has no self-loops, references only known nodes, and is symmetric.
    """
    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("graph must be a non-empty mapping of node -> neighbours")
    adjacency: dict[str, set[str]] = {}
    for node, neighbours in graph.items():
        key = str(node)
        if isinstance(neighbours, (str, bytes)):
            raise ValueError(f"neighbours of {key!r} must be a sequence of nodes")
        adjacency.setdefault(key, set())
        for neighbour in neighbours:
            adjacency[key].add(str(neighbour))
    nodes = tuple(sorted(adjacency))
    node_set = set(nodes)
    for node, neighbours in adjacency.items():
        if node in neighbours:
            raise ValueError(f"graph has a self-loop at node {node!r}")
        unknown = neighbours - node_set
        if unknown:
            raise ValueError(
                f"neighbour(s) {sorted(unknown)!r} of {node!r} are not a node in the graph"
            )
    for node in nodes:
        for neighbour in adjacency[node]:
            if node not in adjacency[neighbour]:
                raise ValueError(
                    f"graph is not symmetric: {neighbour!r} lists {node!r} "
                    f"or vice versa is missing"
                )
    position = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    for node in nodes:
        for neighbour in adjacency[node]:
            rows.append(position[node])
            cols.append(position[neighbour])
    data = np.ones(len(rows), dtype=float)
    w = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return nodes, w


def load_graph_file(path: str | os.PathLike) -> dict[str, list[str]]:
    """Parse an R-INLA / latte ``.graph`` file into a neighbour dict.

    Format: first token is the node count ``n``; then for each node a line
    ``id k nb_1 ... nb_k`` (1-indexed ids). Returns string-keyed neighbours.
    """
    tokens = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            tokens.extend(line.split())
    if not tokens:
        raise ValueError("graph file is empty")
    cursor = 0

    def take_int(what: str) -> int:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError(f"graph file ended while reading {what}")
        token = tokens[cursor]
        cursor += 1
        try:
            return int(token)
        except ValueError as error:
            raise ValueError(f"graph file has a non-integer {what}: {token!r}") from error

    count = take_int("node count")
    if count <= 0:
        raise ValueError(f"graph file node count must be positive, got {count}")
    graph: dict[str, list[str]] = {}
    for _ in range(count):
        node = take_int("node id")
        if not 1 <= node <= count:
            raise ValueError(f"graph file node id {node} out of range 1..{count}")
        degree = take_int("neighbour count")
        if degree < 0:
            raise ValueError(f"graph file neighbour count must be non-negative, got {degree}")
        neighbours = []
        for _ in range(degree):
            neighbour = take_int("neighbour id")
            if not 1 <= neighbour <= count:
                raise ValueError(
                    f"graph file neighbour id {neighbour} out of range 1..{count}"
                )
            neighbours.append(str(neighbour))
        graph[str(node)] = neighbours
    if cursor != len(tokens):
        raise ValueError("graph file has trailing tokens beyond the declared nodes")
    if len(graph) != count:
        raise ValueError("graph file has duplicate or missing node ids")
    return graph

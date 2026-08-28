"""Neighbourhood-graph adjacency for spatial (CAR) effects."""

import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real

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
) -> tuple[tuple[str, tuple], ...]:
    """Validate a graph and return an immutable, canonically ordered form.

    Spatial effect specs are frozen dataclasses, so they cannot hold a mutable
    mapping and stay hashable. ``dict()`` on the result reconstructs a neighbour
    mapping that ``normalize_graph`` accepts again. When the graph is unweighted
    (every edge weight exactly ``1.0``) each node's neighbours are a sorted tuple
    of bare labels, byte-identical to the historical output; otherwise they are
    a sorted tuple of ``(label, weight)`` pairs so weights survive the freeze.
    """
    nodes, w = normalize_graph(graph)
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


def _valid_weight(key: str, weight: object) -> float:
    """Coerce and validate one edge weight: a finite, strictly positive real."""
    try:
        value = float(weight)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"neighbour weight for node {key!r} must be a real number, got {weight!r}"
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"neighbour weight for node {key!r} must be finite and > 0, got {value}"
        )
    return value


def _parse_neighbours(key: str, neighbours: object) -> list[tuple[str, float]]:
    """Read one node's neighbours value into ``[(label, weight), ...]``.

    Accepts a ``{neighbour: weight}`` mapping, or a sequence whose elements are
    each a bare label (weight ``1.0``) or a ``(label, weight)`` pair. A ``str``/
    ``bytes`` neighbours value is rejected (it is not a sequence of nodes).
    """
    if isinstance(neighbours, Mapping):
        items: list[tuple[object, object]] = list(neighbours.items())
    elif isinstance(neighbours, (str, bytes)):
        raise ValueError(f"neighbours of {key!r} must be a sequence of nodes")
    else:
        items = []
        for element in neighbours:  # type: ignore[assignment]
            if (
                not isinstance(element, (str, bytes))
                and isinstance(element, Sequence)
                and len(element) == 2
                and isinstance(element[1], Real)
            ):
                items.append((element[0], element[1]))
            else:
                items.append((element, 1.0))
    return [(str(label), _valid_weight(key, weight)) for label, weight in items]


def normalize_graph(
    graph: Mapping[object, Sequence[object]],
) -> tuple[tuple[str, ...], csr_matrix]:
    """Return canonically sorted string node labels and a symmetric adjacency.

    The node set is the mapping's keys (stringified). Edges may carry weights
    (see ``_parse_neighbours``); an unweighted graph yields an all-ones ``W``,
    identical to before. Validates that the graph is non-empty, has no
    self-loops, references only known nodes, uses finite positive weights, and
    is symmetric (each edge present both ways with equal weight within
    ``rel_tol=1e-9, abs_tol=1e-12``; the stored value is the mean, to absorb
    float round-off).
    """
    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("graph must be a non-empty mapping of node -> neighbours")
    adjacency: dict[str, dict[str, float]] = {}
    for node, neighbours in graph.items():
        key = str(node)
        adjacency.setdefault(key, {})
        for neighbour, weight in _parse_neighbours(key, neighbours):
            adjacency[key][neighbour] = weight
    nodes = tuple(sorted(adjacency))
    node_set = set(nodes)
    for node, neighbours in adjacency.items():
        if node in neighbours:
            raise ValueError(f"graph has a self-loop at node {node!r}")
        unknown = set(neighbours) - node_set
        if unknown:
            raise ValueError(
                f"neighbour(s) {sorted(unknown)!r} of {node!r} are not a node in the graph"
            )
    position = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for node in nodes:
        for neighbour, w_ij in adjacency[node].items():
            reverse = adjacency[neighbour]
            if node not in reverse:
                raise ValueError(
                    f"graph is not symmetric: {node!r} lists {neighbour!r} "
                    f"but the reverse edge is missing"
                )
            w_ji = reverse[node]
            if not math.isclose(w_ij, w_ji, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"graph is not symmetric: edge {node!r}-{neighbour!r} has "
                    f"unequal weights {w_ij} and {w_ji}"
                )
            rows.append(position[node])
            cols.append(position[neighbour])
            data.append(0.5 * (w_ij + w_ji))
    w = csr_matrix(
        (np.array(data, dtype=float), (rows, cols)), shape=(len(nodes), len(nodes))
    )
    return nodes, w


def load_graph_file(path: str | os.PathLike) -> dict:
    """Parse a neighbour graph from a file, dispatched on the extension.

    ``.json``: a JSON object ``{node: [neighbour, ...]}`` -- or, for a weighted
    graph, ``{node: {neighbour: weight}}`` -- exactly the shape the Python
    ``graph=`` argument accepts. Returned unchanged for the effect builders to
    validate, so an existing JSON adjacency (e.g. an example's ``graph.json``)
    is reusable from the YAML frontend without conversion.

    Otherwise (``.graph`` and anything else): the R-INLA / latte format, where
    the first token is the node count ``n``, then for each node a line
    ``id k nb_1 ... nb_k`` (1-indexed ids). Returns string-keyed neighbours.
    """
    if os.fspath(path).lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            graph = json.load(handle)
        if not isinstance(graph, dict):
            raise ValueError("JSON graph file must be an object mapping node -> neighbours")
        return graph
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

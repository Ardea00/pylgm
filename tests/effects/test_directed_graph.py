import numpy as np
import pytest

from pylgm.effects.directed_graph import (
    canonical_directed_graph,
    normalize_directed_graph,
    row_standardize,
)


def test_asymmetric_graph_is_accepted():
    # a -> b but not b -> a: normalize_graph would reject this; directed keeps it.
    nodes, w = normalize_directed_graph({"a": ["b"], "b": []})
    assert nodes == ("a", "b")
    dense = w.toarray()
    assert dense[0, 1] == 1.0  # b influences a (row a, col b)
    assert dense[1, 0] == 0.0


def test_pure_sink_node_gets_a_column():
    # c is only ever a neighbour, never a key.
    nodes, w = normalize_directed_graph({"a": ["c"]})
    assert nodes == ("a", "c")
    assert w.shape == (2, 2)


def test_weighted_edges_preserved():
    nodes, w = normalize_directed_graph({"a": {"b": 2.0}, "b": {"a": 0.5}})
    dense = w.toarray()
    assert dense[0, 1] == 2.0
    assert dense[1, 0] == 0.5


def test_self_loop_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        normalize_directed_graph({"a": ["a"]})


def test_row_standardize_rows_sum_to_one():
    _, w = normalize_directed_graph({"a": {"b": 3.0, "c": 1.0}, "b": ["a"], "c": []})
    r = row_standardize(w).toarray()
    assert np.isclose(r[0].sum(), 1.0)
    assert np.isclose(r[0, 1], 0.75)  # b: 3/(3+1)
    assert np.isclose(r[0, 2], 0.25)  # c: 1/(3+1)


def test_row_standardize_zero_row_stays_zero():
    _, w = normalize_directed_graph({"a": ["b"], "b": []})  # b is a sink: zero row
    r = row_standardize(w).toarray()
    assert np.allclose(r[1], 0.0)


def test_canonical_round_trips_through_normalize():
    graph = {"a": {"b": 2.0}, "b": ["c"], "c": []}
    canonical = canonical_directed_graph(graph)
    assert isinstance(canonical, tuple)
    hash(canonical)  # frozen-dataclass-safe
    nodes, w = normalize_directed_graph(dict(canonical))
    assert nodes == ("a", "b", "c")
    assert w.toarray()[0, 1] == 2.0

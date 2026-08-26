import numpy as np
import pytest

from pylgm.effects.graph import canonical_graph, normalize_graph


def _dense(graph):
    _, w = normalize_graph(graph)
    return w.toarray()


def test_unweighted_is_all_ones_and_symmetric():
    nodes, w = normalize_graph({"a": ["b"], "b": ["a"]})
    assert nodes == ("a", "b")
    assert np.array_equal(w.toarray(), np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_mapping_form_carries_weights():
    w = _dense({"a": {"b": 2.5}, "b": {"a": 2.5}})
    assert w[0, 1] == 2.5 and w[1, 0] == 2.5


def test_pair_and_mixed_sequence_forms_agree_with_mapping():
    m = _dense({"a": {"b": 2.0, "c": 1.0}, "b": {"a": 2.0}, "c": {"a": 1.0}})
    p = _dense({"a": [("b", 2.0), ("c", 1.0)], "b": [("a", 2.0)], "c": [("a", 1.0)]})
    x = _dense({"a": ["c", ("b", 2.0)], "b": [("a", 2.0)], "c": ["a"]})
    assert np.array_equal(m, p) and np.array_equal(m, x)


def test_weighted_degree_is_sum_of_weights():
    _, w = normalize_graph({"a": {"b": 2.0, "c": 3.0}, "b": {"a": 2.0}, "c": {"a": 3.0}})
    degree = np.asarray(w.sum(axis=1)).ravel()
    assert degree[0] == pytest.approx(5.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_nonpositive_or_nonfinite_weight_rejected(bad):
    with pytest.raises(ValueError, match="weight"):
        normalize_graph({"a": {"b": bad}, "b": {"a": bad}})


def test_asymmetric_weight_rejected():
    with pytest.raises(ValueError, match="symmetric|weight"):
        normalize_graph({"a": {"b": 2.0}, "b": {"a": 3.0}})


def test_asymmetric_weight_within_tolerance_is_averaged():
    _, w = normalize_graph({"a": {"b": 2.0}, "b": {"a": 2.0 + 1e-12}})
    assert w.toarray()[0, 1] == pytest.approx(2.0)


def test_missing_reverse_edge_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        normalize_graph({"a": {"b": 2.0}, "b": {}})


def test_self_loop_and_unknown_neighbour_still_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        normalize_graph({"a": {"a": 1.0}})
    with pytest.raises(ValueError, match="not a node"):
        normalize_graph({"a": {"z": 1.0}, "b": {}})


def test_canonical_unweighted_is_bare_and_unchanged():
    assert canonical_graph({"a": ["b"], "b": ["a"]}) == (("a", ("b",)), ("b", ("a",)))


def test_canonical_weighted_is_pairs_and_round_trips():
    g = {"a": {"b": 2.0, "c": 1.0}, "b": {"a": 2.0}, "c": {"a": 1.0}}
    canon = canonical_graph(g)
    assert canon[0] == ("a", (("b", 2.0), ("c", 1.0)))
    # dict(canon) is re-accepted and yields the same adjacency
    n1, w1 = normalize_graph(g)
    n2, w2 = normalize_graph(dict(canon))
    assert n1 == n2 and np.array_equal(w1.toarray(), w2.toarray())


def test_canonical_is_hashable_for_frozen_spec():
    hash(canonical_graph({"a": {"b": 2.0}, "b": {"a": 2.0}}))  # must not raise

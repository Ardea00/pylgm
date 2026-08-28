import numpy as np
import pytest

from pylgm.effects.graph import load_graph_file, normalize_graph


def _dense(graph):
    nodes, w = normalize_graph(graph)
    return nodes, w.toarray()


def test_normalize_path_graph():
    nodes, w = _dense({"A": ["B"], "B": ["A", "C"], "C": ["B"]})
    assert nodes == ("A", "B", "C")
    assert np.array_equal(w, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))


def test_normalize_is_symmetric_and_sorted():
    nodes, w = normalize_graph({"z": ["a"], "a": ["z"]})
    assert nodes == ("a", "z")
    assert (w != w.T).nnz == 0


def test_normalize_two_islands():
    nodes, w = _dense({"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]})
    assert nodes == ("A", "B", "C", "D")
    assert w[0, 1] == 1 and w[2, 3] == 1 and w[0, 2] == 0


def test_normalize_rejects_asymmetric():
    with pytest.raises(ValueError, match="symmetric"):
        normalize_graph({"A": ["B"], "B": []})


def test_normalize_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop"):
        normalize_graph({"A": ["A"]})


def test_normalize_rejects_unknown_neighbour():
    with pytest.raises(ValueError, match="not a node"):
        normalize_graph({"A": ["B"]})


def test_normalize_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_graph({})


def test_load_graph_file(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("3\n1 1 2\n2 2 1 3\n3 1 2\n")
    assert load_graph_file(path) == {"1": ["2"], "2": ["1", "3"], "3": ["2"]}


def test_load_graph_file_roundtrips_dict(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("2\n1 1 2\n2 1 1\n")
    graph = load_graph_file(path)
    nodes, _ = normalize_graph(graph)
    assert nodes == ("1", "2")


def test_load_graph_file_rejects_bad_count(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("3\n1 1 2\n")
    with pytest.raises(ValueError):
        load_graph_file(path)


def test_load_graph_file_rejects_out_of_range(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("2\n1 1 5\n2 1 1\n")
    with pytest.raises(ValueError):
        load_graph_file(path)


def test_load_graph_file_rejects_negative_degree(tmp_path):
    path = tmp_path / "g.graph"
    path.write_text("2\n1 -1\n2 0\n")
    with pytest.raises(ValueError, match="non-negative"):
        load_graph_file(path)


def test_load_graph_file_json_neighbour_dict(tmp_path):
    path = tmp_path / "g.json"
    path.write_text('{"1": ["2"], "2": ["1", "3"], "3": ["2"]}')
    graph = load_graph_file(path)
    assert graph == {"1": ["2"], "2": ["1", "3"], "3": ["2"]}
    # rides straight into the graph builders (path graph 1-2-3, no conversion)
    nodes, _ = normalize_graph(graph)
    assert nodes == ("1", "2", "3")


def test_load_graph_file_json_weighted_survives(tmp_path):
    path = tmp_path / "g.json"
    path.write_text('{"a": {"b": 5.0}, "b": {"a": 5.0}}')
    _, w = normalize_graph(load_graph_file(path))
    assert w.data.max() == 5.0


def test_load_graph_file_json_rejects_non_object(tmp_path):
    path = tmp_path / "g.json"
    path.write_text('["1", "2"]')
    with pytest.raises(ValueError, match="object mapping"):
        load_graph_file(path)


def test_design_from_graph_one_hot_and_missing():
    import pandas as pd
    from pylgm.effects.graph import design_from_graph

    nodes = ("A", "B", "C")
    frame = pd.DataFrame({"region": ["C", "A", "A"]})
    design = design_from_graph(nodes, frame, "region").toarray()
    assert (design == [[0, 0, 1], [1, 0, 0], [1, 0, 0]]).all()
    with pytest.raises(ValueError, match="not in the graph"):
        design_from_graph(nodes, pd.DataFrame({"region": ["A", "Z"]}), "region")

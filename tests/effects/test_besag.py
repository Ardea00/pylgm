import numpy as np
import pandas as pd
import pytest

from pylgm.effects.besag import build_besag
from pylgm.effects.random_walk import build_random_walk

PATH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
ISLANDS = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}


def _frame(regions):
    return pd.DataFrame({"region": regions})


def test_structure_is_graph_laplacian():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    expected = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    assert np.allclose(block.precision.toarray(), expected)
    assert block.labels == ("A", "B", "C")


def test_design_is_one_hot_over_nodes():
    block = build_besag(_frame(["C", "A", "A"]), "region", "region", PATH, 1.0, scale=False)
    assert np.array_equal(
        block.design.toarray(),
        np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=float),
    )


def test_connected_graph_has_one_constraint():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    assert np.array_equal(block.constraints, np.ones((1, 3)))


def test_two_islands_have_two_constraints():
    block = build_besag(_frame(["A", "B", "C", "D"]), "region", "region", ISLANDS, 1.0, scale=False)
    assert block.constraints.shape == (2, 4)
    assert np.array_equal(block.constraints[0], np.array([1, 1, 0, 0], dtype=float))
    assert np.array_equal(block.constraints[1], np.array([0, 0, 1, 1], dtype=float))


def test_scaling_gives_unit_geometric_mean_variance():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=True)
    r = block.precision.toarray()
    variances = np.diag(np.linalg.pinv(r))
    assert np.isclose(np.exp(np.mean(np.log(variances))), 1.0, atol=1e-8)


def test_scale_false_leaves_raw_laplacian():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 2.0, scale=False)
    expected = 2.0 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    assert np.allclose(block.precision.toarray(), expected)


def test_isolated_node_raises():
    graph = {"A": ["B"], "B": ["A"], "C": []}
    with pytest.raises(ValueError, match="no neighbours"):
        build_besag(_frame(["A", "B", "C"]), "region", "region", graph, 1.0)


def test_observed_region_absent_from_graph_raises():
    with pytest.raises(ValueError, match="not in the graph"):
        build_besag(_frame(["A", "B", "Z"]), "region", "region", PATH, 1.0)


def test_path_graph_matches_rw1_structure():
    besag = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    rw1 = build_random_walk(pd.DataFrame({"region": [0, 1, 2]}), "t", "region", 1.0, 1)
    assert np.allclose(besag.precision.toarray(), rw1.precision.toarray())

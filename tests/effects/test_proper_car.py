import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from pylgm.effects.proper_car import build_proper_car

PATH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def _frame(regions):
    return pd.DataFrame({"region": regions})


def test_structure_is_d_minus_rho_w():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.5, 1.0)
    d = np.diag([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(block.precision.toarray(), d - 0.5 * w)
    assert block.labels == ("A", "B", "C")


def test_no_constraints():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.5, 1.0)
    assert block.constraints.shape == (0, 3)


def test_design_is_one_hot_over_nodes():
    block = build_proper_car(_frame(["C", "A"]), "region", "region", PATH, 0.5, 1.0)
    assert np.array_equal(
        block.design.toarray(), np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    )


def test_rho_zero_gives_degree_matrix():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.0, 1.0)
    assert np.allclose(block.precision.toarray(), np.diag([1.0, 2.0, 1.0]))


def test_precision_scales_the_structure():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.5, 3.0)
    d = np.diag([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.allclose(block.precision.toarray(), 3.0 * (d - 0.5 * w))


def test_in_range_rho_is_positive_definite():
    block = build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 0.9, 1.0)
    cho_factor(block.precision.toarray())  # raises if not PD


def test_out_of_range_rho_raises_naming_interval():
    with pytest.raises(ValueError, match=r"rho must lie in"):
        build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 1.5, 1.0)


def test_rho_at_upper_boundary_raises():
    # rho = 1 is the singular intrinsic boundary (mu_max = 1 for a path graph).
    with pytest.raises(ValueError, match=r"rho must lie in"):
        build_proper_car(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, 1.0)


def test_isolated_node_raises():
    graph = {"A": ["B"], "B": ["A"], "C": []}
    with pytest.raises(ValueError, match="no neighbours"):
        build_proper_car(_frame(["A", "B", "C"]), "region", "region", graph, 0.5, 1.0)


def test_observed_region_absent_from_graph_raises():
    with pytest.raises(ValueError, match="not in the graph"):
        build_proper_car(_frame(["A", "B", "Z"]), "region", "region", PATH, 0.5, 1.0)

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from pylgm.effects.besag import _scaled_structure
from pylgm.effects.bym2 import build_bym2, bym2_precision, bym2_spectrum
from pylgm.effects.graph import normalize_graph

PATH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def _frame(regions):
    return pd.DataFrame({"region": regions})


def _direct_precision(graph, tau, phi):
    nodes, w = normalize_graph(graph)
    p = np.linalg.pinv(_scaled_structure(w, nodes, True))
    return tau * np.linalg.inv((1.0 - phi) * np.eye(len(nodes)) + phi * p)


def test_precision_matches_direct_marginal_formula():
    block = build_bym2(_frame(["A", "B", "C"]), "region", "region", PATH, 2.0, 0.7)
    assert np.allclose(block.precision.toarray(), _direct_precision(PATH, 2.0, 0.7))


def test_no_constraints_and_positive_definite():
    block = build_bym2(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, 0.9)
    assert block.constraints.shape == (0, 3)
    cho_factor(block.precision.toarray())


def test_phi_zero_recovers_scaled_identity():
    block = build_bym2(_frame(["A", "B", "C"]), "region", "region", PATH, 3.0, 0.0)
    assert np.allclose(block.precision.toarray(), 3.0 * np.eye(3))


def test_design_is_one_hot_over_nodes_and_labels_align():
    block = build_bym2(_frame(["C", "A"]), "region", "region", PATH, 1.0, 0.5)
    assert block.labels == ("A", "B", "C")
    assert np.array_equal(
        block.design.toarray(), np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    )


def test_spectrum_reassembly_matches_precision():
    vectors, values = bym2_spectrum(PATH)
    assembled = bym2_precision(vectors, values, 1.5, 0.4).toarray()
    assert np.allclose(assembled, _direct_precision(PATH, 1.5, 0.4))


def test_disconnected_graph_is_supported():
    islands = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    block = build_bym2(_frame(["A", "B", "C", "D"]), "region", "region", islands, 1.0, 0.6)
    assert block.constraints.shape == (0, 4)
    cho_factor(block.precision.toarray())


def test_isolated_node_rejected():
    graph = {"A": ["B"], "B": ["A"], "C": []}
    with pytest.raises(ValueError, match="no neighbours"):
        build_bym2(_frame(["A", "B", "C"]), "region", "region", graph, 1.0, 0.5)


def test_observed_region_absent_from_graph_rejected():
    with pytest.raises(ValueError, match="not in the graph"):
        build_bym2(_frame(["A", "Z"]), "region", "region", PATH, 1.0, 0.5)


@pytest.mark.parametrize("phi", [-0.1, 1.0, 1.5])
def test_phi_outside_unit_interval_rejected(phi):
    with pytest.raises(ValueError, match="phi"):
        build_bym2(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, phi)

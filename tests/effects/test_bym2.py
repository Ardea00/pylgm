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


def test_isolated_node_is_iid_singleton():
    # An isolated region becomes an independent IID unit; BYM2 now builds it
    # instead of rejecting. Dense path stays unconstrained and positive definite.
    graph = {"A": ["B"], "B": ["A"], "C": []}
    block = build_bym2(_frame(["A", "B", "C"]), "region", "region", graph, 1.0, 0.5)
    assert block.constraints.shape == (0, 3)
    cho_factor(block.precision.toarray())


def test_observed_region_absent_from_graph_rejected():
    with pytest.raises(ValueError, match="not in the graph"):
        build_bym2(_frame(["A", "Z"]), "region", "region", PATH, 1.0, 0.5)


@pytest.mark.parametrize("phi", [-0.1, 1.0, 1.5])
def test_phi_outside_unit_interval_rejected(phi):
    with pytest.raises(ValueError, match="phi"):
        build_bym2(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, phi)


def test_bym2_augmented_x_marginal_matches_dense():
    from scipy.linalg import null_space

    from pylgm.effects.bym2 import _build_bym2_augmented

    m = 4
    graph = {}

    def idx(r, c):
        return f"{r * m + c}"

    for r in range(m):
        for c in range(m):
            nb = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < m:
                    nb.append(idx(rr, cc))
            graph[idx(r, c)] = nb
    n = m * m
    frame = pd.DataFrame({"region": [idx(r, c) for r in range(m) for c in range(m)]})
    tau, phi = 1.3, 0.6

    aug = _build_bym2_augmented(frame, "s", "region", graph, tau, phi)
    vectors, values = bym2_spectrum(graph)
    q_dense = bym2_precision(vectors, values, tau, phi).toarray()
    dense_x_cov = np.linalg.inv(q_dense)

    qj = aug.precision.toarray()
    c = np.zeros((1, 2 * n))
    c[0, n:] = 1.0
    basis = null_space(c)
    sig = basis @ np.linalg.inv(basis.T @ qj @ basis) @ basis.T
    aug_x_cov = sig[:n, :n]
    assert np.allclose(np.diag(aug_x_cov), np.diag(dense_x_cov), atol=1e-9)


def test_bym2_augmented_multicomponent_x_marginal_matches_dense():
    from scipy.linalg import null_space

    from pylgm.effects.bym2 import _build_bym2_augmented

    # Two disconnected 3-node chains plus a lone isolated region: exercises the
    # per-component u* sum-to-zero AND an unconstrained IID singleton u*. The
    # augmented x-marginal must still equal the dense spectral one component-wise.
    graph = {
        "a0": ["a1"], "a1": ["a0", "a2"], "a2": ["a1"],
        "b0": ["b1"], "b1": ["b0", "b2"], "b2": ["b1"],
        "iso": [],
    }
    nodes = ["a0", "a1", "a2", "b0", "b1", "b2", "iso"]
    n = len(nodes)
    tau, phi = 1.3, 0.6

    aug = _build_bym2_augmented(_frame(nodes), "s", "region", graph, tau, phi)
    vectors, values = bym2_spectrum(graph)
    dense_x_cov = np.linalg.inv(bym2_precision(vectors, values, tau, phi).toarray())

    # two components of size >= 2 -> two u* sum-to-zero rows; isolated adds none
    assert aug.constraints.shape == (2, 2 * n)
    basis = null_space(aug.constraints)
    sig = basis @ np.linalg.inv(basis.T @ aug.precision.toarray() @ basis) @ basis.T
    assert np.allclose(np.diag(sig[:n, :n]), np.diag(dense_x_cov), atol=1e-9)

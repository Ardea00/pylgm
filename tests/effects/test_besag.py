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


def _robust_marginal_variances(r):
    # Generalized inverse of a connected ICAR Laplacian: drop the single null
    # eigenvalue explicitly. Unlike np.linalg.pinv's default tolerance, this
    # stays correct as the component grows (see the large-ring test below).
    eigenvalues, eigenvectors = np.linalg.eigh(r)
    inverse = np.zeros_like(eigenvalues)
    inverse[1:] = 1.0 / eigenvalues[1:]
    return np.einsum("ij,j,ij->i", eigenvectors, inverse, eigenvectors)


def test_scaling_stays_finite_on_a_large_connected_component():
    # Regression: on an irregular component beyond a handful of nodes,
    # np.linalg.pinv's default tolerance failed to discard the ICAR null
    # eigenvalue and produced huge negative marginal variances, aborting the
    # scaling. A 7x8 lattice (56 nodes) reproduces that exactly; the fixed
    # scaling yields finite, positive variances whose geometric mean is 1
    # (the Sørbye-Rue property).
    rows, cols = 7, 8
    labels = [f"N{r}_{c}" for r in range(rows) for c in range(cols)]
    grid: dict[str, list[str]] = {label: [] for label in labels}
    for r in range(rows):
        for c in range(cols):
            here = f"N{r}_{c}"
            if c + 1 < cols:
                right = f"N{r}_{c + 1}"
                grid[here].append(right)
                grid[right].append(here)
            if r + 1 < rows:
                down = f"N{r + 1}_{c}"
                grid[here].append(down)
                grid[down].append(here)
    block = build_besag(_frame(labels), "region", "region", grid, 1.0, scale=True)
    variances = _robust_marginal_variances(block.precision.toarray())
    assert np.all(np.isfinite(variances)) and np.all(variances > 0)
    assert np.isclose(np.exp(np.mean(np.log(variances))), 1.0, atol=1e-8)


def test_scale_false_leaves_raw_laplacian():
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 2.0, scale=False)
    expected = 2.0 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    assert np.allclose(block.precision.toarray(), expected)


def test_isolated_node_is_iid_singleton():
    # An isolated region (no neighbours) is treated as an independent IID unit:
    # structure diagonal 1, decoupled from the rest, and it gets NO sum-to-zero
    # constraint (unlike the connected component, which keeps one). It no longer
    # aborts the fit -- R-INLA's adjust.for.con.comp default.
    graph = {"A": ["B"], "B": ["A"], "C": []}
    block = build_besag(_frame(["A", "B", "C"]), "region", "region", graph, 1.0)
    r = block.precision.toarray()
    assert r[2, 2] == 1.0  # C: unit-variance IID
    assert np.allclose(r[2, :2], 0.0) and np.allclose(r[:2, 2], 0.0)  # decoupled
    assert block.constraints.shape == (1, 3)  # one for {A,B}, none for isolated C
    assert np.array_equal(block.constraints[0], np.array([1, 1, 0], dtype=float))


def test_sparse_scaled_structure_matches_dense():
    # The augmented (large-graph) BYM2 path scales R* component-by-component in
    # sparse form; it must reproduce the validated dense scaler bit-for-bit,
    # including the isolated-node IID diagonal.
    from pylgm.effects.besag import _scaled_structure, _sparse_scaled_structure
    from pylgm.effects.graph import normalize_graph

    graph = {
        "a0": ["a1"], "a1": ["a0", "a2"], "a2": ["a1"],
        "b0": ["b1"], "b1": ["b0"], "iso": [],
    }
    nodes, w = normalize_graph(graph)
    dense = _scaled_structure(w, nodes, True)
    sparse = _sparse_scaled_structure(w).toarray()
    assert np.allclose(sparse, dense, atol=1e-12)


def test_observed_region_absent_from_graph_raises():
    with pytest.raises(ValueError, match="not in the graph"):
        build_besag(_frame(["A", "B", "Z"]), "region", "region", PATH, 1.0)


def test_path_graph_matches_rw1_structure():
    besag = build_besag(_frame(["A", "B", "C"]), "region", "region", PATH, 1.0, scale=False)
    rw1 = build_random_walk(pd.DataFrame({"region": [0, 1, 2]}), "t", "region", 1.0, 1)
    assert np.allclose(besag.precision.toarray(), rw1.precision.toarray())

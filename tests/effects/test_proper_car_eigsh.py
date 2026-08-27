import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.effects.proper_car import _validity_interval


def _ring_adjacency(n: int) -> csr_matrix:
    rows, cols = [], []
    for i in range(n):
        rows += [i, i]
        cols += [(i - 1) % n, (i + 1) % n]
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_eigsh_interval_matches_dense_eigvalsh():
    w = _ring_adjacency(12)
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    # Dense reference for the same extreme eigenvalues of the normalized adjacency.
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = (w.toarray() * d_inv_sqrt).T * d_inv_sqrt
    eigenvalues = np.linalg.eigvalsh(normalized)
    lower, upper, mu_min, mu_max = _validity_interval(w, degree)
    assert np.isclose(mu_min, eigenvalues[0], atol=1e-8)
    assert np.isclose(mu_max, eigenvalues[-1], atol=1e-8)
    # ρ interval is the reciprocal of the extreme eigenvalues, unchanged by this task.
    assert lower < 0 < upper
    assert np.isclose(lower, 1.0 / eigenvalues[0], atol=1e-8)
    assert np.isclose(upper, 1.0 / eigenvalues[-1], atol=1e-8)


def test_isolated_node_still_raises():
    # A node with zero degree must still raise the existing isolated-node error.
    w = csr_matrix((np.array([1.0, 1.0]), (np.array([0, 1]), np.array([1, 0]))), shape=(3, 3))
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    with pytest.raises(Exception):
        _validity_interval(w, degree)

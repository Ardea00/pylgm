import numpy as np

from pylgm.effects.scaling import sorbye_rue_scale


def _marginal_variances(structure, null_dim):
    eigenvalues, eigenvectors = np.linalg.eigh(structure)
    inverse = np.zeros_like(eigenvalues)
    inverse[null_dim:] = 1.0 / eigenvalues[null_dim:]
    return np.einsum("ij,j,ij->i", eigenvectors, inverse, eigenvectors)


def test_sorbye_rue_scale_sets_geometric_mean_variance_to_one():
    # RW1 structure DtD on 6 levels: null_dim = 1 (the constant).
    n = 6
    d = np.zeros((n - 1, n))
    for i in range(n - 1):
        d[i, i] = -1.0
        d[i, i + 1] = 1.0
    r = d.T @ d
    scaled = sorbye_rue_scale(r, null_dim=1)
    variances = _marginal_variances(scaled, null_dim=1)
    assert np.isclose(np.exp(np.mean(np.log(variances))), 1.0)


def test_sorbye_rue_scale_rejects_degenerate_structure():
    import pytest

    # All-zero structure has no positive eigenvalues -> non-finite variances.
    with pytest.raises(ValueError):
        sorbye_rue_scale(np.zeros((4, 4)), null_dim=1)

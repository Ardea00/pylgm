"""Tests for Sørbye-Rue variance scaling (dense eigh and sparse fast path)."""


def test_sorbye_sparse_matches_dense_eigh():
    """Sparse null_dim==1 Sørbye scale must equal the dense eigh reference."""
    import numpy as np
    from scipy.sparse import lil_matrix

    from pylgm.effects.scaling import sorbye_rue_scale

    def ring(n):
        q = lil_matrix((n, n))
        for i in range(n):
            q[i, i] = 2.0
            q[i, (i + 1) % n] = -1.0
            q[i, (i - 1) % n] = -1.0
        return q

    for n in (50, 200):
        sparse_struct = ring(n).tocsc()
        dense_struct = sparse_struct.toarray()
        got = sorbye_rue_scale(sparse_struct, null_dim=1)  # sparse in -> sparse out
        want = sorbye_rue_scale(dense_struct, null_dim=1)  # dense eigh reference
        assert np.allclose(got.toarray(), want, atol=1e-9)

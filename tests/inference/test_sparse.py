import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.exceptions import NumericalError
from pylgm.inference.sparse import SparseSpdFactor, _partition_blocks


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_logdet_matches_slogdet():
    m = _spd(6)
    factor = SparseSpdFactor(csr_matrix(m), "test")
    sign, logdet = np.linalg.slogdet(m)
    assert sign > 0
    assert np.isclose(factor.logdet, logdet, atol=1e-8)


def test_solve_matches_dense():
    m = _spd(6)
    b = np.arange(6, dtype=float)
    factor = SparseSpdFactor(csr_matrix(m), "test")
    assert np.allclose(factor.solve(b), np.linalg.solve(m, b), atol=1e-8)
    # Multi-column solve (used for the Schur block B -> A_s^{-1} B).
    rhs = np.eye(6)[:, :2]
    assert np.allclose(factor.solve(rhs), np.linalg.solve(m, rhs), atol=1e-8)


def test_non_spd_raises():
    m = csr_matrix(np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(NumericalError):
        SparseSpdFactor(m, "indefinite")


def test_partition_splits_fixed_from_field(besag_with_intercept_model):
    # Fixture builds Fixed("1") + Besag(...) compiled model (see conftest).
    sparse_index, dense_index = _partition_blocks(besag_with_intercept_model)
    labels = besag_with_intercept_model.labels
    # The intercept (Fixed) has an all-ones design touching every
    # observation, so it lands in the dense block by design-column density.
    # Fixed("1") compiles the intercept column as "Intercept" (formula-library
    # convention), so the qualified label is "fixed:Intercept", not "fixed:1".
    dense_labels = {labels[i] for i in dense_index}
    assert any(label.endswith(":Intercept") for label in dense_labels)
    # Every Besag column has an incidence design (one cell each), so
    # it lands in the sparse block.
    assert len(sparse_index) + len(dense_index) == len(labels)
    assert set(sparse_index).isdisjoint(dense_index)

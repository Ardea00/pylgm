"""Pin the two-step contraction against the naive three-operand einsum."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.inference.result import quadratic_form_diagonal


@pytest.mark.parametrize("sparse", [False, True])
def test_matches_three_operand_einsum(sparse):
    rng = np.random.default_rng(0)
    n, p = 300, 120
    dense = rng.normal(size=(n, p))
    dense[rng.random((n, p)) > 0.05] = 0.0  # ~5% dense, like a real design
    root = rng.normal(size=(p, p))
    covariance = root @ root.T / p

    weights = csr_matrix(dense) if sparse else dense
    expected = np.einsum("ij,jk,ik->i", dense, covariance, dense)

    assert quadratic_form_diagonal(weights, covariance) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import cho_factor

from pylgm.effects.ar1 import ar1_structure, build_ar1


def _frame(levels):
    return pd.DataFrame({"t": levels})


@pytest.mark.parametrize(("n", "tau", "rho"), [(5, 1.0, 0.7), (6, 2.5, -0.4), (8, 0.3, 0.95)])
def test_precision_inverse_is_the_stationary_ar1_covariance(n, tau, rho):
    # Independent oracle: Cov_ij = (1/tau) * rho^|i-j| for the stationary AR1.
    block = build_ar1(_frame(list(range(n))), "trend", "t", tau, rho)
    index = np.arange(n)
    expected = (1.0 / tau) * rho ** np.abs(index[:, None] - index[None, :])
    assert np.allclose(np.linalg.inv(block.precision.toarray()), expected)


def test_tau_is_the_marginal_precision():
    block = build_ar1(_frame(list(range(6))), "trend", "t", 4.0, 0.8)
    variances = np.diag(np.linalg.inv(block.precision.toarray()))
    assert np.allclose(variances, 0.25)


def test_rho_zero_gives_scaled_identity():
    block = build_ar1(_frame(list(range(4))), "trend", "t", 3.0, 0.0)
    assert np.allclose(block.precision.toarray(), 3.0 * np.eye(4))


def test_no_constraints_and_positive_definite():
    for rho in (-0.99, 0.0, 0.99):
        block = build_ar1(_frame(list(range(5))), "trend", "t", 1.0, rho)
        assert block.constraints.shape == (0, 5)
        cho_factor(block.precision.toarray())


def test_structure_is_tridiagonal_and_symmetric():
    structure = ar1_structure(5, 0.6)
    assert np.allclose(structure, structure.T)
    # everything more than one step from the diagonal is zero
    index = np.arange(5)
    far = np.abs(index[:, None] - index[None, :]) > 1
    assert np.allclose(structure[far], 0.0)


def test_design_is_one_hot_over_ordered_levels():
    block = build_ar1(_frame([2, 0, 1, 0]), "trend", "t", 1.0, 0.5)
    assert block.labels == ("0", "1", "2")
    assert np.array_equal(
        block.design.toarray(),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float),
    )


def test_fewer_than_two_levels_raises():
    with pytest.raises(ValueError, match="two"):
        build_ar1(_frame([0, 0]), "trend", "t", 1.0, 0.5)


@pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5])
def test_rho_outside_the_open_unit_interval_raises(rho):
    with pytest.raises(ValueError, match="rho"):
        build_ar1(_frame(list(range(4))), "trend", "t", 1.0, rho)

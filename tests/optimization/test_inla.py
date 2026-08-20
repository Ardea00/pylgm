import numpy as np
import pytest

from pylgm.exceptions import OptimizationError
from pylgm.optimization.inla import _build_grid, _finite_difference_hessian


def test_finite_difference_hessian_recovers_quadratic():
    A = np.array([[2.0, 0.3], [0.3, 1.5]])
    center = np.array([0.4, -0.2])
    # s(u) = -0.5 (u-c)^T A (u-c); Hessian is -A everywhere
    def s(u):
        d = u - center
        return -0.5 * d @ A @ d
    H = _finite_difference_hessian(s, center, step=1e-3)
    np.testing.assert_allclose(H, -A, atol=1e-4)


def test_build_grid_centers_and_bounds_count():
    center = np.array([0.0])
    hessian = np.array([[-4.0]])  # -H = 4 -> sigma = 0.5 in u-space
    grid = _build_grid(center, hessian, grid_step=1.0, radius=3)
    assert grid.shape == (7, 1)
    # the mode (z=0) is present
    assert np.any(np.all(np.isclose(grid, center), axis=1))
    # step of one lattice unit is 1/sqrt(4) = 0.5 in u-space
    offsets = np.sort(np.unique(np.round(grid[:, 0] - center[0], 6)))
    np.testing.assert_allclose(offsets, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])


def test_build_grid_guards_dimensionality():
    center = np.zeros(6)
    hessian = -np.eye(6)
    with pytest.raises(OptimizationError, match="grid"):
        _build_grid(center, hessian, radius=3, max_grid_points=4096)  # 7**6 >> 4096

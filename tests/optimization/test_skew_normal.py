import numpy as np
from pylgm.optimization.inla import _fit_skew_normal


def _sn_stats(xi, omega, a):
    d = a / np.sqrt(1 + a * a)
    mean = xi + omega * d * np.sqrt(2 / np.pi)
    var = omega ** 2 * (1 - 2 * d * d / np.pi)
    return mean, var


def test_fit_skew_normal_zero_reduces_to_gaussian():
    xi, omega, a, clamped = _fit_skew_normal(np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(xi, [0.0])
    np.testing.assert_allclose(omega, [1.0])
    np.testing.assert_allclose(a, [0.0])
    assert not clamped.any()


def test_fit_skew_normal_matches_mean_variance_and_cubic():
    g1 = np.array([0.15])
    g3 = np.array([0.4])
    xi, omega, a, _ = _fit_skew_normal(g1, g3)
    mean, var = _sn_stats(xi[0], omega[0], a[0])
    np.testing.assert_allclose(mean, g1[0], atol=1e-6)
    np.testing.assert_allclose(var, 1.0, atol=1e-6)
    C = (4 - np.pi) * np.sqrt(2) / np.pi ** 1.5
    np.testing.assert_allclose(C * (a[0] / omega[0]) ** 3, g3[0], rtol=1e-6)


def test_fit_skew_normal_clamps_extreme_skew():
    _, _, _, clamped = _fit_skew_normal(np.array([0.0]), np.array([1e6]))
    assert clamped.all()

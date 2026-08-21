import numpy as np
from scipy.stats import norm, skewnorm
from pylgm.inference.result import TabulatedMarginals


def _grid_density(dist, lo, hi, n=4001):
    x = np.linspace(lo, hi, n)
    return x, dist.pdf(x)


def test_tabulated_matches_normal_moments_and_quantiles():
    x, d = _grid_density(norm(loc=0.5, scale=2.0), -12, 13)
    m = TabulatedMarginals(x[None, :], d[None, :])
    np.testing.assert_allclose(m.mean, [0.5], atol=1e-3)
    np.testing.assert_allclose(m.variance, [4.0], atol=1e-2)
    np.testing.assert_allclose(m.skewness, [0.0], atol=1e-3)
    np.testing.assert_allclose(m.quantile(0.975), [0.5 + norm.ppf(0.975) * 2.0], atol=2e-2)
    np.testing.assert_allclose(m.cdf(m.quantile(0.3))[0], 0.3, atol=1e-3)


def test_tabulated_matches_skewnormal_skewness():
    x, d = _grid_density(skewnorm(a=6.0, loc=0.0, scale=1.0), -6, 8)
    m = TabulatedMarginals(x[None, :], d[None, :])
    assert m.skewness[0] > 0.4   # skewnorm(a=6) skewness ~0.77
    np.testing.assert_allclose(m.mean, [skewnorm.mean(6.0)], atol=1e-2)


def test_tabulated_normalizes_and_is_immutable():
    x = np.linspace(-5, 5, 2001)
    m = TabulatedMarginals(x[None, :], norm.pdf(x)[None, :] * 7.0)  # unnormalized
    np.testing.assert_allclose(np.trapz(m.density[0], m.x[0]), 1.0, atol=1e-6)
    with __import__("pytest").raises(ValueError):
        m.density[0, 0] = 1.0

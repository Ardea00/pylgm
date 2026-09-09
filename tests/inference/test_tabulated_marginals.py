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
    np.testing.assert_allclose(np.trapezoid(m.density[0], m.x[0]), 1.0, atol=1e-6)
    with __import__("pytest").raises(ValueError):
        m.density[0, 0] = 1.0


def test_pdf_and_cdf_are_elementwise_across_components():
    """One convention across all three marginal representations: cdf(x) takes one
    value per component and returns F_i(x_i), shape (p,).

    TabulatedMarginals used to return the (p, len(x)) cross product instead,
    which silently broke any code written against the LatentMarginals protocol --
    it would read component j's CDF where component i's was meant.
    """
    import pytest

    support = np.linspace(-9, 9, 3001)
    components = 3
    m = TabulatedMarginals(
        np.tile(support, (components, 1)), np.tile(norm.pdf(support), (components, 1))
    )
    x = np.array([-1.0, 0.0, 1.0])

    assert m.cdf(x).shape == (components,)
    np.testing.assert_allclose(m.cdf(x), norm.cdf(x), atol=1e-4)
    assert m.pdf(x).shape == (components,)
    np.testing.assert_allclose(m.pdf(x), norm.pdf(x), atol=1e-4)

    with pytest.raises(ValueError, match="must match the marginal shape"):
        m.cdf(np.zeros(components + 1))
    with pytest.raises(ValueError, match="must match the marginal shape"):
        m.pdf(np.zeros(components + 1))


def test_all_marginal_representations_share_the_cdf_convention():
    """The point of the change: the three classes are interchangeable behind the
    LatentMarginals protocol, so a caller need not know which one it holds."""
    from pylgm.inference.result import GaussianMarginals, SkewNormalMarginals

    x = np.array([-1.0, 0.0, 1.0])
    p, grid = 3, 5
    support = np.linspace(-9, 9, 3001)
    representations = [
        GaussianMarginals(np.zeros(p), np.ones(p)),
        SkewNormalMarginals(
            np.full((p, grid), 1.0 / grid), np.zeros((p, grid)),
            np.ones((p, grid)), np.zeros((p, grid)),
        ),
        TabulatedMarginals(np.tile(support, (p, 1)), np.tile(norm.pdf(support), (p, 1))),
    ]
    for marginals in representations:
        values = marginals.cdf(x)
        assert values.shape == (p,), type(marginals).__name__
        np.testing.assert_allclose(values, norm.cdf(x), atol=1e-4,
                                   err_msg=type(marginals).__name__)

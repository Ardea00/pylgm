import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp

from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.inference import fit_laplace
from pylgm.optimization.inla import _simplified_laplace_marginals


class _Fit:
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, float)
        self.covariance = np.asarray(cov, float)


def test_sla_gaussian_likelihood_is_exact_gaussian():
    design = csr_matrix(np.eye(2))
    offset = np.zeros(2)
    y = np.array([0.3, -0.4])
    fit = _Fit([0.2, -0.1], np.diag([0.3, 0.5]))
    loc, scale, shape, w, clamped = _simplified_laplace_marginals(
        design, offset, y, [(1.0, fit, CompiledGaussian(1.0))])
    np.testing.assert_allclose(shape, 0.0, atol=1e-12)   # d3=0 -> no skew
    np.testing.assert_allclose(loc[:, 0], fit.mean)
    np.testing.assert_allclose(scale[:, 0], np.sqrt(np.diag(fit.covariance)))


def test_sla_poisson_matches_true_marginal_better_than_gaussian():
    # p=1 intercept, Poisson counts; the true latent posterior is 1-D -> exact quadrature.
    n = 12
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, size=n).astype(float)
    design = csr_matrix(np.ones((n, 1)))
    prior_prec = 1.0
    block = LatentBlock("b", ("i",), design, eye(1, format="csr") * prior_prec,
                        np.empty((0, 1), dtype=float))
    family = CompiledFamily(y=y, observed=np.array([True] * n), offset=np.zeros(n),
                            blocks=(ScalableBlock(block, None, 1.0),), parameter_names=(),
                            likelihood_factory=lambda r: CompiledPoisson())
    compiled = family.materialize({})
    fit = fit_laplace(compiled)  # single conditional fit at fixed theta
    like = CompiledPoisson()
    loc, scale, shape, w, _ = _simplified_laplace_marginals(
        design, np.zeros(n), y, [(1.0, fit, like)])
    # SLA skew-normal for the single latent:
    from pylgm.inference.result import SkewNormalMarginals
    sla = SkewNormalMarginals(weights=w, location=loc, scale=scale, shape=shape)

    # TRUE 1-D posterior of x: pi(x) ∝ exp(-0.5 prior_prec x^2) * prod Poisson(y_j; exp(x))
    xs = np.linspace(-2.0, 4.0, 8001)
    logp = -0.5 * prior_prec * xs ** 2 + np.array(
        [np.sum(y * x - np.exp(x)) for x in xs])
    wq = np.exp(logp - logsumexp(logp))
    wq = wq / (wq.sum())
    true_mean = np.sum(wq * xs)
    true_var = np.sum(wq * (xs - true_mean) ** 2)
    true_skew = np.sum(wq * (xs - true_mean) ** 3) / true_var ** 1.5

    gauss_mean = fit.mean[0]
    # SLA closer in mean & skewness than the Gaussian strategy; correct skew sign
    assert abs(sla.mean[0] - true_mean) <= abs(gauss_mean - true_mean) + 1e-9
    assert np.sign(sla.skewness[0]) == np.sign(true_skew)
    assert abs(sla.skewness[0] - true_skew) < abs(0.0 - true_skew)  # Gaussian skew is 0

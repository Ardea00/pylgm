import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from pylgm.exceptions import NumericalError
from pylgm.ir.family import CompiledFamily, ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.inference import fit_gaussian, fit_laplace
from pylgm.optimization.inla import _full_laplace_marginals, _simplified_laplace_marginals, _logdet_spd
from pylgm.inference.result import SkewNormalMarginals


def test_full_laplace_gaussian_likelihood_is_exact_gaussian():
    # eq 13's conditional-mean shortcut is exact only when the scaffold fit IS the
    # true mode/Hessian, so build a self-consistent scaffold via the real Gaussian
    # solver (rather than a hand-picked, inconsistent mean/covariance pair).
    n = 6
    X = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    y = np.array([0.4, -0.3, 0.8, 0.1, -0.6, 0.2])
    Xs = csr_matrix(X)
    prior_prec = 0.5
    block = LatentBlock("b", ("i", "s"), Xs, eye(2, format="csr") * prior_prec,
                        np.empty((0, 2), dtype=float))
    fam = CompiledFamily(y=y, observed=np.array([True] * n), offset=np.zeros(n),
                         blocks=(ScalableBlock(block, None, 1.0),), parameter_names=(),
                         likelihood_factory=lambda r: CompiledGaussian(1.0))
    compiled = fam.materialize({})
    fit = fit_gaussian(compiled)
    Q = np.eye(2) * prior_prec
    tab = _full_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, CompiledGaussian(1.0), Q)],
                                  n_abscissae=7)
    # exact-Gaussian: tabulated mean/variance match the Gaussian marginal
    np.testing.assert_allclose(tab.mean, fit.mean, atol=5e-3)
    np.testing.assert_allclose(tab.variance, np.diag(fit.covariance), rtol=5e-2)
    np.testing.assert_allclose(tab.skewness, [0.0, 0.0], atol=2e-2)


def test_full_laplace_multi_grid_uses_per_theta_precision():
    # Two theta-grid points with DIFFERENT prior precisions Q1 != Q2, each with its
    # own self-consistent Gaussian scaffold fit. Since the Gaussian likelihood makes
    # full-Laplace exact per grid point (see test above), the correct mixed
    # mean/variance is the closed-form grid-weighted mixture of the two per-theta
    # Gaussian marginals. If the engine reused a single Q for both grid points (the
    # bug this guards against), grid point 2's tabulated marginal would no longer be
    # an exact Gaussian(fit2.mean, fit2.covariance) and this would fail.
    n = 6
    X = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    y = np.array([0.4, -0.3, 0.8, 0.1, -0.6, 0.2])
    Xs = csr_matrix(X)
    like = CompiledGaussian(1.0)

    def scaffold(prior_prec):
        block = LatentBlock("b", ("i", "s"), Xs, eye(2, format="csr") * prior_prec,
                            np.empty((0, 2), dtype=float))
        fam = CompiledFamily(y=y, observed=np.array([True] * n), offset=np.zeros(n),
                             blocks=(ScalableBlock(block, None, 1.0),), parameter_names=(),
                             likelihood_factory=lambda r: CompiledGaussian(1.0))
        compiled = fam.materialize({})
        fit = fit_gaussian(compiled)
        return fit, np.eye(2) * prior_prec

    fit1, Q1 = scaffold(0.5)
    fit2, Q2 = scaffold(4.0)
    w1, w2 = 0.3, 0.7
    grid = [(w1, fit1, like, Q1), (w2, fit2, like, Q2)]
    tab = _full_laplace_marginals(Xs, np.zeros(n), y, grid, n_abscissae=9)

    v1, v2 = np.diag(fit1.covariance), np.diag(fit2.covariance)
    expected_mean = w1 * fit1.mean + w2 * fit2.mean
    expected_var = w1 * (v1 + fit1.mean ** 2) + w2 * (v2 + fit2.mean ** 2) - expected_mean ** 2

    np.testing.assert_allclose(tab.mean, expected_mean, atol=1e-2)
    np.testing.assert_allclose(tab.variance, expected_var, rtol=8e-2)
    assert np.all(np.abs(tab.skewness) < 0.15)


def test_full_laplace_beats_or_matches_sla_vs_true_marginal():
    # tiny UNCONSTRAINED Poisson: Fixed("1 + x"), p=2 -> true marginal by 2-D quadrature
    n = 15
    rng = np.random.default_rng(0)
    xcov = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([np.ones(n), xcov])
    true_beta = np.array([0.5, 0.8])
    y = rng.poisson(np.exp(X @ true_beta)).astype(float)
    Xs = csr_matrix(X)
    prior_prec = 1.0
    block = LatentBlock("b", ("i", "s"), Xs, eye(2, format="csr") * prior_prec,
                        np.empty((0, 2), dtype=float))
    fam = CompiledFamily(y=y, observed=np.array([True] * n), offset=np.zeros(n),
                         blocks=(ScalableBlock(block, None, 1.0),), parameter_names=(),
                         likelihood_factory=lambda r: CompiledPoisson())
    compiled = fam.materialize({})
    fit = fit_laplace(compiled)
    Q = np.eye(2) * prior_prec
    like = CompiledPoisson()

    tab = _full_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, like, Q)], n_abscissae=9)
    loc, scale, shape, w, _ = _simplified_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, like)])
    sla = SkewNormalMarginals(weights=w, location=loc, scale=scale, shape=shape)

    # TRUE marginal of x_0 (intercept) via 2-D quadrature of the joint posterior
    g0 = np.linspace(fit.mean[0] - 6 * np.sqrt(fit.covariance[0, 0]),
                     fit.mean[0] + 6 * np.sqrt(fit.covariance[0, 0]), 241)
    g1 = np.linspace(fit.mean[1] - 6 * np.sqrt(fit.covariance[1, 1]),
                     fit.mean[1] + 6 * np.sqrt(fit.covariance[1, 1]), 241)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    eta = G0[..., None] * X[:, 0] + G1[..., None] * X[:, 1]      # (241,241,n)
    loglik = np.sum(y * eta - np.exp(eta), axis=-1)
    logprior = -0.5 * prior_prec * (G0 ** 2 + G1 ** 2)
    dens = np.exp((loglik + logprior) - (loglik + logprior).max())
    marg0 = dens.sum(axis=1)
    marg0 /= np.trapz(marg0, g0)
    tm = float(np.trapz(g0 * marg0, g0))
    tv = float(np.trapz((g0 - tm) ** 2 * marg0, g0))
    tskew = float(np.trapz((g0 - tm) ** 3 * marg0, g0) / tv ** 1.5)

    full_skew_err = abs(tab.skewness[0] - tskew)
    sla_skew_err = abs(sla.skewness[0] - tskew)
    gauss_skew_err = abs(0.0 - tskew)
    assert np.sign(tab.skewness[0]) == np.sign(tskew)
    assert full_skew_err <= sla_skew_err + 1e-3     # full LA at least as good as SLA on skew
    assert full_skew_err < gauss_skew_err            # and better than the Gaussian strategy
    np.testing.assert_allclose(tab.mean[0], tm, atol=2e-2)


def test_logdet_spd_raises_on_non_pd_matrix():
    # non-positive-definite matrix should raise NumericalError
    non_pd = np.array([[1.0, 2.0], [2.0, 1.0]])  # has negative eigenvalue
    with pytest.raises(NumericalError, match="must be positive definite"):
        _logdet_spd(non_pd)

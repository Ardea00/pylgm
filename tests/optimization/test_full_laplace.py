import numpy as np
from scipy.sparse import csr_matrix, eye

from pylgm.ir.family import CompiledFamily, ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson
from pylgm.inference import fit_gaussian, fit_laplace
from pylgm.optimization.inla import _full_laplace_marginals, _simplified_laplace_marginals
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
    tab = _full_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, CompiledGaussian(1.0))],
                                  Q, n_abscissae=7)
    # exact-Gaussian: tabulated mean/variance match the Gaussian marginal
    np.testing.assert_allclose(tab.mean, fit.mean, atol=5e-3)
    np.testing.assert_allclose(tab.variance, np.diag(fit.covariance), rtol=5e-2)
    np.testing.assert_allclose(tab.skewness, [0.0, 0.0], atol=2e-2)


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

    tab = _full_laplace_marginals(Xs, np.zeros(n), y, [(1.0, fit, like)], Q, n_abscissae=9)
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

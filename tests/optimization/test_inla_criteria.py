import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
from pylgm.likelihoods import CompiledGaussian
from pylgm.optimization.inla import _model_criteria


class _Fit:
    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(cov, dtype=float)


def test_single_grid_gaussian_matches_closed_forms():
    # 2 observed rows, identity design (eta_i = x_i), latent posterior N(mean, cov diag)
    design = csr_matrix(np.eye(2))
    offset = np.zeros(2)
    y = np.array([0.4, -0.7])
    mean = np.array([0.5, -0.5])
    v = np.array([0.1, 0.2])
    fit = _Fit(mean, np.diag(v))
    sigma = 1.3
    var = sigma ** 2
    like = CompiledGaussian(sigma)
    crit = _model_criteria(design, offset, y, [(1.0, fit, like)], n_nodes=64)

    m = mean  # eta mean = design@mean = mean
    # WAIC ingredients (closed forms)
    elog = -0.5 * (np.log(2 * np.pi * var) + ((y - m) ** 2 + v) / var)
    lppd = norm.logpdf(y, loc=m, scale=np.sqrt(v + var))
    var_logp = (v ** 2 + 2 * v * (y - m) ** 2) / (2 * var ** 2)
    waic = -2.0 * np.sum(lppd - var_logp)
    np.testing.assert_allclose(crit.waic, waic, rtol=1e-6)
    np.testing.assert_allclose(crit.waic_effective_parameters, var_logp.sum(), rtol=1e-6)
    # DIC: D_bar = -2 sum elog ; D(mbar) = -2 sum logdensity(m, y) ; p_D = Dbar - Dmean
    d_bar = -2.0 * elog.sum()
    d_mean = -2.0 * like.pointwise_log_density(m, y).sum()
    np.testing.assert_allclose(crit.dic, d_bar + (d_bar - d_mean), rtol=1e-6)

    # CPO/PIT vs independent fine Gauss-Hermite (harmonic / reweighted definitions)
    from numpy.polynomial.hermite import hermgauss
    nodes, w = hermgauss(200)
    w = w / np.sqrt(np.pi)
    for i in range(2):
        en = m[i] + np.sqrt(2 * v[i]) * nodes
        lp = like.pointwise_log_density(en, np.full_like(en, y[i]))
        einv = np.sum(w / np.exp(lp))
        cdf = like.cdf(en, np.full_like(en, y[i]))
        f = np.sum(w * cdf / np.exp(lp))
        np.testing.assert_allclose(crit.cpo[i], 1.0 / einv, rtol=1e-4)
        np.testing.assert_allclose(crit.pit[i], f / einv, rtol=1e-4)


def test_cpo_failure_flag_on_diffuse_latent():
    design = csr_matrix(np.eye(1))
    offset = np.zeros(1)
    y = np.array([0.0])
    fit = _Fit(np.array([0.0]), np.array([[4.0]]))  # v=4 >> sigma^2=0.25 -> 1/p heavy tail
    crit = _model_criteria(design, offset, y, [(1.0, fit, CompiledGaussian(0.5))], n_nodes=64)
    assert crit.cpo_failures >= 1

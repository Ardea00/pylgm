import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp

from pylgm.exceptions import OptimizationError
from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian
from pylgm.inference import fit_gaussian
from pylgm.optimization.empirical_bayes import OptimizationBounds
from pylgm.optimization.inla import _build_grid, _finite_difference_hessian, integrate_inla


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


def _one_hyperparameter_family():
    # informative region-effects model: 50 regions x 8 obs -> interior, near-Gaussian
    # precision posterior that grid integration approximates well.
    rng = np.random.default_rng(0)
    n_regions, per = 50, 8
    effects = rng.normal(0.0, 1.0, size=n_regions)
    rows_region, y = [], []
    for r in range(n_regions):
        for _ in range(per):
            rows_region.append(r)
            y.append(effects[r] + rng.normal(0.0, 1.0))
    y = np.asarray(y)
    n = len(y)
    design = csr_matrix((np.ones(n), (np.arange(n), rows_region)), shape=(n, n_regions))
    block = LatentBlock(
        "g", tuple(f"r{r}" for r in range(n_regions)), design,
        eye(n_regions, format="csr"), np.empty((0, n_regions), dtype=float),
    )
    return CompiledFamily(
        y=y, observed=np.array([True] * n), offset=np.zeros(n),
        blocks=(ScalableBlock(block, "p", 1.0),), parameter_names=("p",),
        likelihood_factory=lambda r: CompiledGaussian(1.0),
    )


def test_integrate_inla_matches_fine_1d_quadrature():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    result = integrate_inla(family, bounds, fit=fit_gaussian, grid_step=0.75, radius=6)

    # independent fine 1-D quadrature over u = log p
    us = np.linspace(np.log(1e-2), np.log(1e2), 1601)
    logw, means, variances = [], [], []
    for u in us:
        fit = fit_gaussian(family.materialize({"p": float(np.exp(u))}))
        logw.append(float(fit.log_marginal_likelihood) + u)  # + Jacobian
        means.append(fit.mean[0])
        variances.append(fit.covariance[0, 0])
    logw = np.asarray(logw)
    w = np.exp(logw - logsumexp(logw))
    ref_mean = float(np.sum(w * np.asarray(means)))
    ref_var = float(np.sum(w * (np.asarray(variances) + np.asarray(means) ** 2)) - ref_mean ** 2)

    np.testing.assert_allclose(result.mean[0], ref_mean, atol=5e-3)
    np.testing.assert_allclose(result.covariance[0, 0], ref_var, rtol=5e-2)
    assert result.hyperparameter_marginals()["p"].mean[0] > 0
    assert result.diagnostics["inla_grid_points"] >= 3


def test_integrate_inla_produces_a_hyperparameter_distribution():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    result = integrate_inla(family, bounds, fit=fit_gaussian, radius=6)
    # integration explored a genuine posterior (multiple weighted grid points,
    # positive spread on the hyperparameter marginal)
    assert result.diagnostics["inla_grid_points"] >= 3
    assert result.hyperparameter_marginals()["p"].variance[0] > 0.0
    assert result.hyperparameter_marginals()["p"].mean[0] > 0.0

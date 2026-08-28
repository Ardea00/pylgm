import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye
from scipy.special import logsumexp

from pylgm.exceptions import DenseReferenceLimitError, OptimizationError
from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian
from pylgm.inference import fit_gaussian
from pylgm.inference.gaussian import _fit_dense, _fit_sparse
import pylgm.inference.gaussian as gaussian_module
from pylgm.optimization.empirical_bayes import OptimizationBounds
from pylgm.optimization.inla import (
    _build_grid,
    _conditional_latent_variances,
    _finite_difference_hessian,
    integrate_inla,
)
from pylgm.optimization.transforms import LogitTransform, LogTransform


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


def test_integrate_inla_matches_fine_1d_quadrature_at_default_grid():
    # Same anchor as test_integrate_inla_matches_fine_1d_quadrature, but at
    # integrate_inla's shipped defaults (grid_step=1.0, radius=3) -- the
    # settings LGM.fit's INLA path (_run_inla) actually uses. The fine-config
    # anchor above measures accuracy at settings the shipped path never calls.
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    result = integrate_inla(family, bounds, fit=fit_gaussian)

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

    # Observed at these defaults: |mean diff| ~= 3.0e-5, var rel diff ~= 2.6e-4 --
    # far tighter than the fine-config anchor's tolerance (atol=5e-3, rtol=5e-2).
    # Using tightened-but-not-brittle tolerances rather than the looser ceiling.
    np.testing.assert_allclose(result.mean[0], ref_mean, atol=1e-3)
    np.testing.assert_allclose(result.covariance[0, 0], ref_var, rtol=1e-2)
    assert result.hyperparameter_marginals()["p"].mean[0] > 0


def test_integrate_inla_produces_a_hyperparameter_distribution():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    result = integrate_inla(family, bounds, fit=fit_gaussian, radius=6)
    # integration explored a genuine posterior (multiple weighted grid points,
    # positive spread on the hyperparameter marginal)
    assert result.diagnostics["inla_grid_points"] >= 3
    assert result.hyperparameter_marginals()["p"].variance[0] > 0.0
    assert result.hyperparameter_marginals()["p"].mean[0] > 0.0


def test_integrate_inla_with_explicit_log_transform_matches_default():
    # Back-compat: an all-LogTransform model must integrate identically whether
    # the transform is left at its OptimizationBounds default or passed
    # explicitly -- the Jacobian generalization is a no-op for LogTransform.
    family = _one_hyperparameter_family()
    default_bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    explicit_bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2, transform=LogTransform())}
    result_default = integrate_inla(family, default_bounds, fit=fit_gaussian, radius=6)
    result_explicit = integrate_inla(family, explicit_bounds, fit=fit_gaussian, radius=6)
    np.testing.assert_allclose(result_default.mean, result_explicit.mean)
    np.testing.assert_allclose(result_default.covariance, result_explicit.covariance)
    np.testing.assert_allclose(
        result_default.log_marginal_likelihood, result_explicit.log_marginal_likelihood
    )


def test_integrate_inla_with_logit_transform_matches_fine_1d_quadrature():
    # A bounded (LogitTransform) hyperparameter's integrated *hyperparameter*
    # marginal (mean/variance of theta itself) must match a direct trapezoidal-
    # style quadrature over theta (uniform prior on theta, so the brute-force
    # reference needs no Jacobian -- its weight is exp(s(theta)) directly).
    #
    # The transform is centred so the empirical-Bayes mode lands near u=0,
    # where d(log_abs_jacobian)/du ~ 0 while d(sum(u))/du = 1 -- the two
    # weightings diverge sharply there, so a wrong Jacobian (e.g. still
    # summing u instead of transform.log_abs_jacobian(u)) is caught with a
    # large margin (observed: correct Jacobian mean/var within ~1e-4 of the
    # brute-force reference; sum(u) instead is off by 0.33 / 0.09).
    family = _one_hyperparameter_family()
    lower, upper = 0.05, 2.15
    # LogitTransform.contains is the open interval, so its own domain must be
    # strictly wider than the [lower, upper] optimization bounds.
    transform = LogitTransform(0.01, 2.2)
    bounds = {"p": OptimizationBounds(1.0, lower, upper, transform=transform)}
    # grid_step/radius/log_density_drop wide enough that the u-grid spans the
    # full internal-space domain (the whitened grid step is grid_step/sqrt(-H),
    # so a narrow default radius would truncate the domain well before its
    # boundary and bias the reference comparison for reasons unrelated to the
    # Jacobian).
    result = integrate_inla(
        family, bounds, fit=fit_gaussian, grid_step=1.0, radius=12, log_density_drop=10.0,
    )

    # independent fine 1-D quadrature directly over theta (natural scale)
    thetas = np.linspace(lower, upper, 4001)
    logw = np.array([
        float(fit_gaussian(family.materialize({"p": float(theta)})).log_marginal_likelihood)
        for theta in thetas
    ])  # uniform prior on theta: no Jacobian needed
    w = np.exp(logw - logsumexp(logw))
    ref_mean = float(np.sum(w * thetas))
    ref_var = float(np.sum(w * thetas ** 2) - ref_mean ** 2)

    hyper = result.hyperparameter_marginals()["p"]
    np.testing.assert_allclose(hyper.mean[0], ref_mean, atol=1e-2)
    np.testing.assert_allclose(hyper.variance[0], ref_var, atol=1e-2)
    assert result.diagnostics["inla_grid_points"] >= 3


def test_conditional_latent_variances_matches_dense_diagonal():
    """_conditional_latent_variances agrees with diag(dense covariance) when a
    conditional fit of the same model is sparse instead of dense."""
    family = _one_hyperparameter_family()
    model = family.materialize({"p": 1.0})
    dense = _fit_dense(model)
    sparse = _fit_sparse(model)
    np.testing.assert_allclose(
        _conditional_latent_variances(sparse),
        np.diag(dense.covariance),
        atol=1e-7,
    )


def test_integrate_inla_sparse_conditional_diagonal_integration(monkeypatch):
    """INLA integrates diagonal latent + predictive variances when conditionals
    are sparse (forced via the same dense-guard monkeypatch used in
    tests/inference/test_sparse.py), matching a dense INLA fit of the same
    model to ~1e-6."""
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}

    dense_result = integrate_inla(family, bounds, fit=fit_gaussian)

    monkeypatch.setattr(gaussian_module, "_MAX_DENSE_LATENT_DIMENSION", 1)
    sparse_result = integrate_inla(family, bounds, fit=fit_gaussian)

    assert sparse_result.diagnostics["inla_conditional_engine"] == "exact_gaussian"

    dense_variance = dense_result.latent_marginals().variance
    sparse_variance = sparse_result.latent_marginals().variance
    assert np.isfinite(sparse_variance).all()
    assert np.isfinite(sparse_result.predictive_variance).all()
    np.testing.assert_allclose(sparse_result.mean, dense_result.mean, atol=1e-6)
    np.testing.assert_allclose(sparse_variance, dense_variance, atol=1e-6)
    np.testing.assert_allclose(
        sparse_result.predictive_variance, dense_result.predictive_variance, atol=1e-6
    )

    with pytest.raises(DenseReferenceLimitError):
        sparse_result.covariance

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
    _explore_grid,
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


def _stub_evaluate(log_density):
    """An `evaluate` returning only a log density; the rest is unused by exploration."""
    return lambda u: (float(log_density(u)), None, None, None)


_WIDE = (np.array([-np.inf]), np.array([np.inf]))


def test_explore_grid_steps_out_until_the_density_drops():
    """The extent is set by the density, not by a fixed radius.

    -H = 4 gives a whitened step of 1/sqrt(4) = 0.5 in u. A Gaussian log density
    -0.5 * 4 * u^2 falls `explore_drop`=6 below the mode at |u| = sqrt(3) ~ 1.73,
    so exploration stops at the first lattice point past it, |u| = 2.0 (z = 4).
    """
    grid, points = _explore_grid(
        np.array([0.0]), np.array([[-4.0]]), _stub_evaluate(lambda u: -0.5 * 4 * u[0] ** 2),
        internal_lower=_WIDE[0], internal_upper=_WIDE[1],
        grid_step=1.0, max_radius=20, explore_drop=6.0,
    )
    offsets = np.sort(np.round(grid[:, 0], 6))
    np.testing.assert_allclose(offsets, np.arange(-4, 5) * 0.5)
    assert len(points) == len(grid) == 9


def test_explore_grid_reaches_further_for_a_heavier_tail():
    """The point of the change: a flatter log density earns a wider grid, which a
    fixed radius could not give it."""
    def extent(log_density):
        grid, _ = _explore_grid(
            np.array([0.0]), np.array([[-1.0]]), _stub_evaluate(log_density),
            internal_lower=_WIDE[0], internal_upper=_WIDE[1],
            grid_step=1.0, max_radius=40, explore_drop=6.0,
        )
        return float(np.max(np.abs(grid[:, 0])))

    gaussian = extent(lambda u: -0.5 * u[0] ** 2)
    heavy = extent(lambda u: -np.log1p(u[0] ** 2))      # Cauchy-like: much flatter
    assert heavy > 3 * gaussian, (gaussian, heavy)


def test_explore_grid_stops_at_the_declared_domain():
    grid, _ = _explore_grid(
        np.array([0.0]), np.array([[-1.0]]), _stub_evaluate(lambda u: 0.0),  # flat: never drops
        internal_lower=np.array([-1.5]), internal_upper=np.array([2.5]),
        grid_step=1.0, max_radius=50, explore_drop=6.0,
    )
    assert grid[:, 0].min() >= -1.5 and grid[:, 0].max() <= 2.5


def test_pruning_skips_predictably_negligible_corners():
    """Whitening makes the local Gaussian isotropic, so a corner is sqrt(d) times
    further out than an axis point with the same per-axis index -- and its
    predicted log-density drop is that much larger. The corner is skipped; the
    axis point at the same index is not."""
    grid, _ = _explore_grid(
        np.zeros(2), -np.eye(2), _stub_evaluate(lambda u: -0.5 * float(u @ u)),
        internal_lower=np.full(2, -np.inf), internal_upper=np.full(2, np.inf),
        grid_step=1.0, max_radius=10, explore_drop=2.5, prune_drop=3.0,
    )
    present = {tuple(np.round(g, 6)) for g in grid}
    assert (2.0, 0.0) in present          # axis:   predicted drop 0.5*4 = 2.0 <= 3.0
    assert (2.0, 2.0) not in present      # corner: predicted drop 0.5*8 = 4.0 >  3.0


def test_pruning_never_drops_an_already_measured_point():
    """A heavier-than-Gaussian tail is exactly where the prediction is wrong, so
    axis probes -- whose density was measured, not assumed -- survive pruning."""
    heavy = _stub_evaluate(lambda u: -np.log1p(float(u @ u)))   # Cauchy-like
    grid, _ = _explore_grid(
        np.zeros(2), -np.eye(2), heavy,
        internal_lower=np.full(2, -np.inf), internal_upper=np.full(2, np.inf),
        grid_step=1.0, max_radius=12, explore_drop=6.0, prune_drop=1.0,  # brutal pruning
    )
    reach = max(abs(g[0]) for g in grid if abs(g[1]) < 1e-9)
    # Gaussian prediction would allow |z| <= sqrt(2) with prune_drop=1.0; the
    # measured probes go far past it because the tail is genuinely flat.
    assert reach > 3.0, reach


def test_explore_grid_guards_dimensionality():
    center = np.zeros(6)
    hessian = -np.eye(6)
    with pytest.raises(OptimizationError, match="grid"):
        _explore_grid(
            center, hessian, _stub_evaluate(lambda u: -0.5 * float(u @ u)),
            internal_lower=np.full(6, -np.inf), internal_upper=np.full(6, np.inf),
            max_radius=10, explore_drop=6.0, max_grid_points=4096,
        )


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
    result = integrate_inla(family, bounds, fit=fit_gaussian, grid_step=0.75, max_radius=6)

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
    result = integrate_inla(family, bounds, fit=fit_gaussian, max_radius=6)
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
    result_default = integrate_inla(family, default_bounds, fit=fit_gaussian, max_radius=6)
    result_explicit = integrate_inla(family, explicit_bounds, fit=fit_gaussian, max_radius=6)
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
    # grid_step/max_radius/explore_drop wide enough that the u-grid spans the
    # full internal-space domain (the whitened grid step is grid_step/sqrt(-H),
    # so a narrow exploration would truncate the domain well before its
    # boundary and bias the reference comparison for reasons unrelated to the
    # Jacobian).
    result = integrate_inla(
        family, bounds, fit=fit_gaussian, grid_step=1.0, max_radius=12,
        explore_drop=10.0, log_density_drop=10.0,
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


def test_pruning_leaves_the_integrated_result_unchanged():
    """The point of pruning: it removes only points the weighting would discard,
    so the answer is bit-identical while the cost is not.

    This is what licenses pruning at all -- it is a cost optimisation, not an
    approximation, and it must be checked as one.
    """
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    pruned = integrate_inla(family, bounds, fit=fit_gaussian, prune_slack=4.0)
    unpruned = integrate_inla(family, bounds, fit=fit_gaussian, prune_slack=1e9,
                              max_grid_points=200_000)
    np.testing.assert_array_equal(pruned.mean, unpruned.mean)
    np.testing.assert_array_equal(pruned.covariance, unpruned.covariance)
    assert pruned.log_marginal_likelihood == unpruned.log_marginal_likelihood


# ---------------------------------------------------------------------------
# CCD: the design used when filling a region is no longer affordable.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [1, 2, 3, 6, 8, 12])
def test_ccd_design_reproduces_a_standard_gaussians_moments(d):
    """The property the weights are derived from: sum_i w_i z_i z_i^T = I.

    Everything else about the design (Hadamard core, axial points, one sphere)
    exists to make this hold with O(d) points, so this is the check that the
    construction is what it claims to be rather than merely plausible.
    """
    from pylgm.optimization.inla import _ccd_design

    points, weights = _ccd_design(d, f0=1.1)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()                      # f0 > 1 keeps the centre positive
    second = np.einsum("i,ij,ik->jk", weights, points, points)
    np.testing.assert_allclose(second, np.eye(d), atol=1e-12)
    radii = np.linalg.norm(points[1:], axis=1)      # rotatable: one sphere
    np.testing.assert_allclose(radii, radii[0])
    assert len(points) <= 4 * d + 4                 # O(d), not O(c^d)


def test_ccd_weighting_is_exact_for_a_gaussian_target():
    """Pins the whole scheme -- design, centring, importance ratio and the
    log-marginal constant -- against a target whose answer is known in closed form.

    With s(u) = -0.5 (u-m)'A(u-m) and a log transform's Jacobian sum(u), the
    posterior in u is exactly Gaussian, so a second-order design must reproduce
    its mean, covariance and normalising constant exactly. Any error here is an
    implementation bug rather than the approximation CCD is entitled to make.
    """
    from scipy.special import logsumexp

    from pylgm.optimization.inla import _ccd_design, _whitening_directions

    rng = np.random.default_rng(0)
    for d in (2, 3, 5):
        basis = rng.normal(size=(d, d))
        precision = basis @ basis.T + d * np.eye(d)
        mode = rng.normal(size=d)
        ones = np.ones(d)

        true_mean = mode + np.linalg.solve(precision, ones)
        true_cov = np.linalg.inv(precision)
        true_lml = (0.5 * d * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(precision)[1]
                    + ones @ mode + 0.5 * ones @ np.linalg.solve(precision, ones))

        centre = mode + np.linalg.solve(precision, ones)   # the code's Newton step
        directions = _whitening_directions(-precision)
        design, design_weights = _ccd_design(d, 1.1)
        us = np.array([centre + directions @ z for z in design])
        s = np.array([-0.5 * (u - mode) @ precision @ (u - mode) for u in us])
        log_w = (np.log(design_weights) + s + us.sum(axis=1)
                 + 0.5 * (design * design).sum(axis=1))
        w = np.exp(log_w - logsumexp(log_w))
        mean = w @ us
        cov = np.einsum("i,ij,ik->jk", w, us - mean, us - mean)
        lml = (logsumexp(log_w) + 0.5 * d * np.log(2 * np.pi)
               - 0.5 * np.sum(np.log(np.linalg.eigvalsh(precision))))

        np.testing.assert_allclose(mean, true_mean, atol=1e-10)
        np.testing.assert_allclose(cov, true_cov, atol=1e-10)
        assert lml == pytest.approx(true_lml, abs=1e-10)


def test_ccd_rejects_a_scaling_that_would_starve_the_centre():
    from pylgm.optimization.inla import _ccd_design

    with pytest.raises(ValueError, match="f0"):
        _ccd_design(3, f0=1.0)      # centre weight 1 - 1/f0^2 would be zero


def test_auto_keeps_the_grid_when_it_is_affordable():
    """`auto` must not change what small models report -- the grid is the more
    accurate scheme and stays in charge wherever it fits the budget."""
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    auto = integrate_inla(family, bounds, fit=fit_gaussian, int_strategy="auto")
    grid = integrate_inla(family, bounds, fit=fit_gaussian, int_strategy="grid")
    assert auto.diagnostics["inla_int_strategy"] == "grid"
    np.testing.assert_array_equal(auto.mean, grid.mean)
    assert auto.log_marginal_likelihood == grid.log_marginal_likelihood


def test_auto_switches_to_a_design_when_the_grid_would_blow_the_budget():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    # A budget the one-dimensional grid cannot meet forces the alternative.
    result = integrate_inla(family, bounds, fit=fit_gaussian, max_grid_points=3)
    assert result.diagnostics["inla_int_strategy"] == "korobov"
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.log_marginal_likelihood)


@pytest.mark.parametrize("count", [64, 256])
def test_korobov_lattice_fills_the_gaussian(count):
    """A lattice rule buys accuracy with equidistribution rather than polynomial
    exactness, so its weights are equal and positive and its second moments are
    only approximately right -- which is the trade that lets `count` tune
    accuracy without changing the design."""
    from pylgm.optimization.inla import _korobov_design

    for d in (2, 6, 12):
        points, weights = _korobov_design(d, count=count)
        assert len(points) == count
        np.testing.assert_allclose(weights, 1.0 / count)   # equal and positive
        assert weights.sum() == pytest.approx(1.0)
        second = np.einsum("i,ij,ik->jk", weights, points, points)
        assert np.abs(second - np.eye(d)).max() < 0.5
        assert np.isfinite(points).all()                   # no inverse-CDF poles


def test_korobov_is_reproducible_and_refinable():
    """Seeded shift, so a fit is reproducible; and more points is the knob a
    fixed design does not have."""
    from pylgm.optimization.inla import _korobov_design

    first, _ = _korobov_design(4, count=64)
    again, _ = _korobov_design(4, count=64)
    np.testing.assert_array_equal(first, again)
    assert len(_korobov_design(4, count=256)[0]) == 256


def test_korobov_rejects_a_degenerate_point_count():
    from pylgm.optimization.inla import _korobov_design

    with pytest.raises(ValueError, match="four points"):
        _korobov_design(3, count=2)


def test_int_strategy_is_validated():
    family = _one_hyperparameter_family()
    bounds = {"p": OptimizationBounds(1.0, 1e-2, 1e2)}
    with pytest.raises(ValueError, match="int_strategy"):
        integrate_inla(family, bounds, fit=fit_gaussian, int_strategy="lattice")

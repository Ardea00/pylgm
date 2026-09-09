import math
from collections.abc import Callable
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor
from scipy.special import logsumexp
from scipy.stats import norm

from pylgm.exceptions import NumericalError, OptimizationError, UnsupportedEngineError
from pylgm.inference import LaplaceResult, fit_gaussian
from pylgm.inference.result import (
    GaussianMarginals,
    INLAResult,
    ModelCriteria,
    SkewNormalMarginals,
    TabulatedMarginals,
    quadratic_form_diagonal,
)
from pylgm.optimization.empirical_bayes import optimize_empirical_bayes

_SN_C = (4.0 - math.pi) * math.sqrt(2.0) / math.pi ** 1.5
_SN_R_MAX = 5.0  # cap on |a/omega| for robustness (|skew| ~ 0.937 at r=5, shy of the ~0.995 supremum)
_SCALE_FLOOR = 1e-12  # degenerate (sigma_i == 0) coordinate: near-point-mass marginal


def _conditional_latent_variances(fit) -> np.ndarray:
    """diag(Sigma) of a conditional fit, dense or sparse."""
    posterior = getattr(fit, "_sparse_posterior", None)
    if posterior is not None and fit._covariance is None:
        return posterior.marginal_variances()
    return np.diag(np.asarray(fit.covariance, float))


def _conditional_predictive_variances(fit, dense_design) -> np.ndarray:
    """diag(design Sigma design^T) of a conditional fit, dense or sparse."""
    posterior = getattr(fit, "_sparse_posterior", None)
    if posterior is not None and fit._covariance is None:
        return posterior.predictive_variances(dense_design)
    cov = np.asarray(fit.covariance, float)
    return quadratic_form_diagonal(dense_design, cov)


def _solve_omega(r: np.ndarray) -> np.ndarray:
    # solve omega^2 (1 - 2 delta(omega)^2 / pi) = 1, delta = r*omega/sqrt(1+r^2 omega^2)
    # monotone increasing in omega on (0, inf); bisection (vectorized).
    lo = np.zeros_like(r)
    hi = np.full_like(r, 1e6)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        a = r * mid
        delta = a / np.sqrt(1.0 + a * a)
        val = mid * mid * (1.0 - 2.0 * delta * delta / math.pi) - 1.0
        hi = np.where(val > 0.0, mid, hi)
        lo = np.where(val > 0.0, lo, mid)
    return 0.5 * (lo + hi)


def _fit_skew_normal(gamma1, gamma3):
    """Appendix-B skew-normal fit (Rue-Martino-Chopin 2009): standardized
    skew-normal (mean=gamma1, variance=1) whose log-density cubic coefficient
    matches gamma3 via RMC eq 32."""
    gamma1 = np.asarray(gamma1, dtype=float)
    gamma3 = np.asarray(gamma3, dtype=float)
    r = np.sign(gamma3) * (np.abs(gamma3) / _SN_C) ** (1.0 / 3.0)
    clamped = np.abs(r) > _SN_R_MAX
    r = np.clip(r, -_SN_R_MAX, _SN_R_MAX)
    omega = np.where(r == 0.0, 1.0, _solve_omega(r))
    a = r * omega
    delta = a / np.sqrt(1.0 + a * a)
    xi = gamma1 - omega * delta * math.sqrt(2.0 / math.pi)
    return xi, omega, a, clamped


def _simplified_laplace_marginals(design, offset, y, grid):
    """RMC (2009) simplified-Laplace skew-normal latent marginals.

    Per grid point (weight, conditional_fit, likelihood): computes the location
    (gamma1) and skewness (gamma3) corrections from the conditional Gaussian fit's
    mean/covariance and the likelihood's third derivative, fits a standardized
    skew-normal (`_fit_skew_normal`), and transforms into x_i coordinates.

    Returns (location, scale, shape, weights, clamped_count), each of the first
    four arrays shape (p, n_grid).
    """
    dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
    offset = np.asarray(offset, float)
    y = np.asarray(y, float)
    points = list(grid)
    p = points[0][1].mean.shape[0]
    n_grid = len(points)
    location = np.zeros((p, n_grid))
    scale = np.zeros((p, n_grid))
    shape = np.zeros((p, n_grid))
    weights = np.zeros((p, n_grid))
    clamped_count = 0
    for k, (w, fit, likelihood) in enumerate(points):
        m = np.asarray(fit.mean, float)
        cov = np.asarray(fit.covariance, float)
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        eta_mean = offset + dense @ m
        d3 = np.asarray(likelihood.third_derivative(eta_mean, y), float)   # (n,)
        cx_eta = cov @ dense.T                                             # (p, n) cov(x_i, eta_j)
        eta_var = np.clip(quadratic_form_diagonal(dense, cov), 0.0, None)  # (n,)
        sigma_eta = np.sqrt(eta_var)
        safe_si = np.where(sigma > 0, sigma, 1.0)
        # gamma3_i = (1/sigma_i^3) sum_j d3_j cov(x_i,eta_j)^3
        gamma3 = (cx_eta ** 3 @ d3) / safe_si ** 3
        # a_ij = cov / (sigma_i sigma_eta_j); gamma1_i = (1/(2 sigma_i)) sum_j sigma_eta_j^2 (1-a_ij^2) d3_j cov
        safe_se = np.where(sigma_eta > 0, sigma_eta, 1.0)
        a_ij = cx_eta / (safe_si[:, None] * safe_se[None, :])
        term = (sigma_eta ** 2)[None, :] * (1.0 - a_ij ** 2) * d3[None, :] * cx_eta
        gamma1 = term.sum(axis=1) / (2.0 * safe_si)
        # degenerate sigma_i -> no correction
        gamma1 = np.where(sigma > 0, gamma1, 0.0)
        gamma3 = np.where(sigma > 0, gamma3, 0.0)
        xi, omega, sn_a, clamped = _fit_skew_normal(gamma1, gamma3)
        clamped_count += int(clamped.sum())
        location[:, k] = m + sigma * xi
        scale[:, k] = np.where(sigma > 0.0, sigma * omega, _SCALE_FLOOR)
        shape[:, k] = sn_a
        weights[:, k] = w
    return location, scale, shape, weights, clamped_count


def _logdet_spd(matrix):
    try:
        factor = cho_factor(matrix, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise NumericalError("full-Laplace P_{-i} must be positive definite") from error
    return 2.0 * np.sum(np.log(np.diag(factor[0])))


def _full_laplace_marginals(design, offset, y, grid, *,
                            n_abscissae=7, grid_points=201, grid_radius=8.0):
    """RMC (2009) full-Laplace tabulated latent marginals (eqs 12/13/16/17).

    Unconstrained models only (caller guards). Per latent i, per theta grid point,
    per Gauss-Hermite abscissa x_i, computes the conditional-mean configuration
    (eq 13), the full-Laplace log-density log pi~_LA via the joint density and the
    pi~_GG Laplace-approximation determinant (eq 12), and a cubic-spline correction
    against the base Gaussian (eq 17). Mixed over the theta-grid into a
    TabulatedMarginals.

    Each `grid` entry is `(weight, conditional_fit, likelihood, precision)` — the
    prior precision Q is theta-dependent (CompiledFamily.materialize scales block
    precisions per hyperparameter), so it travels with its own grid point rather
    than being shared across the whole grid.
    """
    X = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
    offset = np.asarray(offset, float)
    y = np.asarray(y, float)
    points = list(grid)
    p = points[0][1].mean.shape[0]
    std_nodes = np.polynomial.hermite.hermgauss(n_abscissae)[0] * np.sqrt(2.0)  # ~N(0,1) abscissae

    x_out = np.zeros((p, grid_points))
    dens_out = np.zeros((p, grid_points))
    for i in range(p):
        # common per-latent grid from the mixture spread
        mu_bar = sum(w * f.mean[i] for w, f, _, _ in points)
        # use max per-θ std (not average) so the common grid covers the widest mixture component without truncating its tails
        sig_max = max(np.sqrt(max(f.covariance[i, i], 0.0)) for _, f, _, _ in points)
        gx = np.linspace(mu_bar - grid_radius * sig_max, mu_bar + grid_radius * sig_max, grid_points)
        mixed = np.zeros(grid_points)
        keep = [c for c in range(p) if c != i]
        for w, fit, likelihood, precision in points:
            m = np.asarray(fit.mean, float)
            cov = np.asarray(fit.covariance, float)
            Q_k = precision.toarray() if hasattr(precision, "toarray") else np.asarray(precision, float)
            sigma_i = np.sqrt(max(cov[i, i], 0.0))
            if sigma_i <= 0.0:
                continue
            abscissae = m[i] + sigma_i * std_nodes
            delta = np.empty(n_abscissae)
            for a, xi in enumerate(abscissae):
                x = m.copy()
                x[keep] = m[keep] + cov[keep, i] / cov[i, i] * (xi - m[i])   # eq 13 conditional mean
                x[i] = xi
                eta = offset + X @ x
                loglik = float(likelihood.pointwise_log_density(eta, y).sum())
                joint = -0.5 * float(x @ Q_k @ x) + loglik
                wgt = likelihood.working_weights(eta, y)
                info = Q_k + (X.T * wgt) @ X
                info_mi = np.delete(np.delete(info, i, axis=0), i, axis=1)
                half_logdet = 0.5 * _logdet_spd(info_mi) if p > 1 else 0.0
                log_la = joint - half_logdet
                log_g = -0.5 * std_nodes[a] ** 2 - np.log(sigma_i) - 0.5 * np.log(2 * np.pi)
                delta[a] = log_la - log_g
            spline = CubicSpline(abscissae, delta, extrapolate=False)
            s = spline(gx)
            s[gx < abscissae[0]] = delta[0]      # constant tail extrapolation
            s[gx > abscissae[-1]] = delta[-1]
            log_dens = -0.5 * ((gx - m[i]) / sigma_i) ** 2 - np.log(sigma_i) - 0.5 * np.log(2 * np.pi) + s
            dens = np.exp(log_dens - log_dens.max())
            area = np.trapezoid(dens, gx)
            if not np.isfinite(area) or area <= 0:
                raise NumericalError("full-Laplace density normalization failed")
            mixed += w * (dens / area)
        total = np.trapezoid(mixed, gx)
        if not np.isfinite(total) or total <= 0:
            raise NumericalError("full-Laplace mixture normalization failed")
        x_out[i] = gx
        dens_out[i] = mixed / total
    return TabulatedMarginals(x_out, dens_out)


def _finite_difference_hessian(
    func: Callable[[np.ndarray], float], center: np.ndarray, step: float = 1e-3
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    d = center.size
    hessian = np.zeros((d, d))
    f0 = func(center)
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = step
        f_plus = func(center + ei)
        f_minus = func(center - ei)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step * step)
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = step
            f_pp = func(center + ei + ej)
            f_pm = func(center + ei - ej)
            f_mp = func(center - ei + ej)
            f_mm = func(center - ei - ej)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _whitening_directions(hessian: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    negative = -np.asarray(hessian, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(negative)
    clamped = np.clip(eigenvalues, ridge, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(clamped))


def _explore_grid(
    center: np.ndarray, hessian: np.ndarray, evaluate, *,
    internal_lower: np.ndarray, internal_upper: np.ndarray,
    grid_step: float = 1.0, max_radius: int = 10, explore_drop: float = 10.0,
    prune_drop: float | None = None, max_grid_points: int = 4096,
):
    """Explore outward from the mode until the log density drops, and return
    ``(grid, payloads)`` for every point evaluated inside the declared domain.

    This is the grid strategy of Rue, Martino & Chopin (2009, sec. 3.1): step
    along each whitened direction from the mode and stop when the log density
    has fallen ``explore_drop`` below it. The extent is therefore set by the
    posterior rather than by a fixed radius, which matters because the two
    disagree badly for a weakly identified hyperparameter: the curvature at the
    mode understates a long tail, and a fixed radius truncates it.

    A fixed radius is also wasteful in the other direction -- it evaluates the
    whole ``(2r+1)^d`` box and then discards whatever falls below the weighting
    threshold, paying a full conditional fit for each discarded point.

    Every evaluation is cached by lattice index, so the axis probes that measure
    the extent are reused as grid points rather than recomputed.

    Filling the box costs ``prod(extent)`` fits, and almost all of that is
    corners: whitening makes the local Gaussian isotropic, so lattice point ``z``
    has a *predicted* log-density drop of exactly ``0.5 * grid_step^2 * ||z||^2``,
    and a corner in ``d`` dimensions is ``sqrt(d)`` times further out than an axis
    point with the same per-axis index. ``prune_drop`` skips points whose
    predicted drop exceeds it -- they would be dropped from the integration
    weights anyway -- which turns the cost from the volume of a box into the
    volume of an ellipsoid, the difference between the two growing with ``d``.

    A point already measured is never pruned: the axis probes are kept whatever
    the Gaussian predicts for them, because for those the density is known rather
    than assumed, and a heavier-than-Gaussian tail is exactly the case where the
    prediction is wrong.
    """
    center = np.asarray(center, dtype=float)
    d = center.size
    directions = _whitening_directions(hessian)
    cache: dict[tuple[int, ...], object] = {}

    def at(z: tuple[int, ...]):
        if z not in cache:
            offset = grid_step * directions @ np.asarray(z, dtype=float)
            u = center + offset
            inside = bool(np.all(u >= internal_lower) and np.all(u <= internal_upper))
            # A point outside the declared domain would clip to the same boundary
            # theta as its neighbours, where dtheta/du is zero and the correct
            # Jacobian contribution is zero too; including it with the raw `u` as
            # if the mapping stayed invertible over-weights the boundary.
            cache[z] = (u, evaluate(u)) if inside else None
        return cache[z]

    origin = at((0,) * d)
    if origin is None:
        raise OptimizationError("the empirical-Bayes mode lies outside the declared bounds")
    s_mode = origin[1][0]

    extents = np.zeros((d, 2), dtype=int)
    for axis in range(d):
        for slot, sign in ((0, -1), (1, 1)):
            reach = 0
            for step in range(1, max_radius + 1):
                z = [0] * d
                z[axis] = sign * step
                probe = at(tuple(z))
                if probe is None:
                    break
                reach = step
                if probe[1][0] < s_mode - explore_drop:
                    break
            extents[axis, slot] = reach

    total = sum(
        1
        for z in product(*(range(-extents[a, 0], extents[a, 1] + 1) for a in range(d)))
        if z in cache
        or prune_drop is None
        or 0.5 * grid_step ** 2 * float(np.dot(z, z)) <= prune_drop
    )
    if total > max_grid_points:
        raise OptimizationError(
            f"INLA grid would need {total} points for {d} hyperparameters; "
            f"exceeds max_grid_points={max_grid_points}"
        )

    grid, payloads = [], []
    for z in product(*(range(-extents[a, 0], extents[a, 1] + 1) for a in range(d))):
        if z not in cache and prune_drop is not None:
            predicted = 0.5 * grid_step ** 2 * float(np.dot(z, z))
            if predicted > prune_drop:
                continue
        entry = at(z)
        if entry is None:
            continue
        grid.append(entry[0])
        payloads.append(entry[1])
    return np.asarray(grid), payloads


def _theta_marginals(names, grid, s_values, transforms, theta_mean, theta_sq, points=513):
    """Posterior marginals for the hyperparameters.

    With one hyperparameter the grid is a line in ``u``, and ``s(u) = log p(y|theta)
    + log pi(theta)`` is the unnormalised log posterior evaluated on it. Splining
    that log density and mapping it through the transform gives the marginal
    itself, tabulated -- rather than a Gaussian matched to its first two moments,
    which for a positive, right-skewed precision is wrong in both tails.

    Every grid point is used, not only the ones the integration weights retain:
    they have all been evaluated by the time we get here, and the density-drop
    filter that trims the integration would truncate this marginal's tails at
    roughly 2.2 sigma instead of the grid's own 3.

    With more than one hyperparameter the grid is a lattice rotated onto the
    whitened Hessian's directions, so projections onto a single axis scatter and
    a marginal cannot be read off it this way; those keep the moment match.
    """
    moment_matched = {
        name: GaussianMarginals(
            np.array([theta_mean[name]]),
            np.array([max(theta_sq[name] - theta_mean[name] ** 2, 0.0)]),
        )
        for name in names
    }
    if len(names) != 1 or len(s_values) < 4:
        return moment_matched

    u = np.asarray(grid, dtype=float)[:, 0]
    order = np.argsort(u)
    u, s = u[order], np.asarray(s_values, dtype=float)[order]
    if not np.all(np.diff(u) > 0):
        return moment_matched

    fine = np.linspace(u[0], u[-1], points)
    log_density = CubicSpline(u, s - s.max())(fine)
    theta = np.array([transforms[0].from_internal(value) for value in fine])
    if not np.all(np.diff(theta) > 0):   # transform must stay strictly monotone
        return moment_matched
    density = np.exp(log_density - log_density.max())
    if not np.all(np.isfinite(density)) or density.max() <= 0.0:
        return moment_matched
    # p(theta | y) is proportional to exp(s); TabulatedMarginals normalises it
    # over this theta grid, so no Jacobian belongs here -- s is already a
    # function of theta, and the grid carries the change of variable.
    return {names[0]: TabulatedMarginals(theta[None, :], density[None, :])}


def _hadamard(order: int) -> np.ndarray:
    """Sylvester Hadamard matrix of the given power-of-two order."""
    matrix = np.ones((1, 1))
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _ccd_design(d: int, f0: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    """A rotatable central composite design in whitened space: points and weights.

    Filling a region costs points exponential in ``d`` however cleverly the region
    is shaped, so past a handful of hyperparameters the only way out is to stop
    filling and start *designing*. This is the design R-INLA switches to
    (``int.strategy="ccd"``; Rue, Martino & Chopin 2009, sec. 6.5), and it needs
    ``O(d)`` points rather than ``O(c^d)``.

    The design is a factorial core plus axial points plus the mode:

    * the core is ``d`` columns of a Sylvester Hadamard matrix, so its columns are
      orthogonal and every row is a ``+-1`` vector of norm ``sqrt(d)``. A Hadamard
      core needs only the next power of two above ``d``, where a full ``2^d``
      factorial would defeat the purpose;
    * the ``2d`` axial points sit at ``+-sqrt(d)`` on each coordinate, so they
      share that norm;
    * everything off-centre is then scaled by ``f0``, putting the whole design on
      one sphere of radius ``f0 * sqrt(d)`` -- what makes it *rotatable*, i.e.
      equally accurate in every direction.

    The weights follow from requiring the design to reproduce a standard Gaussian's
    second moment, ``sum_i w_i z_i z_i^T = I``. Orthogonality makes the off-centre
    sum ``f0^2 * n_p * I``, so those points share ``w = 1 / (f0^2 * n_p)`` and the
    centre takes the remainder ``1 - 1/f0^2`` -- positive exactly when ``f0 > 1``,
    which is why ``f0`` is set slightly above one.
    """
    if d < 1:
        raise ValueError("CCD needs at least one dimension")
    if f0 <= 1.0:
        raise ValueError("f0 must exceed 1 so the centre keeps positive weight")
    order = 1
    while order < d + 1:
        order *= 2
    # column 0 of a Sylvester Hadamard matrix is all ones; skip it so the core is
    # centred, and take d of the remaining mutually orthogonal columns.
    core = _hadamard(order)[:, 1:d + 1]
    axial = np.sqrt(d) * np.vstack([np.eye(d), -np.eye(d)])
    offcentre = f0 * np.vstack([core, axial])
    count = offcentre.shape[0]
    points = np.vstack([np.zeros((1, d)), offcentre])
    weights = np.concatenate([[1.0 - 1.0 / f0 ** 2], np.full(count, 1.0 / (f0 ** 2 * count))])
    return points, weights


def _ccd_grid(center, hessian, evaluate, *, internal_lower, internal_upper, f0=1.1):
    """Evaluate a CCD design, returning ``(grid, payloads, z_sq, design_weights)``.

    ``z_sq`` is each point's squared whitened radius, which the caller needs for
    the importance correction: the design integrates the *Gaussian* implied by the
    Hessian, and dividing by that Gaussian recovers the true posterior. Without
    that correction CCD would report the Laplace approximation back to itself and
    the evaluated densities would do no work.
    """
    center = np.asarray(center, dtype=float)
    d = center.size
    directions = _whitening_directions(hessian)
    design, weights = _ccd_design(d, f0)

    grid, payloads, z_sq, kept_weights = [], [], [], []
    for z, weight in zip(design, weights, strict=True):
        u = center + directions @ z
        if not (np.all(u >= internal_lower) and np.all(u <= internal_upper)):
            # Outside the declared domain the transform is not invertible, so the
            # point cannot contribute; the surviving weights renormalise below.
            continue
        grid.append(u)
        payloads.append(evaluate(u))
        z_sq.append(float(z @ z))
        kept_weights.append(weight)
    if not grid:
        raise OptimizationError("the CCD design lies entirely outside the declared bounds")
    return (np.asarray(grid), payloads, np.asarray(z_sq), np.asarray(kept_weights))


def _predicted_grid_points(d, *, depth, prune_drop, grid_step, max_radius, cap=2_000_000):
    """How many points the explored-and-pruned box would hold under the Gaussian.

    Pure arithmetic on the design -- no conditional fits. ``auto`` needs to choose
    a strategy *before* paying for one, and the alternative (explore, blow the
    budget, fall back) would spend `O(d * max_radius)` fits to learn what this
    computes for free.
    """
    reach = min(int(np.ceil(np.sqrt(2.0 * depth) / grid_step)), max_radius)
    if (2 * reach + 1) ** d > cap:
        return cap + 1
    return sum(
        1
        for z in product(range(-reach, reach + 1), repeat=d)
        if 0.5 * grid_step ** 2 * float(np.dot(z, z)) <= prune_drop
    )


def _designed_grid(center, hessian, evaluate, design, weights, *,
                   internal_lower, internal_upper):
    """Evaluate a fixed design, returning ``(grid, payloads, z_sq, weights)``."""
    center = np.asarray(center, dtype=float)
    directions = _whitening_directions(hessian)
    grid, payloads, z_sq, kept = [], [], [], []
    for z, weight in zip(design, weights, strict=True):
        u = center + directions @ z
        if not (np.all(u >= internal_lower) and np.all(u <= internal_upper)):
            # Outside the declared domain the transform is not invertible.
            continue
        grid.append(u)
        payloads.append(evaluate(u))
        z_sq.append(float(z @ z))
        kept.append(weight)
    if not grid:
        raise OptimizationError("the integration design lies entirely outside the bounds")
    return np.asarray(grid), payloads, np.asarray(z_sq), np.asarray(kept)


def _korobov_design(d: int, count: int = 128, seed: int = 0):
    """A randomly shifted rank-1 lattice mapped to ``N(0, I_d)``: points, weights.

    Where CCD and Smolyak buy accuracy with polynomial exactness, a lattice rule
    buys it with *equidistribution*: ``count`` points spread to fill the Gaussian
    evenly, each with weight ``1/count``. That makes it the only one of the three
    whose weights are all positive and equal, which matters here for two reasons
    beyond conditioning -- the model criteria treat the integration weights as a
    probability mixture, and a negative weight makes CPO non-finite; and accuracy
    is tuned by raising ``count`` rather than by moving to a fundamentally more
    expensive design.

    The generating vector is Korobov's ``(1, a, a^2, ...) mod count``. ``a`` is
    chosen by a small search maximising the minimum spectral distance, which is
    cheap here because it runs on the lattice alone -- no conditional fits.

    A random shift keeps the rule from aligning with any structure in the
    integrand; the shift is seeded, so a fit stays reproducible.
    """
    if d < 1:
        raise ValueError("a lattice needs at least one dimension")
    if count < 4:
        raise ValueError("a lattice needs at least four points")
    best_a, best_score = 1, -np.inf
    candidates = [a for a in range(2, count) if math.gcd(a, count) == 1]
    for a in candidates[: 512]:
        powers = np.array([pow(a, j, count) for j in range(d)], dtype=float)
        # spectral-style score: keep the generator away from small-index aliases
        k = np.arange(1, min(count, 64))
        residues = np.minimum((np.outer(k, powers) % count) / count,
                              1.0 - (np.outer(k, powers) % count) / count)
        score = float(np.min(residues.sum(axis=1)))
        if score > best_score:
            best_a, best_score = a, score
    powers = np.array([pow(best_a, j, count) for j in range(d)], dtype=float)
    indices = np.arange(count, dtype=float)[:, None]
    shift = np.random.default_rng(seed).random(d)[None, :]
    unit = np.mod(indices * powers[None, :] / count + shift, 1.0)
    # Keep the inverse CDF away from its poles; the endpoints map to +-infinity.
    unit = np.clip(unit, 1.0 / (2 * count), 1.0 - 1.0 / (2 * count))
    points = norm.ppf(unit)
    weights = np.full(count, 1.0 / count)
    return points, weights


def integrate_inla(
    family, bounds, *, initial=None, fit=None, penalty=None, allow_large_dense=False,
    grid_step=1.0, max_radius=10, explore_drop=10.0, log_density_drop=12.0,
    prune_slack=4.0, int_strategy="auto", ccd_f0=1.1, korobov_points=128,
    max_grid_points=4096, latent_strategy="gaussian",
) -> INLAResult:
    names = tuple(family.parameter_names)
    conditional_fit = fit if fit is not None else fit_gaussian
    transforms = [bounds[name].transform for name in names]
    lower = np.array([bounds[name].lower for name in names])
    upper = np.array([bounds[name].upper for name in names])

    def evaluate(u):
        theta = {
            name: float(np.clip(transforms[i].from_internal(value), lower[i], upper[i]))
            for i, (name, value) in enumerate(zip(names, u, strict=True))
        }
        compiled = family.materialize(theta)
        conditional = (
            conditional_fit(compiled, allow_large_dense=True)
            if allow_large_dense else conditional_fit(compiled)
        )
        s_value = float(conditional.log_marginal_likelihood)
        if penalty is not None:
            s_value += float(penalty(theta))
        return s_value, conditional, theta, compiled

    eb = optimize_empirical_bayes(
        family, bounds, initial=initial, fit=conditional_fit, penalty=penalty,
        allow_large_dense=allow_large_dense,
    )
    u_star = np.array(
        [transforms[i].to_internal(eb.parameters[name]) for i, name in enumerate(names)]
    )
    hessian = _finite_difference_hessian(lambda u: evaluate(u)[0], u_star)
    internal_lower = np.array([transforms[i].to_internal(lower[i]) for i in range(len(names))])
    internal_upper = np.array([transforms[i].to_internal(upper[i]) for i in range(len(names))])
    # Explore only as deep as something consumes. The integration weights drop
    # every point below `log_density_drop`, so exploring past it buys them
    # nothing -- and at d > 1 the extra depth costs (2r+1)^d conditional fits for
    # points that are then discarded. The one consumer that wants the tails is
    # the tabulated hyperparameter marginal, which exists only for a single
    # hyperparameter; there, the extra depth is what makes the marginal cover the
    # posterior instead of truncating it.
    # Explore at least as deep as the weighting will keep, or the grid would be
    # asked for points it never evaluated.
    depth = max(log_density_drop, explore_drop if len(names) == 1 else 0.0)
    prune_drop = max(depth, log_density_drop) + prune_slack
    if int_strategy not in ("auto", "grid", "ccd", "korobov"):
        raise ValueError("int_strategy must be 'auto', 'grid', 'ccd' or 'korobov'")
    if int_strategy == "auto":
        predicted = _predicted_grid_points(
            len(names), depth=depth, prune_drop=prune_drop,
            grid_step=grid_step, max_radius=max_radius,
        )
        int_strategy = "korobov" if predicted > max_grid_points else "grid"

    def jacobian(u):
        return float(sum(transforms[i].log_abs_jacobian(u[i]) for i in range(len(names))))

    if int_strategy in ("ccd", "korobov"):
        # The design integrates a Gaussian, so it must be centred on the density
        # it is actually integrating. `u_star` is the mode of `s` alone, but the
        # posterior *in u* is exp(s + jacobian) -- and for the usual log transform
        # the Jacobian is `u`, a linear tilt that moves the mode a full standard
        # deviation. Centring on `s`'s mode leaves the design systematically
        # off-target, which a few dozen points cannot absorb.
        #
        # One Newton step fixes it, and costs no conditional fits: the Jacobian is
        # analytic, and a linear tilt leaves the curvature alone, so the step is
        # exactly (-H)^-1 grad(jacobian) evaluated at the mode of s.
        gradient = np.zeros(len(names))
        for i in range(len(names)):
            step = 1e-4 * max(1.0, abs(u_star[i]))
            forward, backward = u_star.copy(), u_star.copy()
            forward[i] += step
            backward[i] -= step
            gradient[i] = (jacobian(forward) - jacobian(backward)) / (2.0 * step)
        design_center = u_star + np.linalg.solve(-hessian, gradient)
        if int_strategy == "ccd":
            design, design_weights = _ccd_design(len(names), ccd_f0)
        else:
            design, design_weights = _korobov_design(len(names), korobov_points)
        grid, points, z_sq, design_weights = _designed_grid(
            design_center, hessian, evaluate, design, design_weights,
            internal_lower=internal_lower, internal_upper=internal_upper,
        )
        # Every design point is kept: dropping any would break the balance that
        # gives the design its weights, and the density enters through the
        # importance ratio rather than through a threshold.
        s_values = np.array([s for s, _, _, _ in points])
        kept = [(u, s, cond, theta, compiled)
                for u, (s, cond, theta, compiled) in zip(grid, points, strict=True)]
        # The design integrates the Gaussian implied by the Hessian, so dividing
        # by that Gaussian -- adding 0.5*||z||^2 in logs -- leaves the true
        # posterior. Without it the design would echo the Laplace approximation
        # back and the evaluated densities would do no work.
        log_weights = np.array([
            np.log(abs(w)) + s + jacobian(u) + 0.5 * zz
            for (u, s, _, _, _), w, zz in zip(kept, design_weights, z_sq, strict=True)
        ])
        # Both shipped designs have positive weights, but the signs travel
        # explicitly: a signed rule (a Smolyak sparse grid, say) would otherwise
        # be silently absolute-valued here, and -- worse -- a negative weight
        # makes the CPO criterion non-finite, because the criteria treat the
        # integration weights as a probability mixture.
        weight_signs = np.sign(design_weights)
        volume_element = 0.5 * len(names) * np.log(2.0 * np.pi)
    else:
        grid, points = _explore_grid(
            u_star, hessian, evaluate,
            internal_lower=internal_lower, internal_upper=internal_upper,
            grid_step=grid_step, max_radius=max_radius, explore_drop=depth,
            prune_drop=prune_drop, max_grid_points=max_grid_points,
        )
        if not len(grid):
            grid, points = u_star.reshape(1, -1), [evaluate(u_star)]
        s_values = np.array([s for s, _, _, _ in points])
        s_max = float(s_values.max())
        kept = [(u, s, cond, theta, compiled)
                for u, (s, cond, theta, compiled) in zip(grid, points, strict=True)
                if s >= s_max - log_density_drop]
        if not kept:
            raise NumericalError("INLA grid retained no points above the density threshold")
        # importance-weight Jacobian: log|d theta / d u| summed over parameters
        log_weights = np.array([s + jacobian(u) for (u, s, _, _, _) in kept])
        weight_signs = np.ones(len(kept))
        volume_element = len(names) * np.log(grid_step)

    total_log, total_sign = logsumexp(log_weights, b=weight_signs, return_sign=True)
    if total_sign <= 0:
        raise NumericalError(
            "integration weights summed to a non-positive total; the design's "
            "negative weights overwhelmed its positive ones"
        )
    weights = weight_signs * np.exp(log_weights - total_log)

    reference = kept[0][2]
    is_laplace = isinstance(reference, LaplaceResult)
    link_name = reference.link_name if is_laplace else None

    # All grid conditionals share the model shape, so they are all-sparse or
    # all-dense together -- one flag governs the whole accumulation loop.
    sparse_conditionals = (
        getattr(reference, "_sparse_posterior", None) is not None
        and reference._covariance is None
    )

    mean_acc = np.zeros_like(reference.mean)
    cov_acc = None if sparse_conditionals else np.zeros_like(reference.covariance)
    var_acc = np.zeros_like(reference.mean) if sparse_conditionals else None
    pm_acc = np.zeros_like(reference.predictive_mean)
    pv_acc = np.zeros_like(reference.predictive_variance)
    fitted_acc = np.zeros_like(reference.fitted_mean) if is_laplace else None
    # E[sigma^2] over the grid, so an integrated Gaussian fit can still recover
    # the response-scale variance as predictive_variance + observation_variance.
    observation_acc = None if is_laplace else 0.0
    theta_mean = {name: 0.0 for name in names}
    theta_sq = {name: 0.0 for name in names}

    for (_, _, cond, theta, _), w in zip(kept, weights, strict=True):
        m = cond.mean
        pm = cond.predictive_mean
        mean_acc += w * m
        if sparse_conditionals:
            var_acc += w * (_conditional_latent_variances(cond) + m * m)
        else:
            cov_acc += w * (cond.covariance + np.outer(m, m))
        pm_acc += w * pm
        pv_acc += w * (cond.predictive_variance + pm * pm)
        if is_laplace:
            fitted_acc += w * cond.fitted_mean
        else:
            observation_acc += w * float(cond.observation_variance)
        for name in names:
            theta_mean[name] += w * theta[name]
            theta_sq[name] += w * theta[name] * theta[name]

    mean = mean_acc
    if sparse_conditionals:
        covariance = None
        latent_variance = var_acc - mean * mean
    else:
        covariance = cov_acc - np.outer(mean, mean)
        latent_variance = None
    predictive_mean = pm_acc
    predictive_variance = pv_acc - pm_acc * pm_acc

    hyper_marginals = _theta_marginals(
        names, grid, s_values, transforms, theta_mean, theta_sq
    )

    eigenvalues = np.clip(np.linalg.eigvalsh(-hessian), 1e-6, None)
    integrated_lml = float(
        total_log + volume_element - 0.5 * float(np.sum(np.log(eigenvalues)))
    )

    diagnostics = {
        "inla_grid_points": int(len(kept)),
        "inla_grid_evaluated": int(len(grid)),
        "inla_int_strategy": int_strategy,
        "inla_effective_weight": float(1.0 / np.sum(weights**2)),
        "inla_conditional_engine": "laplace" if is_laplace else "exact_gaussian",
        "inla_active_bounds": ",".join(eb.diagnostics.active_bounds),
        "inla_collapsed": len(kept) == 1,
    }
    for name in names:
        diagnostics[f"inla_mode_{name}"] = float(eb.parameters[name])

    covariance_check = (
        ("latent variance", latent_variance) if sparse_conditionals
        else ("covariance", covariance)
    )
    for label, value in (("mean", mean), covariance_check,
                         ("predictive mean", predictive_mean),
                         ("predictive variance", predictive_variance)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA produced non-finite {label}")
    if not np.isfinite(integrated_lml):
        raise NumericalError("INLA produced non-finite log marginal likelihood")

    full_design = kept[0][4].design
    observed = kept[0][4].observed
    design_obs = full_design[observed]
    offset_obs = kept[0][4].offset[observed]
    y_obs = kept[0][4].y[observed]
    theta_grid = [(w, cond, compiled.likelihood)
                  for (_, _, cond, _, compiled), w in zip(kept, weights, strict=True)]
    crit = _model_criteria(design_obs, offset_obs, y_obs, theta_grid)

    # crit.cpo/pit are computed in canonical order over observed rows only (length
    # n_observed); scatter them into full-length canonical arrays aligned with every
    # row (NaN at unobserved/prediction-target rows) so they can be reordered like
    # predictive_mean/predictive_variance below.
    cpo_full = np.full(observed.size, np.nan)
    pit_full = np.full(observed.size, np.nan)
    cpo_full[observed] = crit.cpo
    pit_full[observed] = crit.pit
    criteria = ModelCriteria(
        crit.dic, crit.dic_effective_parameters,
        crit.waic, crit.waic_effective_parameters,
        cpo_full, pit_full, crit.cpo_failures, crit.log_cpo_sum,
    )

    latent_marginal_table = None
    if latent_strategy == "simplified_laplace":
        # ponytail: simplified-Laplace needs the off-diagonal cov(x_i, eta_j) =
        # Sigma @ design^T, which the diagonal sparse posterior cannot supply.
        # Guard rather than densify. Upgrade path: column-wise Sigma @ design^T
        # via the posterior factor if this strategy is needed above the guard.
        if sparse_conditionals:
            raise UnsupportedEngineError(
                "simplified-Laplace latent marginals are not available above the "
                "sparse guard; use latent_strategy='gaussian'"
            )
        location, scale, shape, sla_weights, clamped_count = _simplified_laplace_marginals(
            design_obs, offset_obs, y_obs, theta_grid,
        )
        latent_marginal_table = SkewNormalMarginals(sla_weights, location, scale, shape)
        diagnostics["skew_clamped"] = int(clamped_count)
    elif latent_strategy == "laplace":
        if kept[0][4].constraints.shape[0] > 0:
            raise UnsupportedEngineError(
                "full Laplace does not support constrained (RW) effects; "
                "use latent_strategy='gaussian' or 'simplified_laplace'"
            )
        if sparse_conditionals:
            raise UnsupportedEngineError(
                "full Laplace latent marginals are not available above the sparse "
                "guard; use latent_strategy='gaussian' or 'simplified_laplace'"
            )
        laplace_grid = [
            (w, cond, compiled.likelihood, compiled.precision)
            for (_, _, cond, _, compiled), w in zip(kept, weights, strict=True)
        ]
        latent_marginal_table = _full_laplace_marginals(design_obs, offset_obs, y_obs, laplace_grid)

    return INLAResult(
        labels=reference.labels,
        mean=mean, covariance=covariance, log_marginal_likelihood=integrated_lml,
        predictive_mean=predictive_mean, predictive_variance=predictive_variance,
        hyperparameter_marginals=hyper_marginals,
        criteria=criteria,
        fitted_mean=fitted_acc, link_name=link_name,
        observation_variance=observation_acc,
        block_slices=dict(reference.block_slices), diagnostics=diagnostics,
        latent_marginal_table=latent_marginal_table, latent_variances=latent_variance,
    )


def _model_criteria(design, offset, y, grid, *, n_nodes=21, cpo_failure_threshold=0.5):
    dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, dtype=float)
    offset = np.asarray(offset, dtype=float)
    y = np.asarray(y, dtype=float)
    nodes, gh = hermgauss(n_nodes)
    gh = gh / np.sqrt(np.pi)
    log_gh = np.log(gh)
    n = y.size

    pd_sum = np.zeros(n)     # Sum_k w_k E[p]
    elog = np.zeros(n)       # Sum_k w_k E[log p]
    elog2 = np.zeros(n)      # Sum_k w_k E[(log p)^2]
    mbar = np.zeros(n)
    # 1/p can be enormous in the tails (a diffuse latent posterior combined with a
    # tight likelihood), so accumulate the reciprocal terms in log space instead of
    # exponentiating -logp directly: exp(-logp) overflows float64 long before the
    # harmonic-mean estimator itself becomes numerically meaningless.
    log_inv_terms = []        # per grid point: log(weight) + log(gh_j) - logp_ij
    log_cdf_inv_terms = []    # ... + log(cdf_ij)
    points = list(grid)

    for weight, fit, likelihood in points:
        m = offset + np.asarray(dense @ fit.mean).reshape(-1)
        v = np.clip(_conditional_predictive_variances(fit, dense), 0.0, None)
        mbar += weight * m
        eta = m[:, None] + np.sqrt(2.0 * v)[:, None] * nodes[None, :]     # (n, n_nodes)
        logp = np.stack([likelihood.pointwise_log_density(eta[:, j], y) for j in range(n_nodes)], axis=1)
        cdf = np.stack([likelihood.cdf(eta[:, j], y) for j in range(n_nodes)], axis=1)
        p = np.exp(logp)
        pd_sum += weight * (gh * p).sum(axis=1)
        elog += weight * (gh * logp).sum(axis=1)
        elog2 += weight * (gh * logp ** 2).sum(axis=1)

        base = np.log(weight) + log_gh[None, :] - logp   # (n, n_nodes)
        log_inv_terms.append(base)
        with np.errstate(divide="ignore"):
            log_cdf = np.log(cdf)   # -inf where cdf == 0; contributes 0 after exp
        log_cdf_inv_terms.append(base + log_cdf)

    lppd = np.log(pd_sum)
    var_logp = np.clip(elog2 - elog ** 2, 0.0, None)
    waic = float(-2.0 * np.sum(lppd - var_logp))
    p_waic = float(var_logp.sum())

    d_bar = float(-2.0 * elog.sum())
    d_mean = 0.0
    for weight, _, likelihood in points:
        d_mean += weight * float(likelihood.pointwise_log_density(mbar, y).sum())
    d_mean = -2.0 * d_mean
    p_d = d_bar - d_mean
    dic = d_bar + p_d

    all_inv = np.concatenate(log_inv_terms, axis=1)          # (n, n_nodes * n_grid_points)
    all_cdf_inv = np.concatenate(log_cdf_inv_terms, axis=1)
    log_einv = logsumexp(all_inv, axis=1)
    log_ecdf_over_p = logsumexp(all_cdf_inv, axis=1)

    # Reliability of the harmonic-mean CPO estimator: flag observation i when the
    # single largest (grid point, quadrature node) contribution to E[1/p_i] exceeds
    # cpo_failure_threshold times the total E[1/p_i] (all in log space to avoid
    # overflow from exp(-logp)).
    max_log_contrib = all_inv.max(axis=1)
    cpo_failure_mask = max_log_contrib > (np.log(cpo_failure_threshold) + log_einv)

    cpo = np.exp(-log_einv)
    pit = np.exp(log_ecdf_over_p - log_einv)
    cpo_failures = int(np.sum(cpo_failure_mask))
    log_cpo_sum = float(np.sum(-log_einv))

    for name, value in (("waic", waic), ("dic", dic), ("cpo", cpo), ("pit", pit)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA criteria produced non-finite {name}")

    return ModelCriteria(dic, p_d, waic, p_waic, cpo, pit, cpo_failures, log_cpo_sum)

from collections.abc import Callable
from itertools import product

import numpy as np
from scipy.special import logsumexp

from pylgm.exceptions import NumericalError, OptimizationError
from pylgm.inference import LaplaceResult, fit_gaussian
from pylgm.inference.result import GaussianMarginals, INLAResult
from pylgm.optimization.empirical_bayes import optimize_empirical_bayes


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


def _build_grid(
    center: np.ndarray, hessian: np.ndarray, *,
    grid_step: float = 1.0, radius: int = 3, max_grid_points: int = 4096,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    d = center.size
    candidate_count = (2 * radius + 1) ** d
    if candidate_count > max_grid_points:
        raise OptimizationError(
            f"INLA grid would need {candidate_count} points for {d} hyperparameters; "
            f"exceeds max_grid_points={max_grid_points}"
        )
    directions = _whitening_directions(hessian)
    offsets = range(-radius, radius + 1)
    points = [
        center + grid_step * directions @ np.asarray(z, dtype=float)
        for z in product(offsets, repeat=d)
    ]
    return np.asarray(points)


def integrate_inla(
    family, bounds, *, initial=None, fit=None, penalty=None, allow_large_dense=False,
    grid_step=1.0, radius=3, log_density_drop=2.5, max_grid_points=4096,
) -> INLAResult:
    names = tuple(family.parameter_names)
    conditional_fit = fit if fit is not None else fit_gaussian
    lower = np.array([bounds[name].lower for name in names])
    upper = np.array([bounds[name].upper for name in names])

    def evaluate(u):
        theta = {
            name: float(np.clip(np.exp(value), lower[i], upper[i]))
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
        return s_value, conditional, theta

    eb = optimize_empirical_bayes(
        family, bounds, initial=initial, fit=conditional_fit, penalty=penalty,
        allow_large_dense=allow_large_dense,
    )
    u_star = np.array([np.log(eb.parameters[name]) for name in names])
    hessian = _finite_difference_hessian(lambda u: evaluate(u)[0], u_star)
    grid = _build_grid(
        u_star, hessian, grid_step=grid_step, radius=radius, max_grid_points=max_grid_points,
    )
    # A grid point whose raw log-theta lands outside the declared [lower, upper] domain
    # gets clipped to the same boundary theta as its neighbours: dθ/du is zero there, so
    # its correct Jacobian contribution is zero too. Including it with the raw `u` as if
    # the mapping stayed invertible double-counts (and, as radius grows, exponentially
    # over-weights) the boundary — drop such points before weighting instead.
    log_lower, log_upper = np.log(lower), np.log(upper)
    in_domain = grid[np.all((grid >= log_lower) & (grid <= log_upper), axis=1)]
    if in_domain.size == 0:
        in_domain = u_star.reshape(1, -1)
    grid = in_domain

    points = [evaluate(u) for u in grid]
    s_values = np.array([s for s, _, _ in points])
    s_max = float(s_values.max())
    kept = [(u, s, cond, theta) for u, (s, cond, theta) in zip(grid, points, strict=True)
            if s >= s_max - log_density_drop]
    if not kept:
        raise NumericalError("INLA grid retained no points above the density threshold")

    log_weights = np.array([s + float(np.sum(u)) for (u, s, _, _) in kept])
    weights = np.exp(log_weights - logsumexp(log_weights))

    reference = kept[0][2]
    is_laplace = isinstance(reference, LaplaceResult)
    link_name = reference.link_name if is_laplace else None

    mean_acc = np.zeros_like(reference.mean)
    cov_acc = np.zeros_like(reference.covariance)
    pm_acc = np.zeros_like(reference.predictive_mean)
    pv_acc = np.zeros_like(reference.predictive_variance)
    fitted_acc = np.zeros_like(reference.fitted_mean) if is_laplace else None
    theta_mean = {name: 0.0 for name in names}
    theta_sq = {name: 0.0 for name in names}

    for (_, _, cond, theta), w in zip(kept, weights, strict=True):
        m = cond.mean
        pm = cond.predictive_mean
        mean_acc += w * m
        cov_acc += w * (cond.covariance + np.outer(m, m))
        pm_acc += w * pm
        pv_acc += w * (cond.predictive_variance + pm * pm)
        if is_laplace:
            fitted_acc += w * cond.fitted_mean
        for name in names:
            theta_mean[name] += w * theta[name]
            theta_sq[name] += w * theta[name] * theta[name]

    mean = mean_acc
    covariance = cov_acc - np.outer(mean, mean)
    predictive_mean = pm_acc
    predictive_variance = pv_acc - pm_acc * pm_acc

    hyper_marginals = {
        name: GaussianMarginals(
            np.array([theta_mean[name]]),
            np.array([max(theta_sq[name] - theta_mean[name] ** 2, 0.0)]),
        )
        for name in names
    }

    eigenvalues = np.clip(np.linalg.eigvalsh(-hessian), 1e-6, None)
    integrated_lml = float(
        logsumexp(log_weights) + len(names) * np.log(grid_step)
        - 0.5 * float(np.sum(np.log(eigenvalues)))
    )

    diagnostics = {
        "inla_grid_points": int(len(kept)),
        "inla_effective_weight": float(1.0 / np.sum(weights**2)),
        "inla_conditional_engine": "laplace" if is_laplace else "exact_gaussian",
        "inla_active_bounds": ",".join(eb.diagnostics.active_bounds),
    }
    for name in names:
        diagnostics[f"inla_mode_{name}"] = float(eb.parameters[name])

    for label, value in (("mean", mean), ("covariance", covariance),
                         ("predictive mean", predictive_mean),
                         ("predictive variance", predictive_variance)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA produced non-finite {label}")

    return INLAResult(
        labels=reference.labels,
        mean=mean, covariance=covariance, log_marginal_likelihood=integrated_lml,
        predictive_mean=predictive_mean, predictive_variance=predictive_variance,
        hyperparameter_marginals=hyper_marginals,
        fitted_mean=fitted_acc, link_name=link_name,
        block_slices=dict(reference.block_slices), diagnostics=diagnostics,
    )

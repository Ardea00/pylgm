from collections.abc import Callable
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from pylgm.exceptions import NumericalError, OptimizationError
from pylgm.inference import LaplaceResult, fit_gaussian
from pylgm.inference.result import GaussianMarginals, INLAResult, ModelCriteria
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
        return s_value, conditional, theta, compiled

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
    s_values = np.array([s for s, _, _, _ in points])
    s_max = float(s_values.max())
    kept = [(u, s, cond, theta, compiled)
            for u, (s, cond, theta, compiled) in zip(grid, points, strict=True)
            if s >= s_max - log_density_drop]
    if not kept:
        raise NumericalError("INLA grid retained no points above the density threshold")

    log_weights = np.array([s + float(np.sum(u)) for (u, s, _, _, _) in kept])
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

    for (_, _, cond, theta, _), w in zip(kept, weights, strict=True):
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
        "inla_collapsed": len(kept) == 1,
    }
    for name in names:
        diagnostics[f"inla_mode_{name}"] = float(eb.parameters[name])

    for label, value in (("mean", mean), ("covariance", covariance),
                         ("predictive mean", predictive_mean),
                         ("predictive variance", predictive_variance)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA produced non-finite {label}")
    if not np.isfinite(integrated_lml):
        raise NumericalError("INLA produced non-finite log marginal likelihood")

    observed = kept[0][4].observed
    design_obs = kept[0][4].design[observed]
    offset_obs = kept[0][4].offset[observed]
    y_obs = kept[0][4].y[observed]
    criteria_grid = [(w, cond, compiled.likelihood)
                     for (_, _, cond, _, compiled), w in zip(kept, weights, strict=True)]
    criteria = _model_criteria(design_obs, offset_obs, y_obs, criteria_grid)

    return INLAResult(
        labels=reference.labels,
        mean=mean, covariance=covariance, log_marginal_likelihood=integrated_lml,
        predictive_mean=predictive_mean, predictive_variance=predictive_variance,
        hyperparameter_marginals=hyper_marginals,
        criteria=criteria,
        fitted_mean=fitted_acc, link_name=link_name,
        block_slices=dict(reference.block_slices), diagnostics=diagnostics,
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
        v = np.clip(np.einsum("ij,jk,ik->i", dense, np.asarray(fit.covariance), dense), 0.0, None)
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

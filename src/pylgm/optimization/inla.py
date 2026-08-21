import math
from collections.abc import Callable
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor
from scipy.special import logsumexp

from pylgm.exceptions import NumericalError, OptimizationError
from pylgm.inference import LaplaceResult, fit_gaussian
from pylgm.inference.result import (
    GaussianMarginals,
    INLAResult,
    ModelCriteria,
    SkewNormalMarginals,
    TabulatedMarginals,
)
from pylgm.optimization.empirical_bayes import optimize_empirical_bayes

_SN_C = (4.0 - math.pi) * math.sqrt(2.0) / math.pi ** 1.5
_SN_R_MAX = 5.0  # cap on |a/omega| for robustness (|skew| ~ 0.937 at r=5, shy of the ~0.995 supremum)
_SCALE_FLOOR = 1e-12  # degenerate (sigma_i == 0) coordinate: near-point-mass marginal


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
        eta_var = np.clip(np.einsum("ij,jk,ik->i", dense, cov, dense), 0.0, None)  # (n,)
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


def _full_laplace_marginals(design, offset, y, grid, precision, *,
                            n_abscissae=7, grid_points=201, grid_radius=8.0):
    """RMC (2009) full-Laplace tabulated latent marginals (eqs 12/13/16/17).

    Unconstrained models only (caller guards). Per latent i, per theta grid point,
    per Gauss-Hermite abscissa x_i, computes the conditional-mean configuration
    (eq 13), the full-Laplace log-density log pi~_LA via the joint density and the
    pi~_GG Laplace-approximation determinant (eq 12), and a cubic-spline correction
    against the base Gaussian (eq 17). Mixed over the theta-grid into a
    TabulatedMarginals.
    """
    X = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
    offset = np.asarray(offset, float)
    y = np.asarray(y, float)
    Q = precision.toarray() if hasattr(precision, "toarray") else np.asarray(precision, float)
    points = list(grid)
    p = points[0][1].mean.shape[0]
    std_nodes = np.polynomial.hermite.hermgauss(n_abscissae)[0] * np.sqrt(2.0)  # ~N(0,1) abscissae

    x_out = np.zeros((p, grid_points))
    dens_out = np.zeros((p, grid_points))
    for i in range(p):
        # common per-latent grid from the mixture spread
        mu_bar = sum(w * f.mean[i] for w, f, _ in points)
        # use max per-θ std (not average) so the common grid covers the widest mixture component without truncating its tails
        sig_max = max(np.sqrt(max(f.covariance[i, i], 0.0)) for _, f, _ in points)
        gx = np.linspace(mu_bar - grid_radius * sig_max, mu_bar + grid_radius * sig_max, grid_points)
        mixed = np.zeros(grid_points)
        keep = [c for c in range(p) if c != i]
        for w, fit, likelihood in points:
            m = np.asarray(fit.mean, float)
            cov = np.asarray(fit.covariance, float)
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
                joint = -0.5 * float(x @ Q @ x) + loglik
                wgt = likelihood.working_weights(eta, y)
                info = Q + (X.T * wgt) @ X
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
            area = np.trapz(dens, gx)
            if not np.isfinite(area) or area <= 0:
                raise NumericalError("full-Laplace density normalization failed")
            mixed += w * (dens / area)
        total = np.trapz(mixed, gx)
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
    latent_strategy="gaussian",
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
        location, scale, shape, sla_weights, clamped_count = _simplified_laplace_marginals(
            design_obs, offset_obs, y_obs, theta_grid,
        )
        latent_marginal_table = SkewNormalMarginals(sla_weights, location, scale, shape)
        diagnostics["skew_clamped"] = int(clamped_count)

    return INLAResult(
        labels=reference.labels,
        mean=mean, covariance=covariance, log_marginal_likelihood=integrated_lml,
        predictive_mean=predictive_mean, predictive_variance=predictive_variance,
        hyperparameter_marginals=hyper_marginals,
        criteria=criteria,
        fitted_mean=fitted_acc, link_name=link_name,
        block_slices=dict(reference.block_slices), diagnostics=diagnostics,
        latent_marginal_table=latent_marginal_table,
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

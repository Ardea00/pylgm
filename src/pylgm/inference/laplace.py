import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.sparse import csr_matrix, vstack

from pylgm.exceptions import InferenceConvergenceError, NumericalError
from pylgm.inference import gaussian as _gaussian
from pylgm.inference.gaussian import (
    _block_slices,
    _condition_on_data_constraints,
    _constraint_null_space,
    _constraint_particular_solution,
    _factor_positive_definite,
    _require_finite,
)
from pylgm.inference.result import LaplaceResult, quadratic_form_diagonal
from pylgm.inference.sampling import GridSampler
from pylgm.ir.model import CompiledLGM


# Latent dimension above which the sparse engine beats the dense one (measured,
# see fit_laplace); a sparse fit keeps no dense covariance.
_SPARSE_LAPLACE_MIN_LATENT = 1000


def _variational_mean_shift(reduced_design, reduced_covariance, factor, eta, y, likelihood):
    """The leading-order variational correction from the mode toward the mean.

    A Laplace approximation reports the *mode* of the conditional posterior. The
    mode is not its mean whenever the likelihood is skewed, and the gap is
    systematic rather than noisy -- always in the same direction, and shrinking
    only as O(1/n) per observation.

    Minimising ``KL(N(x* + d, Sigma) || p)`` over a mean shift ``d``, holding the
    Laplace covariance fixed, gives a gradient at ``d = 0`` of
    ``A' (g1(eta*) - E[g1(eta)])``, where ``g1`` is the likelihood score.
    Expanding that expectation to second order -- its first-order term vanishes
    because ``E[eta - eta*] = 0`` -- leaves ``-A' (sigma_eta^2 * g3) / 2`` with
    ``g3`` the third derivative, and one Newton step against the same Hessian the
    fit already factored gives

        d = 0.5 * H^-1 A' (sigma_eta^2 * g3)

    Everything on the right is already computed: the predictive variances, the
    likelihood's third derivative, and the Cholesky factor. No new solve of a
    different matrix and no second pass over the data, which is what makes this
    close to free.

    This is the "mean" strategy of Van Niekerk & Rue (JMLR 2024,
    arXiv:2111.12945) taken to leading order, rather than their full low-rank
    optimisation. On a Poisson likelihood with a flat prior -- where the exact
    answer is ``digamma(y)`` against a mode of ``log y`` -- it turns an O(1/y)
    error into an O(1/y^2) one.

    The covariance is deliberately left alone: this corrects where the
    approximating Gaussian sits, not its shape.
    """
    eta_variance = np.clip(quadratic_form_diagonal(reduced_design, reduced_covariance), 0.0, None)
    third = np.asarray(likelihood.third_derivative(eta, y), dtype=float)
    return cho_solve(factor, reduced_design.T @ (0.5 * eta_variance * third))


def _observed_likelihood(model: CompiledLGM):
    """The likelihood bound to the observed rows, response validated.

    Binomial carries a per-row trials vector; the fit loop works on the observed
    rows, so bind their trials. For every other likelihood this returns self.
    `restrict` re-indexes any row-indexed internal state (a mixture's masks)
    into observed-row space first; it is a no-op for every other likelihood.
    The model's own `likelihood` stays full-row, since `response_prediction`
    runs over every row.
    """
    observed = model.observed
    observed_likelihood = model.likelihood.restrict(observed)
    _trials = getattr(observed_likelihood, "trials", None)
    lk_obs = observed_likelihood.for_observations(
        {"trials": _trials[observed]} if _trials is not None else None
    )
    lk_obs.validate_response(model.y[observed])
    return lk_obs


def _prediction_outputs(model: CompiledLGM, mean: np.ndarray, predictive_variance: np.ndarray):
    """``(predictive_mean, fitted_mean)`` on the prediction grid.

    Report on `prediction_design`/`prediction_offset`, not the fit `design`/
    `offset`: for an unmodified model the two are identical (CompiledLGM's
    default), but a model augmented with LinearObservation pseudo-rows (Joint)
    fits against extra rows appended to `design` that must not leak into the
    reported predictions. A projected joint model appends those pseudo-rows
    after the prediction-grid rows, so the grid's likelihood is that prefix.
    """
    predictive_mean = np.asarray(model.prediction_offset + model.prediction_design @ mean).reshape(-1)
    prediction_likelihood = model.likelihood
    if model.prediction_design.shape[0] != model.design.shape[0]:
        prediction_likelihood = model.likelihood.restrict(
            np.arange(model.design.shape[0]) < model.prediction_design.shape[0]
        )
    return predictive_mean, prediction_likelihood.response_prediction(
        predictive_mean, predictive_variance
    )


def _stalled(current: float, candidate: float, scale: float) -> bool:
    """A Newton step the line search had to shorten and that still moved the
    objective by no more than its round-off.

    A full step near the mode is fine even when the objective cannot register
    it (quadratic convergence keeps shrinking the gradient). A shortened one
    that gains nothing is the stall: a gradient test on the data's scale can
    stay just above its tolerance for every remaining iteration, each paying a
    line search that halves to nothing. Two in a row end the loop (one can be a
    rounding accident the next full step recovers from); the Newton-decrement
    rescue after it is the right judge, and ``_polished`` fixes the mode.
    """
    return scale < 1.0 and current - candidate <= 4 * np.finfo(float).eps * max(abs(current), 1.0)


def _polished(objective, point: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """``candidate`` (one more Newton step) unless it raises the objective past round-off.

    The Laplace log marginal likelihood depends on the mode to first order
    (through ``logdet H(x)``), so stopping anywhere under the gradient tolerance
    leaves up to ~1e-9 in it -- which a finite-difference Hessian over theta
    (step 1e-3) amplifies to ~1e-3, and which differs with the Newton start
    (a warm start stops at a different point than a cold one). Near the mode
    Newton converges quadratically, so one extra step removes that dependence.
    """
    try:
        current, proposed = objective(point), objective(candidate)
    except FloatingPointError:
        return point
    if np.isfinite(proposed) and proposed <= current + 1e-12 * max(1.0, abs(current)):
        return candidate
    return point


def _reported_predictions(model, mean, variances, wanted: bool):
    """``(predictive_variance, predictive_mean, fitted_mean)`` on the prediction grid.

    ``wanted=False`` is for intermediate objective evaluations, which read only
    the log marginal likelihood: the variances are skipped (``None``) and the
    fitted mean, which needs them, is NaN.
    """
    if not wanted:
        predictive_mean = np.asarray(
            model.prediction_offset + model.prediction_design @ mean
        ).reshape(-1)
        return None, predictive_mean, np.full(predictive_mean.shape, np.nan)
    predictive_variance = variances()
    predictive_mean, fitted_mean = _prediction_outputs(model, mean, predictive_variance)
    _require_finite("predictive variance", predictive_variance)
    _require_finite("fitted mean", fitted_mean)
    return predictive_variance, predictive_mean, fitted_mean


def _curvature_rows(likelihood, design, eta: np.ndarray, y: np.ndarray, *, psd: bool = False):
    """``(rows, weights)`` with ``rows^T diag(weights) rows`` the likelihood curvature.

    A row-separable likelihood gives ``(design, working_weights)``. A coupled one
    owns each pair's two rows outright, so its 2x2 curvature block
    ``[[w_i, c], [c, w_j]]`` is replaced by its eigen-pair rows
    ``cos t d_i + sin t d_j`` and ``-sin t d_i + cos t d_j`` (``tan 2t = 2c / (w_i - w_j)``),
    weighted by the eigenvalues -- the same matrix, and every engine keeps its
    ``(design, weights)`` form. ``psd`` clips negative eigenvalues: a Newton
    direction for an iterate where the (not log-concave) block is indefinite.
    """
    weights = np.asarray(likelihood.working_weights(eta, y), dtype=float)
    coupled = getattr(likelihood, "cross_weights", None)
    pairs = coupled(eta, y) if coupled is not None else None
    if pairs is None:
        return design, weights
    i, j, c = pairs
    w_i, w_j = weights[i], weights[j]
    angle = 0.5 * np.arctan2(2.0 * c, w_i - w_j)
    cos, sin = np.cos(angle), np.sin(angle)
    first = w_i * cos * cos + w_j * sin * sin + 2.0 * c * sin * cos
    second = w_i * sin * sin + w_j * cos * cos - 2.0 * c * sin * cos
    if psd:
        first, second = np.maximum(first, 0.0), np.maximum(second, 0.0)
    weights = weights.copy()
    weights[i] = weights[j] = 0.0
    d_i, d_j = design[i], design[j]
    return (
        vstack([design, d_i.multiply(cos[:, None]) + d_j.multiply(sin[:, None]),
                d_j.multiply(cos[:, None]) - d_i.multiply(sin[:, None])], format="csr"),
        np.concatenate([weights, first, second]),
    )


def _prior_data_log_density(model: CompiledLGM, latent_size: int, precision) -> float:
    """``log p(e)``: the density of the data rows ``A x = e`` under the prior.

    The prior is conditioned on the structural rows only (intrinsic sum-to-zero,
    model-level ``constraints=``), which make an intrinsic prior proper on the
    reduced space. ``precision`` may be dense or sparse.
    """
    structural = model.constraints.shape[0] - model.data_constraint_count
    rows, rhs = model.constraints[:structural], model.constraint_rhs[:structural]
    basis = _constraint_null_space(rows, latent_size)
    x_p = _constraint_particular_solution(rows, rhs, latent_size)
    reduced = basis.T @ (precision @ basis)
    factor, logdet = _factor_positive_definite(np.asarray(reduced), "reduced prior precision")
    data_rows = model.constraints[structural:]
    data_rhs = model.constraint_rhs[structural:]
    mean = np.zeros(basis.shape[1])
    if x_p is not None:
        mean = cho_solve(factor, -(basis.T @ (precision @ x_p)))
        data_rhs = data_rhs - data_rows @ x_p
    *_, log_density = _condition_on_data_constraints(
        mean, logdet, reduced, data_rows @ basis, data_rhs,
    )
    return log_density


def _fit_laplace_dense(
    model: CompiledLGM, max_iterations: int, tolerance: float, mean_correction: bool = False,
    initial_mode: np.ndarray | None = None, predictive_variances: bool = True,
) -> LaplaceResult:
    likelihood = model.likelihood
    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    design = model.design
    y = model.y
    offset = model.offset
    observed = model.observed
    y_obs = y[observed]
    offset_obs = offset[observed]
    lk_obs = _observed_likelihood(model)

    # A trailing ``data_constraint_count`` rows of ``model.constraints`` are
    # exact data (a ``LinearConstraint``), not pure conditioning. They enter as
    # ``log p(y, e) = log p(e) + log p(y | e)``: the prior density of ``A x`` at
    # ``e`` is exact Gaussian, and the Laplace step runs on the prior conditioned
    # on every row, so the Newton search stays on ``A x = e`` and lands on the
    # constrained mode. (Conditioning the *unconstrained* Laplace posterior
    # afterwards is only first-order accurate for a non-Gaussian likelihood, and
    # the error grows with how far the data rows move the mode.)
    data_log_density = 0.0
    if model.data_constraint_count:
        data_log_density = _prior_data_log_density(model, latent_size, precision)
    constraints, constraint_rhs = model.constraints, model.constraint_rhs

    basis = _constraint_null_space(constraints, latent_size)
    identity = not constraints.shape[0]
    reduced_dim = basis.shape[1]
    # The observed design stays sparse: eta = Z (B z), the gradient B^T (Z^T g) and
    # the curvature B^T (Z^T W Z) B cost O(nnz(Z)) and O(p^2 d), where the dense
    # reduced design Z B would cost O(n d) per objective and O(n d^2) per Hessian.
    observed_design = design[observed]

    def predictor(z):
        return np.asarray(observed_design @ (z if identity else basis @ z)).reshape(-1) + offset_obs

    def pulled_back(values):
        pulled = np.asarray(observed_design.T @ values).reshape(-1)
        return pulled if identity else basis.T @ pulled

    def curvature(eta, psd=False):
        rows, weights = _curvature_rows(lk_obs, observed_design, eta, y_obs, psd=psd)
        data = (rows.T @ rows.multiply(weights[:, None])).toarray()
        return reduced_precision + (data if identity else basis.T @ data @ basis)

    reduced_precision = basis.T @ precision @ basis
    prior_factor, logdet_prior = _factor_positive_definite(
        reduced_precision, "reduced prior precision"
    )

    # Nonzero-rhs constraint: x = x_p + basis @ z shifts the observed predictor by
    # design @ x_p and adds the prior linear term b_p = basis.T @ (Q x_p); the
    # induced prior mean of z is m = -Q_r^-1 b_p (conditioning by kriging).
    x_p = _constraint_particular_solution(constraints, constraint_rhs, latent_size)
    prior_linear = np.zeros(reduced_dim)
    prior_mean = np.zeros(reduced_dim)
    if x_p is not None:
        offset_obs = offset_obs + np.asarray(design[observed] @ x_p).reshape(-1)
        prior_linear = basis.T @ (precision @ x_p)
        if reduced_dim:
            assert prior_factor is not None
            prior_mean = cho_solve(prior_factor, -prior_linear)

    def objective(z: np.ndarray) -> float:
        eta = predictor(z)
        return (
            -lk_obs.log_likelihood(eta, y_obs)
            + 0.5 * float(z @ reduced_precision @ z)
            + float(z @ prior_linear)
        )

    z = np.zeros(reduced_dim)
    if initial_mode is not None and reduced_dim:
        # Warm start: the reduced coordinates of the given latent vector.
        shift = initial_mode if x_p is None else initial_mode - x_p
        z = basis.T @ shift
    gradient_norm = 0.0
    newton_decrement = None
    iterations = 0
    converged = reduced_dim == 0
    if reduced_dim:
        current = objective(z)
        stalls = 0
        for iterations in range(1, max_iterations + 1):
            eta = predictor(z)
            grad_ll = lk_obs.gradient(eta, y_obs)
            gradient = reduced_precision @ z + prior_linear - pulled_back(grad_ll)
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm < tolerance:
                converged = True
                break
            try:
                factor, _ = _factor_positive_definite(curvature(eta), "reduced posterior precision")
            except NumericalError:
                # Only a coupled likelihood changes under psd; any other re-raises.
                factor, _ = _factor_positive_definite(
                    curvature(eta, psd=True), "reduced posterior precision"
                )
            step = cho_solve(factor, -gradient)
            slope = float(gradient @ step)
            scale = 1.0
            for _ in range(50):
                candidate = z + scale * step
                try:
                    candidate_obj = objective(candidate)
                except FloatingPointError:
                    candidate_obj = np.inf
                if np.isfinite(candidate_obj) and candidate_obj <= current + 1e-4 * scale * slope:
                    break
                scale *= 0.5
            else:
                raise NumericalError("Laplace line search failed to reduce the objective")
            stalls = stalls + 1 if _stalled(current, candidate_obj, scale) else 0
            z = candidate
            current = candidate_obj
            if stalls == 2:
                break
        if not converged:
            eta = predictor(z)
            gradient = (
                reduced_precision @ z + prior_linear
                - pulled_back(lk_obs.gradient(eta, y_obs))
            )
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm < tolerance:
                converged = True
            else:
                # The gradient's scale follows the data -- for a Poisson model its
                # components are of order the counts -- so an absolute threshold is
                # unreachable on plenty of well-behaved problems: the iteration
                # reaches the mode and then stalls just above it. Fall back to the
                # Newton decrement, which is scale-invariant and bounds the actual
                # suboptimality (f(z) - f* ~ lambda^2 / 2), and accept the point
                # when it is demonstrably optimal in objective terms.
                #
                # This lives only on the failure path on purpose. Breaking on the
                # decrement inside the loop cures the same stalls but stops earlier
                # than the gradient test on well-scaled problems, relocating the
                # mode; here, every fit that converges today is untouched.
                hessian = curvature(eta)
                factor, _ = _factor_positive_definite(hessian, "reduced posterior precision")
                decrement = -0.5 * float(gradient @ cho_solve(factor, -gradient))
                if decrement >= tolerance:
                    raise InferenceConvergenceError(iterations, gradient_norm)
                converged = True
                newton_decrement = decrement
        # Polish: one more Newton step from the accepted point (see _polished).
        eta = predictor(z)
        gradient = reduced_precision @ z + prior_linear - pulled_back(lk_obs.gradient(eta, y_obs))
        hessian = curvature(eta)
        polish_factor, _ = _factor_positive_definite(hessian, "reduced posterior precision")
        z = _polished(objective, z, z + cho_solve(polish_factor, -gradient))
        eta = predictor(z)
        hessian = curvature(eta)
        factor, logdet_posterior = _factor_positive_definite(hessian, "reduced posterior precision")
        loglik_mode = lk_obs.log_likelihood(eta, y_obs)
    else:
        logdet_posterior = 0.0
        loglik_mode = lk_obs.log_likelihood(offset_obs, y_obs)
        factor = prior_factor
        hessian = reduced_precision

    # Only the reported mean moves. `z` stays the mode, because the log marginal
    # likelihood below is a Laplace approximation *at* it and evaluating that
    # expansion anywhere else would stop it being one.
    z_mean = z
    if mean_correction and reduced_dim:
        z_mean = z + _variational_mean_shift(
            np.asarray(observed_design @ basis), cho_solve(factor, np.eye(reduced_dim)), factor,
            eta, y_obs, lk_obs,
        )

    centered = z - prior_mean
    log_marginal_likelihood = float(
        loglik_mode
        - 0.5 * float(centered @ reduced_precision @ centered)
        + 0.5 * logdet_prior
        - 0.5 * logdet_posterior
    ) + model.log_likelihood_normalization + data_log_density

    # Covariance = F F^T with F = basis L^-T; F doubles as the sampling factor.
    # Its columns span the null space of every constraint row, structural or
    # data, so sampler draws satisfy all of them to numerical precision.
    covariance_factor = (
        solve_triangular(factor[0], basis.T, lower=True).T
        if reduced_dim else np.zeros((latent_size, 0))
    )
    mean = basis @ z_mean if x_p is None else x_p + basis @ z_mean
    covariance = covariance_factor @ covariance_factor.T

    prediction_design = model.prediction_design
    prediction_offset = model.prediction_offset
    predictive_variance, predictive_mean, fitted_mean = _reported_predictions(
        model, mean, lambda: quadratic_form_diagonal(prediction_design, covariance),
        predictive_variances,
    )

    _require_finite("posterior mean", mean)
    _require_finite("posterior covariance", covariance)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)

    return LaplaceResult(
        labels=model.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        fitted_mean=fitted_mean,
        # A CompiledMixture has no single link -- its parts may use different
        # ones -- so report a truthful "mixture" rather than borrowing one part's.
        link_name=likelihood.link.name if hasattr(likelihood, "link") else "mixture",
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": int(np.count_nonzero(observed)),
            "constraint_count": int(model.constraints.shape[0]),
            "newton_iterations": int(iterations),
            "final_gradient_norm": float(gradient_norm),
            # Set only when the gradient test failed and the decrement carried
            # the fit, so a rescued mode is auditable in the field.
            "newton_decrement": newton_decrement,
        },
        sampler=GridSampler(
            mean, csr_matrix(prediction_design), prediction_offset, factor=covariance_factor
        ),
    )


def _fit_laplace_sparse(
    model: CompiledLGM, max_iterations: int, tolerance: float, mean_correction: bool = False,
    initial_mode: np.ndarray | None = None, predictive_variances: bool = True,
) -> LaplaceResult:
    """The dense engine's Newton iteration on the partitioned sparse solver.

    One Newton step for ``f(x) = -log p(y | x) + x^T Q x / 2`` on ``C_s x = e_s``
    is the weighted Gaussian solve ``x_N = H^-1 A^T (W A x + g)`` with
    ``H = Q + A^T W A``, kriged onto the structural rows -- ``_sparse_solve``
    with the working weights. Line search, convergence test and the
    Newton-decrement rescue mirror ``_fit_laplace_dense`` so both stop at the
    same mode; see docs/design/specs/2026-09-27-pylgm-sparse-laplace-design.md.
    """
    from pylgm.inference.sparse import _sparse_solve, prior_data_log_density

    # Data rows join the structural ones for the Newton search and the Laplace
    # term, which is then log p(y | e); log p(e) is added at the end (see the
    # dense engine).
    data_model = model if model.data_constraint_count else None
    if data_model is not None:
        model = model._structural_data_rows()

    latent_size = model.precision.shape[0]
    observed = model.observed
    design = model.design[observed]
    y_obs = model.y[observed]
    offset_obs = model.offset[observed]
    q = model.precision
    lk_obs = _observed_likelihood(model)

    structural_count = model.constraints.shape[0] - model.data_constraint_count
    rows = model.constraints[:structural_count]
    rhs = model.constraint_rhs[:structural_count]
    gram = _factor_positive_definite(rows @ rows.T, "constraint gram")[0] if structural_count else None

    def project(v: np.ndarray) -> np.ndarray:
        """Orthogonal projection onto null(C_s): the reduced gradient."""
        return v if gram is None else v - rows.T @ cho_solve(gram, rows @ v)

    def objective(x: np.ndarray) -> float:
        eta = design @ x + offset_obs
        return -lk_obs.log_likelihood(eta, y_obs) + 0.5 * float(x @ (q @ x))

    def weighted_solve(x: np.ndarray, *, final: bool = False, psd: bool = False):
        eta = design @ x + offset_obs
        curvature_rows, weights = _curvature_rows(lk_obs, design, eta, y_obs, psd=psd)
        # final: score = H x, so the solve returns x itself and every kriging /
        # determinant term is evaluated at the mode, as the dense engine does.
        pull = q @ x if final else design.T @ lk_obs.gradient(eta, y_obs)
        return _sparse_solve(model, weights, pull + curvature_rows.T @ (weights * (curvature_rows @ x)),
                             final=final, observed_design=curvature_rows)

    def newton_step(x: np.ndarray) -> np.ndarray:
        try:
            return weighted_solve(x).structural_mean - x
        except NumericalError:
            # As the dense engine: clip an indefinite coupled block for the direction.
            return weighted_solve(x, psd=True).structural_mean - x

    def reduced_gradient(x: np.ndarray) -> np.ndarray:
        eta = design @ x + offset_obs
        return project(q @ x - design.T @ lk_obs.gradient(eta, y_obs))

    # A feasible start: the warm-start mode projected onto C_s x = e_s, or the
    # minimum-norm solution of C_s x = e_s.
    x = np.zeros(latent_size) if initial_mode is None else np.asarray(initial_mode, dtype=float)
    if gram is not None:
        x = x - rows.T @ cho_solve(gram, rows @ x - rhs)
    newton_decrement = None
    iterations = 0
    converged = False
    stalls = 0
    current = objective(x)
    for iterations in range(1, max_iterations + 1):
        gradient = reduced_gradient(x)
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm < tolerance:
            converged = True
            break
        step = newton_step(x)
        slope = float(gradient @ step)
        scale = 1.0
        for _ in range(50):
            candidate = x + scale * step
            try:
                candidate_obj = objective(candidate)
            except FloatingPointError:
                candidate_obj = np.inf
            if np.isfinite(candidate_obj) and candidate_obj <= current + 1e-4 * scale * slope:
                break
            scale *= 0.5
        else:
            raise NumericalError("Laplace line search failed to reduce the objective")
        stalls = stalls + 1 if _stalled(current, candidate_obj, scale) else 0
        x = candidate
        current = candidate_obj
        if stalls == 2:
            break
    if not converged:
        gradient = reduced_gradient(x)
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm >= tolerance:
            # Scale-invariant rescue, exactly as the dense engine (see there).
            newton_decrement = -0.5 * float(gradient @ newton_step(x))
            if newton_decrement >= tolerance:
                raise InferenceConvergenceError(iterations, gradient_norm)

    x = _polished(objective, x, x + newton_step(x))
    solve = weighted_solve(x, final=True)
    eta = design @ x + offset_obs
    posterior = solve.posterior
    centered = x - solve.nu
    log_marginal_likelihood = float(
        lk_obs.log_likelihood(eta, y_obs)
        - 0.5 * float(centered @ (q @ centered))
        + 0.5 * solve.logdet_prior
        - 0.5 * solve.logdet_posterior
    ) + model.log_likelihood_normalization
    if data_model is not None:
        log_marginal_likelihood += prior_data_log_density(data_model, solve.logdet_prior, solve.nu)

    mean = solve.mean
    if mean_correction:
        # Every row is structural here, so this is the dense engine's shift
        # Sigma A^T (sigma_eta^2 g3 / 2) on the fully constrained posterior.
        eta_variance = posterior.predictive_variances(design)
        third = np.asarray(lk_obs.third_derivative(eta, y_obs), dtype=float)
        mean = mean + posterior.covariance_apply(design.T @ (0.5 * eta_variance * third))

    prediction_design = model.prediction_design
    predictive_variance, predictive_mean, fitted_mean = _reported_predictions(
        model, mean, lambda: posterior.predictive_variances(prediction_design),
        predictive_variances,
    )

    _require_finite("posterior mean", mean)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)
    likelihood = model.likelihood
    return LaplaceResult(
        labels=model.labels,
        mean=mean,
        covariance=None,  # never materialised above the guard
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        fitted_mean=fitted_mean,
        link_name=likelihood.link.name if hasattr(likelihood, "link") else "mixture",
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": int(np.count_nonzero(observed)),
            "constraint_count": int(model.constraints.shape[0]),
            "newton_iterations": int(iterations),
            "final_gradient_norm": float(gradient_norm),
            "newton_decrement": newton_decrement,
            "sparse_dimension": solve.sparse_dimension,
            "dense_dimension": solve.dense_dimension,
        },
        sparse_posterior=posterior,
        sampler=GridSampler(
            mean, csr_matrix(prediction_design), model.prediction_offset, posterior=posterior
        ),
    )


def fit_laplace(
    model: CompiledLGM,
    *,
    allow_large_dense: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    mean_correction: bool = False,
    initial_mode: np.ndarray | None = None,
    predictive_variances: bool = True,
) -> LaplaceResult:
    """Fit a latent Gaussian model by a Laplace approximation at fixed hyperparameters.

    Above ~1 000 latents (or the dense memory guard) the fit runs on the sparse
    partitioned solver and keeps no dense ``covariance``; marginals, ``predict``,
    ``linear_combinations`` and ``sample`` work unchanged. ``allow_large_dense=True``
    forces the dense engine.

    ``initial_mode`` warm-starts Newton from a latent vector (the mode at a
    nearby hyperparameter): it changes the iteration count, not the fixed point.
    ``predictive_variances=False`` skips the prediction-grid variances (and so
    the fitted mean, left NaN) for callers that only read the log marginal
    likelihood, as ``fit_gaussian``'s flag does.

    Likelihood-agnostic: any compiled likelihood implementing the GLM protocol works,
    so a Gaussian likelihood is fit exactly and serves as the correctness anchor.

    ``mean_correction`` moves the reported mean from the conditional mode toward
    the conditional mean (see :func:`_variational_mean_shift`). It changes
    ``mean`` and the ``predictive_mean``/``fitted_mean`` derived from it; it
    leaves the covariance and the log marginal likelihood alone, the latter
    because that is a Laplace approximation *at the mode* and would stop being
    one if evaluated anywhere else.
    """
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    # Route to the sparse engine past the dense memory guard (as the exact
    # Gaussian engine does) and, since Laplace refactors every Newton step and
    # grid point, from the measured crossover on (Poisson + Besag: dense 64 s vs
    # sparse 4 s at 2 500 latents, even at ~900). allow_large_dense=True still
    # forces the dense reference.
    sparse = (
        _gaussian._exceeds_dense_threshold(model)
        or model.precision.shape[0] > _SPARSE_LAPLACE_MIN_LATENT
    )
    fit = _fit_laplace_sparse if sparse and not allow_large_dense else _fit_laplace_dense
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return fit(model, max_iterations, tolerance, mean_correction,
                       initial_mode, predictive_variances)
    except FloatingPointError as error:
        raise NumericalError("Laplace numerical calculation was non-finite") from error

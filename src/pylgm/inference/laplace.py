import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.sparse import csr_matrix

from pylgm.exceptions import InferenceConvergenceError, NumericalError, UnsupportedEngineError
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


def _fit_laplace_dense(
    model: CompiledLGM, max_iterations: int, tolerance: float, mean_correction: bool = False
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
    # exact data (a ``LinearConstraint``), not pure conditioning: they enter
    # ``log p(y, e) = log p(y) + log p(e | y)`` as the density of an exact
    # observation of ``A x``, scored below by ``_condition_on_data_constraints``
    # against the *Laplace* posterior -- the same identity ``inference/gaussian.py``
    # uses against the exact Gaussian posterior. Only the structural rows (the
    # intrinsic sum-to-zero and any model-level ``constraints=`` label rows)
    # restrict the Newton fit's search space here.
    structural_count = model.constraints.shape[0] - model.data_constraint_count
    structural_constraints = model.constraints[:structural_count]
    structural_rhs = model.constraint_rhs[:structural_count]

    basis = _constraint_null_space(structural_constraints, latent_size)
    identity = not structural_constraints.shape[0]
    reduced_dim = basis.shape[1]
    reduced_design = np.asarray(design[observed] @ basis)
    reduced_precision = basis.T @ precision @ basis
    prior_factor, logdet_prior = _factor_positive_definite(
        reduced_precision, "reduced prior precision"
    )

    # Nonzero-rhs constraint: x = x_p + basis @ z shifts the observed predictor by
    # design @ x_p and adds the prior linear term b_p = basis.T @ (Q x_p); the
    # induced prior mean of z is m = -Q_r^-1 b_p (conditioning by kriging).
    x_p = _constraint_particular_solution(structural_constraints, structural_rhs, latent_size)
    prior_linear = np.zeros(reduced_dim)
    prior_mean = np.zeros(reduced_dim)
    if x_p is not None:
        offset_obs = offset_obs + np.asarray(design[observed] @ x_p).reshape(-1)
        prior_linear = basis.T @ (precision @ x_p)
        if reduced_dim:
            assert prior_factor is not None
            prior_mean = cho_solve(prior_factor, -prior_linear)

    def objective(z: np.ndarray) -> float:
        eta = reduced_design @ z + offset_obs
        return (
            -lk_obs.log_likelihood(eta, y_obs)
            + 0.5 * float(z @ reduced_precision @ z)
            + float(z @ prior_linear)
        )

    z = np.zeros(reduced_dim)
    gradient_norm = 0.0
    newton_decrement = None
    iterations = 0
    converged = reduced_dim == 0
    if reduced_dim:
        current = objective(z)
        for iterations in range(1, max_iterations + 1):
            eta = reduced_design @ z + offset_obs
            grad_ll = lk_obs.gradient(eta, y_obs)
            weights = lk_obs.working_weights(eta, y_obs)
            gradient = reduced_precision @ z + prior_linear - reduced_design.T @ grad_ll
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm < tolerance:
                converged = True
                break
            hessian = reduced_precision + (reduced_design.T * weights) @ reduced_design
            factor, _ = _factor_positive_definite(hessian, "reduced posterior precision")
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
            z = candidate
            current = candidate_obj
        if not converged:
            eta = reduced_design @ z + offset_obs
            gradient = (
                reduced_precision @ z + prior_linear
                - reduced_design.T @ lk_obs.gradient(eta, y_obs)
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
                weights = lk_obs.working_weights(eta, y_obs)
                hessian = reduced_precision + (reduced_design.T * weights) @ reduced_design
                factor, _ = _factor_positive_definite(hessian, "reduced posterior precision")
                decrement = -0.5 * float(gradient @ cho_solve(factor, -gradient))
                if decrement >= tolerance:
                    raise InferenceConvergenceError(iterations, gradient_norm)
                converged = True
                newton_decrement = decrement
        eta = reduced_design @ z + offset_obs
        weights = lk_obs.working_weights(eta, y_obs)
        hessian = reduced_precision + (reduced_design.T * weights) @ reduced_design
        factor, logdet_posterior = _factor_positive_definite(hessian, "reduced posterior precision")
        # Covariance = F F^T with F = basis L^-T; F doubles as the sampling factor.
        covariance_factor = solve_triangular(factor[0], basis.T, lower=True).T
        loglik_mode = lk_obs.log_likelihood(eta, y_obs)
    else:
        covariance_factor = np.zeros((latent_size, 0))
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
            reduced_design, cho_solve(factor, np.eye(reduced_dim)), factor, eta, y_obs, lk_obs
        )

    centered = z - prior_mean
    log_marginal_likelihood = float(
        loglik_mode
        - 0.5 * float(centered @ reduced_precision @ centered)
        + 0.5 * logdet_prior
        - 0.5 * logdet_posterior
    ) + model.log_likelihood_normalization

    covariance_basis = basis
    z_reported = z_mean
    if model.data_constraint_count:
        # log p(y, e) = log p(y) + log p(e | y): the Laplace posterior at the
        # mode (mean=z, precision=hessian) stands in for the exact Gaussian
        # posterior _condition_on_data_constraints was written against; the
        # identity itself is purely array-based (mean/logdet/precision plus the
        # data rows/rhs), so it applies unchanged to a Laplace mode.
        data_rows = model.constraints[structural_count:]
        data_rhs = model.constraint_rhs[structural_count:]
        if x_p is not None:
            data_rhs = data_rhs - data_rows @ x_p
        # `factor` is reassigned here to the *conditioned* factor (of
        # ``null.T @ hessian @ null``, shape ``null.shape[1]``), not the
        # pre-conditioning Hessian factor computed above -- covariance after
        # conditioning is ``N (N^T H N)^-1 N^T``, so the sampling factor below
        # must be built from that smaller factor, exactly as
        # ``inference/gaussian.py`` reassigns its own ``factor`` at the
        # matching step.
        z_reported, null, factor, data_log_density = _condition_on_data_constraints(
            z_reported, logdet_posterior, hessian,
            data_rows if identity else data_rows @ basis, data_rhs,
        )
        log_marginal_likelihood += data_log_density
        covariance_basis = null if identity else basis @ null

    # Covariance factor after any data-constraint conditioning: its columns span
    # the *data*-constrained null space, so sampler draws satisfy every extra
    # constraint row, structural or data, to numerical precision.
    covariance_factor = (
        solve_triangular(factor[0], covariance_basis.T, lower=True).T
        if reduced_dim and covariance_basis.shape[1]
        else np.zeros((latent_size, covariance_basis.shape[1]))
    )
    mean = basis @ z_reported if x_p is None else x_p + basis @ z_reported
    covariance = covariance_factor @ covariance_factor.T

    prediction_design = model.prediction_design
    prediction_offset = model.prediction_offset
    predictive_variance = quadratic_form_diagonal(prediction_design, covariance)
    predictive_mean, fitted_mean = _prediction_outputs(model, mean, predictive_variance)

    _require_finite("posterior mean", mean)
    _require_finite("posterior covariance", covariance)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)
    _require_finite("predictive variance", predictive_variance)
    _require_finite("fitted mean", fitted_mean)

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
    model: CompiledLGM, max_iterations: int, tolerance: float, mean_correction: bool = False
) -> LaplaceResult:
    """The dense engine's Newton iteration on the partitioned sparse solver.

    One Newton step for ``f(x) = -log p(y | x) + x^T Q x / 2`` on ``C_s x = e_s``
    is the weighted Gaussian solve ``x_N = H^-1 A^T (W A x + g)`` with
    ``H = Q + A^T W A``, kriged onto the structural rows -- ``_sparse_solve``
    with the working weights. Line search, convergence test and the
    Newton-decrement rescue mirror ``_fit_laplace_dense`` so both stop at the
    same mode; see docs/design/specs/2026-09-27-pylgm-sparse-laplace-design.md.
    """
    from pylgm.inference.sparse import _sparse_solve

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

    def newton_target(x: np.ndarray, *, final: bool = False):
        eta = design @ x + offset_obs
        weights = lk_obs.working_weights(eta, y_obs)
        # final: score = H x, so the solve returns x itself and every kriging /
        # determinant term is evaluated at the mode, as the dense engine does.
        pull = q @ x if final else design.T @ lk_obs.gradient(eta, y_obs)
        solve = _sparse_solve(model, weights, pull + design.T @ (weights * (design @ x)),
                              final=final)
        return solve, eta

    def reduced_gradient(x: np.ndarray) -> np.ndarray:
        eta = design @ x + offset_obs
        return project(q @ x - design.T @ lk_obs.gradient(eta, y_obs))

    # A feasible start: the minimum-norm solution of C_s x = e_s.
    x = rows.T @ cho_solve(gram, rhs) if gram is not None and np.any(rhs) else np.zeros(latent_size)
    gradient_norm = 0.0
    newton_decrement = None
    iterations = 0
    converged = False
    current = objective(x)
    for iterations in range(1, max_iterations + 1):
        gradient = reduced_gradient(x)
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm < tolerance:
            converged = True
            break
        step = newton_target(x)[0].structural_mean - x
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
        x = candidate
        current = candidate_obj
    if not converged:
        gradient = reduced_gradient(x)
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm >= tolerance:
            # Scale-invariant rescue, exactly as the dense engine (see there).
            decrement = -0.5 * float(gradient @ (newton_target(x)[0].structural_mean - x))
            if decrement >= tolerance:
                raise InferenceConvergenceError(iterations, gradient_norm)
            newton_decrement = decrement
        converged = True

    solve, eta = newton_target(x, final=True)
    posterior = solve.posterior
    centered = x - solve.nu
    log_marginal_likelihood = float(
        lk_obs.log_likelihood(eta, y_obs)
        - 0.5 * float(centered @ (q @ centered))
        + 0.5 * solve.logdet_prior
        - 0.5 * solve.logdet_posterior
    ) + model.log_likelihood_normalization + solve.data_log_density

    mean = solve.mean
    if mean_correction:
        if model.data_constraint_count:
            # ponytail: the dense engine shifts before conditioning on data rows,
            # using the structural-only covariance, which SparsePosterior does not
            # expose. Add a structural-only covariance_apply if this is needed.
            raise UnsupportedEngineError(
                "mean_correction with data constraints is not available above the "
                "sparse guard; use mean_correction=False"
            )
        eta_variance = posterior.predictive_variances(design)
        third = np.asarray(lk_obs.third_derivative(eta, y_obs), dtype=float)
        mean = mean + posterior.covariance_apply(design.T @ (0.5 * eta_variance * third))

    prediction_design = model.prediction_design
    predictive_variance = posterior.predictive_variances(prediction_design)
    predictive_mean, fitted_mean = _prediction_outputs(model, mean, predictive_variance)

    _require_finite("posterior mean", mean)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)
    _require_finite("predictive variance", predictive_variance)
    _require_finite("fitted mean", fitted_mean)
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
) -> LaplaceResult:
    """Fit a latent Gaussian model by a Laplace approximation at fixed hyperparameters.

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
    # Past the dense threshold, route to the sparse engine (as the exact Gaussian
    # engine does); allow_large_dense=True still forces the dense reference.
    fit = (
        _fit_laplace_sparse
        if not allow_large_dense and _gaussian._exceeds_dense_threshold(model)
        else _fit_laplace_dense
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return fit(model, max_iterations, tolerance, mean_correction)
    except FloatingPointError as error:
        raise NumericalError("Laplace numerical calculation was non-finite") from error

import numpy as np
from scipy.linalg import cho_solve

from pylgm.exceptions import InferenceConvergenceError, NumericalError
from pylgm.inference.gaussian import (
    _block_slices,
    _constraint_null_space,
    _constraint_particular_solution,
    _factor_positive_definite,
    _require_finite,
    preflight_dense_reference,
)
from pylgm.inference.result import LaplaceResult
from pylgm.ir.model import CompiledLGM


def _fit_laplace_dense(model: CompiledLGM, max_iterations: int, tolerance: float) -> LaplaceResult:
    likelihood = model.likelihood
    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    design = model.design
    y = model.y
    offset = model.offset
    observed = model.observed
    y_obs = y[observed]
    offset_obs = offset[observed]
    # Binomial carries a per-row trials vector; the fit loop works on the observed
    # rows, so bind their trials. For every other likelihood this returns self.
    _trials = getattr(likelihood, "trials", None)
    lk_obs = likelihood.for_observations({"trials": _trials[observed]} if _trials is not None else None)
    lk_obs.validate_response(y_obs)

    basis = _constraint_null_space(model.constraints, latent_size)
    reduced_dim = basis.shape[1]
    reduced_design = np.asarray(design[observed] @ basis)
    reduced_precision = basis.T @ precision @ basis
    prior_factor, logdet_prior = _factor_positive_definite(
        reduced_precision, "reduced prior precision"
    )

    # Nonzero-rhs constraint: x = x_p + basis @ z shifts the observed predictor by
    # design @ x_p and adds the prior linear term b_p = basis.T @ (Q x_p); the
    # induced prior mean of z is m = -Q_r^-1 b_p (conditioning by kriging).
    x_p = _constraint_particular_solution(model.constraints, model.constraint_rhs, latent_size)
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
        reduced_covariance = cho_solve(factor, np.eye(reduced_dim))
        loglik_mode = lk_obs.log_likelihood(eta, y_obs)
    else:
        reduced_covariance = np.empty((0, 0))
        logdet_posterior = 0.0
        loglik_mode = lk_obs.log_likelihood(offset_obs, y_obs)

    mean = basis @ z if x_p is None else x_p + basis @ z
    covariance = basis @ reduced_covariance @ basis.T
    centered = z - prior_mean
    log_marginal_likelihood = float(
        loglik_mode
        - 0.5 * float(centered @ reduced_precision @ centered)
        + 0.5 * logdet_prior
        - 0.5 * logdet_posterior
    )

    predictive_mean = np.asarray(offset + design @ mean).reshape(-1)
    design_dense = design.toarray()
    predictive_variance = np.einsum("ij,jk,ik->i", design_dense, covariance, design_dense)
    fitted_mean = likelihood.response_prediction(predictive_mean, predictive_variance)

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
        link_name=likelihood.link.name,
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
    )


def fit_laplace(
    model: CompiledLGM,
    *,
    allow_large_dense: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> LaplaceResult:
    """Fit a latent Gaussian model by a Laplace approximation at fixed hyperparameters.

    Likelihood-agnostic: any compiled likelihood implementing the GLM protocol works,
    so a Gaussian likelihood is fit exactly and serves as the correctness anchor.
    """
    preflight_dense_reference(model, allow_large_dense=allow_large_dense)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _fit_laplace_dense(model, max_iterations, tolerance)
    except FloatingPointError as error:
        raise NumericalError("Laplace numerical calculation was non-finite") from error

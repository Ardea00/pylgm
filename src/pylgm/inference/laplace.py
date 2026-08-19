import numpy as np
from scipy.linalg import cho_solve

from pylgm.exceptions import InferenceConvergenceError, NumericalError
from pylgm.inference.gaussian import (
    _block_slices,
    _constraint_null_space,
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
    likelihood.validate_response(y_obs)

    basis = _constraint_null_space(model.constraints, latent_size)
    reduced_dim = basis.shape[1]
    reduced_design = np.asarray(design[observed] @ basis)
    reduced_precision = basis.T @ precision @ basis
    _, logdet_prior = _factor_positive_definite(reduced_precision, "reduced prior precision")

    def objective(z: np.ndarray) -> float:
        eta = reduced_design @ z + offset_obs
        return -likelihood.log_likelihood(eta, y_obs) + 0.5 * float(z @ reduced_precision @ z)

    z = np.zeros(reduced_dim)
    gradient_norm = 0.0
    iterations = 0
    converged = reduced_dim == 0
    if reduced_dim:
        current = objective(z)
        for iterations in range(1, max_iterations + 1):
            eta = reduced_design @ z + offset_obs
            grad_ll = likelihood.gradient(eta, y_obs)
            weights = likelihood.working_weights(eta, y_obs)
            gradient = reduced_precision @ z - reduced_design.T @ grad_ll
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
                candidate_obj = objective(candidate)
                if np.isfinite(candidate_obj) and candidate_obj <= current + 1e-4 * scale * slope:
                    break
                scale *= 0.5
            else:
                raise NumericalError("Laplace line search failed to reduce the objective")
            z = candidate
            current = candidate_obj
        if not converged:
            eta = reduced_design @ z + offset_obs
            gradient = reduced_precision @ z - reduced_design.T @ likelihood.gradient(eta, y_obs)
            gradient_norm = float(np.max(np.abs(gradient)))
            if gradient_norm < tolerance:
                converged = True
            else:
                raise InferenceConvergenceError(iterations, gradient_norm)
        eta = reduced_design @ z + offset_obs
        weights = likelihood.working_weights(eta, y_obs)
        hessian = reduced_precision + (reduced_design.T * weights) @ reduced_design
        factor, logdet_posterior = _factor_positive_definite(hessian, "reduced posterior precision")
        reduced_covariance = cho_solve(factor, np.eye(reduced_dim))
        loglik_mode = likelihood.log_likelihood(eta, y_obs)
    else:
        reduced_covariance = np.empty((0, 0))
        logdet_posterior = 0.0
        loglik_mode = likelihood.log_likelihood(offset_obs, y_obs)

    mean = basis @ z
    covariance = basis @ reduced_covariance @ basis.T
    log_marginal_likelihood = float(
        loglik_mode
        - 0.5 * float(z @ reduced_precision @ z)
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

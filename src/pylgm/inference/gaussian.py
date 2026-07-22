import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space

from pylgm.inference.result import GaussianResult
from pylgm.ir.model import CompiledLGM


def _factor_positive_definite(
    matrix: np.ndarray, name: str
) -> tuple[tuple[np.ndarray, bool] | None, float]:
    if not matrix.shape[0]:
        return None, 0.0
    try:
        factor = cho_factor(matrix, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise np.linalg.LinAlgError(f"{name} must be positive definite") from error
    return factor, float(2.0 * np.log(np.diag(factor[0])).sum())


def fit_gaussian(model: CompiledLGM) -> GaussianResult:
    """Fit the exact Gaussian posterior conditional on the model hyperparameters."""
    if not np.isfinite(model.sigma) or model.sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    if precision.shape != (latent_size, latent_size):
        raise ValueError("precision must be square")
    if not np.allclose(precision, precision.T):
        raise ValueError("precision must be symmetric")
    constraints = model.constraints
    if constraints.ndim != 2 or constraints.shape[1] != latent_size:
        raise ValueError("constraints must have one column per latent variable")

    basis = null_space(constraints) if constraints.shape[0] else np.eye(latent_size)
    design = model.design
    observed = model.observed
    observed_design = design[observed]
    reduced_design = np.asarray(observed_design @ basis)
    reduced_precision = basis.T @ precision @ basis
    residual = model.y[observed] - model.offset[observed]
    variance = model.sigma**2
    posterior_precision = reduced_precision + reduced_design.T @ reduced_design / variance

    _, logdet_prior = _factor_positive_definite(
        reduced_precision, "reduced prior precision"
    )
    factor, logdet_posterior = _factor_positive_definite(
        posterior_precision, "reduced posterior precision"
    )

    if basis.shape[1]:
        score = np.asarray(reduced_design.T @ residual / variance).reshape(-1)
        assert factor is not None
        reduced_mean = cho_solve(factor, score)
        reduced_covariance = cho_solve(factor, np.eye(basis.shape[1]))
    else:
        score = np.empty(0)
        reduced_mean = np.empty(0)
        reduced_covariance = np.empty((0, 0))

    mean = basis @ reduced_mean
    covariance = basis @ reduced_covariance @ basis.T
    quadratic = residual @ residual / variance - score @ reduced_mean
    n_observed = int(observed.sum())
    log_marginal_likelihood = -0.5 * (
        n_observed * np.log(2 * np.pi * variance)
        - logdet_prior
        + logdet_posterior
        + quadratic
    )
    predictive_mean = model.offset + design @ mean
    design_dense = design.toarray()
    predictive_variance = np.einsum(
        "ij,jk,ik->i", design_dense, covariance, design_dense
    ) + variance

    return GaussianResult(
        labels=model.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=np.asarray(predictive_mean).reshape(-1),
        predictive_variance=predictive_variance,
    )

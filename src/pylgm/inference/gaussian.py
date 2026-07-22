import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space

from pylgm.inference.result import GaussianResult
from pylgm.ir.model import CompiledLGM


# Accept only floating-point roundoff relative to the scale of the precision.
_SYMMETRY_RTOL = 1e-12
_SYMMETRY_ATOL = 1e-14


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


def _constraint_null_space(constraints: np.ndarray, latent_size: int) -> np.ndarray:
    if not np.isfinite(constraints).all():
        raise ValueError("constraints must be finite")

    normalized_rows = []
    for row in constraints:
        scale = np.abs(row).max()
        if scale == 0.0:
            continue
        scaled_row = row / scale
        normalized_rows.append(scaled_row / np.linalg.norm(scaled_row))

    if not normalized_rows:
        return np.eye(latent_size)
    return null_space(np.asarray(normalized_rows))


def fit_gaussian(model: CompiledLGM) -> GaussianResult:
    """Fit the exact Gaussian posterior conditional on the model hyperparameters."""
    sigma = float(model.sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        variance = float(np.square(sigma))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("sigma squared must be finite and positive")

    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    if precision.shape != (latent_size, latent_size):
        raise ValueError("precision must be square")
    if not np.isfinite(precision).all():
        raise ValueError("precision must be finite")
    if not np.allclose(
        precision,
        precision.T,
        rtol=_SYMMETRY_RTOL,
        atol=_SYMMETRY_ATOL,
    ):
        raise ValueError("precision must be symmetric")
    precision = 0.5 * (precision + precision.T)

    constraints = model.constraints
    if constraints.ndim != 2 or constraints.shape[1] != latent_size:
        raise ValueError("constraints must have one column per latent variable")
    design = model.design
    y = model.y
    offset = model.offset
    observed = model.observed
    if observed.ndim != 1 or not np.issubdtype(observed.dtype, np.bool_):
        raise ValueError("observed must be a one-dimensional boolean array")
    if y.ndim != 1 or offset.ndim != 1:
        raise ValueError("y and offset must be one-dimensional arrays")
    if not (y.size == observed.size == offset.size == design.shape[0]):
        raise ValueError("y, observed, offset, and design must have matching row counts")
    if design.shape[1] != latent_size:
        raise ValueError("design must have one column per latent variable")
    if len(model.labels) != latent_size:
        raise ValueError("labels must have one entry per latent variable")

    basis = _constraint_null_space(constraints, latent_size)
    observed_design = design[observed]
    reduced_design = np.asarray(observed_design @ basis)
    reduced_precision = basis.T @ precision @ basis
    residual = y[observed] - offset[observed]
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
    n_observed = int(np.count_nonzero(observed))
    log_marginal_likelihood = -0.5 * (
        n_observed * np.log(2 * np.pi * variance)
        - logdet_prior
        + logdet_posterior
        + quadratic
    )
    predictive_mean = offset + design @ mean
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

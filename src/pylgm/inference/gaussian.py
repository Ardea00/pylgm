from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space

from pylgm.exceptions import DenseReferenceLimitError, NumericalError, UnsupportedEngineError
from pylgm.inference.result import GaussianResult
from pylgm.ir.model import CompiledLGM
from pylgm.likelihoods import CompiledGaussian


_MAX_DENSE_LATENT_DIMENSION = 4_096
_MAX_DENSE_BYTES = 512 * 1024 * 1024


def _estimated_dense_bytes(observation_count: int, latent_size: int) -> int:
    """Conservatively estimate peak arrays used by the dense reference algorithm."""
    float_count = (
        6 * latent_size * latent_size
        + 2 * observation_count * latent_size
        + 4 * latent_size
        + observation_count
    )
    return np.dtype(np.float64).itemsize * float_count


def preflight_dense_reference(model: CompiledLGM, *, allow_large_dense: bool) -> None:
    """Reject unsafe dense-reference work unless an exact boolean override is set."""
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if allow_large_dense:
        return
    latent_size = model.precision.shape[0]
    if latent_size > _MAX_DENSE_LATENT_DIMENSION:
        raise DenseReferenceLimitError(
            "exact Gaussian dense reference latent dimension "
            f"{latent_size} exceeds {_MAX_DENSE_LATENT_DIMENSION}; "
            "pass allow_large_dense=True to opt in"
        )
    estimated_bytes = _estimated_dense_bytes(model.design.shape[0], latent_size)
    if estimated_bytes > _MAX_DENSE_BYTES:
        raise DenseReferenceLimitError(
            "exact Gaussian estimated dense workspace "
            f"{estimated_bytes} bytes exceeds {_MAX_DENSE_BYTES}; "
            "pass allow_large_dense=True to opt in"
        )


# ponytail: mirrors preflight_dense_reference's thresholds as a plain bool for
# routing. preflight keeps its two detailed raises (tested message text), so it
# is intentionally not collapsed into this predicate.
def _exceeds_dense_threshold(model: CompiledLGM) -> bool:
    latent_size = model.precision.shape[0]
    if latent_size > _MAX_DENSE_LATENT_DIMENSION:
        return True
    return _estimated_dense_bytes(model.design.shape[0], latent_size) > _MAX_DENSE_BYTES


def _factor_positive_definite(
    matrix: np.ndarray, name: str
) -> tuple[tuple[np.ndarray, bool] | None, float]:
    if not matrix.shape[0]:
        return None, 0.0
    try:
        factor = cho_factor(matrix, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as error:
        raise NumericalError(f"{name} must be positive definite") from error
    return factor, float(2.0 * np.log(np.diag(factor[0])).sum())


def _constraint_null_space(constraints: np.ndarray, latent_size: int) -> np.ndarray:
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


def _constraint_particular_solution(
    constraints: np.ndarray, rhs: np.ndarray, latent_size: int
) -> np.ndarray | None:
    """A least-norm ``x_p`` with ``constraints @ x_p == rhs``.

    Returns ``None`` for the homogeneous case (``rhs`` all zero) so the caller
    keeps the exact ``A x = 0`` code path unchanged. The choice of particular
    solution is otherwise immaterial: the engines add the induced prior linear
    term, which makes the fit match conditioning-by-kriging regardless of ``x_p``.
    """
    # ponytail: no feasibility guard; contradictory constraints degrade to the
    # least-squares x_p, which is the closest satisfiable field anyway.
    if not constraints.shape[0] or not np.any(rhs):
        return None
    x_p, *_ = np.linalg.lstsq(constraints, rhs, rcond=None)
    return x_p


def _require_finite(name: str, value: np.ndarray | float) -> None:
    if not np.isfinite(value).all():
        raise NumericalError(f"exact Gaussian produced non-finite {name}")


def _block_slices(model: CompiledLGM) -> Mapping[str, slice]:
    start = 0
    result: dict[str, slice] = {}
    for block in model.blocks:
        stop = start + block.design.shape[1]
        result[block.name] = slice(start, stop)
        start = stop
    return result


def _fit_dense(model: CompiledLGM) -> GaussianResult:
    variance = float(model.likelihood.variance)
    if not np.isfinite(variance) or variance <= 0:
        raise NumericalError("sigma squared must be finite and positive")

    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    constraints = model.constraints
    design = model.design
    y = model.y
    offset = model.offset
    observed = model.observed

    basis = _constraint_null_space(constraints, latent_size)
    x_p = _constraint_particular_solution(constraints, model.constraint_rhs, latent_size)
    observed_design = design[observed]
    reduced_design = np.asarray(observed_design @ basis)
    reduced_precision = basis.T @ precision @ basis
    residual = y[observed] - offset[observed]
    # Nonzero-rhs constraint: x = x_p + basis @ z shifts the likelihood residual
    # by design @ x_p and adds the prior linear term b_p = basis.T @ (Q x_p),
    # whose prior mean of z is m = -Q_r^-1 b_p (conditioning by kriging).
    prior_linear = np.zeros(basis.shape[1])
    prior_mean = np.zeros(basis.shape[1])
    if x_p is not None:
        residual = residual - np.asarray(observed_design @ x_p).reshape(-1)
        prior_linear = basis.T @ (precision @ x_p)
    posterior_precision = reduced_precision + reduced_design.T @ reduced_design / variance

    prior_factor, logdet_prior = _factor_positive_definite(
        reduced_precision, "reduced prior precision"
    )
    factor, logdet_posterior = _factor_positive_definite(
        posterior_precision, "reduced posterior precision"
    )

    if basis.shape[1]:
        score = np.asarray(reduced_design.T @ residual / variance).reshape(-1) - prior_linear
        assert factor is not None
        reduced_mean = cho_solve(factor, score)
        reduced_covariance = cho_solve(factor, np.eye(basis.shape[1]))
        if x_p is not None:
            assert prior_factor is not None
            prior_mean = cho_solve(prior_factor, -prior_linear)
    else:
        reduced_mean = np.empty(0)
        reduced_covariance = np.empty((0, 0))

    mean = basis @ reduced_mean if x_p is None else x_p + basis @ reduced_mean
    covariance = basis @ reduced_covariance @ basis.T
    posterior_residual = residual - reduced_design @ reduced_mean
    centered = reduced_mean - prior_mean
    quadratic = float(
        posterior_residual @ posterior_residual / variance
        + centered @ reduced_precision @ centered
    )
    n_observed = int(np.count_nonzero(observed))
    log_marginal_likelihood = -0.5 * (
        n_observed * np.log(2 * np.pi * variance) - logdet_prior + logdet_posterior + quadratic
    )
    predictive_mean = np.asarray(offset + design @ mean).reshape(-1)
    design_dense = design.toarray()
    predictive_variance = np.einsum("ij,jk,ik->i", design_dense, covariance, design_dense)

    _require_finite("posterior mean", mean)
    _require_finite("posterior covariance", covariance)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)
    _require_finite("predictive variance", predictive_variance)
    return GaussianResult(
        labels=model.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        observation_variance=variance,
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": int(np.count_nonzero(observed)),
            "constraint_count": int(constraints.shape[0]),
        },
    )


def _fit_sparse(model: CompiledLGM) -> GaussianResult:
    # ponytail: import sparse_constrained_gaussian lazily here, NOT at module
    # top. sparse.py imports _block_slices/_factor_positive_definite from this
    # module at its top level; a top-level back-import would be circular and
    # fail at import time (the needed names are not yet bound). A function-local
    # import breaks the cycle.
    from pylgm.inference.sparse import sparse_constrained_gaussian

    variance = float(model.likelihood.variance)
    fit = sparse_constrained_gaussian(model)
    _require_finite("posterior mean", fit.mean)
    _require_finite("log marginal likelihood", fit.log_marginal_likelihood)
    _require_finite("predictive mean", fit.predictive_mean)
    return GaussianResult(
        labels=model.labels,
        mean=fit.mean,
        covariance=None,                       # pending E-sparse-C
        log_marginal_likelihood=fit.log_marginal_likelihood,
        predictive_mean=fit.predictive_mean,
        predictive_variance=None,              # pending E-sparse-C
        observation_variance=variance,
        block_slices=fit.block_slices,
        diagnostics=fit.diagnostics,
    )


def fit_gaussian(model: CompiledLGM, *, allow_large_dense: bool = False) -> GaussianResult:
    """Fit the small/medium exact Gaussian dense reference engine.

    The explicit override disables conservative memory and dimension guards. It does
    not change the algorithm's O(p^2) covariance storage or O(p^3) dense solve cost.
    """
    if not isinstance(model.likelihood, CompiledGaussian):
        raise UnsupportedEngineError("exact Gaussian inference requires a Gaussian likelihood")
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if not allow_large_dense and _exceeds_dense_threshold(model):
        return _fit_sparse(model)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _fit_dense(model)
    except FloatingPointError as error:
        raise NumericalError("exact Gaussian numerical calculation was non-finite") from error

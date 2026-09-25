from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space, solve_triangular

from pylgm.exceptions import DenseReferenceLimitError, NumericalError, UnsupportedEngineError
from pylgm.inference.result import GaussianResult, quadratic_form_diagonal
from pylgm.inference.sampling import GridSampler
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


def _augmented_split(labels: tuple[str, ...]) -> int | None:
    """Half-width if ``labels`` are an augmented (x, u*) block, else ``None``.

    The augmented BYM2 builder lays out its 2n labels as n x-labels followed by
    their ``__u`` twins (``"3"`` -> ``"3__u"``). Detecting that exact layout lets
    ``_block_slices`` report the x-effect and the structured ``u*`` component as
    two named slices without threading extra metadata through every block rebuild.
    """
    n = len(labels)
    if n == 0 or n % 2 != 0:
        return None
    half = n // 2
    head, tail = labels[:half], labels[half:]
    if all(t == f"{h}__u" for h, t in zip(head, tail, strict=False)):
        return half
    return None


def _block_slices(model: CompiledLGM) -> Mapping[str, slice]:
    start = 0
    result: dict[str, slice] = {}
    for block in model.blocks:
        stop = start + block.design.shape[1]
        split = _augmented_split(block.labels)
        if split is None:
            result[block.name] = slice(start, stop)
        else:
            # Augmented BYM2: x-effect is the first half, structured u* the second.
            result[block.name] = slice(start, start + split)
            result[f"{block.name}.structured"] = slice(start + split, stop)
        start = stop
    return result


def _condition_on_data_constraints(
    reduced_mean: np.ndarray,
    logdet_posterior: float,
    precision: np.ndarray,
    rows: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple | None, float]:
    """Score and impose data rows ``rows @ z = rhs`` on ``z ~ N(reduced_mean, precision^-1)``.

    Returns the conditioned mean, the basis ``N`` of ``null(rows)`` with the factor
    of ``N^T P N`` (so the conditioned covariance is ``N (N^T P N)^-1 N^T``), and
    ``log N(rhs; rows @ mean, rows P^-1 rows^T)``. Everything stays in precision
    form: a vague direction (``Fixed`` prior) has a tiny precision, never a huge
    covariance to difference, so the density stays exact where kriging would
    cancel catastrophically. ``logdet(rows P^-1 rows^T)`` comes from the identity
    ``logdet(N^T P N) = logdet(P) + logdet(rows P^-1 rows^T) - logdet(rows rows^T)``.
    """
    # One SVD gives null(rows), the least-norm particular solution and the gram
    # logdet; the rows are full rank (redundant ones were dropped at projection).
    left, singular, right = np.linalg.svd(rows)
    null = right[rows.shape[0]:].T
    particular = right[: rows.shape[0]].T @ ((left.T @ rhs) / singular)
    logdet_gram = float(2.0 * np.log(singular).sum())
    null_factor, logdet_null = _factor_positive_definite(
        null.T @ precision @ null, "data-constrained posterior precision"
    )
    if null.shape[1]:
        step = cho_solve(null_factor, null.T @ (precision @ (reduced_mean - particular)))
        conditioned = particular + null @ step
    else:
        conditioned = particular
    gap = conditioned - reduced_mean
    log_density = -0.5 * (
        rhs.size * np.log(2 * np.pi) - logdet_posterior + logdet_null + logdet_gram
        + gap @ precision @ gap
    )
    return conditioned, null, null_factor, float(log_density)


def _fit_dense(model: CompiledLGM, *, predictive_variances: bool = True) -> GaussianResult:
    variance = float(model.likelihood.variance)
    if not np.isfinite(variance) or variance <= 0:
        raise NumericalError("sigma squared must be finite and positive")

    precision = model.precision.toarray()
    latent_size = precision.shape[0]
    design = model.design
    y = model.y
    offset = model.offset
    observed = model.observed
    # Condition on the structural rows by reduction; the trailing data rows are
    # exact observations of A_D x, scored and then imposed by a second reduction.
    structural_count = model.constraints.shape[0] - model.data_constraint_count
    constraints = model.constraints[:structural_count]
    constraint_rhs = model.constraint_rhs[:structural_count]

    basis = _constraint_null_space(constraints, latent_size)
    x_p = _constraint_particular_solution(constraints, constraint_rhs, latent_size)
    observed_design = design[observed]
    # With no structural rows the basis is the identity: skip the O(p^3) products.
    identity = not constraints.shape[0]
    reduced_design = np.asarray(observed_design.toarray() if identity else observed_design @ basis)
    reduced_precision = precision if identity else basis.T @ precision @ basis
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
        if x_p is not None:
            assert prior_factor is not None
            prior_mean = cho_solve(prior_factor, -prior_linear)
    else:
        reduced_mean = np.empty(0)

    posterior_residual = residual - reduced_design @ reduced_mean
    centered = reduced_mean - prior_mean
    quadratic = float(
        posterior_residual @ posterior_residual / variance
        + centered @ reduced_precision @ centered
    )
    n_observed = int(np.count_nonzero(observed))
    log_marginal_likelihood = -0.5 * (
        n_observed * np.log(2 * np.pi * variance) - logdet_prior + logdet_posterior + quadratic
    ) + model.log_likelihood_normalization

    covariance_basis = basis
    if model.data_constraint_count:
        # log p(y, e) = log p(y) + log p(e | y): the data rows are exact observations.
        data_rows = model.constraints[structural_count:]
        data_rhs = model.constraint_rhs[structural_count:]
        if x_p is not None:
            data_rhs = data_rhs - data_rows @ x_p
        reduced_mean, null, factor, data_log_density = _condition_on_data_constraints(
            reduced_mean, logdet_posterior, posterior_precision,
            data_rows if identity else data_rows @ basis, data_rhs,
        )
        log_marginal_likelihood += data_log_density
        covariance_basis = null if identity else basis @ null

    # Covariance = F F^T with F = basis L^-T: F doubles as the sampling factor, and
    # its columns span the constraint null space, so every draw meets A x = e.
    covariance_factor = (
        solve_triangular(factor[0], covariance_basis.T, lower=True).T
        if covariance_basis.shape[1] else np.zeros((latent_size, 0))
    )
    mean = basis @ reduced_mean if x_p is None else x_p + basis @ reduced_mean
    covariance = covariance_factor @ covariance_factor.T
    prediction_design = model.prediction_design
    predictive_mean = np.asarray(
        model.prediction_offset + prediction_design @ mean
    ).reshape(-1)
    predictive_variance = (
        quadratic_form_diagonal(prediction_design, covariance)
        if predictive_variances else None
    )

    _require_finite("posterior mean", mean)
    _require_finite("posterior covariance", covariance)
    _require_finite("log marginal likelihood", log_marginal_likelihood)
    _require_finite("predictive mean", predictive_mean)
    if predictive_variance is not None:
        _require_finite("predictive variance", predictive_variance)
    return GaussianResult(
        labels=model.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        observation_variance=(
            variance
            if model.prediction_observation_variance is None
            else model.prediction_observation_variance
        ),
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": int(np.count_nonzero(observed)),
            "constraint_count": int(model.constraints.shape[0]),
        },
        sampler=GridSampler(
            mean, prediction_design, model.prediction_offset, factor=covariance_factor
        ),
    )


def _fit_sparse(model: CompiledLGM, *, predictive_variances: bool = True) -> GaussianResult:
    # ponytail: import sparse_constrained_gaussian lazily here, NOT at module
    # top. sparse.py imports _block_slices/_factor_positive_definite from this
    # module at its top level; a top-level back-import would be circular and
    # fail at import time (the needed names are not yet bound). A function-local
    # import breaks the cycle.
    from pylgm.inference.sparse import sparse_constrained_gaussian

    variance = float(model.likelihood.variance)
    fit = sparse_constrained_gaussian(model)
    predictive_variance = (
        fit.posterior.predictive_variances(model.prediction_design)
        if predictive_variances else None
    )
    _require_finite("posterior mean", fit.mean)
    _require_finite("log marginal likelihood", fit.log_marginal_likelihood)
    _require_finite("predictive mean", fit.predictive_mean)
    if predictive_variance is not None:
        _require_finite("predictive variance", predictive_variance)
    return GaussianResult(
        labels=model.labels,
        mean=fit.mean,
        covariance=None,                       # never materialised above the guard
        log_marginal_likelihood=fit.log_marginal_likelihood,
        predictive_mean=fit.predictive_mean,
        predictive_variance=predictive_variance,
        observation_variance=(
            variance
            if model.prediction_observation_variance is None
            else model.prediction_observation_variance
        ),
        block_slices=fit.block_slices,
        diagnostics=fit.diagnostics,
        sparse_posterior=fit.posterior,
        sampler=GridSampler(
            fit.mean, model.prediction_design, model.prediction_offset, posterior=fit.posterior
        ),
    )


def fit_gaussian(
    model: CompiledLGM, *, allow_large_dense: bool = False, predictive_variances: bool = True
) -> GaussianResult:
    """Fit the small/medium exact Gaussian dense reference engine.

    The explicit override disables conservative memory and dimension guards. It does
    not change the algorithm's O(p^2) covariance storage or O(p^3) dense solve cost.

    ``predictive_variances=False`` skips the predictive-variance computation
    (``GaussianResult.predictive_variance`` is then ``None``), for callers that
    only need ``log_marginal_likelihood`` -- e.g. intermediate empirical-Bayes
    objective evaluations, where predictive variances are never read but can
    dominate the per-evaluation cost.
    """
    if not isinstance(model.likelihood, CompiledGaussian):
        raise UnsupportedEngineError("exact Gaussian inference requires a Gaussian likelihood")
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if type(predictive_variances) is not bool:
        raise TypeError("predictive_variances must be a boolean")
    if not allow_large_dense and _exceeds_dense_threshold(model):
        return _fit_sparse(model, predictive_variances=predictive_variances)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _fit_dense(model, predictive_variances=predictive_variances)
    except FloatingPointError as error:
        raise NumericalError("exact Gaussian numerical calculation was non-finite") from error

"""Sequential conditioning of an exact-Gaussian posterior on new rows.

At fixed hyperparameters, k new observations ``y = A x + offset + e`` with
``e ~ N(0, sigma^2 I)`` update the posterior ``N(mu, Sigma)`` exactly:

    S      = A Sigma A^T + sigma^2 I                (k x k)
    mu'    = mu + Sigma A^T S^-1 r,    r = y - offset - A mu
    Sigma' = Sigma - Sigma A^T S^-1 A Sigma
    lml'   = lml + log N(r; 0, S)                   (= log p(y_new | y_old))

``Sigma A^T`` costs k solves against the factor already held, so no new
factorisation is made; the constraint projection is inside ``Sigma``.
``Sigma'`` is kept as the base posterior minus a rank-k term, and draws use
Matheron's rule, so repeated updates chain.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pylgm.exceptions import NumericalError


def _quadratic_diagonal(left: np.ndarray, s_factor) -> np.ndarray:
    """diag(left S^-1 left^T)."""
    return np.einsum("ij,ji->i", left, cho_solve(s_factor, left.T))


@dataclass(frozen=True)
class _DenseFactor:
    """A dense posterior's sampling factor, ``Sigma = factor @ factor.T``."""

    factor: np.ndarray

    def sample_deviations(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n, self.factor.shape[1])) @ self.factor.T


@dataclass(frozen=True)
class UpdatedPosterior:
    """``Sigma' = Sigma_base - V S^-1 V^T`` with ``V = Sigma_base A^T``.

    Duck-types ``SparsePosterior`` (variances, sampling, covariance products),
    so it can serve as the base of a further update.
    """

    base: object
    v: np.ndarray
    design: np.ndarray
    s_factor: tuple
    noise_sd: float

    def covariance_apply(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float)
        return self.base.covariance_apply(rhs) - self.v @ cho_solve(self.s_factor, self.v.T @ rhs)

    def marginal_variances(self) -> np.ndarray:
        base = self.base.marginal_variances()
        return np.clip(base - _quadratic_diagonal(self.v, self.s_factor), 0.0, None)

    def predictive_variances(self, design) -> np.ndarray:
        dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
        base = self.base.predictive_variances(dense)
        return np.clip(base - _quadratic_diagonal(dense @ self.v, self.s_factor), 0.0, None)

    def linear_combination_variances(self, weights) -> np.ndarray:
        return self.predictive_variances(weights)

    def sample_deviations(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Matheron: x' = x - Sigma A^T S^-1 (A x + e) is exactly N(0, Sigma').
        base = self.base.sample_deviations(n, rng)
        noise = self.noise_sd * rng.standard_normal((n, self.design.shape[0]))
        residual = base @ self.design.T + noise
        return base - cho_solve(self.s_factor, residual.T).T @ self.v.T


def condition_on_rows(result, new_data):
    """Return ``result`` conditioned on ``new_data``'s observed rows (see module doc)."""
    from pylgm.inference.prediction import PredictionContext, _design_for, _offset_for
    from pylgm.inference.result import GaussianResult, quadratic_form_diagonal
    from pylgm.inference.sampling import GridSampler

    context = result.prediction_context
    sampler = result._sampler
    if not isinstance(context, PredictionContext) or context.response is None or sampler is None:
        raise ValueError("update() is available on single-response results produced by LGM.fit")
    if context.response not in new_data.columns:
        raise ValueError(f"update() new_data is missing the response column {context.response!r}")
    y = np.asarray(new_data[context.response], dtype=float)
    observed = np.isfinite(y)
    if not observed.any():
        return result
    rows = new_data[observed]
    design = _design_for(context, rows)
    residual = y[observed] - _offset_for(context, rows) - design @ result.mean

    dense = result._covariance is not None
    v = result._covariance @ design.T if dense else result._sparse_posterior.covariance_apply(design.T)
    variance = float(result.observation_variance)
    s = design @ v + variance * np.eye(design.shape[0])
    try:
        s_factor = cho_factor(s, lower=True)
    except np.linalg.LinAlgError as error:
        raise NumericalError("update() innovation covariance is not positive definite") from error
    alpha = cho_solve(s_factor, residual)
    logdet = 2.0 * float(np.sum(np.log(np.diag(s_factor[0]))))

    posterior = UpdatedPosterior(
        base=_DenseFactor(sampler.factor) if sampler.factor is not None else sampler.posterior,
        v=v, design=design, s_factor=s_factor, noise_sd=float(np.sqrt(variance)),
    )
    # The fitted grid: canonical rows of the sampler's design, in caller order.
    grid = sampler.design if sampler.row_order is None else sampler.design[sampler.row_order]
    # diag(G S^-1 G^T), G = P V: through the p x p Delta = V S^-1 V^T when that is
    # cheaper (O(p^2 k + nnz(P) p)) than the n x k route (O(n k^2)); the dense
    # path needs Delta for the covariance anyway.
    covariance = None
    (p, k), n = v.shape, grid.shape[0]
    via_delta = dense or p * p * k + grid.nnz * p < n * k * k
    delta = v @ cho_solve(s_factor, v.T) if via_delta else None
    if dense:
        covariance = result._covariance - delta
        covariance = 0.5 * (covariance + covariance.T)
    predictive_variance = result._predictive_variance
    if predictive_variance is not None:
        shrink = (
            quadratic_form_diagonal(grid, delta) if delta is not None
            else _quadratic_diagonal(np.asarray(grid @ v), s_factor)
        )
        predictive_variance = np.clip(predictive_variance - shrink, 0.0, None)
    mean = result.mean + v @ alpha
    diagnostics = dict(result.diagnostics)
    diagnostics["updated_rows"] = int(diagnostics.get("updated_rows", 0)) + int(observed.sum())
    return GaussianResult(
        labels=result.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=result.log_marginal_likelihood
        - 0.5 * (design.shape[0] * np.log(2 * np.pi) + logdet + residual @ alpha),
        predictive_mean=result.predictive_mean + np.asarray(grid @ (v @ alpha)).ravel(),
        predictive_variance=predictive_variance,
        observation_variance=result.observation_variance,
        block_slices=result.block_slices,
        diagnostics=diagnostics,
        prediction_keys=result.prediction_keys,
        hyperparameters=result.hyperparameters,
        prediction_context=context,
        sparse_posterior=None if dense else posterior,
        sampler=GridSampler(mean, sampler.design, sampler.offset, posterior=posterior,
                            row_order=sampler.row_order),
    )

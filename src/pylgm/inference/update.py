"""Sequential conditioning of a fitted posterior on new rows.

At fixed hyperparameters the fitted posterior ``N(mu, Sigma)`` becomes the
prior for k new rows ``y ~ p(y | eta)``, ``eta = offset + A x``. The new rows
see ``x`` only through ``eta``, whose prior is ``N(m, S0)`` with
``m = offset + A mu`` and ``S0 = A V``, ``V = Sigma A^T``, so the mode is a
k-dimensional Newton problem (Rasmussen & Williams 2006, Alg. 3.1):

    B      = I + W^1/2 S0 W^1/2,     W = -d^2 log p(y | eta) at the mode
    mu'    = mu + V a,               a = d log p(y | eta) at the mode
    Sigma' = Sigma - V W^1/2 B^-1 W^1/2 V^T
    lml'   = lml + log p(y | eta^) - a^T (eta^ - m) / 2 - log|B| / 2

For a Gaussian likelihood this is exact (one Newton step) and equals a refit on
all rows. For any other likelihood it is the Laplace approximation of
``p(y_new | y_old)`` with the old rows' curvature frozen at the old mode: a
refit would re-linearise the old rows around the new mode, so the two differ
at second order in the mode shift (through the likelihood's third derivative).

``V`` costs k solves against the factor already held, so no new factorisation
is made; the constraint projection is inside ``Sigma``. ``Sigma'`` is kept as
the base posterior minus a rank-k term, and draws use Matheron's rule, so
repeated updates chain; a dense chain collapses to one factor once it holds
more rows than latents.

``update_integrated`` lifts this to an INLA result: each grid point's
conditional is updated and the grid is reweighted by its
``p(y_new | y_old, theta_k)`` -- Bayes' rule on the hyperparameter grid.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from pylgm.exceptions import InferenceConvergenceError, NumericalError, UnsupportedEngineError


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


def _chain_rank(posterior) -> int:
    """Rows accumulated in a chain of ``UpdatedPosterior`` terms."""
    rank = 0
    while isinstance(posterior, UpdatedPosterior):
        rank += posterior.v.shape[1]
        posterior = posterior.base
    return rank


def _psd_factor(covariance: np.ndarray) -> np.ndarray:
    """``F`` with ``F F^T = covariance``, dropping its (constraint) null directions."""
    values, vectors = np.linalg.eigh(covariance)
    keep = values > 1e-12 * max(float(values[-1]), 0.0)
    return vectors[:, keep] * np.sqrt(values[keep])


def _rows_mode(m: np.ndarray, s0: np.ndarray, y: np.ndarray, likelihood, max_iterations=100):
    """Mode of ``p(y | eta) N(eta; m, S0)``: ``(a, W^1/2, cho(B), log p(y))`` (module doc)."""
    k = m.size

    def curvature(eta):
        weights = np.asarray(likelihood.working_weights(eta, y), dtype=float)
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise NumericalError(
                "update() needs a log-concave likelihood at the new rows (non-negative "
                "working weights); refit instead"
            )
        root = np.sqrt(weights)
        try:
            return weights, root, cho_factor(np.eye(k) + root[:, None] * s0 * root, lower=True)
        except np.linalg.LinAlgError as error:
            raise NumericalError("update() innovation covariance is not positive definite") from error

    a = np.zeros(k)
    eta = m.copy()
    psi = likelihood.log_likelihood(eta, y)
    for _ in range(max_iterations):
        weights, root, b_factor = curvature(eta)
        b = weights * (eta - m) + likelihood.gradient(eta, y)
        step = b - root * cho_solve(b_factor, root * (s0 @ b)) - a
        for _ in range(50):   # step halving; psi is concave in a, so this ends
            trial = a + step
            trial_eta = m + s0 @ trial
            trial_psi = likelihood.log_likelihood(trial_eta, y) - 0.5 * trial @ (trial_eta - m)
            if np.isfinite(trial_psi) and trial_psi >= psi - 1e-12 * (1.0 + abs(psi)):
                break
            step *= 0.5
        else:
            raise NumericalError("update() line search failed to raise the objective")
        done = abs(trial_psi - psi) <= 1e-12 * (1.0 + abs(psi))
        a, eta, psi = trial, trial_eta, trial_psi
        if done:
            break
    else:
        raise InferenceConvergenceError(max_iterations, float(np.max(np.abs(step))))
    _, root, b_factor = curvature(eta)
    return a, root, b_factor, psi - float(np.sum(np.log(np.diag(b_factor[0]))))


def _rows_likelihood(result, context, rows):
    """The likelihood the new rows are scored under, bound to their own trials."""
    from pylgm.inference.prediction import _prediction_likelihood
    from pylgm.inference.result import GaussianResult
    from pylgm.likelihoods import CompiledGaussian, CompiledWeibullSurv

    if isinstance(result, GaussianResult):
        return CompiledGaussian(float(np.sqrt(result.observation_variance)))
    if isinstance(context.likelihood, CompiledWeibullSurv):
        raise UnsupportedEngineError("update() does not support survival likelihoods; refit instead")
    return _prediction_likelihood(context, rows)


def _fitted_mean(likelihood, predictive_mean, predictive_variance, row_order):
    """``E[g^-1(eta)]`` per grid row in caller order; ``likelihood`` is in canonical order."""
    if row_order is None:
        return likelihood.response_prediction(predictive_mean, predictive_variance)
    canonical = np.empty_like(predictive_mean)
    canonical[row_order] = predictive_mean
    variance = None
    if predictive_variance is not None:
        variance = np.empty_like(predictive_variance)
        variance[row_order] = predictive_variance
    return np.asarray(likelihood.response_prediction(canonical, variance))[row_order]


@dataclass(frozen=True)
class _Conditioning:
    """The k new rows' conditioning of a fitted posterior (module doc symbols)."""

    rows: object          # the observed new rows
    design: np.ndarray    # A
    m: np.ndarray         # prior mean of eta at the rows
    s0: np.ndarray        # A Sigma A^T
    v: np.ndarray         # Sigma A^T
    a: np.ndarray
    root: np.ndarray      # W^1/2
    s_factor: tuple       # cho(B)
    log_evidence: float


def _condition(result, new_data, caller="update"):
    """``_Conditioning`` for ``new_data``'s observed rows, or ``None`` if there are none."""
    from pylgm.inference.prediction import PredictionContext, _design_for, _offset_for

    context = result.prediction_context
    if (not isinstance(context, PredictionContext) or context.response is None
            or result._sampler is None):
        raise ValueError(f"{caller}() is available on single-response results produced by LGM.fit")
    if context.response not in new_data.columns:
        raise ValueError(
            f"{caller}() new_data is missing the response column {context.response!r}"
        )
    y = np.asarray(new_data[context.response], dtype=float)
    observed = np.isfinite(y)
    if not observed.any():
        return None
    rows = new_data[observed]
    likelihood = _rows_likelihood(result, context, rows)
    likelihood.validate_response(y[observed])
    design = _design_for(context, rows)
    if result._covariance is not None:
        v = result._covariance @ design.T
    else:
        v = result._sparse_posterior.covariance_apply(design.T)
    m = _offset_for(context, rows) + design @ result.mean
    s0 = design @ v
    a, root, s_factor, log_evidence = _rows_mode(m, s0, y[observed], likelihood)
    return _Conditioning(rows, design, m, s0, v, a, root, s_factor, log_evidence)


def condition_on_rows(result, new_data):
    """Return ``result`` conditioned on ``new_data``'s observed rows (see module doc)."""
    conditioning = _condition(result, new_data)
    return result if conditioning is None else _conditioned(result, conditioning)


def _conditioned(result, c: _Conditioning):
    from pylgm.inference.result import GaussianResult, LaplaceResult, quadratic_form_diagonal
    from pylgm.inference.sampling import GridSampler

    context = result.prediction_context
    sampler = result._sampler
    design, s_factor, root = c.design, c.s_factor, c.root
    dense = result._covariance is not None
    shift = c.v @ c.a
    # Whitening the rows by W^1/2 turns S into B with unit noise.
    v = c.v * root
    posterior = UpdatedPosterior(
        base=_DenseFactor(sampler.factor) if sampler.factor is not None else sampler.posterior,
        v=v, design=design * root[:, None], s_factor=s_factor, noise_sd=1.0,
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
        if _chain_rank(posterior) >= p:
            # Past p accumulated rows the low-rank chain costs more than a dense
            # factor of Sigma' itself: collapse it (O(p^3) once per ~p rows).
            posterior = _DenseFactor(_psd_factor(covariance))
    predictive_variance = result._predictive_variance
    if predictive_variance is not None:
        shrink = (
            quadratic_form_diagonal(grid, delta) if delta is not None
            else _quadratic_diagonal(np.asarray(grid @ v), s_factor)
        )
        predictive_variance = np.clip(predictive_variance - shrink, 0.0, None)
    mean = result.mean + shift
    predictive_mean = result.predictive_mean + np.asarray(grid @ shift).ravel()
    diagnostics = dict(result.diagnostics)
    diagnostics["updated_rows"] = int(diagnostics.get("updated_rows", 0)) + len(c.rows)
    common = dict(
        labels=result.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=result.log_marginal_likelihood + c.log_evidence,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        block_slices=result.block_slices,
        diagnostics=diagnostics,
        prediction_keys=result.prediction_keys,
        hyperparameters=result.hyperparameters,
        prediction_context=context,
        sparse_posterior=None if dense else posterior,
        sampler=GridSampler(mean, sampler.design, sampler.offset, posterior=posterior,
                            row_order=sampler.row_order),
    )
    if isinstance(result, GaussianResult):
        return GaussianResult(observation_variance=result.observation_variance, **common)
    return LaplaceResult(
        fitted_mean=_fitted_mean(context.likelihood, predictive_mean, predictive_variance,
                                 sampler.row_order),
        link_name=result.link_name, **common,
    )


@dataclass(frozen=True)
class IntegrationGrid:
    """What an INLA result keeps to be updated: its retained grid points.

    ``conditionals[k]()`` returns point k's conditional fit (refitted lazily
    until an update makes it concrete); ``s`` is the unnormalised log posterior
    of theta there and ``weights`` the normalised integration weights.
    ``marginals(grid, s_values, theta_mean, theta_sq)`` rebuilds the
    hyperparameter marginals and ``context(marginals)`` the prediction context
    whose plug-in hyperparameters they fix.
    """

    thetas: tuple
    u: np.ndarray
    s: np.ndarray
    weights: np.ndarray
    conditionals: tuple
    marginals: object
    context: object = None


def update_integrated(result, new_data):
    """Update every grid point's conditional and reweight the grid (module doc)."""
    grid = result._grid
    if grid is None:
        raise ValueError("update() is available on integrated results produced by LGM.fit")
    if result.latent_marginal_table is not None:
        # The skewed marginals are fitted on the old rows' data, which the result
        # does not keep; the updated grid mixture is what remains exact.
        warnings.warn(
            "update() reports the Gaussian grid-mixture latent marginals; refit for "
            "the skewed latent_strategy on all rows.",
            UserWarning, stacklevel=3,
        )
    before = [conditional() for conditional in grid.conditionals]
    return _reweighted(result, before, [conditional.update(new_data) for conditional in before])


def _reweighted(result, before, after):
    """The integrated result whose grid conditionals moved from ``before`` to ``after``."""
    from pylgm.inference.result import INLAResult
    from pylgm.optimization.inla import _integrated_moments, _theta_moments

    grid = result._grid
    evidence = np.array([
        new.log_marginal_likelihood - old.log_marginal_likelihood
        for old, new in zip(before, after, strict=True)
    ])
    if not evidence.any():
        return result
    # p(theta | y_old, y_new) ~ p(theta | y_old) p(y_new | y_old, theta), on the grid.
    total, sign = logsumexp(evidence, b=grid.weights, return_sign=True)
    if sign <= 0:
        raise NumericalError("update() reweighted the integration grid to a non-positive total")
    weights = grid.weights * np.exp(evidence - total)
    s = grid.s + evidence
    moments = _integrated_moments(after, weights)
    names = tuple(grid.thetas[0])
    theta_mean, theta_sq = _theta_moments(names, grid.thetas, weights)
    # ponytail: the tabulated one-hyperparameter marginal now spans only the
    # retained points (the evaluated tails carry no updated conditional); refit
    # when those tails matter.
    marginals = grid.marginals(grid.u, s, theta_mean, theta_sq)

    diagnostics = dict(result.diagnostics)
    # A tighter posterior on the same grid always sheds effective points; it is
    # resolution that fails, once only a couple of points carry the weight.
    effective = float(1.0 / np.sum(weights ** 2))
    if effective < 3.0 <= float(diagnostics.get("inla_effective_weight", 0.0)):
        warnings.warn(
            f"update() left {effective:.1f} effective integration points: the "
            "hyperparameter posterior has outgrown the fitted grid; refit to re-centre it.",
            UserWarning, stacklevel=3,
        )
    diagnostics["inla_effective_weight"] = effective
    diagnostics["updated_rows"] = max(int(new.diagnostics.get("updated_rows", 0)) for new in after)
    return INLAResult(
        labels=result.labels,
        mean=moments["mean"], covariance=moments["covariance"],
        log_marginal_likelihood=result.log_marginal_likelihood + total,
        predictive_mean=moments["predictive_mean"],
        predictive_variance=moments["predictive_variance"],
        hyperparameter_marginals=marginals,
        criteria=None,
        fitted_mean=moments["fitted_mean"], link_name=result.link_name,
        block_slices=result.block_slices, diagnostics=diagnostics,
        prediction_keys=result.prediction_keys,
        hyperparameters=result.hyperparameters,
        prediction_context=(
            result.prediction_context if grid.context is None else grid.context(marginals)
        ),
        observation_variance=moments["observation_variance"],
        latent_variances=moments["latent_variance"],
        grid=IntegrationGrid(
            thetas=grid.thetas, u=grid.u, s=s, weights=weights,
            conditionals=tuple((lambda new=new: new) for new in after),
            marginals=grid.marginals, context=grid.context,
        ),
    )


# --- news decomposition -------------------------------------------------------


@dataclass(frozen=True)
class News:
    """How a release of new rows revises a fitted posterior, row by row.

    ``releases`` has one row per observed new row: its ``actual`` response, the
    response ``expected`` before the release, and the ``news`` on the
    linear-predictor scale, where the decomposition is additive: ``y - E[eta]``
    for a Gaussian row, and for any other the working response at the new mode
    minus ``E[eta]`` -- the linearisation the Laplace update makes, so the
    revisions still add up exactly. ``latent`` (indexed by
    ``(block, label)``) and ``prediction`` (indexed by the target rows) hold the
    revision each release causes, one column per released row, so every row of
    them sums to that target's total revision -- exactly. An integrated result
    adds a ``hyperparameters`` column: the part of the revision that comes from
    the release moving the hyperparameter posterior. ``updated`` is the result
    after the release, the same as ``result.update(new_data)``.
    """

    releases: object
    latent: object
    prediction: object
    updated: object


def _point_news(result, new_data, at):
    """``(updated, conditioning, news, latent and prediction impacts, target, expected)``."""
    from pylgm.inference.prediction import _design_for

    c = _condition(result, new_data, caller="news")
    if c is None:
        raise ValueError("news() new_data has no row with an observed response")
    positive = c.root > 0
    # W^1/2 (z - m), z the working response: the news in the whitened metric.
    # At the mode a = W^1/2 B^-1 W^1/2 (z - m), so V a splits over the rows.
    whitened = c.root * (c.s0 @ c.a) + np.divide(c.a, c.root, out=np.zeros_like(c.a), where=positive)
    news = np.divide(whitened, c.root, out=np.full_like(c.a, np.nan), where=positive)
    latent = ((c.v * c.root) @ cho_solve(c.s_factor, np.eye(c.a.size))) * whitened
    if at is None:
        sampler = result._sampler
        target = sampler.design if sampler.row_order is None else sampler.design[sampler.row_order]
    else:
        target = _design_for(result.prediction_context, at)
    expected = _rows_likelihood(result, result.prediction_context, c.rows).response_prediction(
        c.m, np.diag(c.s0)
    )
    return (_conditioned(result, c), c, news, latent, np.asarray(target @ latent), target,
            np.asarray(expected, dtype=float))


def news(result, new_data, at=None) -> News:
    """Decompose ``result``'s revision by ``new_data``'s rows (see ``News``)."""
    import pandas as pd

    from pylgm.inference.result import INLAResult

    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("news() new_data must be a pandas DataFrame")
    if not new_data.index.is_unique:
        raise ValueError("news() needs a unique new_data index: its labels name the releases")
    if isinstance(result, INLAResult):
        return _integrated_news(result, new_data, at)
    updated, c, row_news, latent, prediction, _, expected = _point_news(result, new_data, at)
    return _news_frames(result, updated, c.rows, c.rows[result.prediction_context.response],
                        expected, row_news, latent, prediction, at)


def _integrated_news(result, new_data, at):
    grid = result._grid
    if grid is None:
        raise ValueError("news() is available on integrated results produced by LGM.fit")
    before = [conditional() for conditional in grid.conditionals]
    parts = [_point_news(conditional, new_data, at) for conditional in before]
    updated = _reweighted(result, before, [part[0] for part in parts])
    old, new = grid.weights, updated._grid.weights
    c = parts[0][1]
    latent = sum(w * part[3] for w, part in zip(new, parts, strict=True))
    prediction = sum(w * part[4] for w, part in zip(new, parts, strict=True))
    # sum_k w'_k mu'_k - sum_k w_k mu_k = sum_k w'_k (mu'_k - mu_k) + sum_k (w'_k - w_k) mu_k
    moved = new - old
    latent_theta = sum(d * cond.mean for d, cond in zip(moved, before, strict=True))
    prediction_theta = sum(
        d * np.asarray(part[5] @ cond.mean).ravel()
        for d, cond, part in zip(moved, before, parts, strict=True)
    )
    expected = sum(w * part[6] for w, part in zip(old, parts, strict=True))
    row_news = sum(w * part[2] for w, part in zip(old, parts, strict=True))
    return _news_frames(
        result, updated, c.rows, c.rows[result.prediction_context.response], expected, row_news,
        np.column_stack([latent, latent_theta]), np.column_stack([prediction, prediction_theta]),
        at, extra="hyperparameters",
    )


def _news_frames(result, updated, rows, actual, expected, row_news, latent, prediction, at,
                 extra=None):
    import pandas as pd

    columns = list(rows.index) + ([extra] if extra else [])
    blocks = np.empty(len(result.labels), dtype=object)
    for block, where in result.block_slices.items():
        blocks[where] = block
    latent_index = pd.MultiIndex.from_arrays([blocks, list(result.labels)], names=["block", "label"])
    if at is not None:
        prediction_index = at.index
    elif result.prediction_keys is not None:
        prediction_index = pd.MultiIndex.from_frame(result.prediction_keys)
    else:
        prediction_index = pd.RangeIndex(prediction.shape[0])
    return News(
        releases=pd.DataFrame(
            {"actual": np.asarray(actual, dtype=float), "expected": expected, "news": row_news},
            index=rows.index,
        ),
        latent=pd.DataFrame(latent, index=latent_index, columns=columns),
        prediction=pd.DataFrame(prediction, index=prediction_index, columns=columns),
        updated=updated,
    )

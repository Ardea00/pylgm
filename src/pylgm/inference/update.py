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

An INLA result lifts this: each grid point's
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
        # Matheron: x' = x - Sigma A^T S^-1 (A x + e) is exactly N(0, Sigma'); the
        # rows are whitened by W^1/2, so e has unit variance.
        base = self.base.sample_deviations(n, rng)
        noise = rng.standard_normal((n, self.design.shape[0]))
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


def _rows_mode(m: np.ndarray, s0: np.ndarray, y: np.ndarray, likelihood):
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
    for _ in range(100):
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
        raise InferenceConvergenceError(100, float(np.max(np.abs(step))))
    _, root, b_factor = curvature(eta)
    return a, root, b_factor, psi - float(np.sum(np.log(np.diag(b_factor[0]))))


@dataclass(frozen=True)
class _StackedLikelihood:
    """Several outcomes' rows scored together, each part under its own likelihood."""

    parts: tuple   # (slice, likelihood)

    def _each(self, method, *arrays):
        return np.concatenate([
            np.asarray(getattr(likelihood, method)(*(a[rows] for a in arrays)), dtype=float)
            for rows, likelihood in self.parts
        ])

    def log_likelihood(self, eta, y):
        return float(sum(lk.log_likelihood(eta[rows], y[rows]) for rows, lk in self.parts))

    def gradient(self, eta, y):
        return self._each("gradient", eta, y)

    def working_weights(self, eta, y):
        return self._each("working_weights", eta, y)

    def response_prediction(self, eta, variance):
        return self._each("response_prediction", eta, variance)


@dataclass(frozen=True)
class _Rows:
    """Observed new rows, stacked across a joint model's outcomes."""

    labels: object        # new_data's index; (outcome, index) on a joint model
    y: np.ndarray
    design: np.ndarray    # A
    offset: np.ndarray
    likelihood: object    # scoring them, bound to their own trials


def _observed_rows(result, frame, caller="update"):
    """``frame``'s rows with an observed response, or ``None`` if there are none."""
    import pandas as pd

    from pylgm.inference.prediction import (
        JointPredictionContext,
        PredictionContext,
        _design_for,
        _offset_for,
        _prediction_likelihood,
    )
    from pylgm.inference.result import GaussianResult
    from pylgm.likelihoods import CompiledGaussian, CompiledWeibullSurv

    context = result.prediction_context
    joint = isinstance(context, JointPredictionContext)
    contexts = list(context.contexts.values()) if joint else [context]
    if (result._sampler is None or not all(
            isinstance(c, PredictionContext) and c.response for c in contexts)):
        raise ValueError(f"{caller}() is available on results produced by LGM.fit or Joint.fit")
    present = [c for c in contexts if c.response in frame.columns]
    if not present:
        names = ", ".join(repr(c.response) for c in contexts)
        raise ValueError(f"{caller}() new_data has no response column ({names})")
    labels, ys, designs, offsets, parts, start = [], [], [], [], [], 0
    for c in present:
        y = np.asarray(frame[c.response], dtype=float)
        observed = np.isfinite(y)
        if not observed.any():
            continue
        rows = frame[observed]
        if isinstance(result, GaussianResult):
            likelihood = CompiledGaussian(float(np.sqrt(result.observation_variance)))
        elif isinstance(c.likelihood, CompiledWeibullSurv):
            raise UnsupportedEngineError(
                f"{caller}() does not support survival likelihoods; refit instead"
            )
        else:
            likelihood = _prediction_likelihood(c, rows)
        likelihood.validate_response(y[observed])
        labels.append([(c.response, label) for label in rows.index] if joint else list(rows.index))
        ys.append(y[observed])
        designs.append(_design_for(c, rows))
        offsets.append(_offset_for(c, rows))
        parts.append((slice(start, start + len(rows)), likelihood))
        start += len(rows)
    if not parts:
        return None
    labels = sum(labels, [])
    return _Rows(
        labels=pd.MultiIndex.from_tuples(labels, names=["outcome", None]) if joint
        else pd.Index(labels, name=frame.index.name),
        y=np.concatenate(ys),
        design=np.vstack(designs),
        offset=np.concatenate(offsets),
        likelihood=parts[0][1] if len(parts) == 1 else _StackedLikelihood(tuple(parts)),
    )


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

    rows: _Rows
    m: np.ndarray         # prior mean of eta at the rows
    s0: np.ndarray        # A Sigma A^T
    v: np.ndarray         # Sigma A^T
    a: np.ndarray
    root: np.ndarray      # W^1/2
    s_factor: tuple       # cho(B)
    log_evidence: float


def _condition(result, new_data, caller="update"):
    """``_Conditioning`` for ``new_data``'s observed rows, or ``None`` if there are none."""
    rows = _observed_rows(result, new_data, caller)
    if rows is None:
        return None
    v = _covariance_apply(result, rows.design.T)
    m = rows.offset + rows.design @ result.mean
    s0 = rows.design @ v
    a, root, s_factor, log_evidence = _rows_mode(m, s0, rows.y, rows.likelihood)
    return _Conditioning(rows, m, s0, v, a, root, s_factor, log_evidence)


def _covariance_apply(result, rhs):
    if result._covariance is not None:
        return result._covariance @ rhs
    return result._sparse_posterior.covariance_apply(rhs)


@dataclass(frozen=True)
class _Revision:
    """Already-fitted Gaussian rows whose values were revised (``previous -> revised``)."""

    rows: _Rows           # the revised values
    previous: np.ndarray
    latent: np.ndarray    # p x r: each row's shift of the latent mean
    log_evidence: float   # change in the log marginal likelihood


def _revision(result, revisions, caller):
    """``_Revision`` for ``revisions = (previous_rows, revised_rows)``, or ``None``.

    Revising Gaussian rows moves the mean linearly and leaves the covariance
    alone: ``dmu = Sigma A^T D^-1 dy``, ``D`` their noise variances. The log
    marginal likelihood swaps ``log p(y_prev | rest)`` for ``log p(y_new | rest)``;
    with ``P = A Sigma A^T`` the leave-these-out predictive is
    ``N(y_prev - D (D - P)^-1 e, D (D - P)^-1 D)``, ``e = y_prev - A mu``.
    Exact for a Gaussian fit; in a Laplace fit (a joint model's Gaussian
    outcomes) the other rows' curvature stays where it was.
    """
    from pylgm.likelihoods import CompiledGaussian

    if revisions is None:
        return None
    try:
        before, after = revisions
    except (TypeError, ValueError) as error:
        raise TypeError(f"{caller}() revisions must be a (previous, revised) pair of frames") from error
    previous = _observed_rows(result, before, caller)
    revised = _observed_rows(result, after, caller)
    if previous is None or revised is None or not previous.labels.equals(revised.labels):
        raise ValueError(
            f"{caller}() revisions must give the same observed rows before and after"
        )
    parts = getattr(revised.likelihood, "parts", ((slice(None), revised.likelihood),))
    noise = np.empty(revised.y.size)
    for rows, likelihood in parts:
        if not isinstance(likelihood, CompiledGaussian):
            raise UnsupportedEngineError(f"{caller}() revises Gaussian rows only")
        noise[rows] = likelihood.variance
    v = _covariance_apply(result, revised.design.T)
    change = revised.y - previous.y
    residual = previous.y - previous.offset - previous.design @ result.mean
    try:
        loo = cho_factor(np.diag(noise) - revised.design @ v, lower=True)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            f"{caller}() cannot leave the revised rows out: the rest does not identify them"
        ) from error
    old = noise * cho_solve(loo, residual)      # y_prev minus its leave-out mean
    new = old + change

    def quadratic(r):                           # r^T C^-1 r, C^-1 = D^-1 (D - P) D^-1
        scaled = r / noise
        return scaled @ (np.diag(noise) - revised.design @ v) @ scaled

    return _Revision(
        rows=revised, previous=previous.y, latent=v * (change / noise),
        log_evidence=-0.5 * (quadratic(new) - quadratic(old)),
    )


def update(result, new_data, revisions=None):
    """``result`` conditioned on ``new_data``'s observed rows (see module doc)."""
    from pylgm.inference.result import INLAResult

    if isinstance(result, INLAResult):
        return _update_integrated(result, new_data, revisions)
    return _step(result, new_data, revisions, "update")[0]


def _step(result, new_data, revisions, caller):
    """``(updated, revision, conditioning)`` at fixed hyperparameters."""
    revision = _revision(result, revisions, caller)
    if revision is not None:
        result = _assembled(result, revision.latent.sum(axis=1), revision.log_evidence, 0)
    c = None if new_data is None else _condition(result, new_data, caller)
    if c is not None:
        # Whitening the rows by W^1/2 turns S into B with unit noise.
        result = _assembled(
            result, c.v @ c.a, c.log_evidence, c.rows.y.size,
            low_rank=(c.v * c.root, c.rows.design * c.root[:, None], c.s_factor),
        )
    return result, revision, c


def _assembled(result, shift, log_evidence, added_rows, low_rank=None):
    """``result`` with its mean moved by ``shift`` and, given ``low_rank = (V, A, cho(S))``,
    ``V S^-1 V^T`` taken off its covariance (``UpdatedPosterior``)."""
    from pylgm.inference.result import GaussianResult, LaplaceResult, quadratic_form_diagonal
    from pylgm.inference.sampling import GridSampler

    context = result.prediction_context
    sampler = result._sampler
    dense = result._covariance is not None
    covariance = result._covariance
    predictive_variance = result._predictive_variance
    posterior = result._sparse_posterior
    factor = sampler.factor
    # The fitted grid: canonical rows of the sampler's design, in caller order.
    grid = sampler.design if sampler.row_order is None else sampler.design[sampler.row_order]
    if low_rank is not None:
        v, design, s_factor = low_rank
        posterior = UpdatedPosterior(
            base=_DenseFactor(factor) if factor is not None else sampler.posterior,
            v=v, design=design, s_factor=s_factor,
        )
        factor = None
        # diag(G S^-1 G^T), G = P V: through the p x p Delta = V S^-1 V^T when that
        # is cheaper (O(p^2 k + nnz(P) p)) than the n x k route (O(n k^2)); the
        # dense path needs Delta for the covariance anyway.
        (p, k), n = v.shape, grid.shape[0]
        via_delta = dense or p * p * k + grid.nnz * p < n * k * k
        delta = v @ cho_solve(s_factor, v.T) if via_delta else None
        if dense:
            covariance = result._covariance - delta
            covariance = 0.5 * (covariance + covariance.T)
            if _chain_rank(posterior) >= p:
                # Past p accumulated rows the low-rank chain costs more than a
                # dense factor of Sigma' itself: collapse it (O(p^3) per ~p rows).
                posterior = _DenseFactor(_psd_factor(covariance))
        if predictive_variance is not None:
            shrink = (
                quadratic_form_diagonal(grid, delta) if delta is not None
                else _quadratic_diagonal(np.asarray(grid @ v), s_factor)
            )
            predictive_variance = np.clip(predictive_variance - shrink, 0.0, None)
    mean = result.mean + shift
    predictive_mean = result.predictive_mean + np.asarray(grid @ shift).ravel()
    diagnostics = dict(result.diagnostics)
    diagnostics["updated_rows"] = int(diagnostics.get("updated_rows", 0)) + added_rows
    common = dict(
        labels=result.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=result.log_marginal_likelihood + log_evidence,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        block_slices=result.block_slices,
        diagnostics=diagnostics,
        prediction_keys=result.prediction_keys,
        hyperparameters=result.hyperparameters,
        prediction_context=context,
        sparse_posterior=None if dense else posterior,
        sampler=GridSampler(
            mean, sampler.design, sampler.offset, factor=factor,
            posterior=None if factor is not None else (posterior if low_rank else sampler.posterior),
            row_order=sampler.row_order,
        ),
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


def _update_integrated(result, new_data, revisions):
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
            UserWarning, stacklevel=4,   # user -> result.update -> update -> here
        )
    before = [conditional() for conditional in grid.conditionals]
    return _reweighted(
        result, before, [_step(cond, new_data, revisions, "update")[0] for cond in before]
    )


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
            UserWarning, stacklevel=5,   # user -> result.update/news -> ... -> here
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
    """How a release of new rows, and revisions of old ones, move a fitted posterior.

    ``releases`` has one row per observed new row: its ``actual`` response, the
    response ``expected`` before the release, and the ``news`` on the
    linear-predictor scale, where the decomposition is additive: ``y - E[eta]``
    for a Gaussian row, and for any other the working response at the new mode
    minus ``E[eta]`` -- the linearisation the Laplace update makes. ``revisions``
    lists the revised rows (``previous``, ``revised``), or is ``None``.

    ``latent`` (indexed by ``(block, label)``) and ``prediction`` (indexed by the
    targets) hold each released row's revision of every latent effect and
    target, then one ``revision <label>`` column per revised row; every row of
    them sums to that target's total revision exactly. ``by_block`` splits each
    target's revision by the latent block it passes through,
    ``G[:, block] @ dmu[block]`` (indexed by ``(target, block)``); its blocks add
    up to ``prediction``. Summing a block's own rows of ``latent`` is not that:
    an RW1 or Besag block is constrained to sum to zero.

    ``uncertainty`` is each release's reduction of the target variances, taking
    the rows in ``new_data``'s order (the sequential attribution: row j's share
    given the rows before it); it adds up to the total reduction. Revisions of
    Gaussian rows leave the variances alone. An integrated result adds a
    ``hyperparameters`` column to all three: what the release changes by moving
    the hyperparameter posterior. ``updated`` is the result after it all, as
    ``result.update(new_data, revisions)`` returns it.
    """

    releases: object
    revisions: object
    latent: object
    prediction: object
    by_block: object
    uncertainty: object
    updated: object


def _targets(result, at, weights):
    """``(G, index)``: the target design and its labels (``news`` docstring)."""
    from collections.abc import Mapping

    import pandas as pd

    from pylgm.inference.prediction import JointPredictionContext, _design_for

    context = result.prediction_context
    if at is None:
        sampler = result._sampler
        design = sampler.design if sampler.row_order is None else sampler.design[sampler.row_order]
        index = (pd.MultiIndex.from_frame(result.prediction_keys)
                 if result.prediction_keys is not None else pd.RangeIndex(design.shape[0]))
    elif isinstance(context, JointPredictionContext):
        if not isinstance(at, Mapping) or not set(at) <= set(context.contexts):
            raise ValueError(
                f"news() at= on a joint model maps outcomes {context.outcomes} to rows"
            )
        design = np.vstack([_design_for(context.contexts[o], rows) for o, rows in at.items()])
        index = pd.MultiIndex.from_tuples(
            [(o, label) for o, rows in at.items() for label in rows.index], names=["outcome", None]
        )
    else:
        design, index = _design_for(context, at), at.index
    if weights is None:
        return design, index
    missing = [label for label in weights.columns if label not in index]
    if missing:
        raise ValueError(f"news() weights name target rows that are not in at=: {missing[:3]}")
    matrix = weights.reindex(columns=index, fill_value=0.0).to_numpy(dtype=float)
    return np.asarray(matrix @ design), weights.index


@dataclass(frozen=True)
class _PointNews:
    """One fixed-theta decomposition; arrays are targets or latents x columns."""

    updated: object
    releases: _Rows | None
    revision: _Revision | None
    news: np.ndarray
    expected: np.ndarray
    target: object          # the target design G
    latent: np.ndarray      # p x (releases + revisions)
    by_block: dict          # block -> G[:, block] @ latent[block]
    reduction: np.ndarray   # targets x releases: sequential variance reductions


def _point_news(result, new_data, revisions, targets) -> _PointNews:
    from scipy.linalg import solve_triangular

    target, _ = targets(result)
    updated, revision, c = _step(result, new_data, revisions, "news")
    if c is None and revision is None:
        raise ValueError("news() needs a released row with an observed response, or revisions")
    columns, news, expected = [], np.empty(0), np.empty(0)
    reduction = np.zeros((target.shape[0], 0))
    if c is not None:
        positive = c.root > 0
        # W^1/2 (z - m), z the working response: the news in the whitened metric.
        # At the mode a = W^1/2 B^-1 W^1/2 (z - m), so V a splits over the rows.
        whitened = c.root * (c.s0 @ c.a) + np.divide(
            c.a, c.root, out=np.zeros_like(c.a), where=positive
        )
        gain = (c.v * c.root) @ cho_solve(c.s_factor, np.eye(c.a.size))
        columns.append(gain * whitened)
        news = np.divide(whitened, c.root, out=np.full_like(c.a, np.nan), where=positive)
        expected = np.asarray(
            c.rows.likelihood.response_prediction(c.m, np.diag(c.s0)), dtype=float
        )
        # Sigma' = Sigma - U U^T, U = V W^1/2 L^-T with B = L L^T: column j of U is
        # row j's innovation given the rows before it, so the variance a target
        # loses splits over the rows in order.
        projected = np.asarray(target @ (c.v * c.root))
        reduction = solve_triangular(c.s_factor[0], projected.T, lower=True).T ** 2
    if revision is not None:
        columns.append(revision.latent)
    latent = np.column_stack(columns)
    return _PointNews(
        updated=updated, releases=None if c is None else c.rows, revision=revision,
        news=news, expected=expected, target=target, latent=latent,
        by_block={
            block: np.asarray(target[:, where] @ latent[where])
            for block, where in result.block_slices.items()
        },
        reduction=reduction,
    )


def news(result, new_data, at=None, *, weights=None, revisions=None) -> News:
    """Decompose the revision of ``result`` by ``new_data``'s rows (see ``News``)."""
    import pandas as pd

    from pylgm.inference.result import INLAResult

    if new_data is not None:
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError("news() new_data must be a pandas DataFrame")
        if not new_data.index.is_unique:
            raise ValueError("news() needs a unique new_data index: its labels name the releases")
    if weights is not None and not isinstance(weights, pd.DataFrame):
        raise TypeError("news() weights must be a DataFrame: aggregates x target rows")

    def targets(point):
        return _targets(point, at, weights)

    if isinstance(result, INLAResult):
        return _integrated_news(result, new_data, revisions, targets)
    point = _point_news(result, new_data, revisions, targets)
    return _news_frames(
        result, point, targets(result)[1], point.news, point.expected, point.latent,
        point.by_block, point.reduction,
    )


def _integrated_news(result, new_data, revisions, targets):
    from pylgm.optimization.inla import _conditional_predictive_variances

    grid = result._grid
    if grid is None:
        raise ValueError("news() is available on integrated results produced by LGM.fit")
    before = [conditional() for conditional in grid.conditionals]
    points = [_point_news(cond, new_data, revisions, targets) for cond in before]
    updated = _reweighted(result, before, [point.updated for point in points])
    old, new = grid.weights, updated._grid.weights
    # sum_k w'_k mu'_k - sum_k w_k mu_k = sum_k w'_k (mu'_k - mu_k) + sum_k (w'_k - w_k) mu_k:
    # the rows' part, then the hyperparameters' part as a last column.
    moved = new - old

    def mixed(values, weights=new):
        return sum(w * value for w, value in zip(weights, values, strict=True))

    def theta_part(values):
        return sum(d * value for d, value in zip(moved, values, strict=True))

    latent = np.column_stack([
        mixed([point.latent for point in points]), theta_part([cond.mean for cond in before]),
    ])
    by_block = {
        block: np.column_stack([
            mixed([point.by_block[block] for point in points]),
            theta_part([
                np.asarray(point.target[:, where] @ cond.mean[where]).ravel()
                for point, cond in zip(points, before, strict=True)
            ]),
        ])
        for block, where in result.block_slices.items()
    }
    # Var of the mixture before minus after; the rows' part is sum_k w'_k (row reductions).
    dense_targets = [np.asarray(point.target.todense() if hasattr(point.target, "todense")
                                else point.target) for point in points]

    def mixture_variance(weights, conditionals):
        means = [t @ cond.mean for t, cond in zip(dense_targets, conditionals, strict=True)]
        variances = [_conditional_predictive_variances(cond, t)
                     for t, cond in zip(dense_targets, conditionals, strict=True)]
        first = mixed(means, weights)
        return mixed([v + m * m for v, m in zip(variances, means, strict=True)], weights) - first ** 2

    rows_part = mixed([point.reduction for point in points])
    total = (mixture_variance(old, before)
             - mixture_variance(new, [point.updated for point in points]))
    reduction = np.column_stack([rows_part, total - rows_part.sum(axis=1)])
    return _news_frames(
        result, points[0], targets(result)[1],
        mixed([point.news for point in points], old), mixed([point.expected for point in points], old),
        latent, by_block, reduction, updated=updated, extra="hyperparameters",
    )


def _news_frames(result, point, targets, row_news, expected, latent, by_block, reduction,
                 updated=None, extra=None):
    import pandas as pd

    releases = [] if point.releases is None else list(point.releases.labels)
    revised = [] if point.revision is None else [
        f"revision {label}" for label in point.revision.rows.labels
    ]
    columns = releases + revised + ([extra] if extra else [])
    uncertainty_columns = releases + ([extra] if extra else [])
    blocks = np.empty(len(result.labels), dtype=object)
    for block, where in result.block_slices.items():
        blocks[where] = block
    latent_index = pd.MultiIndex.from_arrays([blocks, list(result.labels)], names=["block", "label"])
    names = list(by_block)
    return News(
        releases=None if point.releases is None else pd.DataFrame(
            {"actual": point.releases.y, "expected": expected, "news": row_news},
            index=point.releases.labels,
        ),
        revisions=None if point.revision is None else pd.DataFrame(
            {"previous": point.revision.previous, "revised": point.revision.rows.y},
            index=point.revision.rows.labels,
        ),
        latent=pd.DataFrame(latent, index=latent_index, columns=columns),
        prediction=pd.DataFrame(sum(by_block.values()), index=targets, columns=columns),
        # (target..., block): each target's blocks together, targets in the given order.
        by_block=pd.DataFrame(
            np.stack([by_block[name] for name in names], axis=1).reshape(-1, len(columns)),
            index=pd.MultiIndex.from_tuples(
                [(*(t if isinstance(t, tuple) else (t,)), name) for t in targets for name in names],
                names=[*targets.names, "block"],
            ),
            columns=columns,
        ),
        uncertainty=pd.DataFrame(reduction, index=targets, columns=uncertainty_columns),
        updated=point.updated if updated is None else updated,
    )

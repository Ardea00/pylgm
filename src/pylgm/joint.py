"""Joint latent Gaussian models: several responses stacked into one CompiledLGM.

A joint model is an ordinary :class:`~pylgm.ir.model.CompiledLGM` with more
rows. Responses stack as ``y = (y^(1), ..., y^(K))``; sub-model ``k`` occupies a
contiguous row slice. Private latent blocks are zero-padded outside their slice,
shared blocks carry scaled rows in every slice they enter, and the likelihood
becomes a row-dispatching :class:`~pylgm.likelihoods.CompiledMixture`. Both IR
invariants -- ``design == hstack(blocks)`` and ``precision == block_diag(blocks)``
-- are preserved.
"""

import collections.abc
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.sparse import csr_matrix, vstack

from pylgm.effects import Copy, Weighted
from pylgm.exceptions import ModelValidationError, UnsupportedEngineError
from pylgm.ir.model import LatentBlock
from pylgm.observations import (
    LinearConstraint,
    LinearObservation,
    _ProjectedMixtureFamily,
    _RelinearizedFamily,
    _aligned,
)
from pylgm.parameters import Hyperparameter


def _pad_block_rows(block: LatentBlock, before: int, after: int) -> LatentBlock:
    """Zero-pad a block's design rows into the stacked row space.

    Precision, labels and constraints are row-independent and pass through
    untouched, so a Besag sum-to-zero still constrains exactly what it did.
    """
    if before == 0 and after == 0:
        return block
    width = block.design.shape[1]
    pieces = []
    if before:
        pieces.append(csr_matrix((before, width)))
    pieces.append(block.design)
    if after:
        pieces.append(csr_matrix((after, width)))
    return LatentBlock(
        block.name,
        block.labels,
        vstack(pieces, format="csr"),
        block.precision,
        block.constraints,
    )


# A scaled shared field enters slice k as `scale_k * u`. The sentinel
# ("<name>", "inverse") means "the reciprocal of the hyperparameter <name>",
# which is how the Knorr-Held & Best (delta, delta^-1) pairing is carried
# through compilation without inventing an expression language (see
# Shared.scales_for).


@dataclass(frozen=True)
class Shared:
    """One latent field entering several sub-models with a per-sub-model scaling.

    ``scale`` is a float (broadcast to every sub-model), a ``Hyperparameter``
    (shorthand for the Knorr-Held & Best ``(delta, delta^-1)`` pairing, and
    therefore valid only for exactly two sub-models), or an explicit
    per-sub-model tuple of floats and/or ``Hyperparameter``s.
    """

    effect: object
    scale: object = 1.0
    allow_ragged: bool = False
    """Accept a shared index whose level set differs between sub-models.

    The latent always spans the union of levels; this only silences the report.
    Off by default because an unintended mismatch weakens the shared field
    without any visible symptom.
    """

    def __post_init__(self) -> None:
        if not hasattr(self.effect, "name"):
            raise TypeError("Shared effect must be a latent effect spec")
        if isinstance(self.effect, Copy):
            raise TypeError(
                f"Shared effect {self.effect!r} is a Copy, which has no block of its "
                "own -- it folds into an existing target block instead, so Joint has "
                "no target for a shared copy to fold into. Copy cannot be shared."
            )
        if not hasattr(self.effect, "index"):
            if isinstance(self.effect, Weighted):
                raise TypeError(
                    f"Shared effect {self.effect!r} is a Weighted wrapper, which has "
                    "no `index` of its own -- a shared effect must be indexed (IID, "
                    "RW1/RW2, AR1, Seasonal, Besag, ProperCAR, SAR, BYM2), so it can "
                    "be summed across sub-models over a common level set. Weighted "
                    "cannot be shared."
                )
            raise TypeError(
                f"Shared effect {self.effect!r} has no `index` -- a shared effect "
                "must be indexed (IID, RW1/RW2, AR1, Seasonal, Besag, ProperCAR, "
                "SAR, BYM2), so it can be summed across sub-models over a common "
                "level set. Fixed/MIDAS/MIDASParametric/SpaceTime/"
                "DynamicSpatialPanel/Replicated/Grouped effects cannot be shared."
            )
        if not isinstance(self.allow_ragged, bool):
            raise TypeError("Shared allow_ragged must be a bool")
        scale = self.scale
        if isinstance(scale, (tuple, list)):
            entries = tuple(scale)
            if not entries:
                raise ValueError("Shared scale tuple must be non-empty")
            for entry in entries:
                if not isinstance(entry, (int, float, Hyperparameter)):
                    raise TypeError(
                        "Shared scale entries must be floats or Hyperparameters"
                    )
            object.__setattr__(self, "scale", entries)
        elif not isinstance(scale, (int, float, Hyperparameter)):
            raise TypeError("Shared scale must be a float, Hyperparameter, or tuple")

    @property
    def name(self) -> str:
        return self.effect.name

    def scales_for(self, count: int) -> tuple:
        """Expand ``scale`` to one entry per sub-model."""
        scale = self.scale
        if isinstance(scale, tuple):
            if len(scale) != count:
                raise ValueError(
                    f"Shared {self.name!r} scale tuple has length {len(scale)}, "
                    f"but the joint has {count} sub-models"
                )
            return scale
        if isinstance(scale, Hyperparameter):
            if count != 2:
                raise ValueError(
                    f"Shared {self.name!r} has a scalar Hyperparameter scale, which is "
                    "the (delta, delta^-1) shorthand and requires exactly two "
                    f"sub-models; this joint has {count}. Pass an explicit "
                    "per-sub-model tuple instead."
                )
            return (scale, (scale.name, "inverse"))
        return tuple(float(scale) for _ in range(count))


def _linear_inputs(self: "Joint", argument, kind: type, name: str) -> dict:
    """Validate and normalize an ``observations``/``constraints`` mapping.

    Returns ``{outcome: tuple(items)}`` with only non-empty entries.
    """
    if argument is None:
        return {}
    if not isinstance(argument, collections.abc.Mapping):
        raise TypeError(
            f"{name} must be a mapping from outcome name to a list of {kind.__name__}"
        )
    result = {}
    for key, value in argument.items():
        if key not in self.outcomes:
            raise ModelValidationError(
                f"{name} names unknown outcome {key!r}; the joint's outcomes are {self.outcomes}"
            )
        try:
            items = tuple(value)
        except TypeError as error:
            raise TypeError(
                f"{name} must be a mapping from outcome name to a list of {kind.__name__}"
            ) from error
        if any(not isinstance(item, kind) for item in items):
            raise TypeError(f"{name}[{key!r}] must contain only {kind.__name__} instances")
        if items:
            result[key] = items
    return result


def _hold_out(self: "Joint", argument, rows: int) -> dict:
    """Validate a ``hold_out`` mapping: outcome -> boolean mask over the frame's rows."""
    if argument is None:
        return {}
    if not isinstance(argument, collections.abc.Mapping):
        raise TypeError("hold_out must be a mapping from outcome name to a boolean row mask")
    masks = {}
    for outcome, mask in argument.items():
        if outcome not in self.outcomes:
            raise ModelValidationError(
                f"hold_out names unknown outcome {outcome!r}; the joint's outcomes are "
                f"{self.outcomes}"
            )
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (rows,):
            raise ValueError(f"hold_out[{outcome!r}] must be a boolean mask of length {rows}")
        masks[outcome] = mask
    return masks


@dataclass(frozen=True)
class CensoredHurdle:
    """Edges absent from a register that reports amounts at or above a threshold.

    ``link`` names a Bernoulli outcome (1 for an edge in the register) and
    ``amount`` a Gaussian outcome on its log amount. A row where the boolean
    column ``censored`` is true is a candidate edge *absent* from the register:
    it either does not exist or exists below the threshold, so both responses
    must be NaN there and the row contributes

        log[(1 - p) + p Phi((threshold - eta_amount) / sigma)],  p = expit(eta_link),

    with ``sigma`` the amount outcome's (fixed or estimated) Gaussian sigma.
    ``threshold`` is the log reporting threshold: a float, or a column name.
    The two outcomes must share their predictor grid row for row, which they do
    when both come from the same frame.
    """

    link: str
    amount: str
    censored: str
    threshold: float | str

    def __post_init__(self) -> None:
        for name in ("link", "amount", "censored"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"CensoredHurdle {name} must be a non-empty column name")
        if self.link == self.amount:
            raise ValueError("CensoredHurdle link and amount must be different outcomes")
        if not isinstance(self.threshold, str) and not np.isfinite(self.threshold):
            raise ValueError("CensoredHurdle threshold must be finite or a column name")

    def rows(self, frame) -> tuple[np.ndarray, np.ndarray]:
        """The censored mask and the per-row log threshold, validated on ``frame``."""
        from pylgm.exceptions import DataContractError

        if self.censored not in frame.columns:
            raise DataContractError(f"censored column {self.censored!r} is missing")
        censored = frame[self.censored]
        if censored.dtype != bool:
            raise DataContractError(f"censored column {self.censored!r} must be boolean")
        censored = censored.to_numpy(copy=True)
        for outcome in (self.link, self.amount):
            if outcome in frame.columns and frame.loc[censored, outcome].notna().any():
                raise DataContractError(
                    f"outcome {outcome!r} must be NaN on censored rows: absence from "
                    "the register is the only datum there"
                )
        if isinstance(self.threshold, str):
            if self.threshold not in frame.columns:
                raise DataContractError(f"threshold column {self.threshold!r} is missing")
            threshold = frame[self.threshold].to_numpy(dtype=float)
        else:
            threshold = np.full(len(frame), float(self.threshold))
        if not np.isfinite(threshold[censored]).all():
            raise DataContractError("CensoredHurdle threshold must be finite on censored rows")
        return censored, threshold


@dataclass(frozen=True)
class _BelowThresholdMass:
    """The map behind ``scale="below_threshold"``: frame rows of censored edges.

    For a censored edge with link predictor ``a`` and log-amount ``b``,

        g = E[W 1{link} 1{W < c} | absent] = p M(b) / (1 - p S(b)),
        M(b) = exp(b + sigma^2 / 2) Phi((t - b - sigma^2) / sigma),

    so with ``u = (t - b - sigma^2) / sigma`` and the hurdle's ``q`` and ``m``,

        d log g / da = (1 - p)(1 + q),
        d log g / db = 1 - phi(u) / (sigma Phi(u)) + m.

    ``link`` and ``amount`` give each frame row's stacked grid row (-1 where the
    outcome does not hold it); ``sigma`` is read off the materialized hurdle.
    """

    link: np.ndarray
    amount: np.ndarray
    threshold: np.ndarray

    @classmethod
    def bind(cls, rows: int, threshold, link, amount) -> "_BelowThresholdMass":
        def stacked(original, start):
            out = np.full(rows, -1)
            out[original] = start + np.arange(original.size)
            return out

        return cls(stacked(*link), stacked(*amount), np.asarray(threshold, dtype=float))

    def _log_mass(self, eta, model):
        """``(held, log g, dlog g/da, dlog g/db, d2log g/da2, d2log g/dadb, d2log g/db2)``."""
        from scipy.special import log_expit, log_ndtr

        from pylgm.likelihoods import hurdle_curvature, hurdle_terms

        sigma = next(lk.sigma for _, lk in model.likelihood.parts if hasattr(lk, "cross_weights")
                     and hasattr(lk, "sigma"))
        held = (self.link >= 0) & (self.amount >= 0)
        a, b, t = eta[self.link[held]], eta[self.amount[held]], self.threshold[held]
        w, one_minus_p, log_absent, q, m = hurdle_terms(a, b, t, sigma)
        l_aa, l_ab, l_bb = hurdle_curvature(one_minus_p, w, sigma, q, m)
        u = (t - b - sigma * sigma) / sigma
        log_phi_u = log_ndtr(u)
        log_mass = log_expit(a) + b + 0.5 * sigma * sigma + log_phi_u - log_absent
        mills = np.exp(-0.5 * u * u - 0.5 * np.log(2 * np.pi) - log_phi_u)
        p = 1.0 - one_minus_p
        return (
            held, log_mass, one_minus_p * (1.0 + q), 1.0 - mills / sigma + m,
            -p * one_minus_p - l_aa, -l_ab, -mills * (u + mills) / sigma**2 - l_bb,
        )

    def value(self, eta, model) -> np.ndarray:
        held, log_mass, *_ = self._log_mass(eta, model)
        out = np.zeros(self.link.size)
        out[held] = np.exp(log_mass)
        return out

    def jacobian(self, eta, model) -> csr_matrix:
        held, log_mass, d_a, d_b, *_ = self._log_mass(eta, model)
        rows = np.flatnonzero(held)
        mass = np.exp(log_mass)
        return csr_matrix(
            (np.concatenate([mass * d_a, mass * d_b]),
             (np.concatenate([rows, rows]),
              np.concatenate([self.link[held], self.amount[held]]))),
            shape=(self.link.size, len(eta)),
        )

    def hessian(self, eta, model, weights):
        """``sum_e weights_e grad^2 g_e`` over grid rows: ``(diagonal, (i, j, c))``.

        ``grad^2 g = g (grad L grad L^T + grad^2 L)`` with ``L = log g``.
        """
        held, log_mass, d_a, d_b, d_aa, d_ab, d_bb = self._log_mass(eta, model)
        scaled = weights[held] * np.exp(log_mass)
        link, amount = self.link[held], self.amount[held]
        diagonal = np.zeros(len(eta))
        np.add.at(diagonal, link, scaled * (d_a * d_a + d_aa))
        np.add.at(diagonal, amount, scaled * (d_b * d_b + d_bb))
        return diagonal, (link, amount, scaled * (d_a * d_b + d_ab))


def _validate_censoring(censoring, submodels) -> None:
    from pylgm.likelihoods import Bernoulli, Gaussian

    if not isinstance(censoring, CensoredHurdle):
        raise TypeError("Joint censoring must be a CensoredHurdle")
    by_response = {model.response: model for model in submodels}
    for role, kind in (("link", Bernoulli), ("amount", Gaussian)):
        outcome = getattr(censoring, role)
        if outcome not in by_response:
            raise ModelValidationError(f"CensoredHurdle {role} {outcome!r} is not an outcome")
        if not isinstance(by_response[outcome].likelihood, kind):
            raise ModelValidationError(
                f"CensoredHurdle {role} outcome {outcome!r} needs a {kind.__name__} likelihood"
            )


@dataclass(frozen=True)
class Joint:
    """Several `LGM` sub-models fitted as one stacked latent Gaussian model."""

    submodels: tuple = ()
    shared: tuple = ()
    censoring: "CensoredHurdle | None" = None

    def __init__(self, submodels, shared=(), censoring=None) -> None:
        submodels = tuple(submodels)
        if len(submodels) < 2:
            raise ValueError("Joint requires at least two sub-models")
        responses = [model.response for model in submodels]
        if len(responses) != len(set(responses)):
            raise ValueError("Joint sub-model response names must be unique")
        shared = tuple(shared)
        for entry in shared:
            if not isinstance(entry, Shared):
                raise TypeError("Joint shared entries must be Shared instances")
            entry.scales_for(len(submodels))
        shared_names = [entry.name for entry in shared]
        if len(shared_names) != len(set(shared_names)):
            raise ValueError("Joint shared effect names must be unique")
        if censoring is not None:
            _validate_censoring(censoring, submodels)
        object.__setattr__(self, "submodels", submodels)
        object.__setattr__(self, "shared", shared)
        object.__setattr__(self, "censoring", censoring)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(model.response for model in self.submodels)

    @classmethod
    def _unchecked(cls, submodels, shared=()):
        """Bypass the two-sub-model minimum. Test-only: used by the reduction test."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "submodels", tuple(submodels))
        object.__setattr__(obj, "shared", tuple(shared))
        object.__setattr__(obj, "censoring", None)
        return obj

    def fit(self, frame, engine: str = "laplace", *, hyperparameters: str = "optimize",
            latent_strategy: str = "gaussian", mean_correction: bool = False,
            observations=None, constraints=None,
            num_workers: int = 1, blas_threads: int | None = None, warm_start=None,
            hold_out=None):
        """Compile and fit this joint model. Only ``engine='laplace'`` is supported.

        ``observations`` and ``constraints`` are mappings from sub-model
        response name to a list of :class:`~pylgm.observations.LinearObservation`
        / :class:`~pylgm.observations.LinearConstraint` respectively. ``None``
        or ``{}`` behaves exactly like today (bit-identical).

        Operator columns follow the caller's ``frame`` rows in caller order
        (width == ``len(frame)``), exactly like ``LGM.fit``. Operators act on
        that outcome's linear predictor ``eta`` (link scale, e.g. log for
        Poisson), not on the response mean.

        Row rule: sub-model ``k`` keeps the rows where its response is
        non-null, plus, if ``k`` is named in ``observations``/``constraints``,
        every row that a nonzero operator entry of ``k`` references -- these
        become unobserved-but-predicted rows of ``k`` (e.g. the quarters to
        nowcast), and every row ``hold_out[k]`` (a boolean mask over ``frame``)
        selects -- future periods, say, so the latent field has their levels
        for ``update``. Other NaN rows are dropped. A response column absent
        from ``frame`` is treated as all-NaN.

        See docs/joint-models.md for the full semantics, including
        ``LinearObservation`` pseudo-rows, ``LinearConstraint`` conditioning,
        and estimating a ``LinearObservation`` sigma by empirical Bayes.

        ``num_workers`` runs a hyperparameter search's independent conditional
        fits concurrently on a thread pool; results are bit-identical to
        ``num_workers=1, blas_threads=1``. See ``LGM.fit`` / docs/empirical-bayes.md.
        ``warm_start`` starts from an earlier result, as in ``LGM.fit``.
        """
        import pandas as pd

        from pylgm.optimization.inla import require_single_mean_shift

        require_single_mean_shift(latent_strategy, mean_correction)
        from pylgm.compiler import compile_joint, compile_joint_family, build_joint_prediction_contexts
        from pylgm.config.schema import DataConfig
        from pylgm.data.panel import CanonicalPanel
        from pylgm.exceptions import DataContractError
        from pylgm.inference.laplace import fit_laplace
        from pylgm.model import _finished, _fit_family
        from pylgm.observations import (
            observation_hyperparameters,
            project_gaussian_family,
            project_mixture_model,
        )
        from pylgm.parallel import validate_blas_threads, validate_workers

        num_workers = validate_workers(num_workers, "num_workers")
        blas_threads = validate_blas_threads(blas_threads)

        if engine != "laplace":
            raise UnsupportedEngineError(
                "Joint models require engine='laplace'; the exact_gaussian engine "
                "needs a single CompiledGaussian likelihood, and a mixture is not one. "
                "Laplace is exact for an all-Gaussian stack anyway."
            )
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("frame must be a Pandas DataFrame")

        hold_out = _hold_out(self, hold_out, len(frame))
        observations = _linear_inputs(self, observations, LinearObservation, "observations")
        constraints = _linear_inputs(self, constraints, LinearConstraint, "constraints")
        for mapping in (observations, constraints):
            for outcome, items in mapping.items():
                for item in items:
                    _aligned(
                        item.operator, len(frame),
                        f"{type(item).__name__} operator for outcome {outcome!r}",
                    )
        hurdle = self.censoring
        censored_mask = threshold = None
        if hurdle is not None:
            censored_mask, threshold = hurdle.rows(frame)
        panels = {}
        positions = {}
        for model in self.submodels:
            # Each sub-model sees only the rows carrying its own response.
            #
            # This deliberately differs from LGM.fit, which keeps NaN-response
            # rows as held-out-but-fitted. In the long-stacked layout that joint
            # models are normally given -- one row per (outcome, unit) pair, so
            # every row is NaN for every *other* outcome -- a NaN means "this row
            # belongs to another outcome", not "hold this observation out".
            # Keeping those rows would double the stacked design and produce
            # fitted values for observations that do not exist (docs/joint-models.md,
            # "Held-out rows").
            #
            # Two exceptions keep a NaN row as an unobserved-but-predicted row of
            # its outcome: a row that outcome's observations/constraints operator
            # references (e.g. the months a quarterly observation aggregates),
            # and a row its `hold_out` mask selects (e.g. future periods, so the
            # latent field has their levels for update()).
            response = model.response
            named = response in observations or response in constraints
            keep = (frame[response].notna().to_numpy(copy=True) if response in frame.columns
                    else np.zeros(len(frame), dtype=bool))
            for item in (*observations.get(response, ()), *constraints.get(response, ())):
                operator = item.operator.copy()
                operator.eliminate_zeros()
                keep[np.unique(operator.indices)] = True
            keep |= hold_out.get(response, False)
            censored_role = hurdle is not None and response in (hurdle.link, hurdle.amount)
            if censored_role:
                keep |= censored_mask
            positions[response] = np.flatnonzero(keep)
            sub = frame.iloc[positions[response]].reset_index(drop=True)
            if response not in sub.columns:
                sub = sub.assign(**{response: np.nan})
            time = model.time or "__pylgm_row__"
            if model.time is None:
                sub = sub.assign(**{time: range(len(sub))})
            panels[model.response] = CanonicalPanel.from_frame(
                sub, DataConfig(time=time, response=model.response, panel=model.panel),
                require_observed=not (named or censored_role),
            )

        for model in self.submodels:
            named = model.response in observations or model.response in constraints
            if (
                named
                and hasattr(model.likelihood, "sigma")
                and isinstance(model.likelihood.sigma, Hyperparameter)
                and not panels[model.response].observed.any()
            ):
                raise ModelValidationError(
                    f"the Gaussian sigma {model.likelihood.sigma.name!r} cannot be estimated: "
                    f"outcome {model.response!r} has no row responses, only linear observations "
                    "or constraints; give it a fixed value, or estimate a LinearObservation "
                    "sigma instead"
                )

        sizes = [len(panels[outcome].frame) for outcome in self.outcomes]
        starts, total = [], 0
        for size in sizes:
            starts.append(total)
            total += size

        def original_rows(outcome):
            """The caller's frame row behind each canonical row of ``outcome``."""
            return positions[outcome][panels[outcome].source_positions]

        censoring = None
        if hurdle is not None:
            # Pair the link and amount rows of each censored edge by its frame row.
            pairs = []
            for outcome in (hurdle.link, hurdle.amount):
                original = original_rows(outcome)
                local = np.flatnonzero(censored_mask[original])
                local = local[np.argsort(original[local], kind="stable")]
                pairs.append((starts[self.outcomes.index(outcome)] + local, original[local]))
            (link_rows, link_original), (amount_rows, _) = pairs
            censoring = (link_rows, amount_rows, threshold[link_original])

        linear = bool(observations or constraints)
        stacked_observations: tuple = ()
        stacked_constraints: tuple = ()
        if linear:
            selections = {}
            for index, outcome in enumerate(self.outcomes):
                n_k = sizes[index]
                selections[outcome] = csr_matrix(
                    (np.ones(n_k), (original_rows(outcome), starts[index] + np.arange(n_k))),
                    shape=(len(frame), total),
                )

            def stacked(item, outcome):
                """The item on the stacked grid: its operator, or its map, absorbs the rows."""
                if item.scale != "below_threshold":
                    return item.operator @ selections[outcome], item.scale
                if hurdle is None:
                    raise ModelValidationError(
                        "scale='below_threshold' needs Joint(..., censoring=CensoredHurdle(...))"
                    )
                referenced = np.unique(item.operator.tocsc().nonzero()[1])
                if not censored_mask[referenced].all():
                    raise ModelValidationError(
                        "a scale='below_threshold' operator may reference censored edges only"
                    )
                return item.operator, below_threshold

            if hurdle is not None:
                below_threshold = _BelowThresholdMass.bind(
                    len(frame), threshold,
                    *(
                        (original_rows(outcome), starts[self.outcomes.index(outcome)])
                        for outcome in (hurdle.link, hurdle.amount)
                    ),
                )
            stacked_observations = tuple(
                LinearObservation(item.values, operator, item.sigma, scale=scale)
                for outcome in self.outcomes
                for item in observations.get(outcome, ())
                for operator, scale in (stacked(item, outcome),)
            )
            stacked_constraints = tuple(
                LinearConstraint(operator, item.rhs, scale=scale)
                for outcome in self.outcomes
                for item in constraints.get(outcome, ())
                for operator, scale in (stacked(item, outcome),)
            )

        compiled = compile_joint(self, panels, censoring=censoring)
        if any(item.scale != "identity" for item in (*stacked_observations, *stacked_constraints)):
            family = project_gaussian_family(
                compile_joint_family(self, panels, censoring=censoring), stacked_observations, stacked_constraints,
                base_model=compiled,
                family_type=partial(
                    _RelinearizedFamily, project=project_mixture_model,
                    inner_fit=partial(fit_laplace, predictive_variances=False),
                    curvature=True,
                ),
            )

            def direct():
                return family.materialize({})
        else:
            family = compile_joint_family(self, panels, censoring=censoring)
            if observation_hyperparameters(stacked_observations):
                family = project_gaussian_family(
                    family, stacked_observations, stacked_constraints,
                    base_model=compiled if family is None else None,
                    family_type=_ProjectedMixtureFamily,
                )
            elif family is not None and linear:
                family = project_gaussian_family(
                    family, stacked_observations, stacked_constraints,
                    family_type=_ProjectedMixtureFamily,
                )

            def direct():
                if linear:
                    return project_mixture_model(compiled, stacked_observations, stacked_constraints)
                return compiled
        result = _fit_family(
            family, direct,
            partial(fit_laplace, mean_correction=True) if mean_correction else fit_laplace,
            self._declared_hyperparameters(),
            hyperparameters=hyperparameters, latent_strategy=latent_strategy,
            warm_start=warm_start, num_workers=num_workers, blas_threads=blas_threads,
            stacklevel=3,
        )

        return _finished(
            result, lambda estimates, table: build_joint_prediction_contexts(
                self, panels, compiled, estimates
            ),
        )

    def _declared_hyperparameters(self) -> list:
        """Every sub-model's Hyperparameters plus the shared scales."""
        from pylgm.compiler import _model_hyperparameters

        declared = []
        for model in self.submodels:
            declared.extend(hp for _, hp in _model_hyperparameters(model))
        for entry in self.shared:
            for scale in entry.scales_for(len(self.submodels)):
                if isinstance(scale, Hyperparameter):
                    declared.append(scale)
        return declared

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
    _ProjectedJointFamily,
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


@dataclass(frozen=True)
class Joint:
    """Several `LGM` sub-models fitted as one stacked latent Gaussian model."""

    submodels: tuple = ()
    shared: tuple = ()

    def __init__(self, submodels, shared=()) -> None:
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
        object.__setattr__(self, "submodels", submodels)
        object.__setattr__(self, "shared", shared)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return tuple(model.response for model in self.submodels)

    @classmethod
    def _unchecked(cls, submodels, shared=()):
        """Bypass the two-sub-model minimum. Test-only: used by the reduction test."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "submodels", tuple(submodels))
        object.__setattr__(obj, "shared", tuple(shared))
        return obj

    def fit(self, frame, engine: str = "laplace", *, hyperparameters: str = "optimize",
            latent_strategy: str = "gaussian", mean_correction: bool = False,
            observations=None, constraints=None):
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
        nowcast). Unreferenced NaN rows are still dropped, as today. If ``k``
        is named and its response column is absent from ``frame``, it is
        treated as all-NaN.

        See docs/joint-models.md for the full semantics, including
        ``LinearObservation`` pseudo-rows, ``LinearConstraint`` conditioning,
        and estimating a ``LinearObservation`` sigma by empirical Bayes.
        """
        import pandas as pd

        from pylgm.compiler import compile_joint, compile_joint_family, build_joint_prediction_contexts
        from pylgm.config.schema import DataConfig
        from pylgm.data.panel import CanonicalPanel
        from pylgm.exceptions import DataContractError
        from pylgm.inference.laplace import fit_laplace
        from pylgm.model import _rebuild_result
        from pylgm.observations import (
            observation_hyperparameters,
            project_gaussian_family,
            project_joint_model,
        )

        if engine != "laplace":
            raise UnsupportedEngineError(
                "Joint models require engine='laplace'; the exact_gaussian engine "
                "needs a single CompiledGaussian likelihood, and a mixture is not one. "
                "Laplace is exact for an all-Gaussian stack anyway."
            )
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("frame must be a Pandas DataFrame")

        observations = _linear_inputs(self, observations, LinearObservation, "observations")
        constraints = _linear_inputs(self, constraints, LinearConstraint, "constraints")
        for mapping in (observations, constraints):
            for outcome, items in mapping.items():
                for item in items:
                    _aligned(
                        item.operator, len(frame),
                        f"{type(item).__name__} operator for outcome {outcome!r}",
                    )
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
            # fitted values for observations that do not exist. The cost is that
            # the LGM.fit hold-out idiom does not carry over to a Joint; that is
            # documented in docs/joint-models.md under "Not supported yet".
            #
            # An outcome named in observations/constraints is the exception: a
            # NaN row it references through a nonzero operator entry becomes an
            # unobserved-but-predicted row of that outcome (e.g. the quarters a
            # nowcast observation aggregates), rather than being dropped.
            named = model.response in observations or model.response in constraints
            if not named:
                sub = frame[frame[model.response].notna()].reset_index(drop=True)
                positions[model.response] = np.flatnonzero(
                    frame[model.response].notna().to_numpy()
                )
            else:
                if model.response in frame.columns:
                    keep = frame[model.response].notna().to_numpy().copy()
                else:
                    keep = np.zeros(len(frame), dtype=bool)
                for item in (*observations.get(model.response, ()), *constraints.get(model.response, ())):
                    operator = item.operator.copy()
                    operator.eliminate_zeros()
                    keep[np.unique(operator.indices)] = True
                positions[model.response] = np.flatnonzero(keep)
                sub = frame.iloc[positions[model.response]].reset_index(drop=True)
                if model.response not in sub.columns:
                    sub = sub.assign(**{model.response: np.nan})
            time = model.time or "__pylgm_row__"
            if model.time is None:
                sub = sub.assign(**{time: range(len(sub))})
            panels[model.response] = CanonicalPanel.from_frame(
                sub, DataConfig(time=time, response=model.response, panel=model.panel),
                require_observed=not named,
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

        linear = bool(observations or constraints)
        stacked_observations: tuple = ()
        stacked_constraints: tuple = ()
        if linear:
            sizes = [len(panels[outcome].frame) for outcome in self.outcomes]
            starts, total = [], 0
            for size in sizes:
                starts.append(total)
                total += size

            selections = {}
            for index, outcome in enumerate(self.outcomes):
                p_k = positions[outcome]
                s_k = panels[outcome].source_positions
                n_k = len(s_k)
                selections[outcome] = csr_matrix(
                    (
                        np.ones(n_k),
                        (p_k[s_k], starts[index] + np.arange(n_k)),
                    ),
                    shape=(len(frame), total),
                )

            stacked_observations = tuple(
                LinearObservation(
                    item.values, item.operator @ selections[outcome], item.sigma
                )
                for outcome in self.outcomes
                for item in observations.get(outcome, ())
            )
            stacked_constraints = tuple(
                LinearConstraint(item.operator @ selections[outcome], item.rhs)
                for outcome in self.outcomes
                for item in constraints.get(outcome, ())
            )

        family = compile_joint_family(self, panels)
        if observation_hyperparameters(stacked_observations):
            family = project_gaussian_family(
                family, stacked_observations, stacked_constraints,
                base_model=compile_joint(self, panels) if family is None else None,
                family_type=_ProjectedJointFamily,
            )
        elif family is not None and linear:
            family = project_gaussian_family(
                family, stacked_observations, stacked_constraints,
                family_type=_ProjectedJointFamily,
            )
        if hyperparameters == "integrate":
            if family is None:
                raise ValueError(
                    "hyperparameters='integrate' requires a declared Hyperparameter"
                )
            result = self._run_inla(family, latent_strategy, mean_correction)
            compiled = compile_joint(self, panels)
        elif family is None:
            compiled = compile_joint(self, panels)
            fitted = (
                project_joint_model(compiled, stacked_observations, stacked_constraints)
                if linear else compiled
            )
            result = fit_laplace(fitted, mean_correction=mean_correction)
        else:
            result = self._run_empirical_bayes(family, mean_correction)
            compiled = compile_joint(self, panels)

        contexts = build_joint_prediction_contexts(self, panels, compiled, result)
        return _rebuild_result(result, prediction_context=contexts)

    def _family_optimization_inputs(self, family):
        """Bounds, initial values and the prior penalty for this joint's hyperparameters.

        Mirrors ``LGM._family_optimization_inputs`` (model.py:405-426) but reads
        the declared Hyperparameters from every sub-model plus the shared scales,
        which is where a joint's parameters actually live.
        """
        from pylgm.compiler import _model_hyperparameters
        from pylgm.optimization.empirical_bayes import OptimizationBounds

        declared = []
        for model in self.submodels:
            declared.extend(hp for _, hp in _model_hyperparameters(model))
        for entry in self.shared:
            for scale in entry.scales_for(len(self.submodels)):
                if isinstance(scale, Hyperparameter):
                    declared.append(scale)
        declared.extend(getattr(family, "hyperparameters", ()))

        bounds = (
            dict(family.parameter_bounds)
            if family.parameter_bounds
            else {hp.name: OptimizationBounds(hp.initial, hp.lower, hp.upper) for hp in declared}
        )
        initial = {hp.name: hp.initial for hp in declared if hp.name in family.parameter_names}
        family_priors = dict(getattr(family, "parameter_priors", {}) or {})
        priored = [hp for hp in declared if hp.prior is not None]
        penalty = None
        if family_priors or priored:
            def penalty(values, priored=priored):
                return sum(float(hp.prior.logpdf(values[hp.name])) for hp in priored)
        return bounds, initial, penalty

    def _run_empirical_bayes(self, family, mean_correction: bool = False):
        """Type-II ML / MAP-II fit. Mirrors LGM._run_empirical_bayes (model.py:428)."""
        import warnings

        from pylgm.inference.laplace import fit_laplace
        from pylgm.model import _attach_estimates, _parameters_at_bound
        from pylgm.optimization.empirical_bayes import optimize_empirical_bayes

        bounds, initial, penalty = self._family_optimization_inputs(family)
        eb = optimize_empirical_bayes(
            family, bounds, initial=initial,
            fit=partial(fit_laplace, mean_correction=mean_correction) if mean_correction
            else fit_laplace,
            penalty=penalty,
        )
        diagnostics = dict(eb.fit.diagnostics)
        diagnostics["empirical_bayes_converged"] = eb.diagnostics.converged
        diagnostics["empirical_bayes_evaluations"] = eb.diagnostics.evaluations
        diagnostics["hyperparameter_penalized"] = penalty is not None
        pinned = _parameters_at_bound(dict(eb.parameters), bounds)
        diagnostics["hyperparameters_at_bound"] = ", ".join(pinned)
        if pinned:
            warnings.warn(
                f"empirical-Bayes estimate(s) {list(pinned)} landed on the edge of "
                "the declared interval, so the bound rather than the data is "
                "setting the value. Widen lower/upper on those Hyperparameters "
                "and refit.",
                UserWarning,
                stacklevel=3,
            )
        return _attach_estimates(eb.fit, dict(eb.parameters), diagnostics)

    def _run_inla(self, family, latent_strategy: str = "gaussian",
                  mean_correction: bool = False):
        """INLA grid integration. Mirrors LGM._run_inla (model.py:450)."""
        from pylgm.inference.laplace import fit_laplace
        from pylgm.optimization.inla import integrate_inla

        bounds, initial, penalty = self._family_optimization_inputs(family)
        conditional = (partial(fit_laplace, mean_correction=True) if mean_correction
                       else fit_laplace)
        return integrate_inla(
            family, bounds, initial=initial, fit=conditional, penalty=penalty,
            latent_strategy=latent_strategy,
        )

"""Joint latent Gaussian models: several responses stacked into one CompiledLGM.

A joint model is an ordinary :class:`~pylgm.ir.model.CompiledLGM` with more
rows. Responses stack as ``y = (y^(1), ..., y^(K))``; sub-model ``k`` occupies a
contiguous row slice. Private latent blocks are zero-padded outside their slice,
shared blocks carry scaled rows in every slice they enter, and the likelihood
becomes a row-dispatching :class:`~pylgm.likelihoods.CompiledMixture`. Both IR
invariants -- ``design == hstack(blocks)`` and ``precision == block_diag(blocks)``
-- are preserved.
"""

from dataclasses import dataclass

from scipy.sparse import csr_matrix, vstack

from pylgm.ir.model import LatentBlock
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
# through compilation without inventing an expression language.
InverseOf = tuple[str, str]


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
        if not hasattr(self.effect, "index"):
            raise TypeError(
                f"Shared effect {self.effect!r} has no `index` -- a shared effect "
                "must be indexed (IID, RW1/RW2, AR1, Seasonal, Besag, ProperCAR, "
                "SAR, BYM2), so it can be summed across sub-models over a common "
                "level set. Fixed/MIDAS/MIDASParametric/SpaceTime/"
                "DynamicSpatialPanel effects cannot be shared."
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
            latent_strategy: str = "gaussian"):
        """Compile and fit this joint model. Only ``engine='laplace'`` is supported."""
        import pandas as pd

        from pylgm.compiler import compile_joint, compile_joint_family, build_joint_prediction_contexts
        from pylgm.config.schema import DataConfig
        from pylgm.data.panel import CanonicalPanel
        from pylgm.exceptions import DataContractError, UnsupportedEngineError
        from pylgm.inference.laplace import fit_laplace
        from pylgm.model import _rebuild_result

        if engine != "laplace":
            raise UnsupportedEngineError(
                "Joint models require engine='laplace'; the exact_gaussian engine "
                "needs a single CompiledGaussian likelihood, and a mixture is not one. "
                "Laplace is exact for an all-Gaussian stack anyway."
            )
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError("frame must be a Pandas DataFrame")

        panels = {}
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
            sub = frame[frame[model.response].notna()].reset_index(drop=True)
            time = model.time or "__pylgm_row__"
            if model.time is None:
                sub = sub.assign(**{time: range(len(sub))})
            panels[model.response] = CanonicalPanel.from_frame(
                sub, DataConfig(time=time, response=model.response, panel=model.panel)
            )

        family = compile_joint_family(self, panels)
        if hyperparameters == "integrate":
            if family is None:
                raise ValueError(
                    "hyperparameters='integrate' requires a declared Hyperparameter"
                )
            result = self._run_inla(family, latent_strategy)
            compiled = compile_joint(self, panels)
        elif family is None:
            compiled = compile_joint(self, panels)
            result = fit_laplace(compiled)
        else:
            result = self._run_empirical_bayes(family)
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

    def _run_empirical_bayes(self, family):
        """Type-II ML / MAP-II fit. Mirrors LGM._run_empirical_bayes (model.py:428)."""
        import warnings

        from pylgm.inference.laplace import fit_laplace
        from pylgm.model import _attach_estimates, _parameters_at_bound
        from pylgm.optimization.empirical_bayes import optimize_empirical_bayes

        bounds, initial, penalty = self._family_optimization_inputs(family)
        eb = optimize_empirical_bayes(
            family, bounds, initial=initial, fit=fit_laplace, penalty=penalty
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

    def _run_inla(self, family, latent_strategy: str = "gaussian"):
        """INLA grid integration. Mirrors LGM._run_inla (model.py:450)."""
        from pylgm.inference.laplace import fit_laplace
        from pylgm.optimization.inla import integrate_inla

        bounds, initial, penalty = self._family_optimization_inputs(family)
        return integrate_inla(
            family, bounds, initial=initial, fit=fit_laplace, penalty=penalty,
            latent_strategy=latent_strategy,
        )

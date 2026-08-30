import warnings
from typing import TYPE_CHECKING

import numpy as np
from formulaic import model_matrix
from formulaic.errors import FormulaicError
from scipy.sparse import bmat, block_diag, csr_matrix, diags, hstack, identity

from pylgm.config import RunConfig
from pylgm.config.schema import DataConfig, ModelConfig
from pylgm.data import CanonicalPanel
from pylgm.data.scalars import ordered_observed_levels
from pylgm.effects import (
    AR1,
    Besag,
    BYM2,
    DynamicSpatialPanel,
    Fixed,
    IID,
    MIDAS,
    MIDASParametric,
    ProperCAR,
    RW1,
    RW2,
    SAR,
    SpaceTime,
    build_ar1,
    build_besag,
    build_bym2,
    build_fixed,
    build_iid,
    build_midas,
    build_midas_parametric,
    build_proper_car,
    build_random_walk,
    build_sar,
    build_spacetime,
    midas_penalty,
    midas_weights,
    normalize_graph,
)
from pylgm.effects.ar1 import ar1_structure
from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.sar import (
    build_dynamic_spatial_panel, _panel_networks, _sdpd_operator,
    _sar_operator, _gram_precision,
)
from pylgm.effects.bym2 import (
    _build_bym2_augmented,
    _BYM2_AUGMENT_NODES,
    bym2_precision,
    bym2_spectrum,
)
from pylgm.effects.proper_car import car_rho_interval
from pylgm.effects.scaling import sorbye_rue_scale
from pylgm.exceptions import CompilationError, DataContractError, ModelValidationError
from pylgm.ir import (
    CompiledFamily,
    CompiledGaussianFamily,
    CompiledLGM,
    Hyperparameters,
    ParametricBlock,
    ParametricDesignBlock,
    ScalableBlock,
)
from pylgm.inference.prediction import PredictionContext
from pylgm.ir.model import LatentBlock, _block_constraints
from pylgm.likelihoods import (
    Bernoulli,
    Beta,
    Binomial,
    CompiledGaussian,
    ExponentialSurv,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    WeibullSurv,
)
from pylgm.optimization.empirical_bayes import OptimizationBounds
from pylgm.optimization.transforms import IdentityTransform, LogitTransform, LogTransform
from pylgm.parameters import Hyperparameter

if TYPE_CHECKING:
    from pylgm.model import LGM


def resolve_constraints(
    rows: tuple[tuple[dict[str, float], float], ...], labels: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Turn label-keyed ``A x = e`` constraints into a raw ``(m, width)`` matrix and rhs.

    ``labels`` are the qualified latent labels in column order; each row is a
    ``(mapping, rhs)`` pair mapping those labels to coefficients. Unknown labels
    are rejected here because this is the first point where the full latent
    ordering is known.
    """
    width = len(labels)
    if not rows:
        return np.empty((0, width)), np.empty(0)
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(rows), width))
    rhs = np.zeros(len(rows))
    for position, (row, row_rhs) in enumerate(rows):
        rhs[position] = row_rhs
        for label, coefficient in row.items():
            column = index.get(label)
            if column is None:
                raise CompilationError(
                    f"constraint references unknown latent label {label!r}; "
                    "labels are qualified as 'effect:level'"
                )
            matrix[position, column] = coefficient
    return matrix, rhs


def _structured_blocks(
    model: ModelConfig,
    panel: CanonicalPanel,
    optimized: tuple[str, ...] = (),
) -> list[LatentBlock]:
    blocks: list[LatentBlock] = []
    frame = panel.frame
    for effect in model.effects:
        parameter = f"{effect.name}.precision"
        precision = 1.0 if parameter in optimized else effect.precision
        try:
            if effect.type == "iid":
                block = build_iid(frame, effect.name, effect.index, precision)
            else:
                order = 1 if effect.type == "rw1" else 2
                block = build_random_walk(
                    frame,
                    effect.name,
                    effect.index,
                    precision,
                    order,
                )
        except (DataContractError, ModelValidationError, TypeError, ValueError) as error:
            raise CompilationError(f"failed to compile effect {effect.name!r}: {error}") from error
        blocks.append(block)
    return blocks


def _qualified_labels(blocks: list[LatentBlock]) -> tuple[str, ...]:
    labels = tuple(f"{block.name}:{label}" for block in blocks for label in block.labels)
    if len(labels) != len(set(labels)):
        raise CompilationError("duplicate latent labels after qualification")
    return labels


def _validate_required_columns(data: DataConfig, model: ModelConfig, frame_columns: object) -> None:
    columns = set(frame_columns)
    required_data = {*data.panel, data.time, data.response}
    missing_data = sorted(required_data.difference(columns))
    if missing_data:
        raise DataContractError(f"missing configured data columns: {missing_data}")
    missing_indexes = sorted({effect.index for effect in model.effects}.difference(columns))
    if missing_indexes:
        raise DataContractError(f"missing configured effect index columns: {missing_indexes}")


def _validate_optimized_names(model: ModelConfig, optimized: tuple[str, ...]) -> None:
    if any(not isinstance(name, str) for name in optimized):
        raise CompilationError("optimized parameter names must be strings")
    if len(optimized) != len(set(optimized)):
        raise CompilationError("optimized parameter names must be unique")
    allowed = {"sigma", *(f"{effect.name}.precision" for effect in model.effects)}
    unknown = sorted(set(optimized).difference(allowed))
    if unknown:
        raise CompilationError(f"unknown optimized parameter names: {unknown}")


def compile_gaussian_family(
    data: DataConfig,
    model: ModelConfig,
    panel: CanonicalPanel,
    optimized: tuple[str, ...],
) -> CompiledGaussianFamily:
    optimized = tuple(optimized)
    _validate_optimized_names(model, optimized)
    if panel.response != data.response:
        raise DataContractError(
            "panel response metadata does not match configuration: "
            f"{panel.response!r} != {data.response!r}"
        )
    expected_keys = (*data.panel, data.time)
    if panel.key_columns != expected_keys:
        raise DataContractError(
            "panel key metadata does not match configuration: "
            f"{panel.key_columns!r} != {expected_keys!r}"
        )
    frame = panel.frame
    _validate_required_columns(data, model, frame.columns)
    try:
        fixed = build_fixed(
            frame,
            model.fixed,
            model.fixed_prior_precision,
        )
    except (
        DataContractError,
        FormulaicError,
        ModelValidationError,
        TypeError,
        ValueError,
    ) as error:
        raise CompilationError(f"failed to compile fixed formula: {error}") from error
    structured_optimized = tuple(name for name in optimized if name.endswith(".precision"))
    blocks = [fixed]
    if structured_optimized:
        blocks.extend(_structured_blocks(model, panel, optimized))
    else:
        blocks.extend(_structured_blocks(model, panel))
    wrong_rows = [block.name for block in blocks if block.design.shape[0] != len(frame)]
    if wrong_rows:
        raise CompilationError(
            f"latent block design row count does not match the panel: {wrong_rows}"
        )
    _qualified_labels(blocks)
    try:
        y = frame[panel.response].fillna(0.0).to_numpy(dtype=float)
        scalable_blocks = tuple(
            ScalableBlock(
                block,
                f"{block.name}.precision" if f"{block.name}.precision" in optimized else None,
                1.0,
            )
            for block in blocks
        )
        return CompiledGaussianFamily(
            y=y,
            observed=panel.observed,
            offset=np.zeros(len(frame)),
            blocks=scalable_blocks,
            parameter_names=optimized,
            initial=Hyperparameters(
                sigma=float(model.sigma),
                precisions={effect.name: float(effect.precision) for effect in model.effects},
            ),
        )
    except (TypeError, ValueError) as error:
        raise CompilationError(f"compiled Gaussian family is invalid: {error}") from error


def compile_model(config: RunConfig, panel: CanonicalPanel) -> CompiledLGM:
    return compile_gaussian_family(config.data, config.model, panel, optimized=()).materialize({})


def _resolved_precision(value: float | Hyperparameter) -> float:
    return value.initial if isinstance(value, Hyperparameter) else value


def _warn_missing_spacetime_main_effects(effects) -> None:
    """Warn if a SpaceTime interaction lacks its spatial/temporal main effects.

    The interaction's sum-to-zero constraints assume the main effects absorb the
    spatial and temporal marginals; omitting one is a modelling error, so warn
    (do not crash) naming the effect and the missing companion.
    """
    spatial_indices = {e.index for e in effects if isinstance(e, (Besag, ProperCAR, BYM2))}
    temporal_indices = {e.index for e in effects if isinstance(e, (RW1, RW2, AR1))}
    for effect in effects:
        if not isinstance(effect, SpaceTime):
            continue
        missing = []
        if effect.space not in spatial_indices:
            missing.append(f"a spatial main effect on {effect.space!r}")
        if effect.time not in temporal_indices:
            missing.append(f"a temporal main effect on {effect.time!r}")
        if missing:
            warnings.warn(
                f"SpaceTime effect {effect.name!r} has no {' and no '.join(missing)}; "
                "its identifiability constraints assume the main effects absorb the "
                "marginals. Add them, or the variance split is not identified.",
                UserWarning,
                stacklevel=2,
            )


def _likelihood_columns(model: "LGM", frame: object) -> "dict | None":
    """Extract and validate the per-row auxiliary data a likelihood binds.

    Returns ``{"trials": ...}`` for Binomial and ``None`` for every family that
    binds no per-row data (its ``for_observations`` hook is then a no-op).
    """
    like = model.likelihood
    if isinstance(like, Binomial):
        column = like.trials
        if column not in frame.columns:
            raise DataContractError(f"trials column not found: {column!r}")
        trials = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(trials)) or np.any(trials < 1.0) or np.any(trials != np.floor(trials)):
            raise CompilationError("binomial trials column must be positive integers")
        return {"trials": trials}
    if isinstance(like, (WeibullSurv, ExponentialSurv)):
        if not np.all(np.isfinite(frame[model.response].to_numpy(dtype=float))):
            raise DataContractError(
                f"survival response {model.response!r} (follow-up time) must be observed for every row"
            )
        if like.event not in frame.columns:
            raise DataContractError(f"event column not found: {like.event!r}")
        event = frame[like.event].to_numpy(dtype=float)
        if not np.all(np.isin(event, (0.0, 1.0))):
            raise DataContractError("survival event indicator must be 0 or 1")
        aux: dict = {"event": event, "entry": None}
        if like.entry is not None:
            if like.entry not in frame.columns:
                raise DataContractError(f"entry column not found: {like.entry!r}")
            entry = frame[like.entry].to_numpy(dtype=float)
            t = frame[model.response].to_numpy(dtype=float)
            if not np.all(np.isfinite(entry)) or np.any(entry < 0.0) or np.any(entry >= t):
                raise DataContractError("survival entry time must satisfy 0 <= entry < time")
            aux["entry"] = entry
        return aux
    return None


def _estimable_scalar(likelihood: object) -> "Hyperparameter | None":
    """The likelihood's optimisable scalar (dispersion phi or Weibull shape), if any."""
    for attr in ("phi", "shape"):
        value = getattr(likelihood, attr, None)
        if isinstance(value, Hyperparameter):
            return value
    return None


def _slice_aux(aux: "dict | None", observed: "np.ndarray") -> "dict | None":
    """Slice each bound vector to the observed rows (None entries stay None)."""
    if aux is None:
        return None
    return {k: (v[observed] if v is not None else None) for k, v in aux.items()}


def compile_lgm(model: "LGM", panel: CanonicalPanel) -> CompiledLGM:
    """Compile a declarative model through the existing sparse effect builders."""
    if panel.response != model.response:
        raise DataContractError(
            "panel response metadata does not match model: "
            f"{panel.response!r} != {model.response!r}"
        )
    frame = panel.frame
    blocks: list[LatentBlock] = []
    precisions: dict[str, float] = {}
    for effect in model.predictor.effects:
        try:
            if isinstance(effect, Fixed):
                block = build_fixed(frame, effect.formula, effect.prior_precision)
            elif isinstance(effect, IID):
                precision = _resolved_precision(effect.precision)
                block = build_iid(frame, effect.name, effect.index, precision)
                precisions[effect.name] = precision
            elif isinstance(effect, Besag):
                precision = _resolved_precision(effect.precision)
                block = build_besag(
                    frame, effect.name, effect.index, dict(effect.graph), precision, effect.scale
                )
                precisions[effect.name] = precision
            elif isinstance(effect, ProperCAR):
                precision = _resolved_precision(effect.precision)
                rho = _resolved_precision(effect.rho)
                block = build_proper_car(
                    frame, effect.name, effect.index, dict(effect.graph), rho, precision
                )
                precisions[effect.name] = precision
            elif isinstance(effect, SAR):
                precision = _resolved_precision(effect.precision)
                rho = _resolved_precision(effect.rho)
                block = build_sar(
                    frame, effect.name, effect.index, dict(effect.graph), rho, precision
                )
                precisions[effect.name] = precision
            elif isinstance(effect, DynamicSpatialPanel):
                precision = _resolved_precision(effect.precision)
                block = build_dynamic_spatial_panel(
                    frame, effect.name, effect.unit, effect.time,
                    {t: dict(g) for t, g in dict(effect.graphs).items()},
                    _resolved_precision(effect.rho),
                    _resolved_precision(effect.gamma),
                    _resolved_precision(effect.eta),
                    precision,
                )
                precisions[effect.name] = precision
            elif isinstance(effect, BYM2):
                precision = _resolved_precision(effect.precision)
                phi = _resolved_precision(effect.phi) if isinstance(effect.phi, Hyperparameter) else effect.phi
                block = build_bym2(
                    frame, effect.name, effect.index, dict(effect.graph), precision, phi
                )
                precisions[effect.name] = precision
            elif isinstance(effect, AR1):
                precision = _resolved_precision(effect.precision)
                rho = _resolved_precision(effect.rho) if isinstance(effect.rho, Hyperparameter) else effect.rho
                block = build_ar1(
                    frame, effect.name, effect.index, precision, rho, effect.group
                )
                precisions[effect.name] = precision
            elif isinstance(effect, (RW1, RW2)):
                precision = _resolved_precision(effect.precision)
                order = 1 if isinstance(effect, RW1) else 2
                block = build_random_walk(
                    frame, effect.name, effect.index, precision, order
                )
                precisions[effect.name] = precision
            elif isinstance(effect, MIDAS):
                precision = _resolved_precision(effect.precision)
                block = build_midas(
                    frame, effect.name, effect.columns, precision, effect.order, effect.ridge
                )
                precisions[effect.name] = precision
            elif isinstance(effect, MIDASParametric):
                theta = (_resolved_precision(effect.shape1), _resolved_precision(effect.shape2))
                block = build_midas_parametric(
                    frame, effect.name, effect.columns, effect.kernel, theta, effect.prior_precision
                )
            elif isinstance(effect, SpaceTime):
                precision = _resolved_precision(effect.precision)
                block = build_spacetime(
                    frame, effect.name, effect.space, effect.time,
                    dict(effect.graph) if effect.graph is not None else None,
                    effect.interaction, effect.order, precision, effect.scale,
                )
                precisions[effect.name] = precision
            else:
                # An unrecognized effect must not fall through to the random-walk
                # builder: that silently mis-compiles it as an RW2.
                raise CompilationError(
                    f"unsupported effect type: {type(effect).__name__}"
                )
        except (
            DataContractError,
            FormulaicError,
            ModelValidationError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise CompilationError(
                f"failed to compile effect {effect.name!r}: {error}"
            ) from error
        blocks.append(block)

    _warn_missing_spacetime_main_effects(model.predictor.effects)

    wrong_rows = [block.name for block in blocks if block.design.shape[0] != len(frame)]
    if wrong_rows:
        raise CompilationError(
            f"latent block design row count does not match the panel: {wrong_rows}"
        )
    labels = _qualified_labels(blocks)
    extra_constraints, extra_constraint_rhs = resolve_constraints(model.constraints, labels)

    if model.offset is not None and model.offset not in frame.columns:
        raise DataContractError(f"offset column not found: {model.offset!r}")
    offset = (
        frame[model.offset].to_numpy(dtype=float)
        if model.offset is not None
        else np.zeros(len(frame))
    )
    if not np.isfinite(offset).all():
        raise CompilationError("offset column must be finite")
    y = frame[panel.response].fillna(0.0).to_numpy(dtype=float)

    if isinstance(model.likelihood, Gaussian):
        sigma = (
            model.likelihood.sigma.initial
            if isinstance(model.likelihood.sigma, Hyperparameter)
            else model.likelihood.sigma
        )
        try:
            family = CompiledGaussianFamily(
                y=y,
                observed=panel.observed,
                offset=offset,
                blocks=tuple(ScalableBlock(block, None, 1.0) for block in blocks),
                parameter_names=(),
                initial=Hyperparameters(sigma=sigma, precisions=precisions),
                extra_constraints=extra_constraints,
                extra_constraint_rhs=extra_constraint_rhs,
            )
            return family.materialize({})
        except (TypeError, ValueError) as error:
            raise CompilationError(f"compiled declarative model is invalid: {error}") from error

    if isinstance(model.likelihood, (Poisson, Bernoulli, Binomial, NegativeBinomial,
                                     Gamma, Beta, WeibullSurv, ExponentialSurv)):
        # A phi/shape-family with an optimisable scalar compiles here at its
        # initial value (the EB/INLA fit refines it); a fixed-scalar family
        # ignores the mapping.
        scalar = _estimable_scalar(model.likelihood)
        values = {scalar.name: scalar.initial} if scalar is not None else {}
        compiled_likelihood = model.likelihood.materialize(values)
        observed = panel.observed
        # Binomial carries a per-row trials vector; binding is a no-op otherwise.
        aux = _likelihood_columns(model, frame)
        compiled_likelihood.for_observations(_slice_aux(aux, observed)).validate_response(y[observed])
        compiled_likelihood = compiled_likelihood.for_observations(aux)
        if not blocks:
            raise CompilationError("model must contain at least one latent effect")
        width = sum(block.design.shape[1] for block in blocks)
        design = hstack([block.design for block in blocks], format="csr")
        precision = block_diag([block.precision for block in blocks], format="csr")
        constraints = _block_constraints(tuple(blocks), width)
        if extra_constraints.shape[0]:
            constraints = np.vstack([constraints, extra_constraints])
        try:
            return CompiledLGM(
                y=y,
                observed=observed,
                offset=offset,
                design=design,
                precision=precision,
                constraints=constraints,
                labels=labels,
                likelihood=compiled_likelihood,
                blocks=tuple(blocks),
                extra_constraints=extra_constraints,
                extra_constraint_rhs=extra_constraint_rhs,
            )
        except (TypeError, ValueError, ModelValidationError) as error:
            raise CompilationError(f"compiled declarative model is invalid: {error}") from error

    raise CompilationError("unsupported likelihood for declarative compilation")


def _model_hyperparameters(model: "LGM") -> list[tuple[str, Hyperparameter]]:
    """Return (target_name, Hyperparameter) pairs declared on the model."""
    found: list[tuple[str, Hyperparameter]] = []
    if isinstance(model.likelihood, Gaussian) and isinstance(model.likelihood.sigma, Hyperparameter):
        found.append(("sigma", model.likelihood.sigma))
    phi = getattr(model.likelihood, "phi", None)  # NB/Gamma/Beta dispersion/precision
    if isinstance(phi, Hyperparameter):
        found.append(("phi", phi))
    shape = getattr(model.likelihood, "shape", None)  # Weibull survival shape
    if isinstance(shape, Hyperparameter):
        found.append(("shape", shape))
    for effect in model.predictor.effects:
        precision = getattr(effect, "precision", None)
        if isinstance(precision, Hyperparameter):
            found.append((effect.name, precision))
        if isinstance(effect, ProperCAR) and isinstance(effect.rho, Hyperparameter):
            found.append((effect.name, effect.rho))
        if isinstance(effect, BYM2) and isinstance(effect.phi, Hyperparameter):
            found.append((effect.name, effect.phi))
        if isinstance(effect, AR1) and isinstance(effect.rho, Hyperparameter):
            found.append((effect.name, effect.rho))
        if isinstance(effect, SAR) and isinstance(effect.rho, Hyperparameter):
            found.append((effect.name, effect.rho))
        if isinstance(effect, DynamicSpatialPanel):
            for coeff in (effect.rho, effect.gamma, effect.eta):
                if isinstance(coeff, Hyperparameter):
                    found.append((effect.name, coeff))
        if isinstance(effect, MIDASParametric):
            for shape in (effect.shape1, effect.shape2):
                if isinstance(shape, Hyperparameter):
                    found.append((effect.name, shape))
    return found


def _log_bounds(hp: Hyperparameter) -> OptimizationBounds:
    """Bounds for a log-transform (positive) hyperparameter: sigma or a precision.

    Identical to the ``OptimizationBounds(hp.initial, hp.lower, hp.upper)`` that
    ``model.py`` used to build directly (default transform is already
    ``LogTransform``), so existing optimise/integrate behaviour is unchanged.
    """
    if hp.transform == "logit":
        raise CompilationError(
            f"hyperparameter {hp.name!r} declares transform='logit' but is not attached "
            "to an effect that supplies a bounded interval; only a bounded effect "
            "parameter (proper CAR rho, BYM2 phi, AR1 rho) resolves one"
        )
    return OptimizationBounds(hp.initial, hp.lower, hp.upper, transform=LogTransform())


def _real_bounds(hp: Hyperparameter) -> OptimizationBounds:
    """Bounds for a real-line (identity-transform) hyperparameter, e.g. an
    exp-Almon MIDAS weight shape. The Hyperparameter already carries finite
    lower/upper (defaulted symmetrically) under transform='identity'."""
    if hp.transform != "identity":
        raise CompilationError(
            f"exp-Almon MIDAS shape {hp.name!r} must be declared transform='identity'; "
            f"got transform={hp.transform!r}"
        )
    return OptimizationBounds(hp.initial, hp.lower, hp.upper, transform=IdentityTransform())


def _bounded_parameter(
    hyperparameter: Hyperparameter,
    lower: float,
    upper: float,
    *,
    label: str,
    inset: float,
) -> OptimizationBounds:
    """Bounds for a hyperparameter confined to an open interval.

    Shared by proper CAR's rho, BYM2's phi and AR1's rho: all three are bounded
    parameters inferred on a logit scale, and all three must reject a log
    transform, which would silently inherit positive-only default bounds and
    confine the parameter to a wrong, one-sided interval.
    """
    if hyperparameter.transform != "logit":
        raise CompilationError(
            f"{label} {hyperparameter.name!r} must be declared with transform='logit' "
            f"(it is bounded to ({lower:.6g}, {upper:.6g})); got "
            f"transform={hyperparameter.transform!r}"
        )
    low = lower + inset if hyperparameter.lower is None else max(hyperparameter.lower, lower + inset)
    high = upper - inset if hyperparameter.upper is None else min(hyperparameter.upper, upper - inset)
    if not low <= hyperparameter.initial <= high:
        raise CompilationError(
            f"initial value for {label} {hyperparameter.name!r} must lie in "
            f"({low:.6g}, {high:.6g}); got {hyperparameter.initial}"
        )
    return OptimizationBounds(
        float(hyperparameter.initial), low, high, transform=LogitTransform(lower, upper)
    )


def _compiled_block(name: str, builder, *args) -> object:
    """Call an effect builder, wrapping failures the same way ``compile_lgm`` does.

    Without this, the same malformed effect (e.g. a single-level AR1) raises a
    bare exception here but a ``CompilationError`` naming the effect from
    ``compile_lgm`` -- purely because this one declares a ``Hyperparameter``.
    """
    try:
        return builder(*args)
    except (
        DataContractError,
        FormulaicError,
        ModelValidationError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise CompilationError(f"failed to compile effect {name!r}: {error}") from error


def compile_family(model: "LGM", panel: CanonicalPanel) -> CompiledFamily | None:
    """Build an optimisable family, or None when no Hyperparameter is declared."""
    if not _model_hyperparameters(model):
        return None
    frame = panel.frame
    if model.offset is not None and model.offset not in frame.columns:
        raise DataContractError(f"offset column not found: {model.offset!r}")
    offset = (
        frame[model.offset].to_numpy(dtype=float)
        if model.offset is not None
        else np.zeros(len(frame))
    )
    y = frame[panel.response].fillna(0.0).to_numpy(dtype=float)

    scalable: list[ScalableBlock | ParametricBlock | ParametricDesignBlock] = []
    parameter_names: list[str] = []
    parameter_bounds: dict[str, OptimizationBounds] = {}
    parameter_priors: dict[str, object] = {}
    for effect in model.predictor.effects:
        if isinstance(effect, Fixed):
            block = _compiled_block(
                effect.name, build_fixed, frame, effect.formula, effect.prior_precision
            )
            scalable.append(ScalableBlock(block, None, 1.0))
            continue
        if isinstance(effect, MIDASParametric):
            # No `.precision` field (unlike every other effect below): the beta
            # loading's prior precision is a fixed constant, not a named
            # Hyperparameter, so this branch must precede `effect.precision`.
            theta_init = (_resolved_precision(effect.shape1), _resolved_precision(effect.shape2))
            template = _compiled_block(
                effect.name, build_midas_parametric,
                frame, effect.name, effect.columns, effect.kernel, theta_init, effect.prior_precision,
            )
            shapes = (effect.shape1, effect.shape2)
            estimated = [s for s in shapes if isinstance(s, Hyperparameter)]
            if not estimated:
                # both shapes fixed: design bakes in, no per-theta rebuild
                scalable.append(ScalableBlock(template, None, 1.0))
                continue
            name1 = effect.shape1.name if isinstance(effect.shape1, Hyperparameter) else None
            name2 = effect.shape2.name if isinstance(effect.shape2, Hyperparameter) else None
            fixed1 = None if name1 else float(effect.shape1)
            fixed2 = None if name2 else float(effect.shape2)
            columns, kernel, prior_precision = effect.columns, effect.kernel, effect.prior_precision
            frame_ref = frame

            def build(values, columns=columns, kernel=kernel, prior_precision=prior_precision,
                      frame_ref=frame_ref, name1=name1, name2=name2, fixed1=fixed1, fixed2=fixed2):
                theta = (
                    values[name1] if name1 else fixed1,
                    values[name2] if name2 else fixed2,
                )
                V = frame_ref[list(columns)].to_numpy(dtype=float)
                w = midas_weights(kernel, len(columns), theta)
                return csr_matrix((V @ w).reshape(-1, 1))

            param_names = tuple(s.name for s in estimated)
            scalable.append(ParametricDesignBlock(template, param_names, build))
            for shape in estimated:
                parameter_names.append(shape.name)
                parameter_bounds[shape.name] = (
                    _log_bounds(shape) if shape.transform == "log" else _real_bounds(shape)
                )
                if shape.prior is not None:
                    parameter_priors[shape.name] = shape.prior
            continue
        precision = effect.precision
        optimized = isinstance(precision, Hyperparameter)
        value = 1.0 if optimized else precision
        if isinstance(effect, Besag):
            block = _compiled_block(
                effect.name, build_besag,
                frame, effect.name, effect.index, dict(effect.graph), value, effect.scale,
            )
            scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            continue
        if isinstance(effect, ProperCAR):
            rho_is_hp = isinstance(effect.rho, Hyperparameter)
            if not rho_is_hp:
                block = _compiled_block(
                    effect.name, build_proper_car,
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, value,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            # rho is a Hyperparameter -> a ParametricBlock over (tau, rho), bounded
            # to the graph's positive-definiteness interval via a Logit transform.
            nodes, w = normalize_graph(dict(effect.graph))
            degree = np.asarray(w.sum(axis=1)).ravel()
            a, b = car_rho_interval(dict(effect.graph))
            rho_bounds = _bounded_parameter(
                effect.rho, a, b, label="proper CAR rho", inset=1e-6 * (b - a)
            )
            rho_initial = float(effect.rho.initial)
            template = _compiled_block(
                effect.name, build_proper_car,
                frame, effect.name, effect.index, dict(effect.graph),
                rho_initial, value if not optimized else 1.0,
            )
            deg = degree
            wmat = w
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            rho_name = effect.rho.name

            def build(
                values,
                deg=deg,
                wmat=wmat,
                tau_name=tau_name,
                tau_fixed=tau_fixed,
                rho_name=rho_name,
            ) -> csr_matrix:
                tau = values[tau_name] if tau_name else tau_fixed
                rho = values[rho_name]
                return csr_matrix(tau * (diags(deg) - rho * wmat))

            params = tuple(name for name in (tau_name, rho_name) if name)
            scalable.append(ParametricBlock(template, params, build))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            parameter_names.append(rho_name)
            parameter_bounds[rho_name] = rho_bounds
            continue
        if isinstance(effect, BYM2):
            phi_is_hp = isinstance(effect.phi, Hyperparameter)
            if not phi_is_hp:
                block = _compiled_block(
                    effect.name, build_bym2,
                    frame, effect.name, effect.index, dict(effect.graph), value, effect.phi,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            # phi is a Hyperparameter -> a ParametricBlock over (tau, phi), bounded
            # to (0, 1) via a Logit transform (BYM2 is unconstrained: no graph
            # interval to resolve, unlike proper CAR's rho).
            phi_bounds = _bounded_parameter(effect.phi, 0.0, 1.0, label="BYM2 phi", inset=1e-6)
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            phi_name = effect.phi.name
            nodes_bym2, w_bym2 = normalize_graph(dict(effect.graph))
            augmented = len(nodes_bym2) > _BYM2_AUGMENT_NODES
            if augmented:
                template = _compiled_block(
                    effect.name, _build_bym2_augmented,
                    frame, effect.name, effect.index, dict(effect.graph),
                    value, float(effect.phi.initial),
                )
                degree = np.asarray(w_bym2.sum(axis=1)).ravel()
                rstar = sorbye_rue_scale((diags(degree) - w_bym2).tocsc(), null_dim=1)
                ident = identity(len(nodes_bym2), format="csr")

                def build(
                    values_map,
                    rstar=rstar,
                    ident=ident,
                    tau_name=tau_name,
                    tau_fixed=tau_fixed,
                    phi_name=phi_name,
                ) -> csr_matrix:
                    tau = values_map[tau_name] if tau_name else tau_fixed
                    phi = values_map[phi_name]
                    a_ = 1.0 / (1.0 - phi)
                    b_ = -np.sqrt(phi) / (1.0 - phi)
                    d_ = phi / (1.0 - phi)
                    return (tau * bmat([[a_ * ident, b_ * ident],
                                        [b_ * ident, rstar + d_ * ident]], format="csr")).tocsr()
            else:
                vectors, values_ = bym2_spectrum(dict(effect.graph))
                template = _compiled_block(
                    effect.name, build_bym2,
                    frame, effect.name, effect.index, dict(effect.graph),
                    value, float(effect.phi.initial),
                )

                def build(
                    values_map,
                    vectors=vectors,
                    spectrum=values_,
                    tau_name=tau_name,
                    tau_fixed=tau_fixed,
                    phi_name=phi_name,
                ) -> csr_matrix:
                    tau = values_map[tau_name] if tau_name else tau_fixed
                    return bym2_precision(vectors, spectrum, tau, values_map[phi_name])

            params = tuple(name for name in (tau_name, phi_name) if name)
            scalable.append(ParametricBlock(template, params, build))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            parameter_names.append(phi_name)
            parameter_bounds[phi_name] = phi_bounds
            if effect.phi.prior is not None and hasattr(effect.phi.prior, "bind"):
                if augmented:
                    raise NotImplementedError(
                        "a PC prior on BYM2 phi needs the graph spectrum, which the "
                        "large-graph augmented path does not compute; raise "
                        "_BYM2_AUGMENT_NODES to use the dense path, or drop the PC prior"
                    )
                parameter_priors[phi_name] = effect.phi.prior.bind(values_[values_ > 1e-10])
            elif effect.phi.prior is not None:
                parameter_priors[phi_name] = effect.phi.prior
            continue
        if isinstance(effect, AR1):
            rho_is_hp = isinstance(effect.rho, Hyperparameter)
            if not rho_is_hp:
                block = _compiled_block(
                    effect.name, build_ar1,
                    frame, effect.name, effect.index, value, effect.rho, effect.group,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            rho_bounds = _bounded_parameter(effect.rho, -1.0, 1.0, label="AR1 rho", inset=1e-6)
            level_count = len(ordered_observed_levels(frame[effect.index]))
            group_count = (
                1 if effect.group is None else frame[effect.group].astype(str).nunique()
            )
            template = _compiled_block(
                effect.name, build_ar1,
                frame, effect.name, effect.index, value, float(effect.rho.initial),
                effect.group,
            )
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            rho_name = effect.rho.name

            def build(
                values,
                level_count=level_count,
                group_count=group_count,
                tau_name=tau_name,
                tau_fixed=tau_fixed,
                rho_name=rho_name,
            ) -> csr_matrix:
                tau = values[tau_name] if tau_name else tau_fixed
                return csr_matrix(
                    tau * ar1_structure(level_count, values[rho_name], group_count)
                )

            params = tuple(n for n in (tau_name, rho_name) if n)
            scalable.append(ParametricBlock(template, params, build))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            parameter_names.append(rho_name)
            parameter_bounds[rho_name] = rho_bounds
            continue
        if isinstance(effect, SAR):
            if not isinstance(effect.rho, Hyperparameter):
                block = _compiled_block(
                    effect.name, build_sar,
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, value,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            rho_bounds = _bounded_parameter(effect.rho, -1.0, 1.0, label="SAR rho", inset=1e-6)
            nodes, w = normalize_directed_graph(dict(effect.graph))
            w = row_standardize(w)
            template = _compiled_block(
                effect.name, build_sar,
                frame, effect.name, effect.index, dict(effect.graph),
                float(effect.rho.initial), value if not optimized else 1.0,
            )
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            rho_name = effect.rho.name
            wmat = w
            n_nodes = len(nodes)

            def build(values, wmat=wmat, n_nodes=n_nodes, tau_name=tau_name,
                      tau_fixed=tau_fixed, rho_name=rho_name) -> csr_matrix:
                tau = values[tau_name] if tau_name else tau_fixed
                m = _sar_operator(wmat, values[rho_name])
                return _gram_precision(m, tau)

            params = tuple(nm for nm in (tau_name, rho_name) if nm)
            scalable.append(ParametricBlock(template, params, build))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            parameter_names.append(rho_name)
            parameter_bounds[rho_name] = rho_bounds
            continue
        if isinstance(effect, DynamicSpatialPanel):
            graphs = {t: dict(g) for t, g in dict(effect.graphs).items()}
            coeff_is_hp = any(
                isinstance(p, Hyperparameter) for p in (effect.rho, effect.gamma, effect.eta)
            )
            if not coeff_is_hp:
                block = _compiled_block(
                    effect.name, build_dynamic_spatial_panel,
                    frame, effect.name, effect.unit, effect.time, graphs,
                    effect.rho, effect.gamma, effect.eta, value,
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            _, _, ws = _panel_networks(graphs)

            def _coeff(param):
                if isinstance(param, Hyperparameter):
                    return param.name, None
                return None, float(param)

            rho_name, rho_fixed = _coeff(effect.rho)
            gamma_name, gamma_fixed = _coeff(effect.gamma)
            eta_name, eta_fixed = _coeff(effect.eta)
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value

            template = _compiled_block(
                effect.name, build_dynamic_spatial_panel,
                frame, effect.name, effect.unit, effect.time, graphs,
                rho_fixed if rho_fixed is not None else float(effect.rho.initial),
                gamma_fixed if gamma_fixed is not None else float(effect.gamma.initial),
                eta_fixed if eta_fixed is not None else float(effect.eta.initial),
                value if not optimized else 1.0,
            )

            def build(values, ws=ws,
                      rho_name=rho_name, rho_fixed=rho_fixed,
                      gamma_name=gamma_name, gamma_fixed=gamma_fixed,
                      eta_name=eta_name, eta_fixed=eta_fixed,
                      tau_name=tau_name, tau_fixed=tau_fixed) -> csr_matrix:
                rho = values[rho_name] if rho_name else rho_fixed
                gamma = values[gamma_name] if gamma_name else gamma_fixed
                eta = values[eta_name] if eta_name else eta_fixed
                tau = values[tau_name] if tau_name else tau_fixed
                m = _sdpd_operator(ws, rho, gamma, eta)
                return _gram_precision(m, tau)

            params = tuple(nm for nm in (tau_name, rho_name, gamma_name, eta_name) if nm)
            scalable.append(ParametricBlock(template, params, build))
            if rho_name:
                parameter_names.append(rho_name)
                parameter_bounds[rho_name] = _bounded_parameter(
                    effect.rho, -1.0, 1.0, label="SDPD rho", inset=1e-6
                )
            if gamma_name:
                parameter_names.append(gamma_name)
                parameter_bounds[gamma_name] = _real_bounds(effect.gamma)
            if eta_name:
                parameter_names.append(eta_name)
                parameter_bounds[eta_name] = _real_bounds(effect.eta)
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            continue
        if isinstance(effect, MIDAS):
            # Q(tau) = tau * DtD + delta * P0. The delta * P0 term does not scale
            # with tau, so an estimated tau needs a ParametricBlock (rebuild per
            # tau) rather than a ScalableBlock; a fixed tau bakes into one matrix.
            dtd, projector = midas_penalty(len(effect.columns), effect.order)
            delta = effect.ridge
            if not optimized:
                block = _compiled_block(
                    effect.name, build_midas,
                    frame, effect.name, effect.columns, value, effect.order, delta,
                )
                scalable.append(ScalableBlock(block, None, 1.0))
                continue
            tau_name = precision.name
            template = _compiled_block(
                effect.name, build_midas,
                frame, effect.name, effect.columns, float(precision.initial), effect.order, delta,
            )

            def build(values, dtd=dtd, projector=projector, delta=delta, tau_name=tau_name) -> csr_matrix:
                return csr_matrix(values[tau_name] * dtd + delta * projector)

            scalable.append(ParametricBlock(template, (tau_name,), build))
            parameter_names.append(tau_name)
            parameter_bounds[tau_name] = _log_bounds(precision)
            continue
        if isinstance(effect, SpaceTime):
            block = _compiled_block(
                effect.name, build_spacetime,
                frame, effect.name, effect.space, effect.time,
                dict(effect.graph) if effect.graph is not None else None,
                effect.interaction, effect.order, value, effect.scale,
            )
            scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            continue
        if isinstance(effect, IID):
            block = _compiled_block(effect.name, build_iid, frame, effect.name, effect.index, value)
        elif isinstance(effect, (RW1, RW2)):
            order = 1 if isinstance(effect, RW1) else 2
            block = _compiled_block(
                effect.name, build_random_walk, frame, effect.name, effect.index, value, order
            )
        else:
            # As in compile_lgm: never let an unrecognized effect become an RW2.
            raise CompilationError(f"unsupported effect type: {type(effect).__name__}")
        scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
        if optimized:
            parameter_names.append(precision.name)
            parameter_bounds[precision.name] = _log_bounds(precision)

    if isinstance(model.likelihood, Gaussian):
        sigma = model.likelihood.sigma
        if isinstance(sigma, Hyperparameter):
            parameter_names.append(sigma.name)
            parameter_bounds[sigma.name] = _log_bounds(sigma)
            sigma_name = sigma.name

            def factory(resolved: dict, sigma_name: str = sigma_name) -> CompiledGaussian:
                return CompiledGaussian(resolved[sigma_name])
        else:

            def factory(resolved: dict, sigma: float = sigma) -> CompiledGaussian:
                return CompiledGaussian(sigma)
    else:
        likelihood = model.likelihood
        scalar = _estimable_scalar(likelihood)  # NB/Gamma/Beta phi or Weibull shape
        if scalar is not None:
            parameter_names.append(scalar.name)
            parameter_bounds[scalar.name] = _log_bounds(scalar)

        # materialize is lenient (picks the scalar out of the joint mapping); a
        # scalar-less family (Poisson/Bernoulli) ignores `resolved` and returns
        # unchanged. Binomial's trials vector is data, bound onto the compiled
        # likelihood.
        aux = _likelihood_columns(model, frame)

        def factory(resolved: dict, likelihood: object = likelihood, aux=aux) -> object:
            return likelihood.materialize(resolved).for_observations(aux)

    constraint_labels = _qualified_labels([item.block for item in scalable])
    extra_constraints, extra_constraint_rhs = resolve_constraints(
        model.constraints, constraint_labels
    )
    return CompiledFamily(
        y=y,
        observed=panel.observed,
        offset=offset,
        blocks=tuple(scalable),
        parameter_names=tuple(parameter_names),
        likelihood_factory=factory,
        parameter_bounds=parameter_bounds,
        parameter_priors=parameter_priors,
        extra_constraints=extra_constraints,
        extra_constraint_rhs=extra_constraint_rhs,
    )


def build_prediction_context(
    model: "LGM", panel: CanonicalPanel, compiled: CompiledLGM
) -> PredictionContext:
    """Capture what ``result.predict(new_data)`` needs to rebuild a design.

    Structured labels come from the already-compiled blocks so the (sometimes
    expensive) effect builders are not run twice; only each fixed effect's
    formulaic spec is recomputed, which is what reproduces the fitted
    categorical encoding on new rows.
    """
    blocks = {block.name: block for block in compiled.blocks}
    entries: list[tuple[str, object]] = []
    implied_labels: list[str] = []
    for effect in model.predictor.effects:
        block = blocks[effect.name]
        if isinstance(effect, Fixed):
            spec = model_matrix(effect.formula, panel.frame).model_spec
            entries.append(("fixed", spec))
        elif isinstance(effect, MIDAS):
            # No index/one-hot: the design is the raw lag columns, rebuilt directly.
            entries.append(("midas", (effect.name, effect.columns)))
        elif isinstance(effect, MIDASParametric):
            theta_spec = tuple(
                s.name if isinstance(s, Hyperparameter) else float(s)
                for s in (effect.shape1, effect.shape2)
            )
            entries.append(("midas_parametric", (effect.name, effect.columns, effect.kernel, theta_spec)))
        elif isinstance(effect, SpaceTime):
            area_labels = tuple(label.split("|", 1)[0] for label in block.labels)
            time_labels = tuple(label.split("|", 1)[1] for label in block.labels)
            entries.append(
                ("spacetime", (effect.name, effect.space, effect.time,
                               tuple(dict.fromkeys(area_labels)),
                               tuple(dict.fromkeys(time_labels))))
            )
        elif isinstance(effect, DynamicSpatialPanel):
            unit_labels = tuple(dict.fromkeys(label.split("@", 1)[0] for label in block.labels))
            time_labels = tuple(dict.fromkeys(label.split("@", 1)[1] for label in block.labels))
            entries.append(
                ("dynamic_spatial_panel",
                 (effect.name, effect.unit, effect.time, unit_labels, time_labels))
            )
        elif isinstance(effect, AR1) and effect.group is not None:
            group_labels = tuple(dict.fromkeys(la.split("@", 1)[0] for la in block.labels))
            level_labels = tuple(dict.fromkeys(la.split("@", 1)[1] for la in block.labels))
            entries.append(
                ("grouped_structured",
                 (effect.name, effect.group, effect.index, group_labels, level_labels))
            )
        else:
            entries.append(("structured", (effect.name, effect.index, block.labels)))
        implied_labels.extend(f"{block.name}:{label}" for label in block.labels)
    if tuple(implied_labels) != compiled.labels:
        raise CompilationError(
            "prediction context entry order does not match the compiled block "
            "order; this would silently misalign predict() designs"
        )
    return PredictionContext(
        entries=tuple(entries),
        likelihood=compiled.likelihood,
        offset=model.offset,
        trials=model.likelihood.trials if isinstance(model.likelihood, Binomial) else None,
        width=compiled.design.shape[1],
    )

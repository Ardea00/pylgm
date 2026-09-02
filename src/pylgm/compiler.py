import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from formulaic import model_matrix
from formulaic.errors import FormulaicError
from scipy.sparse import bmat, block_diag, csr_matrix, diags, hstack, identity, vstack

from pylgm.config import RunConfig
from pylgm.config.schema import DataConfig, ModelConfig
from pylgm.data import CanonicalPanel
from pylgm.data.scalars import ordered_observed_levels
from pylgm.effects import (
    AR1,
    Besag,
    BYM2,
    Copy,
    DynamicSpatialPanel,
    Fixed,
    IID,
    MIDAS,
    MIDASParametric,
    ProperCAR,
    RW1,
    RW2,
    SAR,
    Seasonal,
    SpaceTime,
    Weighted,
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
    build_seasonal,
    build_spacetime,
    midas_penalty,
    midas_weights,
    seasonal_penalty,
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
from pylgm.inference.prediction import JointPredictionContext, PredictionContext
from pylgm.ir.model import LatentBlock, _block_constraints
from pylgm.joint import Joint, _pad_block_rows
from pylgm.likelihoods import (
    Bernoulli,
    Beta,
    Binomial,
    CompiledGaussian,
    CompiledMixture,
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


def _effect_failure_detail(error: Exception, frame) -> str:
    """Render a builder failure for humans.

    A missing column surfaces as a bare ``KeyError`` whose ``str`` is just the
    quoted column name, which reads as noise ("failed to compile effect 'a':
    'tt'"). Name what was looked up and what is actually there instead.
    """
    if isinstance(error, KeyError) and len(error.args) == 1:
        available = ", ".join(
            str(c) for c in frame.columns if not str(c).startswith("__pylgm")
        )
        return (
            f"column {error.args[0]!r} is not in the data; available columns: {available}"
        )
    return str(error)


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
    # Unwrap Weighted so a spatially-varying-coefficient main effect (or
    # interaction) is still recognized -- weighting changes how the field
    # enters the predictor, not what kind of effect it is.
    unwrapped = [effect.effect if isinstance(effect, Weighted) else effect for effect in effects]
    spatial_indices = {e.index for e in unwrapped if isinstance(e, (Besag, ProperCAR, BYM2))}
    temporal_indices = {e.index for e in unwrapped if isinstance(e, (RW1, RW2, AR1))}
    for effect in unwrapped:
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


def _weight_vector(frame, effect) -> np.ndarray:
    """The validated weight column for a Weighted effect.

    Rejected rather than tolerated: a missing or non-numeric column is a data
    contract error, and an all-zero column makes the effect contribute nothing
    while still consuming latent dimensions, which fits happily and reports a
    field the data never informed.
    """
    if effect.by not in frame.columns:
        raise DataContractError(
            f"weight column {effect.by!r} for effect {effect.name!r} not found"
        )
    column = frame[effect.by]
    if not is_numeric_dtype(column):
        raise DataContractError(
            f"weight column {effect.by!r} for effect {effect.name!r} must be "
            "numeric and finite"
        )
    try:
        values = column.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise DataContractError(
            f"weight column {effect.by!r} for effect {effect.name!r} must be "
            "numeric and finite"
        ) from error
    if not np.isfinite(values).all():
        raise DataContractError(
            f"weight column {effect.by!r} for effect {effect.name!r} must be "
            "numeric and finite"
        )
    if not np.any(values):
        raise CompilationError(
            f"weight column {effect.by!r} for effect {effect.name!r} is all zero, "
            "so the effect cannot contribute to the predictor while still "
            "consuming latent dimensions"
        )
    return values


def _scaled_design_block(block: LatentBlock, weights: np.ndarray) -> LatentBlock:
    """Scale a latent block's design row-wise by ``diag(weights)``.

    The two-site invariant a ``Weighted`` effect relies on: weighting scales
    the design and leaves name, labels, precision and constraints untouched.
    Shared by the compile-side (``_build_effect_block``) and family-side
    (``_weighted_family_block``) weighted-block construction so the two never
    drift apart.
    """
    return LatentBlock(
        block.name,
        block.labels,
        csr_matrix(diags(weights) @ block.design),
        block.precision,
        block.constraints,
    )


def _build_effect_block(effect, frame) -> "tuple[LatentBlock, float | None]":
    """Build one latent block from an effect spec. Shared by compile_lgm and compile_joint.

    Returns ``(block, precision)`` where ``precision`` is the resolved precision
    value to record under the effect's own name, or ``None`` for effects that
    do not carry one (Fixed, MIDASParametric).
    """
    try:
        if isinstance(effect, Weighted):
            block, precision = _build_effect_block(effect.effect, frame)
            weights = _weight_vector(frame, effect)
            return (_scaled_design_block(block, weights), precision)
        elif isinstance(effect, Fixed):
            block = build_fixed(frame, effect.formula, effect.prior_precision)
            precision = None
        elif isinstance(effect, IID):
            precision = _resolved_precision(effect.precision)
            block = build_iid(frame, effect.name, effect.index, precision)
        elif isinstance(effect, Besag):
            precision = _resolved_precision(effect.precision)
            block = build_besag(
                frame, effect.name, effect.index, dict(effect.graph), precision, effect.scale
            )
        elif isinstance(effect, ProperCAR):
            precision = _resolved_precision(effect.precision)
            rho = _resolved_precision(effect.rho)
            block = build_proper_car(
                frame, effect.name, effect.index, dict(effect.graph), rho, precision
            )
        elif isinstance(effect, SAR):
            precision = _resolved_precision(effect.precision)
            rho = _resolved_precision(effect.rho)
            block = build_sar(
                frame, effect.name, effect.index, dict(effect.graph), rho, precision
            )
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
        elif isinstance(effect, BYM2):
            precision = _resolved_precision(effect.precision)
            phi = _resolved_precision(effect.phi) if isinstance(effect.phi, Hyperparameter) else effect.phi
            block = build_bym2(
                frame, effect.name, effect.index, dict(effect.graph), precision, phi
            )
        elif isinstance(effect, AR1):
            precision = _resolved_precision(effect.precision)
            rho = _resolved_precision(effect.rho) if isinstance(effect.rho, Hyperparameter) else effect.rho
            block = build_ar1(
                frame, effect.name, effect.index, precision, rho, effect.group
            )
        elif isinstance(effect, (RW1, RW2)):
            precision = _resolved_precision(effect.precision)
            order = 1 if isinstance(effect, RW1) else 2
            block = build_random_walk(
                frame, effect.name, effect.index, precision, order
            )
        elif isinstance(effect, Seasonal):
            precision = _resolved_precision(effect.precision)
            block = build_seasonal(
                frame, effect.name, effect.index, precision, effect.period, effect.ridge
            )
        elif isinstance(effect, MIDAS):
            precision = _resolved_precision(effect.precision)
            block = build_midas(
                frame, effect.name, effect.columns, precision, effect.order, effect.ridge
            )
        elif isinstance(effect, MIDASParametric):
            precision = None
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
            f"failed to compile effect {effect.name!r}: "
            f"{_effect_failure_detail(error, frame)}"
        ) from error
    return block, precision


def _copy_incidence(frame, copy, labels: "tuple[str, ...]") -> csr_matrix:
    """The incidence A_index of a copy's index over its target block's levels.

    A copy reuses an existing latent field, so every value of its index column
    must already be a level of the target. A value outside that set would mean
    creating a new latent component, which a copy by definition cannot do.
    """
    if copy.index not in frame.columns:
        raise CompilationError(
            f"copy index column {copy.index!r} for target block {copy.name!r} not found"
        )
    position = {label: column for column, label in enumerate(labels)}
    values = frame[copy.index].astype(str)
    unknown = sorted({value for value in values if value not in position})
    if unknown:
        raise CompilationError(
            f"copy of {copy.name!r} indexes level(s) {unknown!r} through column "
            f"{copy.index!r} that the target block does not have. A copy reuses an "
            "existing latent field and cannot add a level to it."
        )
    rows = np.arange(len(frame))
    columns = np.array([position[value] for value in values])
    return csr_matrix(
        (np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(labels))
    )


def _fold_copies(blocks, copies, frame, resolved) -> "list[LatentBlock]":
    """Add each copy's scaled incidence to the columns of the block it names.

    Copies produce no block of their own: ``design == hstack(blocks)`` gives each
    block a disjoint column span, and a copy shares its target's columns by
    definition. Folding here is what keeps that invariant true.
    """
    by_name = {block.name: index for index, block in enumerate(blocks)}
    folded = list(blocks)
    for copy in copies:
        if copy.name not in by_name:
            raise CompilationError(
                f"copy targets block {copy.name!r}, which this model does not "
                f"declare. Declared blocks: {sorted(by_name)!r}"
            )
        position = by_name[copy.name]
        target = folded[position]
        scale = _resolve_scale(copy.scale, resolved, default=1.0)
        incidence = _copy_incidence(frame, copy, target.labels)
        folded[position] = LatentBlock(
            target.name,
            target.labels,
            csr_matrix(target.design + scale * incidence),
            target.precision,
            target.constraints,
        )
    return folded


def _split_copies(effects) -> "tuple[list, list]":
    """Separate ordinary effects from copies, preserving declaration order.

    Several copies may target the same block -- one field entering the predictor
    three or more times, at different indices. They simply fold in turn. A copy
    of a *copy* is not rejected here because it is not expressible: a copy has no
    name of its own, so nothing can reference one.
    """
    ordinary, copies = [], []
    for effect in effects:
        (copies if isinstance(effect, Copy) else ordinary).append(effect)
    return ordinary, copies


def compile_lgm(model: "LGM", panel: CanonicalPanel) -> CompiledLGM:
    """Compile a declarative model through the existing sparse effect builders."""
    if panel.response != model.response:
        raise DataContractError(
            "panel response metadata does not match model: "
            f"{panel.response!r} != {model.response!r}"
        )
    frame = panel.frame
    ordinary, copies = _split_copies(model.predictor.effects)
    blocks: list[LatentBlock] = []
    precisions: dict[str, float] = {}
    for effect in ordinary:
        block, precision = _build_effect_block(effect, frame)
        if precision is not None:
            precisions[effect.name] = precision
        blocks.append(block)
    blocks = _fold_copies(blocks, copies, frame, {})

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


def _offset_vector(model: "LGM", frame) -> np.ndarray:
    if model.offset is None:
        return np.zeros(len(frame))
    if model.offset not in frame.columns:
        raise DataContractError(f"offset column not found: {model.offset!r}")
    values = frame[model.offset].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CompilationError(f"offset column {model.offset!r} must be finite")
    return values


def _shared_incidences(entry, frames, starts, sizes, total, outcomes=()):
    """Per-slice incidence matrices A_k of the shared index over the union of levels.

    The latent spans the union, so a level seen in only some sub-models still
    gets a latent entry -- it is simply informed by fewer rows. That is
    legitimate for genuinely ragged data and a data bug otherwise, so it is
    reported rather than silently absorbed.

    Levels are kept in the column's *original* dtype (not stringified): builders
    that key their natural order off the index (RW1/RW2, Seasonal, AR1,
    SpaceTime's time axis, via ``ordered_observed_levels``) need the real
    int/float/ordered-categorical values to sort or order correctly -- a string
    column sorts lexicographically, silently reordering the latent. Every
    sub-model must agree on that dtype; a genuine mismatch is a data bug, not
    something to coerce silently.
    """
    index = entry.effect.index
    columns = []
    for frame in frames:
        if index not in frame.columns:
            raise CompilationError(
                f"shared effect {entry.name!r} indexes column {index!r}, "
                "which is missing from at least one sub-model's frame"
            )
        columns.append(frame[index])
    for column in columns[1:]:
        if column.dtype != columns[0].dtype:
            raise CompilationError(
                f"shared effect {entry.name!r} indexes column {index!r} with "
                f"inconsistent dtypes across sub-models ({columns[0].dtype!r} vs "
                f"{column.dtype!r}). Align the column's dtype in every sub-model's "
                "frame before sharing it."
            )

    per_frame: list[set] = []
    levels: list = []
    seen_overall = set()
    for column in columns:
        seen = set(column.tolist())
        for value in column.tolist():
            if value not in seen_overall:
                seen_overall.add(value)
                levels.append(value)
        per_frame.append(seen)

    union = set(levels)
    ragged = {
        (outcomes[i] if i < len(outcomes) else str(i)): sorted(union - seen, key=str)
        for i, seen in enumerate(per_frame)
        if union - seen
    }
    if ragged and not entry.allow_ragged:
        detail = "; ".join(
            f"{outcome} is missing {missing[:5]}{'...' if len(missing) > 5 else ''}"
            for outcome, missing in ragged.items()
        )
        raise CompilationError(
            f"shared effect {entry.name!r} has a ragged index {index!r}: {detail}. "
            "The latent spans the union of levels, so this is supported, but it is "
            "reported because an unintended mismatch silently weakens the shared "
            "field. Pass allow_ragged=True on the Shared to accept it."
        )
    position_of = {level: i for i, level in enumerate(levels)}
    incidences = []
    for start, size, frame in zip(starts, sizes, frames):
        rows = np.arange(start, start + size)
        cols = np.array([position_of[v] for v in frame[index].tolist()])
        incidences.append(
            csr_matrix((np.ones(size), (rows, cols)), shape=(total, len(levels)))
        )
    return tuple(levels), incidences


def _resolve_scale(scale, resolved: "dict[str, float]", default: float = 1.0) -> float:
    """Turn a scale entry into a number, given current hyperparameter values.

    ``default`` is the pre-fit fallback for the ("<name>", "inverse") sentinel --
    callers pass the named hyperparameter's own ``initial`` so the sentinel's
    fallback (1/initial) matches the Hyperparameter branch's fallback
    (``scale.initial``) for the same (delta, delta^-1) pairing, instead of a
    bare 1.0 that only happens to agree when initial == 1.0.
    """
    if isinstance(scale, Hyperparameter):
        return float(resolved.get(scale.name, scale.initial))
    if isinstance(scale, tuple):          # the ("<name>", "inverse") sentinel
        name, _ = scale
        return 1.0 / float(resolved.get(name, default))
    return float(scale)


def _effect_hyperparameters(effect) -> list[Hyperparameter]:
    """Every Hyperparameter declared directly on one effect spec.

    Covers ``precision`` (every effect) plus the effect-specific bounded/real
    parameters (``rho``, ``phi``, ``gamma``, ``eta``, the MIDASParametric
    shapes). Shared by ``_model_hyperparameters`` and the shared-effect guard
    in ``_shared_block``, so both stay in sync with which fields can carry one.
    """
    if isinstance(effect, Weighted):
        # Delegate: an unfound inner Hyperparameter would silently pin at its
        # initial value instead of being estimated.
        return _effect_hyperparameters(effect.effect)
    if isinstance(effect, Copy):
        return [effect.scale] if isinstance(effect.scale, Hyperparameter) else []
    found: list[Hyperparameter] = []
    precision = getattr(effect, "precision", None)
    if isinstance(precision, Hyperparameter):
        found.append(precision)
    if isinstance(effect, (ProperCAR, SAR, AR1)) and isinstance(effect.rho, Hyperparameter):
        found.append(effect.rho)
    if isinstance(effect, BYM2) and isinstance(effect.phi, Hyperparameter):
        found.append(effect.phi)
    if isinstance(effect, DynamicSpatialPanel):
        for coeff in (effect.rho, effect.gamma, effect.eta):
            if isinstance(coeff, Hyperparameter):
                found.append(coeff)
    if isinstance(effect, MIDASParametric):
        for shape in (effect.shape1, effect.shape2):
            if isinstance(shape, Hyperparameter):
                found.append(shape)
    return found


def _realign_shared_incidences(entry, levels, template, incidences) -> list:
    """Realign incidence columns to the effect builder's label order.

    The builder is free to reorder/restrict levels (``build_iid`` sorts them
    alphabetically; ``Besag``/``ProperCAR``/``SAR``/``BYM2`` ignore the data
    entirely and take their labels from the graph), but the incidence columns
    above were built off the observed union. Labels are always strings
    (``str(level)``), so compare on that -- comparing raw-dtype ``levels``
    against ``template.labels`` would never match once levels are ints/floats.

    A graph node with no observed row for this shared index (finding I1) is
    ambiguous which column to route it to: raise a clear error rather than
    let ``.index()`` crash with a bare ``ValueError``.
    """
    label_strings = tuple(str(level) for level in levels)
    if template.labels == label_strings:
        return incidences
    missing = [label for label in template.labels if label not in label_strings]
    if missing:
        shown = missing[:5]
        raise CompilationError(
            f"shared effect {entry.name!r} over index {entry.effect.index!r}: its "
            f"latent domain includes {shown}{'...' if len(missing) > 5 else ''}, "
            "which no sub-model observes in that column. A shared spatial effect's "
            "graph must contain exactly the observed levels -- drop the extra "
            "node(s) from the graph, or add rows for them to the data."
        )
    reorder = [label_strings.index(label) for label in template.labels]
    return [incidence[:, reorder] for incidence in incidences]


def _shared_block(entry, joint, frames, starts, sizes, total, resolved) -> LatentBlock:
    """Build the shared latent block: design = sum_k scale_k * A_k over the union index."""
    own_hyperparameters = _effect_hyperparameters(entry.effect)
    if own_hyperparameters:
        names = ", ".join(sorted({hp.name for hp in own_hyperparameters}))
        raise CompilationError(
            f"shared effect {entry.name!r} declares Hyperparameter(s) {names} on its own "
            "precision/rho/phi/gamma/eta/shape -- estimating a shared effect's own "
            "structural parameters is not supported yet. Pass a fixed value for "
            "that field for now; only the Shared `scale` may be a Hyperparameter."
        )
    levels, incidences = _shared_incidences(
        entry, frames, starts, sizes, total, joint.outcomes
    )
    dtype = frames[0][entry.effect.index].dtype
    template, _ = _build_effect_block(
        entry.effect, _levels_frame(entry.effect.index, levels, dtype)
    )
    incidences = _realign_shared_incidences(entry, levels, template, incidences)
    scales = entry.scales_for(len(joint.submodels))
    default = entry.scale.initial if isinstance(entry.scale, Hyperparameter) else 1.0
    design = sum(
        _resolve_scale(scale, resolved, default) * incidence
        for scale, incidence in zip(scales, incidences)
    ).tocsr()
    return LatentBlock(
        entry.name, template.labels, design, template.precision, template.constraints
    )


def _levels_frame(index: str, levels: tuple, dtype=None) -> "pd.DataFrame":
    """A one-row-per-level frame, so the effect builder produces the union-index block.

    ``dtype`` carries the shared column's original dtype (int/float/ordered
    categorical/...) so a builder that orders by the index's natural order
    (``ordered_observed_levels``) sees that order instead of Python's default
    inference, which for an explicit ``list`` of scalars would still be right
    for plain numeric dtypes but would silently drop declared category order.
    """
    return pd.DataFrame({index: pd.Series(list(levels), dtype=dtype)})


def compile_joint(joint: "Joint", panels: "dict[str, CanonicalPanel]") -> CompiledLGM:
    """Compile a Joint into one stacked CompiledLGM.

    Row slices follow sub-model declaration order. Block order is every
    sub-model's private blocks in order, then the shared blocks -- fixed here
    because the prediction contexts assert against it.
    """
    outcomes = joint.outcomes
    frames = [panels[name].frame for name in outcomes]
    sizes = [len(frame) for frame in frames]
    starts, total = [], 0
    for size in sizes:
        starts.append(total)
        total += size

    blocks: list[LatentBlock] = []
    for position, (outcome, model, frame) in enumerate(zip(outcomes, joint.submodels, frames)):
        before, after = starts[position], total - starts[position] - sizes[position]
        for effect in model.predictor.effects:
            try:
                block, _precision = _build_effect_block(effect, frame)
            except CompilationError as error:
                raise CompilationError(f"{error} for outcome {outcome!r}") from error
            named = LatentBlock(
                f"{outcome}:{block.name}", block.labels, block.design,
                block.precision, block.constraints,
            )
            blocks.append(_pad_block_rows(named, before, after))

    for entry in joint.shared:
        blocks.append(
            _shared_block(entry, joint, frames, starts, sizes, total, resolved={})
        )

    y = np.concatenate([
        frame[name].fillna(0.0).to_numpy(dtype=float)
        for name, frame in zip(outcomes, frames)
    ])
    observed = np.concatenate([panels[name].observed for name in outcomes])
    offset = np.concatenate([_offset_vector(model, frame) for model, frame in zip(joint.submodels, frames)])

    parts = []
    for position, (outcome, model, frame) in enumerate(zip(outcomes, joint.submodels, frames)):
        mask = np.zeros(total, dtype=bool)
        mask[starts[position] : starts[position] + sizes[position]] = True
        # Gaussian's estimable scalar is `sigma`, not `phi`/`shape`, so it needs
        # the same special case every other dispatch site in this module gives
        # it (compile_lgm, compile_joint_family's likelihood_factory) --
        # _estimable_scalar only ever resolves phi/shape and would otherwise
        # leave an estimated sigma unresolved, crashing materialize().
        if isinstance(model.likelihood, Gaussian):
            scalar = (
                model.likelihood.sigma
                if isinstance(model.likelihood.sigma, Hyperparameter)
                else None
            )
        else:
            scalar = _estimable_scalar(model.likelihood)
        values = {scalar.name: scalar.initial} if scalar is not None else {}
        compiled = model.likelihood.materialize(values)
        aux = _likelihood_columns(model, frame)
        sub_observed = observed[mask]
        try:
            compiled.for_observations(_slice_aux(aux, sub_observed)).validate_response(
                y[mask][sub_observed]
            )
        except DataContractError as error:
            raise DataContractError(f"{error} for outcome {outcome!r}") from error
        parts.append((mask, compiled.for_observations(aux)))
    likelihood = CompiledMixture(tuple(parts), total)

    width = sum(block.design.shape[1] for block in blocks)
    design = hstack([block.design for block in blocks], format="csr")
    precision = block_diag([block.precision for block in blocks], format="csr")
    constraints = _block_constraints(tuple(blocks), width)
    labels = _qualified_labels(blocks)
    try:
        return CompiledLGM(
            y=y, observed=observed, offset=offset, design=design, precision=precision,
            constraints=constraints, labels=labels, likelihood=likelihood,
            blocks=tuple(blocks),
        )
    except (TypeError, ValueError, ModelValidationError) as error:
        raise CompilationError(f"compiled joint model is invalid: {error}") from error


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
        found.extend((effect.name, hp) for hp in _effect_hyperparameters(effect))
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


def _real_bounds(hp: Hyperparameter, label: str = "real-line parameter") -> OptimizationBounds:
    """Bounds for a real-line (identity-transform) hyperparameter.

    Shared by the exp-Almon MIDAS weight shapes and the SDPD temporal/diffusion
    coefficients, so the caller names the parameter -- the message used to say
    "exp-Almon MIDAS shape" whatever it was handed. The Hyperparameter already
    carries finite lower/upper (defaulted symmetrically) under
    transform='identity'.
    """
    if hp.transform != "identity":
        raise CompilationError(
            f"{label} {hp.name!r} must be declared transform='identity'; "
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


def _weighted_family_block(item, weights: np.ndarray):
    """Apply a Weighted effect's weights to one family block.

    ScalableBlock and ParametricBlock vary only their precision, which
    weighting leaves alone, so scaling the template design is enough.

    A ``ParametricDesignBlock`` is deliberately not handled here: it rebuilds
    its design per hyperparameter draw, and ``_context_with_fitted_weights``
    (``pylgm/model.py``) does not recurse into a ``("weighted", ...)``
    prediction entry, so predict() would silently reuse the *initial* theta
    rather than the fitted one. It is also unreachable today -- the only
    builder that produces one is MIDASParametric, which has no ``.index``, so
    ``Weighted`` already refuses it at construction -- but that combination is
    not a safe capability to half-implement, so it is not handled here either.
    """
    inner = item.block
    scaled = _scaled_design_block(inner, weights)
    if isinstance(item, ParametricBlock):
        return ParametricBlock(scaled, item.parameters, item.build)
    return ScalableBlock(scaled, item.parameter, item.scale)


def _append_family_blocks(
    effect,
    frame,
    scalable: list,
    parameter_names: list[str],
    parameter_bounds: dict,
    parameter_priors: dict,
) -> None:
    """One effect's contribution to a CompiledFamily.

    Extracted verbatim out of compile_family's loop so the Weighted branch can
    build the inner effect through the same path instead of duplicating it. The
    accumulators are mutated in place, exactly as the inline body did.
    """
    if isinstance(effect, Weighted):
        # Routed through _compiled_block so a bad weight column raises the same
        # CompilationError here as it does from _build_effect_block -- without
        # this, an otherwise-identical model raised CompilationError or a raw
        # DataContractError depending only on whether it declared a Hyperparameter.
        weights = _compiled_block(effect.name, _weight_vector, frame, effect)
        first = len(scalable)
        _append_family_blocks(
            effect.effect, frame, scalable, parameter_names, parameter_bounds,
            parameter_priors,
        )
        for position in range(first, len(scalable)):
            scalable[position] = _weighted_family_block(scalable[position], weights)
        return
    if isinstance(effect, Fixed):
        block = _compiled_block(
            effect.name, build_fixed, frame, effect.formula, effect.prior_precision
        )
        scalable.append(ScalableBlock(block, None, 1.0))
        return
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
            return
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
                _log_bounds(shape) if shape.transform == "log"
                else _real_bounds(shape, "exp-Almon MIDAS shape")
            )
            if shape.prior is not None:
                parameter_priors[shape.name] = shape.prior
        return
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
        return
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
            return
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
        return
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
            return
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
        return
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
            return
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
        return
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
            return
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
        return
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
            return
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
            parameter_bounds[gamma_name] = _real_bounds(effect.gamma, "SDPD gamma")
        if eta_name:
            parameter_names.append(eta_name)
            parameter_bounds[eta_name] = _real_bounds(effect.eta, "SDPD eta")
        if optimized:
            parameter_names.append(precision.name)
            parameter_bounds[precision.name] = _log_bounds(precision)
        return
    if isinstance(effect, Seasonal):
        # Q(tau) = tau * StS + delta * P0, exactly like MIDAS below: the
        # delta * P0 term holds the fixed seasonal patterns and does not
        # scale with tau, so an estimated tau needs a ParametricBlock.
        level_count = len(ordered_observed_levels(frame[effect.index]))
        sts, projector = seasonal_penalty(level_count, effect.period)
        delta = effect.ridge
        if not optimized:
            block = _compiled_block(
                effect.name, build_seasonal,
                frame, effect.name, effect.index, value, effect.period, delta,
            )
            scalable.append(ScalableBlock(block, None, 1.0))
            return
        tau_name = precision.name
        template = _compiled_block(
            effect.name, build_seasonal,
            frame, effect.name, effect.index, float(precision.initial),
            effect.period, delta,
        )

        def build(values, sts=sts, projector=projector, delta=delta,
                  tau_name=tau_name) -> csr_matrix:
            return csr_matrix(values[tau_name] * sts + delta * projector)

        scalable.append(ParametricBlock(template, (tau_name,), build))
        parameter_names.append(tau_name)
        parameter_bounds[tau_name] = _log_bounds(precision)
        return
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
            return
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
        return
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
        return
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


def _copied_family_block(item, copy, incidence):
    """Fold a copy into one family block, rebuilding per draw when needed.

    A fixed scale bakes into the template. An estimated scale makes the design
    a function of the hyperparameter, so it must be re-formed on every draw --
    folding only into the template would silently drop the copy from every draw
    after the first.
    """
    inner = item.block
    fixed = not isinstance(copy.scale, Hyperparameter)
    if isinstance(item, ParametricBlock) and not fixed:
        raise CompilationError(
            f"copy of {copy.name!r} has an estimated scale, but that block's "
            "precision is itself a function of hyperparameters. Combining an "
            "estimated copy scale with an estimated structural parameter on the "
            "same block is not supported; fix one of them."
        )
    baked = LatentBlock(
        inner.name,
        inner.labels,
        csr_matrix(inner.design + _resolve_scale(copy.scale, {}, default=1.0) * incidence),
        inner.precision,
        inner.constraints,
    )
    if fixed:
        if isinstance(item, ParametricDesignBlock):
            def build(values, inner_build=item.build, incidence=incidence,
                      scale=float(copy.scale)):
                return csr_matrix(inner_build(values) + scale * incidence)

            return ParametricDesignBlock(baked, item.parameters, build)
        if isinstance(item, ParametricBlock):
            return ParametricBlock(baked, item.parameters, item.build)
        return ScalableBlock(baked, item.parameter, item.scale)

    name = copy.scale.name
    base_design = inner.design
    inner_build = item.build if isinstance(item, ParametricDesignBlock) else None
    parameters = tuple(dict.fromkeys((item.parameters if isinstance(
        item, (ParametricBlock, ParametricDesignBlock)) else ()) + (name,)))

    def build(values, base_design=base_design, incidence=incidence, name=name,
              inner_build=inner_build):
        design = inner_build(values) if inner_build is not None else base_design
        return csr_matrix(design + float(values[name]) * incidence)

    return ParametricDesignBlock(baked, parameters, build)


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
    ordinary, copies = _split_copies(model.predictor.effects)
    for effect in ordinary:
        _append_family_blocks(
            effect, frame, scalable, parameter_names, parameter_bounds, parameter_priors
        )
    for copy in copies:
        matches = [k for k, item in enumerate(scalable) if item.block.name == copy.name]
        if not matches:
            raise CompilationError(
                f"copy targets block {copy.name!r}, which this model does not "
                f"declare. Declared blocks: "
                f"{sorted(item.block.name for item in scalable)!r}"
            )
        position = matches[0]
        incidence = _copy_incidence(frame, copy, scalable[position].block.labels)
        scalable[position] = _copied_family_block(scalable[position], copy, incidence)
        if isinstance(copy.scale, Hyperparameter):
            parameter_names.append(copy.scale.name)
            # Mirrors the exp-Almon MIDAS shape dispatch: a copy scale defaults to
            # transform="log" (Hyperparameter's own default, positive-only), and
            # _real_bounds only accepts "identity" -- so dispatch on the declared
            # transform instead of assuming one.
            parameter_bounds[copy.scale.name] = (
                _log_bounds(copy.scale) if copy.scale.transform == "log"
                else _real_bounds(copy.scale, "copy scale")
            )
            if copy.scale.prior is not None:
                parameter_priors[copy.scale.name] = copy.scale.prior

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


def _prediction_entry(effect, model: "LGM", panel: CanonicalPanel, block: LatentBlock):
    """The predict-time descriptor for one effect.

    Extracted out of ``build_prediction_context`` so ``build_joint_prediction_contexts``
    shares the exact same effect-to-entry dispatch -- the same de-duplication
    Task 4 did for ``_build_effect_block``.
    """
    if isinstance(effect, Weighted):
        # Nest rather than special-case: later modifier wrappers reuse this same
        # rule instead of adding a case per combination.
        #
        # NOTE for the next wrapper: passing the *outer* block down is only
        # sound because Weighted preserves the inner effect's labels verbatim.
        # Replicated/Grouped re-label to (replicate, level) / (group, level)
        # pairs, and several branches below (spacetime, dynamic_spatial_panel,
        # grouped_structured, the fallback "structured" case) derive their
        # level tuples straight from block.labels -- so a re-labelling wrapper
        # must pass its own relabelled block into the recursive call here, not
        # the block it received.
        return ("weighted", (_prediction_entry(effect.effect, model, panel, block), effect.by))
    if isinstance(effect, Fixed):
        spec = model_matrix(effect.formula, panel.frame).model_spec
        return ("fixed", spec)
    if isinstance(effect, MIDAS):
        # No index/one-hot: the design is the raw lag columns, rebuilt directly.
        return ("midas", (effect.name, effect.columns))
    if isinstance(effect, MIDASParametric):
        theta_spec = tuple(
            s.name if isinstance(s, Hyperparameter) else float(s)
            for s in (effect.shape1, effect.shape2)
        )
        return ("midas_parametric", (effect.name, effect.columns, effect.kernel, theta_spec))
    if isinstance(effect, SpaceTime):
        area_labels = tuple(label.split("|", 1)[0] for label in block.labels)
        time_labels = tuple(label.split("|", 1)[1] for label in block.labels)
        return (
            "spacetime",
            (effect.name, effect.space, effect.time,
             tuple(dict.fromkeys(area_labels)), tuple(dict.fromkeys(time_labels))),
        )
    if isinstance(effect, DynamicSpatialPanel):
        unit_labels = tuple(dict.fromkeys(label.split("@", 1)[0] for label in block.labels))
        time_labels = tuple(dict.fromkeys(label.split("@", 1)[1] for label in block.labels))
        return (
            "dynamic_spatial_panel",
            (effect.name, effect.unit, effect.time, unit_labels, time_labels),
        )
    if isinstance(effect, AR1) and effect.group is not None:
        group_labels = tuple(dict.fromkeys(la.split("@", 1)[0] for la in block.labels))
        level_labels = tuple(dict.fromkeys(la.split("@", 1)[1] for la in block.labels))
        return (
            "grouped_structured",
            (effect.name, effect.group, effect.index, group_labels, level_labels),
        )
    return ("structured", (effect.name, effect.index, block.labels))


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
    # A copy produces no block of its own -- it folds into its target's design
    # (_fold_copies) -- so it carries no separate prediction entry either; a
    # naive loop over every effect would look its target block up a second
    # time and duplicate that block's labels here.
    ordinary, _copies = _split_copies(model.predictor.effects)
    for effect in ordinary:
        block = blocks[effect.name]
        entries.append(_prediction_entry(effect, model, panel, block))
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


def build_joint_prediction_contexts(joint: "Joint", panels, compiled: CompiledLGM, result):
    """One PredictionContext per outcome, each spanning the full stacked latent.

    Block order is fixed by compile_joint: every sub-model's private blocks in
    declaration order, then the shared blocks. The column span of each block is
    read back off `compiled.blocks` so the two cannot drift apart silently.
    """
    spans, cursor = {}, 0
    for block in compiled.blocks:
        width = block.design.shape[1]
        spans[block.name] = (cursor, cursor + width)
        cursor += width

    fitted = dict(result.hyperparameters or {})
    contexts = {}
    for outcome, model in zip(joint.outcomes, joint.submodels):
        panel = panels[outcome]
        entries, slices, used_blocks = [], [], []

        for effect in model.predictor.effects:
            block = next(
                b for b in compiled.blocks if b.name == f"{outcome}:{effect.name}"
            )
            entries.append(_prediction_entry(effect, model, panel, block))
            slices.append(spans[block.name])
            used_blocks.append(block.name)

        for entry in joint.shared:
            block = next(b for b in compiled.blocks if b.name == entry.name)
            scales = entry.scales_for(len(joint.submodels))
            scale = scales[joint.outcomes.index(outcome)]
            if isinstance(scale, Hyperparameter):
                spec, value = scale.name, float(fitted.get(scale.name, scale.initial))
            elif isinstance(scale, tuple):            # ("<name>", "inverse")
                name, _ = scale
                default = entry.scale.initial if isinstance(entry.scale, Hyperparameter) else 1.0
                spec, value = name, 1.0 / float(fitted.get(name, default))
            else:
                spec, value = float(scale), float(scale)
            entries.append(
                ("shared", (entry.name, entry.effect.index, block.labels, spec, value))
            )
            slices.append(spans[block.name])
            used_blocks.append(block.name)

        # Cross-check against compiled.blocks independently of how `used_blocks`
        # was assembled: every block compiled for this outcome (its own
        # `outcome:`-prefixed private blocks, plus every shared block) must be
        # referenced exactly once. A predictor effect silently missing its
        # block -- or a compiled block never picked up by any effect -- would
        # otherwise leave a column span untouched (implicitly zero) in the
        # design with nothing else to flag it.
        expected_blocks = {b.name for b in compiled.blocks if b.name.startswith(f"{outcome}:")}
        expected_blocks |= {entry.name for entry in joint.shared}
        if set(used_blocks) != expected_blocks or len(used_blocks) != len(expected_blocks):
            raise CompilationError(
                f"joint prediction context for outcome {outcome!r} does not match "
                "the compiled blocks for that outcome; this would silently misalign "
                "predict()"
            )

        contexts[outcome] = PredictionContext(
            entries=tuple(entries),
            likelihood=_submodel_likelihood(model, panel, fitted),
            offset=model.offset,
            trials=model.likelihood.trials if isinstance(model.likelihood, Binomial) else None,
            width=compiled.design.shape[1],
            column_slices=tuple(slices),
        )
    return JointPredictionContext(contexts)


def _submodel_likelihood(model: "LGM", panel: CanonicalPanel, fitted: dict):
    """That sub-model's own compiled likelihood at the fitted scalar.

    Per-outcome prediction never uses the mixture: new_data is homogeneous, so
    response_prediction, trials and survival aux behave exactly as they do for a
    single-response model.
    """
    # Gaussian's estimable scalar is `sigma`, not `phi`/`shape`, so it needs the
    # same special case every dispatch site in this module gives it (compile_lgm,
    # compile_joint_family's likelihood_factory) -- _estimable_scalar only ever
    # resolves phi/shape and would otherwise leave an estimated sigma unresolved,
    # crashing materialize().
    if isinstance(model.likelihood, Gaussian):
        scalar = model.likelihood.sigma if isinstance(model.likelihood.sigma, Hyperparameter) else None
    else:
        scalar = _estimable_scalar(model.likelihood)
    values = (
        {scalar.name: float(fitted[scalar.name])}
        if scalar is not None and scalar.name in fitted
        else ({scalar.name: scalar.initial} if scalar is not None else {})
    )
    return model.likelihood.materialize(values)


def _restack_family_block(item, outcome: str, before: int, after: int):
    """Rename and row-pad one family block from a sub-model's CompiledFamily.

    ScalableBlock and ParametricBlock vary only their *precision* with the
    hyperparameters, which is row-independent, so padding the template is
    enough. ParametricDesignBlock rebuilds a *design* over the sub-frame's rows,
    so its build output must be padded on every draw too.
    """
    inner = item.block
    named = LatentBlock(
        f"{outcome}:{inner.name}", inner.labels, inner.design,
        inner.precision, inner.constraints,
    )
    padded = _pad_block_rows(named, before, after)

    if isinstance(item, ParametricDesignBlock):
        def build(values, inner_build=item.build, before=before, after=after,
                  width=inner.design.shape[1]):
            design = inner_build(values)
            pieces = []
            if before:
                pieces.append(csr_matrix((before, width)))
            pieces.append(design)
            if after:
                pieces.append(csr_matrix((after, width)))
            return vstack(pieces, format="csr") if len(pieces) > 1 else design

        return ParametricDesignBlock(padded, item.parameters, build)

    if isinstance(item, ParametricBlock):
        return ParametricBlock(padded, item.parameters, item.build)

    return ScalableBlock(padded, item.parameter, item.scale)


def compile_joint_family(joint: "Joint", panels: "dict[str, CanonicalPanel]") -> CompiledFamily | None:
    """Family form of compile_joint: rebuild scale-dependent designs per draw."""
    outcomes = joint.outcomes
    frames = [panels[name].frame for name in outcomes]
    sizes = [len(frame) for frame in frames]
    starts, total = [], 0
    for size in sizes:
        starts.append(total)
        total += size

    scalable: list = []
    parameter_names: list[str] = []
    parameter_bounds: dict[str, OptimizationBounds] = {}
    parameter_priors: dict[str, object] = {}
    # Tracks, per registered name, the exact Hyperparameter declaration that
    # claimed it -- the same object reused across two Shared entries (the
    # (delta, delta^-1) shorthand, or an explicit tuple) dedups; a *different*
    # declaration reusing the name (a sub-model hyperparameter and an unrelated
    # shared scale, say) is a collision and must raise, not silently alias.
    registered: dict[str, Hyperparameter] = {}

    # Reuse compile_family per sub-model rather than duplicating its 130-line
    # effect chain, then pad and rename what it produced. A sub-model with no
    # declared Hyperparameter returns None, in which case its blocks are plain
    # ScalableBlocks built from the compile_joint path.
    for position, (outcome, model, frame) in enumerate(zip(outcomes, joint.submodels, frames)):
        before, after = starts[position], total - starts[position] - sizes[position]
        sub_family = compile_family(model, panels[outcome])
        if sub_family is None:
            for effect in model.predictor.effects:
                block, _ = _build_effect_block(effect, frame)
                named = LatentBlock(
                    f"{outcome}:{block.name}", block.labels, block.design,
                    block.precision, block.constraints,
                )
                scalable.append(ScalableBlock(_pad_block_rows(named, before, after), None, 1.0))
            continue

        for item in sub_family.blocks:
            scalable.append(_restack_family_block(item, outcome, before, after))

        sub_hyperparameters = {hp.name: hp for _, hp in _model_hyperparameters(model)}
        for name in sub_family.parameter_names:
            if name in parameter_names:
                raise CompilationError(
                    f"hyperparameter name {name!r} is declared by more than one "
                    "sub-model. Joint sub-models share one hyperparameter namespace, "
                    "so give each its own name (e.g. 'tau_oral', 'tau_larynx')."
                )
            parameter_names.append(name)
            registered[name] = sub_hyperparameters[name]
            if name in sub_family.parameter_bounds:
                parameter_bounds[name] = sub_family.parameter_bounds[name]
            if name in sub_family.parameter_priors:
                parameter_priors[name] = sub_family.parameter_priors[name]

    for entry in joint.shared:
        scales = entry.scales_for(len(joint.submodels))
        estimated = [s for s in scales if isinstance(s, Hyperparameter)]
        template = _shared_block(entry, joint, frames, starts, sizes, total, resolved={})
        if not estimated:
            scalable.append(ScalableBlock(template, None, 1.0))
            continue

        levels, incidences = _shared_incidences(
            entry, frames, starts, sizes, total, joint.outcomes
        )
        # Same realignment _shared_block applies internally.
        incidences = _realign_shared_incidences(entry, levels, template, incidences)
        default = entry.scale.initial if isinstance(entry.scale, Hyperparameter) else 1.0

        def build(values, scales=scales, incidences=incidences, default=default):
            return sum(
                _resolve_scale(scale, values, default) * incidence
                for scale, incidence in zip(scales, incidences)
            ).tocsr()

        names = tuple(dict.fromkeys(s.name for s in estimated))
        scalable.append(ParametricDesignBlock(template, names, build))
        for hyper in estimated:
            existing = registered.get(hyper.name)
            if existing is not None:
                if existing is hyper:
                    continue
                raise CompilationError(
                    f"hyperparameter name {hyper.name!r} is declared by more than one "
                    "sub-model/shared entry, with a different Hyperparameter object "
                    "for each. Joint hyperparameters share one namespace, so give "
                    "each its own name (e.g. 'tau_oral', 'tau_larynx')."
                )
            registered[hyper.name] = hyper
            parameter_names.append(hyper.name)
            parameter_bounds[hyper.name] = _log_bounds(hyper)
            if hyper.prior is not None:
                parameter_priors[hyper.name] = hyper.prior

    if not parameter_names:
        return None

    y = np.concatenate([
        frame[name].fillna(0.0).to_numpy(dtype=float)
        for name, frame in zip(outcomes, frames)
    ])
    observed = np.concatenate([panels[name].observed for name in outcomes])
    offset = np.concatenate([
        _offset_vector(model, frame) for model, frame in zip(joint.submodels, frames)
    ])

    masks = []
    for position in range(len(outcomes)):
        mask = np.zeros(total, dtype=bool)
        mask[starts[position] : starts[position] + sizes[position]] = True
        masks.append(mask)

    def likelihood_factory(values, masks=masks, submodels=joint.submodels, frames=frames):
        """Rebuild the mixture at the current hyperparameter values.

        Each sub-model's estimable scalar (Gaussian sigma, NegBin/Gamma/Beta phi,
        Weibull alpha) is resolved from `values` if it is optimised, else left at
        its fixed value -- the same resolution compile_lgm does at its initial.
        """
        parts = []
        for mask, model, frame in zip(masks, submodels, frames):
            # Gaussian's estimable scalar is `sigma`, not `phi`/`shape`, so it
            # needs the same special case every other dispatch site in this
            # module gives it (compile_lgm, compile_family) -- _estimable_scalar
            # only ever resolves phi/shape and would otherwise leave an
            # estimated sigma unresolved, crashing materialize().
            if isinstance(model.likelihood, Gaussian):
                scalar = model.likelihood.sigma if isinstance(model.likelihood.sigma, Hyperparameter) else None
            else:
                scalar = _estimable_scalar(model.likelihood)
            resolved = (
                {scalar.name: float(values[scalar.name])}
                if scalar is not None and scalar.name in values
                else ({scalar.name: scalar.initial} if scalar is not None else {})
            )
            compiled = model.likelihood.materialize(resolved)
            parts.append((mask, compiled.for_observations(_likelihood_columns(model, frame))))
        return CompiledMixture(tuple(parts), total)

    return CompiledFamily(
        y=y,
        observed=observed,
        offset=offset,
        blocks=tuple(scalable),
        parameter_names=tuple(parameter_names),
        likelihood_factory=likelihood_factory,
        parameter_bounds=parameter_bounds,
        parameter_priors=parameter_priors,
    )

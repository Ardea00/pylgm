from typing import TYPE_CHECKING

import numpy as np
from formulaic.errors import FormulaicError
from scipy.sparse import block_diag, csr_matrix, diags, hstack

from pylgm.config import RunConfig
from pylgm.config.schema import DataConfig, ModelConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import (
    Besag,
    BYM2,
    Fixed,
    IID,
    ProperCAR,
    RW1,
    RW2,
    build_besag,
    build_bym2,
    build_fixed,
    build_iid,
    build_proper_car,
    build_random_walk,
    normalize_graph,
)
from pylgm.effects.bym2 import bym2_precision, bym2_spectrum
from pylgm.effects.proper_car import car_rho_interval
from pylgm.exceptions import CompilationError, DataContractError, ModelValidationError
from pylgm.ir import (
    CompiledFamily,
    CompiledGaussianFamily,
    CompiledLGM,
    Hyperparameters,
    ParametricBlock,
    ScalableBlock,
)
from pylgm.ir.model import LatentBlock, _block_constraints
from pylgm.likelihoods import Bernoulli, CompiledGaussian, Gaussian, Poisson
from pylgm.optimization.empirical_bayes import OptimizationBounds
from pylgm.optimization.transforms import LogitTransform, LogTransform
from pylgm.parameters import Hyperparameter

if TYPE_CHECKING:
    from pylgm.model import LGM


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
                block = build_proper_car(
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, precision
                )
                precisions[effect.name] = precision
            elif isinstance(effect, BYM2):
                precision = _resolved_precision(effect.precision)
                phi = _resolved_precision(effect.phi) if isinstance(effect.phi, Hyperparameter) else effect.phi
                block = build_bym2(
                    frame, effect.name, effect.index, dict(effect.graph), precision, phi
                )
                precisions[effect.name] = precision
            elif isinstance(effect, (RW1, RW2)):
                precision = _resolved_precision(effect.precision)
                order = 1 if isinstance(effect, RW1) else 2
                block = build_random_walk(
                    frame, effect.name, effect.index, precision, order
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

    wrong_rows = [block.name for block in blocks if block.design.shape[0] != len(frame)]
    if wrong_rows:
        raise CompilationError(
            f"latent block design row count does not match the panel: {wrong_rows}"
        )
    labels = _qualified_labels(blocks)

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
            )
            return family.materialize({})
        except (TypeError, ValueError) as error:
            raise CompilationError(f"compiled declarative model is invalid: {error}") from error

    if isinstance(model.likelihood, (Poisson, Bernoulli)):
        compiled_likelihood = model.likelihood.materialize({})
        observed = panel.observed
        compiled_likelihood.validate_response(y[observed])
        if not blocks:
            raise CompilationError("model must contain at least one latent effect")
        width = sum(block.design.shape[1] for block in blocks)
        design = hstack([block.design for block in blocks], format="csr")
        precision = block_diag([block.precision for block in blocks], format="csr")
        constraints = _block_constraints(tuple(blocks), width)
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
            )
        except (TypeError, ValueError, ModelValidationError) as error:
            raise CompilationError(f"compiled declarative model is invalid: {error}") from error

    raise CompilationError("unsupported likelihood for declarative compilation")


def _model_hyperparameters(model: "LGM") -> list[tuple[str, Hyperparameter]]:
    """Return (target_name, Hyperparameter) pairs declared on the model."""
    found: list[tuple[str, Hyperparameter]] = []
    if isinstance(model.likelihood, Gaussian) and isinstance(model.likelihood.sigma, Hyperparameter):
        found.append(("sigma", model.likelihood.sigma))
    for effect in model.predictor.effects:
        precision = getattr(effect, "precision", None)
        if isinstance(precision, Hyperparameter):
            found.append((effect.name, precision))
        if isinstance(effect, ProperCAR) and isinstance(effect.rho, Hyperparameter):
            found.append((effect.name, effect.rho))
        if isinstance(effect, BYM2) and isinstance(effect.phi, Hyperparameter):
            found.append((effect.name, effect.phi))
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
            "to an effect that supplies a bounded interval; only a proper CAR rho "
            "resolves its interval (from the graph)"
        )
    return OptimizationBounds(hp.initial, hp.lower, hp.upper, transform=LogTransform())


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

    scalable: list[ScalableBlock | ParametricBlock] = []
    parameter_names: list[str] = []
    parameter_bounds: dict[str, OptimizationBounds] = {}
    parameter_priors: dict[str, object] = {}
    for effect in model.predictor.effects:
        if isinstance(effect, Fixed):
            block = build_fixed(frame, effect.formula, effect.prior_precision)
            scalable.append(ScalableBlock(block, None, 1.0))
            continue
        precision = effect.precision
        optimized = isinstance(precision, Hyperparameter)
        value = 1.0 if optimized else precision
        if isinstance(effect, Besag):
            block = build_besag(
                frame, effect.name, effect.index, dict(effect.graph), value, effect.scale
            )
            scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
            if optimized:
                parameter_names.append(precision.name)
                parameter_bounds[precision.name] = _log_bounds(precision)
            continue
        if isinstance(effect, ProperCAR):
            rho_is_hp = isinstance(effect.rho, Hyperparameter)
            if not rho_is_hp:
                block = build_proper_car(
                    frame, effect.name, effect.index, dict(effect.graph), effect.rho, value
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
            if effect.rho.transform != "logit":
                # A log-transform rho would silently inherit positive-only default
                # bounds and be confined to a wrong, one-sided interval.
                raise CompilationError(
                    f"proper CAR rho {effect.rho.name!r} must be declared with "
                    f"transform='logit' (its interval ({a:.6g}, {b:.6g}) is resolved "
                    f"from the graph); got transform={effect.rho.transform!r}"
                )
            inset = 1e-6 * (b - a)
            rho_lower = a + inset if effect.rho.lower is None else max(effect.rho.lower, a + inset)
            rho_upper = b - inset if effect.rho.upper is None else min(effect.rho.upper, b - inset)
            if not rho_lower <= effect.rho.initial <= rho_upper:
                raise CompilationError(
                    f"initial value for proper CAR rho {effect.rho.name!r} must lie in "
                    f"({a:.6g}, {b:.6g}), the graph's positive-definiteness interval; "
                    f"got {effect.rho.initial}"
                )
            rho_initial = float(effect.rho.initial)
            template = build_proper_car(
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
            parameter_bounds[rho_name] = OptimizationBounds(
                rho_initial, rho_lower, rho_upper, transform=LogitTransform(a, b)
            )
            continue
        if isinstance(effect, BYM2):
            phi_is_hp = isinstance(effect.phi, Hyperparameter)
            if not phi_is_hp:
                block = build_bym2(
                    frame, effect.name, effect.index, dict(effect.graph), value, effect.phi
                )
                scalable.append(ScalableBlock(block, precision.name if optimized else None, 1.0))
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                continue
            # phi is a Hyperparameter -> a ParametricBlock over (tau, phi), bounded
            # to (0, 1) via a Logit transform (BYM2 is unconstrained: no graph
            # interval to resolve, unlike proper CAR's rho).
            if effect.phi.transform != "logit":
                raise CompilationError(
                    f"BYM2 phi {effect.phi.name!r} must be declared with transform='logit' "
                    f"(it mixes on (0, 1)); got transform={effect.phi.transform!r}"
                )
            vectors, values_ = bym2_spectrum(dict(effect.graph))
            inset = 1e-6
            phi_lower = inset if effect.phi.lower is None else max(effect.phi.lower, inset)
            phi_upper = 1.0 - inset if effect.phi.upper is None else min(effect.phi.upper, 1.0 - inset)
            if not phi_lower <= effect.phi.initial <= phi_upper:
                raise CompilationError(
                    f"initial value for BYM2 phi {effect.phi.name!r} must lie in "
                    f"({phi_lower:.6g}, {phi_upper:.6g}); got {effect.phi.initial}"
                )
            template = build_bym2(
                frame, effect.name, effect.index, dict(effect.graph),
                value, float(effect.phi.initial),
            )
            tau_name = precision.name if optimized else None
            tau_fixed = None if optimized else value
            phi_name = effect.phi.name

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
            parameter_bounds[phi_name] = OptimizationBounds(
                float(effect.phi.initial), phi_lower, phi_upper,
                transform=LogitTransform(0.0, 1.0),
            )
            if effect.phi.prior is not None and hasattr(effect.phi.prior, "bind"):
                positive = values_[values_ > 1e-10]
                parameter_priors[phi_name] = effect.phi.prior.bind(positive)
            elif effect.phi.prior is not None:
                parameter_priors[phi_name] = effect.phi.prior
            continue
        if isinstance(effect, IID):
            block = build_iid(frame, effect.name, effect.index, value)
        elif isinstance(effect, (RW1, RW2)):
            order = 1 if isinstance(effect, RW1) else 2
            block = build_random_walk(frame, effect.name, effect.index, value, order)
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

        def factory(resolved: dict, likelihood: object = likelihood) -> object:
            return likelihood.materialize({})

    return CompiledFamily(
        y=y,
        observed=panel.observed,
        offset=offset,
        blocks=tuple(scalable),
        parameter_names=tuple(parameter_names),
        likelihood_factory=factory,
        parameter_bounds=parameter_bounds,
        parameter_priors=parameter_priors,
    )

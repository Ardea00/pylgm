from typing import TYPE_CHECKING

import numpy as np
from formulaic.errors import FormulaicError
from scipy.sparse import block_diag, hstack

from pylgm.config import RunConfig
from pylgm.config.schema import DataConfig, ModelConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import Fixed, IID, RW1, build_fixed, build_iid, build_random_walk
from pylgm.exceptions import CompilationError, DataContractError, ModelValidationError
from pylgm.ir import CompiledGaussianFamily, CompiledLGM, Hyperparameters, ScalableBlock
from pylgm.ir.model import LatentBlock, _block_constraints
from pylgm.likelihoods import Bernoulli, Gaussian, Poisson
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
            else:
                precision = _resolved_precision(effect.precision)
                order = 1 if isinstance(effect, RW1) else 2
                block = build_random_walk(
                    frame, effect.name, effect.index, precision, order
                )
                precisions[effect.name] = precision
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

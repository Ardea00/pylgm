from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.sparse import block_diag, hstack

from pylgm.exceptions import ModelValidationError
from pylgm.ir.model import CompiledLGM, LatentBlock


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _ordinary_positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ModelValidationError(f"{name} must be a finite positive real number")
    return float(value)


def _validate_parameter_mapping(
    values: Mapping[str, float], parameter_names: tuple[str, ...]
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ModelValidationError("optimized values must be a mapping")
    supplied = set(values)
    expected = set(parameter_names)
    if supplied != expected:
        missing = tuple(name for name in parameter_names if name not in supplied)
        unknown = tuple(name for name in values if name not in expected)
        raise ModelValidationError(
            f"optimized values must contain exactly {parameter_names!r}; "
            f"missing={missing}, unknown={unknown}"
        )
    return {
        name: _ordinary_positive(values[name], f"optimized parameter {name!r}")
        for name in parameter_names
    }


@dataclass(frozen=True, init=False)
class Hyperparameters:
    sigma: float
    precisions: Mapping[str, float]

    def __init__(self, sigma: float, precisions: Mapping[str, float]) -> None:
        sigma_value = _ordinary_positive(sigma, "sigma")
        if not isinstance(precisions, Mapping):
            raise ModelValidationError("precisions must be a mapping")
        isolated: dict[str, float] = {}
        for name, precision in precisions.items():
            if not isinstance(name, str) or not name:
                raise ModelValidationError("precision names must be non-empty strings")
            isolated[name] = _ordinary_positive(precision, f"precision {name!r}")
        object.__setattr__(self, "sigma", sigma_value)
        object.__setattr__(self, "precisions", MappingProxyType(isolated))


@dataclass(frozen=True, init=False)
class ScalableBlock:
    block: LatentBlock
    parameter: str | None
    scale: float

    def __init__(
        self, block: LatentBlock, parameter: str | None, scale: float
    ) -> None:
        if not isinstance(block, LatentBlock):
            raise ModelValidationError("scalable block must contain a latent block")
        if parameter is not None and (not isinstance(parameter, str) or not parameter):
            raise ModelValidationError("scalable block parameter must be a non-empty string")
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "scale", _ordinary_positive(scale, "block scale"))


def _qualified_labels(blocks: tuple[LatentBlock, ...]) -> tuple[str, ...]:
    labels = tuple(
        f"{block.name}:{label}" for block in blocks for label in block.labels
    )
    if len(labels) != len(set(labels)):
        raise ModelValidationError("duplicate latent labels after qualification")
    return labels


def _assemble_compiled_model(
    y: np.ndarray,
    observed: np.ndarray,
    offset: np.ndarray,
    blocks: tuple[LatentBlock, ...],
    sigma: float,
) -> CompiledLGM:
    widths = [block.design.shape[1] for block in blocks]
    total = sum(widths)
    constraint_rows: list[np.ndarray] = []
    start = 0
    for block, width in zip(blocks, widths, strict=True):
        for local in block.constraints:
            row = np.zeros(total)
            row[start : start + width] = local
            constraint_rows.append(row)
        start += width
    constraints = (
        np.vstack(constraint_rows) if constraint_rows else np.empty((0, total))
    )
    return CompiledLGM(
        y=y,
        observed=observed,
        offset=offset,
        design=hstack([block.design for block in blocks], format="csr"),
        precision=block_diag([block.precision for block in blocks], format="csr"),
        constraints=constraints,
        labels=_qualified_labels(blocks),
        sigma=sigma,
        blocks=blocks,
    )


@dataclass(frozen=True, init=False)
class CompiledGaussianFamily:
    _y: np.ndarray = field(repr=False)
    _observed: np.ndarray = field(repr=False)
    _offset: np.ndarray = field(repr=False)
    blocks: tuple[ScalableBlock, ...]
    parameter_names: tuple[str, ...]
    initial: Hyperparameters

    def __init__(
        self,
        y: np.ndarray,
        observed: np.ndarray,
        offset: np.ndarray,
        blocks: tuple[ScalableBlock, ...],
        parameter_names: tuple[str, ...],
        initial: Hyperparameters,
    ) -> None:
        y_value = np.asarray(y)
        observed_value = np.asarray(observed)
        offset_value = np.asarray(offset)
        if (
            y_value.ndim != 1
            or not np.issubdtype(y_value.dtype, np.number)
            or not np.isrealobj(y_value)
        ):
            raise ModelValidationError("family y must be a one-dimensional numeric array")
        if observed_value.ndim != 1 or not np.issubdtype(observed_value.dtype, np.bool_):
            raise ModelValidationError("family observed must be a one-dimensional boolean array")
        if (
            offset_value.ndim != 1
            or not np.issubdtype(offset_value.dtype, np.number)
            or not np.isrealobj(offset_value)
            or not np.isfinite(offset_value).all()
        ):
            raise ModelValidationError("family offset must be a finite one-dimensional numeric array")
        if not (y_value.size == observed_value.size == offset_value.size):
            raise ModelValidationError("family arrays must have equal row counts")
        if not np.isfinite(y_value[observed_value]).all():
            raise ModelValidationError("family observed y values must be finite")
        try:
            block_values = tuple(blocks)
            names = tuple(parameter_names)
        except TypeError as error:
            raise ModelValidationError("family blocks and parameter names must be iterable") from error
        if any(not isinstance(item, ScalableBlock) for item in block_values):
            raise ModelValidationError("family blocks must be scalable blocks")
        if any(item.block.design.shape[0] != y_value.size for item in block_values):
            raise ModelValidationError("family block design rows must match the response")
        block_names = [item.block.name for item in block_values]
        if len(block_names) != len(set(block_names)):
            raise ModelValidationError("family block names must be unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise ModelValidationError("parameter names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ModelValidationError("parameter names must be unique")
        bindings = {item.parameter for item in block_values if item.parameter is not None}
        if not bindings.issubset(names):
            raise ModelValidationError("scalable block parameters must be declared")
        if not isinstance(initial, Hyperparameters):
            raise ModelValidationError("family initial values must be hyperparameters")
        object.__setattr__(self, "_y", _readonly_array(y_value))
        object.__setattr__(self, "_observed", _readonly_array(observed_value))
        object.__setattr__(self, "_offset", _readonly_array(offset_value))
        object.__setattr__(self, "blocks", block_values)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "initial", initial)

    @property
    def y(self) -> np.ndarray:
        result = self._y.copy()
        result.setflags(write=False)
        return result

    @property
    def observed(self) -> np.ndarray:
        result = self._observed.copy()
        result.setflags(write=False)
        return result

    @property
    def offset(self) -> np.ndarray:
        result = self._offset.copy()
        result.setflags(write=False)
        return result

    def materialize(self, values: Mapping[str, float]) -> CompiledLGM:
        resolved = _validate_parameter_mapping(values, self.parameter_names)
        blocks = tuple(
            LatentBlock(
                item.block.name,
                item.block.labels,
                item.block.design,
                item.block.precision
                * (resolved[item.parameter] if item.parameter else item.scale),
                item.block.constraints,
            )
            for item in self.blocks
        )
        sigma = resolved.get("sigma", self.initial.sigma)
        return _assemble_compiled_model(
            self._y, self._observed, self._offset, blocks, sigma
        )

    def block_slice(self, name: str) -> slice:
        start = 0
        for item in self.blocks:
            stop = start + item.block.design.shape[1]
            if item.block.name == name:
                return slice(start, stop)
            start = stop
        raise KeyError(name)

from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack

from pylgm.exceptions import ModelValidationError, NumericalError
from pylgm.ir.model import CompiledLGM, LatentBlock
from pylgm.likelihoods import CompiledGaussian


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _ordinary_positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ModelValidationError(f"{name} must be a finite positive real number")
    return float(value)


def _ordinary_finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ModelValidationError(f"{name} must be a finite real number")
    return float(value)


def _validate_parameter_mapping(
    values: Mapping[str, float],
    parameter_names: tuple[str, ...],
    parameter_bounds: Mapping[str, object] = MappingProxyType({}),
) -> dict[str, float]:
    """Validate supplied hyperparameter values against their declared domains.

    A parameter's *transform* is the authority on its natural domain: a log
    transform admits only positive values, a bounded (logit) transform admits
    any value strictly inside its interval — so a proper-CAR ``rho`` may be zero
    or negative. Parameters without a declared bound fall back to strict
    positivity, which is the historical behaviour for sigma and block
    precisions.
    """
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
    result = {}
    for name in parameter_names:
        label = f"optimized parameter {name!r}"
        transform = getattr(parameter_bounds.get(name), "transform", None)
        if transform is None:
            result[name] = _ordinary_positive(values[name], label)
            continue
        value = _ordinary_finite(values[name], label)
        if not transform.contains(value):
            raise ModelValidationError(
                f"{label} must lie in {transform.domain_description()}; got {value}"
            )
        result[name] = value
    return result


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


@dataclass(frozen=True, init=False)
class ParametricBlock:
    """A latent block whose precision is an arbitrary function of hyperparameters."""

    block: LatentBlock
    parameters: tuple[str, ...]
    build: Callable[[Mapping[str, float]], csr_matrix]

    def __init__(
        self,
        block: LatentBlock,
        parameters: tuple[str, ...],
        build: Callable[[Mapping[str, float]], csr_matrix],
    ) -> None:
        if not isinstance(block, LatentBlock):
            raise ModelValidationError("parametric block must contain a latent block")
        names = tuple(parameters)
        if not names or any(not isinstance(n, str) or not n for n in names):
            raise ModelValidationError("parametric block parameters must be non-empty strings")
        if not callable(build):
            raise ModelValidationError("parametric block build must be callable")
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "parameters", names)
        object.__setattr__(self, "build", build)

    def materialize(self, resolved: Mapping[str, float]) -> LatentBlock:
        precision = self.build(resolved)
        if not np.isfinite(precision.data).all():
            raise NumericalError(
                f"parametric precision for block {self.block.name!r} must remain finite"
            )
        return LatentBlock(
            self.block.name,
            self.block.labels,
            self.block.design,
            precision,
            self.block.constraints,
        )


@dataclass(frozen=True, init=False)
class ParametricDesignBlock:
    """A latent block whose DESIGN is a function of hyperparameters.

    Mirror of ``ParametricBlock``: ``build`` returns a fresh design column while
    the precision, labels and constraints stay fixed. Used by restricted MIDAS,
    whose single regressor is a theta-weighted aggregate of the HF lags.
    """

    block: LatentBlock
    parameters: tuple[str, ...]
    build: Callable[[Mapping[str, float]], csr_matrix]

    def __init__(
        self,
        block: LatentBlock,
        parameters: tuple[str, ...],
        build: Callable[[Mapping[str, float]], csr_matrix],
    ) -> None:
        if not isinstance(block, LatentBlock):
            raise ModelValidationError("parametric-design block must contain a latent block")
        names = tuple(parameters)
        if not names or any(not isinstance(n, str) or not n for n in names):
            raise ModelValidationError("parametric-design block parameters must be non-empty strings")
        if not callable(build):
            raise ModelValidationError("parametric-design block build must be callable")
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "parameters", names)
        object.__setattr__(self, "build", build)

    def materialize(self, resolved: Mapping[str, float]) -> LatentBlock:
        design = self.build(resolved)
        if not np.isfinite(design.data).all():
            raise NumericalError(
                f"parametric design for block {self.block.name!r} must remain finite"
            )
        return LatentBlock(
            self.block.name,
            self.block.labels,
            design,
            self.block.precision,
            self.block.constraints,
        )


def _family_extra_constraints(value: object, total_width: int) -> np.ndarray:
    """Validate a family's raw ``A x = e`` matrix against the latent width."""
    if value is None:
        return np.empty((0, total_width))
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != total_width:
        raise ModelValidationError(
            "family extra constraints must be a 2D array aligned with the latent width"
        )
    if not np.isfinite(matrix).all():
        raise ModelValidationError("family extra constraints must be finite")
    return matrix


def _family_extra_constraint_rhs(value: object, rows: int) -> np.ndarray:
    """Validate a family's raw ``e`` vector against the number of extra rows."""
    if value is None:
        return np.zeros(rows)
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.shape[0] != rows:
        raise ModelValidationError(
            "family extra constraint rhs must be a 1D array with one entry per extra row"
        )
    if not np.isfinite(vector).all():
        raise ModelValidationError("family extra constraint rhs must be finite")
    return vector


def _qualified_labels(blocks: tuple[LatentBlock, ...]) -> tuple[str, ...]:
    labels = tuple(
        f"{block.name}:{label}" for block in blocks for label in block.labels
    )
    if len(labels) != len(set(labels)):
        raise ModelValidationError("duplicate latent labels after qualification")
    return labels


def _materialize_blocks(
    blocks: "tuple[ScalableBlock | ParametricBlock | ParametricDesignBlock, ...]",
    resolved: Mapping[str, float],
) -> tuple[LatentBlock, ...]:
    materialized: list[LatentBlock] = []
    for item in blocks:
        if isinstance(item, (ParametricBlock, ParametricDesignBlock)):
            materialized.append(item.materialize(resolved))
            continue
        multiplier = resolved[item.parameter] if item.parameter else item.scale
        with np.errstate(over="ignore", invalid="ignore"):
            precision = item.block.precision * multiplier
        if not np.isfinite(precision.data).all():
            raise NumericalError(
                f"precision scaling for block {item.block.name!r} must remain finite"
            )
        materialized.append(
            LatentBlock(
                item.block.name,
                item.block.labels,
                item.block.design,
                precision,
                item.block.constraints,
            )
        )
    return tuple(materialized)


def _assemble_compiled_model(
    y: np.ndarray,
    observed: np.ndarray,
    offset: np.ndarray,
    blocks: tuple[LatentBlock, ...],
    likelihood: object,
    extra_constraints: np.ndarray | None = None,
    extra_constraint_rhs: np.ndarray | None = None,
) -> CompiledLGM:
    if not blocks:
        return CompiledLGM(
            y=y,
            observed=observed,
            offset=offset,
            design=csr_matrix((y.size, 0)),
            precision=csr_matrix((0, 0)),
            constraints=np.empty((0, 0)),
            labels=(),
            likelihood=likelihood,
            blocks=(),
            extra_constraints=extra_constraints,
            extra_constraint_rhs=extra_constraint_rhs,
        )
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
    if extra_constraints is not None and extra_constraints.shape[0]:
        constraints = np.vstack([constraints, extra_constraints])
    return CompiledLGM(
        y=y,
        observed=observed,
        offset=offset,
        design=hstack([block.design for block in blocks], format="csr"),
        precision=block_diag([block.precision for block in blocks], format="csr"),
        constraints=constraints,
        labels=_qualified_labels(blocks),
        likelihood=likelihood,
        blocks=blocks,
        extra_constraints=extra_constraints,
        extra_constraint_rhs=extra_constraint_rhs,
    )


@dataclass(frozen=True, init=False)
class CompiledGaussianFamily:
    _y: np.ndarray = field(repr=False)
    _observed: np.ndarray = field(repr=False)
    _offset: np.ndarray = field(repr=False)
    blocks: tuple[ScalableBlock | ParametricBlock, ...]
    parameter_names: tuple[str, ...]
    initial: Hyperparameters
    parameter_bounds: Mapping[str, object]
    parameter_priors: Mapping[str, object]
    _extra_constraints: np.ndarray = field(repr=False)
    _extra_constraint_rhs: np.ndarray = field(repr=False)

    def __init__(
        self,
        y: np.ndarray,
        observed: np.ndarray,
        offset: np.ndarray,
        blocks: tuple[ScalableBlock | ParametricBlock, ...],
        parameter_names: tuple[str, ...],
        initial: Hyperparameters,
        parameter_bounds: Mapping[str, object] = MappingProxyType({}),
        parameter_priors: Mapping[str, object] = MappingProxyType({}),
        extra_constraints: np.ndarray | None = None,
        extra_constraint_rhs: np.ndarray | None = None,
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
        if any(not isinstance(item, (ScalableBlock, ParametricBlock, ParametricDesignBlock)) for item in block_values):
            raise ModelValidationError("family blocks must be scalable or parametric blocks")
        if any(item.block.design.shape[0] != y_value.size for item in block_values):
            raise ModelValidationError("family block design rows must match the response")
        block_names = [item.block.name for item in block_values]
        if len(block_names) != len(set(block_names)):
            raise ModelValidationError("family block names must be unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise ModelValidationError("parameter names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ModelValidationError("parameter names must be unique")
        if not isinstance(initial, Hyperparameters):
            raise ModelValidationError("family initial values must be hyperparameters")
        bindings = tuple(
            item.parameter
            for item in block_values
            if isinstance(item, ScalableBlock) and item.parameter is not None
        )
        if len(bindings) != len(set(bindings)):
            raise ModelValidationError("scalable block parameters must be unique")
        parametric_names: set[str] = set()
        for item in block_values:
            if isinstance(item, (ParametricBlock, ParametricDesignBlock)):
                if not set(item.parameters) <= set(names):
                    raise ModelValidationError(
                        "parametric block parameters must appear in parameter_names"
                    )
                parametric_names.update(item.parameters)
                continue
            if item.parameter is None:
                continue
            expected = f"{item.block.name}.precision"
            if item.parameter != expected:
                raise ModelValidationError(
                    "scalable block parameter must match its block precision"
                )
            if item.block.name not in initial.precisions:
                raise ModelValidationError(
                    "bound scalable blocks require an initial configured precision"
                )
        expected_names = set(bindings) | parametric_names
        if "sigma" in names:
            expected_names.add("sigma")
        if set(names) != expected_names:
            raise ModelValidationError(
                "parameter names must exactly match bound block precisions and sigma"
            )
        object.__setattr__(self, "_y", _readonly_array(y_value))
        object.__setattr__(self, "_observed", _readonly_array(observed_value))
        object.__setattr__(self, "_offset", _readonly_array(offset_value))
        object.__setattr__(self, "blocks", block_values)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "parameter_bounds", MappingProxyType(dict(parameter_bounds)))
        object.__setattr__(self, "parameter_priors", MappingProxyType(dict(parameter_priors)))
        total_width = sum(item.block.design.shape[1] for item in block_values)
        extra = _family_extra_constraints(extra_constraints, total_width)
        object.__setattr__(self, "_extra_constraints", _readonly_array(extra))
        object.__setattr__(
            self,
            "_extra_constraint_rhs",
            _readonly_array(_family_extra_constraint_rhs(extra_constraint_rhs, extra.shape[0])),
        )

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
        resolved = _validate_parameter_mapping(
            values, self.parameter_names, self.parameter_bounds
        )
        blocks = _materialize_blocks(self.blocks, resolved)
        likelihood = CompiledGaussian(resolved.get("sigma", self.initial.sigma))
        return _assemble_compiled_model(
            self._y, self._observed, self._offset, blocks, likelihood,
            extra_constraints=self._extra_constraints,
            extra_constraint_rhs=self._extra_constraint_rhs,
        )

    def block_slice(self, name: str) -> slice:
        start = 0
        for item in self.blocks:
            stop = start + item.block.design.shape[1]
            if item.block.name == name:
                return slice(start, stop)
            start = stop
        raise KeyError(name)


@dataclass(frozen=True, init=False)
class CompiledFamily:
    """A likelihood-agnostic optimisable family for the declarative path."""

    _y: np.ndarray = field(repr=False)
    _observed: np.ndarray = field(repr=False)
    _offset: np.ndarray = field(repr=False)
    blocks: tuple[ScalableBlock | ParametricBlock, ...]
    parameter_names: tuple[str, ...]
    likelihood_factory: Callable[[Mapping[str, float]], object] = field(repr=False)
    parameter_bounds: Mapping[str, object]
    parameter_priors: Mapping[str, object]
    _extra_constraints: np.ndarray = field(repr=False)
    _extra_constraint_rhs: np.ndarray = field(repr=False)

    def __init__(
        self,
        y: np.ndarray,
        observed: np.ndarray,
        offset: np.ndarray,
        blocks: tuple[ScalableBlock | ParametricBlock, ...],
        parameter_names: tuple[str, ...],
        likelihood_factory: Callable[[Mapping[str, float]], object],
        parameter_bounds: Mapping[str, object] = MappingProxyType({}),
        parameter_priors: Mapping[str, object] = MappingProxyType({}),
        extra_constraints: np.ndarray | None = None,
        extra_constraint_rhs: np.ndarray | None = None,
    ) -> None:
        y_value = np.asarray(y)
        observed_value = np.asarray(observed)
        offset_value = np.asarray(offset)
        if y_value.ndim != 1 or not np.issubdtype(y_value.dtype, np.number) or not np.isrealobj(y_value):
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
        block_values = tuple(blocks)
        names = tuple(parameter_names)
        if any(not isinstance(item, (ScalableBlock, ParametricBlock, ParametricDesignBlock)) for item in block_values):
            raise ModelValidationError("family blocks must be scalable or parametric blocks")
        if any(item.block.design.shape[0] != y_value.size for item in block_values):
            raise ModelValidationError("family block design rows must match the response")
        block_names = [item.block.name for item in block_values]
        if len(block_names) != len(set(block_names)):
            raise ModelValidationError("family block names must be unique")
        if any(not isinstance(name, str) or not name for name in names):
            raise ModelValidationError("parameter names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ModelValidationError("parameter names must be unique")
        bound = tuple(
            item.parameter
            for item in block_values
            if isinstance(item, ScalableBlock) and item.parameter is not None
        )
        if len(bound) != len(set(bound)):
            raise ModelValidationError("scalable block parameters must be unique")
        for parameter in bound:
            if parameter not in names:
                raise ModelValidationError(
                    f"bound block parameter {parameter!r} must appear in parameter_names"
                )
        for item in block_values:
            if isinstance(item, (ParametricBlock, ParametricDesignBlock)) and not set(item.parameters) <= set(names):
                raise ModelValidationError(
                    "parametric block parameters must appear in parameter_names"
                )
        if not callable(likelihood_factory):
            raise ModelValidationError("likelihood_factory must be callable")
        object.__setattr__(self, "_y", _readonly_array(y_value))
        object.__setattr__(self, "_observed", _readonly_array(observed_value))
        object.__setattr__(self, "_offset", _readonly_array(offset_value))
        object.__setattr__(self, "blocks", block_values)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "likelihood_factory", likelihood_factory)
        object.__setattr__(self, "parameter_bounds", MappingProxyType(dict(parameter_bounds)))
        object.__setattr__(self, "parameter_priors", MappingProxyType(dict(parameter_priors)))
        total_width = sum(item.block.design.shape[1] for item in block_values)
        extra = _family_extra_constraints(extra_constraints, total_width)
        object.__setattr__(self, "_extra_constraints", _readonly_array(extra))
        object.__setattr__(
            self,
            "_extra_constraint_rhs",
            _readonly_array(_family_extra_constraint_rhs(extra_constraint_rhs, extra.shape[0])),
        )

    def materialize(self, values: Mapping[str, float]) -> CompiledLGM:
        resolved = _validate_parameter_mapping(
            values, self.parameter_names, self.parameter_bounds
        )
        blocks = _materialize_blocks(self.blocks, resolved)
        likelihood = self.likelihood_factory(resolved)
        return _assemble_compiled_model(
            self._y, self._observed, self._offset, blocks, likelihood,
            extra_constraints=self._extra_constraints,
            extra_constraint_rhs=self._extra_constraint_rhs,
        )

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack

from pylgm.exceptions import ModelValidationError


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _readonly_csr_matrix(value: csr_matrix) -> csr_matrix:
    result = value.copy()
    result.data.setflags(write=False)
    result.indices.setflags(write=False)
    result.indptr.setflags(write=False)
    return result


def _array(value: object, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ModelValidationError(f"{name} must be a regular array") from error


def _numeric_array(
    value: object, name: str, dimensions: int, *, require_finite: bool = True
) -> np.ndarray:
    result = _array(value, name)
    if result.ndim != dimensions:
        raise ModelValidationError(f"{name} must be {dimensions}-dimensional")
    if not np.issubdtype(result.dtype, np.number) or not np.isrealobj(result):
        raise ModelValidationError(f"{name} must have a real numeric dtype")
    if require_finite and not np.isfinite(result).all():
        raise ModelValidationError(f"{name} must be finite")
    return result


def _numeric_csr(value: csr_matrix, name: str) -> csr_matrix:
    if not isinstance(value, csr_matrix):
        raise ModelValidationError(f"{name} must be a CSR sparse matrix")
    if not np.issubdtype(value.dtype, np.number) or not np.isrealobj(value.data):
        raise ModelValidationError(f"{name} must have a real numeric dtype")
    if not np.isfinite(value.data).all():
        raise ModelValidationError(f"{name} data must be finite")
    return value


def _validate_symmetric(value: csr_matrix, name: str) -> None:
    difference = value - value.T
    if difference.nnz and not np.allclose(
        difference.data, 0.0, rtol=1e-12, atol=1e-14
    ):
        raise ModelValidationError(f"{name} must be symmetric")


def _sparse_matches(left: csr_matrix, right: csr_matrix) -> bool:
    difference = left - right
    return not difference.nnz or np.allclose(
        difference.data, 0.0, rtol=1e-12, atol=1e-14
    )


def _block_constraints(blocks: tuple["LatentBlock", ...], width: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    start = 0
    for block in blocks:
        block_width = block.design.shape[1]
        for local in block.constraints:
            row = np.zeros(width)
            row[start : start + block_width] = local
            rows.append(row)
        start += block_width
    return np.vstack(rows) if rows else np.empty((0, width))


@dataclass(frozen=True, init=False)
class LatentBlock:
    name: str
    labels: tuple[str, ...]
    _design: csr_matrix = field(repr=False)
    _precision: csr_matrix = field(repr=False)
    _constraints: np.ndarray = field(repr=False)

    def __init__(
        self,
        name: str,
        labels: tuple[str, ...],
        design: csr_matrix,
        precision: csr_matrix,
        constraints: np.ndarray,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ModelValidationError("block name must be a non-empty string")
        labels = tuple(labels)
        if any(not isinstance(label, str) for label in labels):
            raise ModelValidationError("block labels must be strings")
        if len(labels) != len(set(labels)):
            raise ModelValidationError("block labels must be unique")
        design = _numeric_csr(design, "block design")
        precision = _numeric_csr(precision, "block precision")
        constraints = _numeric_array(constraints, "block constraints", 2)
        width = design.shape[1]
        if len(labels) != width:
            raise ModelValidationError("block labels must align with the design width")
        if precision.shape != (width, width):
            raise ModelValidationError(
                "block precision must be square and align with the design"
            )
        _validate_symmetric(precision, "block precision")
        if constraints.shape[1] != width:
            raise ModelValidationError(
                "block constraints must align with the design width"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "_design", _readonly_csr_matrix(design))
        object.__setattr__(self, "_precision", _readonly_csr_matrix(precision))
        object.__setattr__(self, "_constraints", _readonly_array(constraints))

    @property
    def design(self) -> csr_matrix:
        return _readonly_csr_matrix(self._design)

    @property
    def precision(self) -> csr_matrix:
        return _readonly_csr_matrix(self._precision)

    @property
    def constraints(self) -> np.ndarray:
        return _readonly_array(self._constraints)


@dataclass(frozen=True, init=False)
class CompiledLGM:
    _y: np.ndarray = field(repr=False)
    _observed: np.ndarray = field(repr=False)
    _offset: np.ndarray = field(repr=False)
    _design: csr_matrix = field(repr=False)
    _precision: csr_matrix = field(repr=False)
    _constraints: np.ndarray = field(repr=False)
    labels: tuple[str, ...]
    sigma: float
    blocks: tuple[LatentBlock, ...]

    def __init__(
        self,
        y: np.ndarray,
        observed: np.ndarray,
        offset: np.ndarray,
        design: csr_matrix,
        precision: csr_matrix,
        constraints: np.ndarray,
        labels: tuple[str, ...],
        sigma: float,
        blocks: tuple[LatentBlock, ...],
    ) -> None:
        y = _numeric_array(y, "y", 1, require_finite=False)
        observed = _array(observed, "observed")
        if observed.ndim != 1 or not np.issubdtype(observed.dtype, np.bool_):
            raise ModelValidationError(
                "observed must be a one-dimensional boolean array"
            )
        offset = _numeric_array(offset, "offset", 1)
        design = _numeric_csr(design, "design")
        precision = _numeric_csr(precision, "precision")
        constraints = _numeric_array(constraints, "constraints", 2)
        try:
            labels = tuple(labels)
        except TypeError as error:
            raise ModelValidationError("labels must be an iterable of strings") from error
        try:
            blocks = tuple(blocks)
        except TypeError as error:
            raise ModelValidationError("blocks must be an iterable of latent blocks") from error
        rows, width = design.shape
        if not (y.size == observed.size == offset.size == rows):
            raise ModelValidationError(
                "y, observed, offset, and design row counts must match"
            )
        if not np.isfinite(y[observed]).all():
            raise ModelValidationError("observed y values must be finite")
        if precision.shape != (width, width):
            raise ModelValidationError(
                "precision must be square and align with the design width"
            )
        _validate_symmetric(precision, "precision")
        if constraints.shape[1] != width:
            raise ModelValidationError("constraints must align with the design width")
        if len(labels) != width:
            raise ModelValidationError("labels must align with the design width")
        if any(not isinstance(label, str) for label in labels):
            raise ModelValidationError("labels must be strings")
        if len(labels) != len(set(labels)):
            raise ModelValidationError("labels must be unique")
        sigma_array = _numeric_array(sigma, "sigma", 0)
        sigma_value = float(sigma_array)
        if sigma_value <= 0:
            raise ModelValidationError("sigma must be finite and positive")
        block_names = [block.name for block in blocks]
        if len(block_names) != len(set(block_names)):
            raise ModelValidationError("block names must be unique")
        if blocks:
            if any(block.design.shape[0] != rows for block in blocks):
                raise ModelValidationError(
                    "block design row counts must match the model"
                )
            if sum(block.design.shape[1] for block in blocks) != width:
                raise ModelValidationError(
                    "block widths must align with the model design"
                )
            qualified = tuple(
                f"{block.name}:{label}" for block in blocks for label in block.labels
            )
            if labels != qualified:
                raise ModelValidationError(
                    "labels must align with block names and labels"
                )
            expected_design = hstack(
                [block.design for block in blocks], format="csr"
            )
            if not _sparse_matches(design, expected_design):
                raise ModelValidationError("design must match the latent blocks")
            expected_precision = block_diag(
                [block.precision for block in blocks], format="csr"
            )
            if not _sparse_matches(precision, expected_precision):
                raise ModelValidationError("precision must match the latent blocks")
            expected_constraints = _block_constraints(blocks, width)
            if constraints.shape != expected_constraints.shape or not np.allclose(
                constraints, expected_constraints, rtol=1e-12, atol=1e-14
            ):
                raise ModelValidationError("constraints must match the latent blocks")
        object.__setattr__(self, "_y", _readonly_array(y))
        object.__setattr__(self, "_observed", _readonly_array(observed))
        object.__setattr__(self, "_offset", _readonly_array(offset))
        object.__setattr__(self, "_design", _readonly_csr_matrix(design))
        object.__setattr__(self, "_precision", _readonly_csr_matrix(precision))
        object.__setattr__(self, "_constraints", _readonly_array(constraints))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sigma", sigma_value)
        object.__setattr__(self, "blocks", blocks)

    @property
    def y(self) -> np.ndarray:
        return _readonly_array(self._y)

    @property
    def observed(self) -> np.ndarray:
        return _readonly_array(self._observed)

    @property
    def offset(self) -> np.ndarray:
        return _readonly_array(self._offset)

    @property
    def design(self) -> csr_matrix:
        return _readonly_csr_matrix(self._design)

    @property
    def precision(self) -> csr_matrix:
        return _readonly_csr_matrix(self._precision)

    @property
    def constraints(self) -> np.ndarray:
        return _readonly_array(self._constraints)

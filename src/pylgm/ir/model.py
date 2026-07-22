from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix


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
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "labels", tuple(labels))
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
        object.__setattr__(self, "_y", _readonly_array(y))
        object.__setattr__(self, "_observed", _readonly_array(observed))
        object.__setattr__(self, "_offset", _readonly_array(offset))
        object.__setattr__(self, "_design", _readonly_csr_matrix(design))
        object.__setattr__(self, "_precision", _readonly_csr_matrix(precision))
        object.__setattr__(self, "_constraints", _readonly_array(constraints))
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "blocks", tuple(blocks))

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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class LatentBlock:
    name: str
    labels: tuple[str, ...]
    design: csr_matrix
    precision: csr_matrix
    constraints: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "design", _readonly_csr_matrix(self.design))
        object.__setattr__(self, "precision", _readonly_csr_matrix(self.precision))
        object.__setattr__(self, "constraints", _readonly_array(self.constraints))


@dataclass(frozen=True)
class CompiledLGM:
    y: np.ndarray
    observed: np.ndarray
    offset: np.ndarray
    design: csr_matrix
    precision: csr_matrix
    constraints: np.ndarray
    labels: tuple[str, ...]
    sigma: float
    blocks: tuple[LatentBlock, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", _readonly_array(self.y))
        object.__setattr__(self, "observed", _readonly_array(self.observed))
        object.__setattr__(self, "offset", _readonly_array(self.offset))
        object.__setattr__(self, "design", _readonly_csr_matrix(self.design))
        object.__setattr__(self, "precision", _readonly_csr_matrix(self.precision))
        object.__setattr__(self, "constraints", _readonly_array(self.constraints))

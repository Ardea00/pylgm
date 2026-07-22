from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class LatentBlock:
    name: str
    labels: tuple[str, ...]
    design: csr_matrix
    precision: csr_matrix
    constraints: np.ndarray


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

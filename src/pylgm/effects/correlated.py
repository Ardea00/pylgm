"""A k-variate IID effect: k correlated components per level (R-INLA's ``iidkd``)."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye, kron

from pylgm.ir.model import LatentBlock


def correlation_cholesky(cpcs: np.ndarray, k: int) -> np.ndarray:
    """Cholesky factor of a correlation matrix from canonical partial correlations.

    ``cpcs`` lists ``z[j, i]`` for ``j < i`` row-major, ``(0,1), (0,2), ...,
    (1,2), ...``: the partial correlation of components ``j`` and ``i`` given
    ``0..j-1``. Any values in ``(-1, 1)`` give a positive definite matrix
    (Lewandowski, Kurowicka and Joe 2009), so the optimizer needs no constraint.
    """
    z = np.zeros((k, k))
    z[np.triu_indices(k, 1)] = cpcs
    factor = np.zeros((k, k))
    factor[0, 0] = 1.0
    for i in range(1, k):
        remaining = 1.0
        for j in range(i):
            factor[i, j] = z[j, i] * np.sqrt(remaining)
            remaining -= factor[i, j] ** 2
        factor[i, i] = np.sqrt(remaining)
    return factor


def component_precision(precisions, cpcs) -> np.ndarray:
    """``Sigma^-1`` with ``Sigma = D R D``, ``D = diag(precision^-1/2)``."""
    precisions = np.asarray(precisions, dtype=float)
    k = precisions.size
    factor = correlation_cholesky(np.asarray(cpcs, dtype=float), k)
    inverse = np.linalg.inv(factor)
    root = np.sqrt(precisions)
    return (inverse.T @ inverse) * np.outer(root, root)


def correlated_levels(frame: pd.DataFrame, columns) -> tuple:
    """The sorted union of the index columns' levels (NaN = component absent)."""
    values = pd.concat([frame[column] for column in columns]).dropna()
    return tuple(sorted(values.drop_duplicates().tolist()))


def correlated_design(frame: pd.DataFrame, columns, levels) -> csr_matrix:
    """Component-major one-hot design: component ``c`` occupies columns ``c n .. (c+1) n``."""
    n = len(levels)
    positions = {level: column for column, level in enumerate(levels)}
    rows, cols = [], []
    for c, column in enumerate(columns):
        present = frame[column].notna().to_numpy()
        rows.append(np.flatnonzero(present))
        cols.append(c * n + np.array([positions[v] for v in frame.loc[present, column]], dtype=int))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(rows.size), (rows, cols)), shape=(len(frame), len(columns) * n))


def build_correlated(frame: pd.DataFrame, name: str, columns, components: np.ndarray) -> LatentBlock:
    """``x = (x_1, ..., x_k)`` per level with precision ``components ⊗ I_n``."""
    levels = correlated_levels(frame, columns)
    n = len(levels)
    return LatentBlock(
        name,
        tuple(f"{column}@{level}" for column in columns for level in levels),
        correlated_design(frame, columns, levels),
        kron(csr_matrix(components), eye(n, format="csr"), format="csr"),
        np.empty((0, len(columns) * n)),
    )


def dyad_columns(frame: pd.DataFrame, sender: str, receiver: str) -> pd.DataFrame:
    """``forward``/``backward`` index columns for edge-level reciprocity.

    Each unordered pair ``{i, j}`` (``i < j``) is one level; the row ``i -> j``
    indexes it through ``forward`` and ``j -> i`` through ``backward``, so
    ``Correlated("dyad", index=("forward", "backward"))`` gives every pair its
    ``(u_ij, u_ji)`` with correlation = reciprocity. Self-loops are rejected.
    """
    i, j = frame[sender].to_numpy(dtype=object), frame[receiver].to_numpy(dtype=object)
    if np.any(i == j):
        raise ValueError("dyad_columns needs sender != receiver on every row")
    low = np.array([a if a < b else b for a, b in zip(i, j, strict=True)], dtype=object)
    high = np.array([b if a < b else a for a, b in zip(i, j, strict=True)], dtype=object)
    key = pd.Series([f"{a}|{b}" for a, b in zip(low, high, strict=True)], index=frame.index)
    upward = pd.Series(i == low, index=frame.index)
    return pd.DataFrame({"forward": key.where(upward), "backward": key.where(~upward)})

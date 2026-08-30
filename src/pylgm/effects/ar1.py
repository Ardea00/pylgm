"""Stationary first-order autoregressive (AR1) latent effect."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, identity, kron

from pylgm.data.scalars import ordered_observed_levels
from pylgm.ir.model import LatentBlock


def ar1_structure(level_count: int, rho: float, group_count: int = 1) -> csr_matrix:
    """Return ``T / (1 - rho**2)`` for the stationary AR1 on ``level_count`` levels.

    ``T`` is tridiagonal with unit corners, ``1 + rho**2`` on the interior
    diagonal and ``-rho`` off it, so ``precision * ar1_structure(...)`` inverts
    to ``(1 / precision) * rho ** |i - j|``.

    ``group_count > 1`` returns ``I_G kron T``: ``G`` independent series that
    share ``rho`` and ``precision`` but not their realizations, laid out
    group-major so the result is block diagonal (each group's band is
    contiguous, which is what keeps the sparse factorization cheap).

    Conditioning degrades as ``rho`` approaches +/-1: ``cond(Q)`` runs roughly
    ``2e6 * level_count`` at the 1e-6 inset the compiler bounds an estimated
    ``rho`` to, leaving about seven significant digits at 1000 levels. Nothing
    fails there, but a fit that pins ``rho`` at the bound on a long series is
    really asking for a random walk.
    """
    # Guards direct callers; build_ar1 checks first, with a message naming
    # the effect.
    if level_count < 2:
        raise ValueError("AR1 requires at least two ordered levels")
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError(f"rho must lie strictly inside (-1, 1); got {rho}")
    if group_count < 1:
        raise ValueError(f"group_count must be at least 1; got {group_count}")
    # Sparse like the other structured builders: a dense n x n allocation here
    # costs ~400 MB at n=5000, and this runs again at every hyperparameter the
    # optimiser or INLA grid visits.
    off = np.full(level_count - 1, -rho)
    main = np.full(level_count, 1.0 + rho**2)
    main[0] = main[-1] = 1.0
    structure = diags([off, main, off], [-1, 0, 1], format="csr")
    if group_count > 1:
        structure = kron(identity(group_count, format="csr"), structure, format="csr")
    return structure / (1.0 - rho**2)


def build_ar1(
    frame: pd.DataFrame,
    name: str,
    index: str,
    precision: float,
    rho: float,
    group: str | None = None,
) -> LatentBlock:
    levels = ordered_observed_levels(frame[index])
    if len(levels) < 2:
        raise ValueError(f"{name} requires at least two ordered levels")
    if group is None:
        positions = {level: column for column, level in enumerate(levels)}
        rows = np.arange(len(frame))
        columns = np.array([positions[value] for value in frame[index]])
        design = csr_matrix(
            (np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels))
        )
        precision_matrix = csr_matrix(precision * ar1_structure(len(levels), rho))
        constraints = np.empty((0, len(levels)))
        return LatentBlock(name, tuple(map(str, levels)), design, precision_matrix, constraints)

    if group not in frame.columns:
        raise ValueError(f"{name} group column {group!r} not found")
    if frame[group].isna().any():
        raise ValueError(f"{name} group column {group!r} must not contain null values")
    groups = tuple(sorted({str(value) for value in frame[group]}))
    time_position = {level: column for column, level in enumerate(levels)}
    group_position = {label: row for row, label in enumerate(groups)}
    # Group-major: cell = group * n_levels + level, so the precision is block
    # diagonal with each group's AR1 band contiguous.
    cells = np.array([
        group_position[str(g)] * len(levels) + time_position[t]
        for g, t in zip(frame[group], frame[index])
    ])
    width = len(groups) * len(levels)
    design = csr_matrix(
        (np.ones(len(frame)), (np.arange(len(frame)), cells)), shape=(len(frame), width)
    )
    precision_matrix = csr_matrix(
        precision * ar1_structure(len(levels), rho, group_count=len(groups))
    )
    labels = tuple(f"{g}@{t}" for g in groups for t in levels)
    constraints = np.empty((0, width))
    return LatentBlock(name, labels, design, precision_matrix, constraints)

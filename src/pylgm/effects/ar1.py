"""Stationary first-order autoregressive (AR1) latent effect."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from pylgm.data.scalars import ordered_observed_levels
from pylgm.ir.model import LatentBlock


def ar1_structure(level_count: int, rho: float) -> csr_matrix:
    """Return ``T / (1 - rho**2)`` for the stationary AR1 on ``level_count`` levels.

    ``T`` is tridiagonal with unit corners, ``1 + rho**2`` on the interior
    diagonal and ``-rho`` off it, so ``precision * ar1_structure(...)`` inverts
    to ``(1 / precision) * rho ** |i - j|``.

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
    # Sparse like the other structured builders: a dense n x n allocation here
    # costs ~400 MB at n=5000, and this runs again at every hyperparameter the
    # optimiser or INLA grid visits.
    off = np.full(level_count - 1, -rho)
    main = np.full(level_count, 1.0 + rho**2)
    main[0] = main[-1] = 1.0
    structure = diags([off, main, off], [-1, 0, 1], format="csr")
    return structure / (1.0 - rho**2)


def build_ar1(
    frame: pd.DataFrame,
    name: str,
    index: str,
    precision: float,
    rho: float,
) -> LatentBlock:
    levels = ordered_observed_levels(frame[index])
    if len(levels) < 2:
        raise ValueError(f"{name} requires at least two ordered levels")
    positions = {level: column for column, level in enumerate(levels)}
    rows = np.arange(len(frame))
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix(
        (np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels))
    )
    precision_matrix = csr_matrix(precision * ar1_structure(len(levels), rho))
    constraints = np.empty((0, len(levels)))
    return LatentBlock(name, tuple(map(str, levels)), design, precision_matrix, constraints)

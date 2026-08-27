from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pylgm.effects.random_walk import difference_operator
from pylgm.ir.model import LatentBlock


def midas_penalty(n: int, order: Literal[1, 2]) -> tuple[csr_matrix, csr_matrix]:
    """The MIDAS penalty pieces over ``n`` lags: curvature ``DᵀD`` and the
    orthogonal projector ``P₀`` onto the penalty's null space (level, or
    level+slope), so ``τ·DᵀD + δ·P₀`` smooths curvature while leaving the lag
    curve's level/slope on a fixed, ``τ``-independent precision ``δ``.
    """
    difference = difference_operator(n, order)
    dtd = csr_matrix(difference.T @ difference)
    basis = [np.ones(n)] if order == 1 else [np.ones(n), np.arange(n, dtype=float)]
    t = np.column_stack(basis)
    projector = csr_matrix(t @ np.linalg.pinv(t))
    return dtd, projector


def build_midas(
    frame: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
    precision: float,
    order: Literal[1, 2],
    ridge: float,
) -> LatentBlock:
    if len(columns) <= order:
        raise ValueError(f"{name} requires more than {order} lag columns")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} lag columns not found: {missing}")
    values = frame[list(columns)].to_numpy(dtype=float)
    for offset, column in enumerate(columns):
        if not np.isfinite(values[:, offset]).all():
            raise ValueError(f"{name} lag column {column!r} contains non-finite values")
    design = csr_matrix(values)
    dtd, projector = midas_penalty(len(columns), order)
    precision_matrix = csr_matrix(precision * dtd + ridge * projector)
    constraints = np.empty((0, len(columns)), dtype=float)
    return LatentBlock(name, tuple(columns), design, precision_matrix, constraints)


def _softmax(log_w: np.ndarray) -> np.ndarray:
    shifted = log_w - log_w.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def midas_weights(kernel: str, n_lags: int, theta: tuple[float, float]) -> np.ndarray:
    """Normalized MIDAS lag weights (sum to 1) for a parametric kernel.

    ``kernel="exp_almon"``: log w_k = theta1*k + theta2*k**2 (theta real).
    ``kernel="beta"``: Beta density on an interior grid x_k=(k+1)/(K+1),
    log w_k = (a-1)*log x_k + (b-1)*log(1-x_k) with theta=(a, b), a,b>0.
    Both are softmax-normalized (overflow-safe).
    """
    k = np.arange(n_lags, dtype=float)
    if kernel == "exp_almon":
        theta1, theta2 = theta
        log_w = theta1 * k + theta2 * k * k
    elif kernel == "beta":
        a, b = theta
        x = (k + 1.0) / (n_lags + 1.0)  # strictly inside (0, 1): no log(0)
        log_w = (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x)
    else:
        raise ValueError(f"unknown MIDAS kernel {kernel!r}; expected 'beta' or 'exp_almon'")
    return _softmax(log_w)


def build_midas_parametric(
    frame: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
    kernel: str,
    theta: tuple[float, float],
    prior_precision: float,
) -> LatentBlock:
    """Restricted-MIDAS block: one regressor = theta-weighted aggregate of the
    HF lag columns, with a fixed vague Gaussian prior on the loading."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"MIDAS lag columns missing from frame: {missing!r}")
    values = frame[list(columns)].to_numpy(dtype=float)  # (N, K)
    if not np.isfinite(values).all():
        raise ValueError(f"MIDAS lag columns for {name!r} contain non-finite values")
    weights = midas_weights(kernel, len(columns), theta)
    design = csr_matrix((values @ weights).reshape(-1, 1))
    precision = csr_matrix(np.array([[float(prior_precision)]]))
    constraints = np.empty((0, 1), dtype=float)
    return LatentBlock(name, (name,), design, precision, constraints)

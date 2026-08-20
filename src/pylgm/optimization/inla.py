from collections.abc import Callable
from itertools import product

import numpy as np

from pylgm.exceptions import OptimizationError


def _finite_difference_hessian(
    func: Callable[[np.ndarray], float], center: np.ndarray, step: float = 1e-3
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    d = center.size
    hessian = np.zeros((d, d))
    f0 = func(center)
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = step
        f_plus = func(center + ei)
        f_minus = func(center - ei)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step * step)
        for j in range(i + 1, d):
            ej = np.zeros(d)
            ej[j] = step
            f_pp = func(center + ei + ej)
            f_pm = func(center + ei - ej)
            f_mp = func(center - ei + ej)
            f_mm = func(center - ei - ej)
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def _whitening_directions(hessian: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    negative = -np.asarray(hessian, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(negative)
    clamped = np.clip(eigenvalues, ridge, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(clamped))


def _build_grid(
    center: np.ndarray, hessian: np.ndarray, *,
    grid_step: float = 1.0, radius: int = 3, max_grid_points: int = 4096,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    d = center.size
    candidate_count = (2 * radius + 1) ** d
    if candidate_count > max_grid_points:
        raise OptimizationError(
            f"INLA grid would need {candidate_count} points for {d} hyperparameters; "
            f"exceeds max_grid_points={max_grid_points}"
        )
    directions = _whitening_directions(hessian)
    offsets = range(-radius, radius + 1)
    points = [
        center + grid_step * directions @ np.asarray(z, dtype=float)
        for z in product(offsets, repeat=d)
    ]
    return np.asarray(points)

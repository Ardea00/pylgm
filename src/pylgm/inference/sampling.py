"""Joint posterior draws of the linear predictor on the prediction grid."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class GridSampler:
    """One Gaussian latent posterior, mapped to ``eta = offset + design @ x``.

    Exactly one of ``factor`` (dense: ``covariance = factor @ factor.T``, whose
    columns span the constraint null space) or ``posterior`` (a
    ``SparsePosterior``) is set. ``row_order`` carries the caller-order
    permutation applied to the result's prediction rows.
    """

    mean: np.ndarray
    design: csr_matrix
    offset: np.ndarray
    factor: np.ndarray | None = None
    posterior: object | None = None
    row_order: np.ndarray | None = None

    def reordered(self, order: np.ndarray) -> "GridSampler":
        return replace(self, row_order=order if self.row_order is None else self.row_order[order])

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.factor is not None:
            deviations = rng.standard_normal((n, self.factor.shape[1])) @ self.factor.T
        else:
            deviations = self.posterior.sample_deviations(n, rng)
        eta = self.offset + np.asarray(self.design @ (self.mean + deviations).T).T
        return eta if self.row_order is None else eta[:, self.row_order]


@dataclass(frozen=True)
class RefitSampler:
    """A grid point's conditional posterior, rebuilt only when it is drawn from.

    An integrated result would otherwise keep every grid point's sampling factor
    alive -- ``O(points * p * d)`` memory -- for draws that may never be asked
    for. ``refit`` re-runs that point's conditional fit and returns a
    ``GridSampler``; its cost is one fit per point that receives draws.
    """

    refit: Callable[[], GridSampler]
    row_order: np.ndarray | None = None

    def reordered(self, order: np.ndarray) -> "RefitSampler":
        return replace(self, row_order=order if self.row_order is None else self.row_order[order])

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        sampler = self.refit()
        if self.row_order is not None:
            sampler = sampler.reordered(self.row_order)
        return sampler.draw(n, rng)


def sample_mixture(components, n, rng) -> np.ndarray:
    """``n`` draws from a weighted mixture of ``(weight, GridSampler)`` components.

    Draws are allocated to components by one multinomial over the weights, then
    shuffled so their order carries no information about the component.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    rng = np.random.default_rng(rng)
    weights = np.array([weight for weight, _ in components], dtype=float)
    if np.any(weights < 0):
        raise ValueError("posterior mixture weights must be non-negative to sample")
    counts = rng.multinomial(int(n), weights / weights.sum())
    draws = np.concatenate([
        sampler.draw(int(count), rng)
        for count, (_, sampler) in zip(counts, components, strict=True) if count
    ])
    return draws[rng.permutation(draws.shape[0])] if len(components) > 1 else draws

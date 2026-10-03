"""Joint posterior draws of the linear predictor on the prediction grid."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import csr_matrix


class LazyArray:
    """An array computed on first ``get()`` and cached (shared by every holder)."""

    def __init__(self, compute: Callable[[], np.ndarray]) -> None:
        self._compute: Callable[[], np.ndarray] | None = compute
        self._value: np.ndarray | None = None

    def get(self) -> np.ndarray:
        if self._compute is not None:
            self._value = self._compute()
            self._compute = None  # drop the closed-over inputs once materialised
        return self._value


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
    factor: "np.ndarray | LazyArray | None" = None
    posterior: object | None = None
    row_order: np.ndarray | None = None

    def dense_factor(self) -> np.ndarray | None:
        return self.factor.get() if isinstance(self.factor, LazyArray) else self.factor

    def reordered(self, order: np.ndarray) -> "GridSampler":
        return replace(self, row_order=order if self.row_order is None else self.row_order[order])

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        factor = self.dense_factor()
        if factor is not None:
            deviations = rng.standard_normal((n, factor.shape[1])) @ factor.T
        else:
            deviations = self.posterior.sample_deviations(n, rng)
        eta = self.offset + np.asarray(self.design @ (self.mean + deviations).T).T
        return eta if self.row_order is None else eta[:, self.row_order]


@dataclass(frozen=True)
class RefitSampler:
    """A grid point's conditional posterior, rebuilt only when it is drawn from.

    An integrated result would otherwise keep every grid point's sampling factor
    alive -- ``O(points * p * d)`` memory -- for draws that may never be asked
    for. ``conditional`` returns that point's fit (refitting it until an update
    has made it concrete); its cost is one fit per point that receives draws.
    """

    conditional: Callable[[], object]

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.conditional()._sampler.draw(n, rng)


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

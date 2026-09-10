"""Semi-parametric survival by Poisson augmentation.

A proportional-hazards model with an *arbitrary* baseline hazard does not need a
new likelihood. Split the time axis into intervals, hold the hazard constant
within each, and expand every subject into one pseudo-row per interval it is at
risk in; the piecewise-exponential likelihood is then the Poisson likelihood on
those rows, with the log time-at-risk as an offset. Smoothing the interval
effects with ``RW1``/``RW2`` gives a smooth baseline hazard, and leaving them
``IID`` gives a free one -- either way the covariate effects are the Cox ones.

This is the mechanism R-INLA uses (``inla.coxph``); see Martino, Akerkar & Rue,
*Scand. J. Stat.* (2011).

THE LIKELIHOOD IDENTITY, AND ITS ONE CAVEAT
-------------------------------------------
For subject ``i`` with hazard ``lambda_k = exp(eta_k)`` constant on interval
``k`` and time-at-risk ``r_ik``, the piecewise-exponential contribution is
``d_i eta_k(i) - sum_k r_ik lambda_k``. The Poisson pseudo-rows give
``d_i (log r_i,k(i) + eta_k(i)) - sum_k r_ik lambda_k``. The two differ by
``sum_i d_i log r_i,k(i)`` -- a constant in the parameters, so every estimate
agrees, but the **log marginal likelihood is offset by that constant**. Compare
`log_marginal_likelihood` between two expansions only if they share breakpoints;
:func:`log_likelihood_offset` returns the constant when an absolute value is
needed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pylgm.exceptions import DataContractError

__all__ = ["expand_cox", "log_likelihood_offset", "CoxExpansion"]


@dataclass(frozen=True)
class CoxExpansion:
    """The expanded frame and the column names a model should reference."""

    frame: pd.DataFrame
    interval: str
    exposure: str
    response: str
    breaks: np.ndarray

    @property
    def intervals(self) -> int:
        return len(self.breaks) - 1


def _validated(frame, time, event, entry):
    for column in (time, event) + ((entry,) if entry is not None else ()):
        if column not in frame.columns:
            raise DataContractError(f"column not found: {column!r}")
    t = frame[time].to_numpy(dtype=float)
    d = frame[event].to_numpy(dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise DataContractError(f"follow-up time {time!r} must be finite and positive")
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise DataContractError(f"event indicator {event!r} must be 0 or 1")
    if entry is None:
        e = np.zeros_like(t)
    else:
        e = frame[entry].to_numpy(dtype=float)
        if not np.all(np.isfinite(e)) or np.any(e < 0.0) or np.any(e >= t):
            raise DataContractError(f"entry time {entry!r} must satisfy 0 <= entry < {time}")
    return t, d, e


def _edges(t, d, breaks):
    """Interval boundaries ``0 = a_0 < a_1 < ... < a_K = inf``.

    An integer asks for that many intervals cut at quantiles of the *event*
    times, which is the usual default: it puts roughly equal numbers of events
    in each interval, and an interval with no events carries no information about
    its own baseline level.
    """
    if isinstance(breaks, (int, np.integer)) and not isinstance(breaks, bool):
        if breaks < 1:
            raise ValueError("breaks must be at least 1")
        observed = np.sort(t[d == 1.0])
        if observed.size == 0:
            raise DataContractError("cannot choose breakpoints: no events observed")
        quantiles = np.quantile(observed, np.linspace(0.0, 1.0, breaks + 1)[1:-1])
        interior = np.unique(quantiles[quantiles > 0.0])
    else:
        interior = np.asarray(breaks, dtype=float)
        if interior.ndim != 1 or interior.size == 0:
            raise ValueError("breaks must be a positive integer or a 1-D array of cut points")
        if not np.all(np.isfinite(interior)) or np.any(interior <= 0.0):
            raise ValueError("break points must be finite and positive")
        if np.any(np.diff(interior) <= 0.0):
            raise ValueError("break points must be strictly increasing")
    return np.concatenate([[0.0], interior, [np.inf]])


def expand_cox(
    frame,
    *,
    time: str,
    event: str,
    breaks=8,
    entry: str | None = None,
    interval: str = "interval",
    exposure: str = "log_exposure",
    response: str = "events",
) -> CoxExpansion:
    """Expand survival rows into piecewise-exponential Poisson pseudo-rows.

    Each subject contributes one row per interval it is at risk in, carrying its
    original columns unchanged plus three new ones: the interval index, the log
    time at risk there, and whether the event fell in it.

    >>> expansion = expand_cox(frame, time="t", event="d", breaks=8)   # doctest: +SKIP
    >>> LGM(                                                           # doctest: +SKIP
    ...     response=expansion.response,
    ...     likelihood=Poisson(),
    ...     predictor=Fixed("1 + x") + RW1("baseline", index=expansion.interval,
    ...                                    precision=Hyperparameter("kappa", initial=1.0)),
    ...     offset=expansion.exposure,
    ... ).fit(expansion.frame, engine="laplace")

    ``RW1``/``RW2`` on the interval index smooths the log baseline hazard; ``IID``
    leaves it free; omitting it entirely gives a constant hazard, i.e. the
    exponential model.

    Rows where the subject is not at risk are dropped rather than carried with
    zero exposure -- their offset would be ``log 0``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    for name in (interval, exposure, response):
        if name in frame.columns:
            raise ValueError(f"column {name!r} already exists; pass a different name")

    t, d, e = _validated(frame, time, event, entry)
    edges = _edges(t, d, breaks)
    lower, upper = edges[:-1][None, :], edges[1:][None, :]      # (1, K)

    # Time at risk in each interval, clipped at the subject's entry and exit.
    at_risk = np.clip(
        np.minimum(t[:, None], upper) - np.maximum(e[:, None], lower), 0.0, None
    )
    # The event falls in the interval containing the exit time.
    fell_here = (t[:, None] > lower) & (t[:, None] <= upper) & (d[:, None] == 1.0)

    subject, which = np.nonzero(at_risk > 0.0)
    if subject.size == 0:
        raise DataContractError("expansion produced no rows at risk")

    expanded = frame.iloc[subject].reset_index(drop=True)
    expanded[interval] = which.astype(np.int64)
    expanded[exposure] = np.log(at_risk[subject, which])
    expanded[response] = fell_here[subject, which].astype(float)
    return CoxExpansion(expanded, interval, exposure, response, edges)


def log_likelihood_offset(expansion: CoxExpansion) -> float:
    """``sum_i d_i log r_i,k(i)`` -- what separates the Poisson log likelihood
    from the piecewise-exponential one.

    Add it to a fit's ``log_marginal_likelihood`` to compare against a model that
    was not expanded, or against an expansion with different breakpoints.
    """
    events = expansion.frame[expansion.response].to_numpy(dtype=float)
    exposure = expansion.frame[expansion.exposure].to_numpy(dtype=float)
    return float(-np.sum(events * exposure))

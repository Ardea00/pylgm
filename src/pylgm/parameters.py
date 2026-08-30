"""Hyperparameter definitions for latent Gaussian models."""

from dataclasses import dataclass
from typing import Literal, Protocol

import math


class Prior(Protocol):
    """A prior density evaluated on its native parameter scale."""

    def logpdf(self, value: float) -> float: ...


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


@dataclass(frozen=True)
class Hyperparameter:
    """A named model parameter and its optional prior (positive-only under the
    default ``log`` transform; ``logit``/``identity`` allow any finite value).

    A declared ``Hyperparameter`` is estimated by type-II maximum likelihood
    (empirical Bayes) in ``LGM.fit``, optimizing over ``[lower, upper]`` starting
    from ``initial``. This applies to both the exact-Gaussian and Laplace
    engines; the estimate is reported on ``result.hyperparameters``, keyed by
    ``name``. With ``hyperparameters="integrate"`` the posterior is integrated
    over instead, and ``result.hyperparameter_marginals()`` reports the
    marginals.

    ``transform`` selects the unconstrained scale both the optimizer and the
    INLA grid search in: ``"log"`` (default) for strictly positive parameters
    such as ``sigma`` and effect precisions, and ``"logit"`` for parameters
    confined to a bounded interval, such as a :class:`~pylgm.effects.ProperCAR`
    ``rho`` (whose interval is resolved from the graph at compile time). Under
    ``"logit"`` the ``initial`` value may be any finite real — including zero or
    negative — and ``lower``/``upper`` may be left ``None`` for the effect to
    supply. ``"identity"`` selects the real line (any finite value) — used by exp-Almon MIDAS weight shapes.

    When ``prior`` is declared, its native-scale log density is added as a
    penalty to the objective (no Jacobian correction), turning the estimate into
    a MAP-II estimate; ``result.diagnostics`` records whether any declared
    hyperparameter was penalized this way.
    """

    name: str
    initial: float
    prior: Prior | None = None
    transform: Literal["identity", "log", "logit"] = "log"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.transform not in ("identity", "log", "logit"):
            raise ValueError("transform must be 'identity', 'log', or 'logit'")
        if self.prior is not None and not callable(getattr(self.prior, "logpdf", None)):
            raise ValueError("prior must provide a logpdf method")
        if self.transform == "log":
            if type(self.initial) in (int, float) and math.isfinite(self.initial) and self.initial <= 0:
                # The common first mistake: a bounded parameter (a correlation
                # rho, a mixing phi) declared without transform="logit", where
                # an initial of 0 or a negative value is perfectly sensible.
                raise ValueError(
                    f"initial must be positive under the default transform='log', got "
                    f"{self.initial}; a parameter confined to an interval (such as an "
                    "AR1/SAR/CAR rho or a BYM2 phi) needs transform='logit', which "
                    "accepts any finite initial"
                )
            object.__setattr__(self, "initial", _positive_real(self.initial, "initial"))
            lower = self.initial * 1e-3 if self.lower is None else _positive_real(self.lower, "lower")
            upper = self.initial * 1e3 if self.upper is None else _positive_real(self.upper, "upper")
            if not lower < self.initial:
                raise ValueError("lower bound must be below initial")
            if not self.initial < upper:
                raise ValueError("upper bound must be above initial")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        elif self.transform == "identity":
            # Real-line parameter (e.g. exp-Almon weight shape): any finite
            # initial; default to a finite symmetric window so OptimizationBounds
            # (which requires finite lower/upper under IdentityTransform) is valid.
            if type(self.initial) not in (int, float) or not math.isfinite(self.initial):
                raise ValueError("initial must be a finite real value")
            object.__setattr__(self, "initial", float(self.initial))
            lower = self.initial - 10.0 if self.lower is None else float(self.lower)
            upper = self.initial + 10.0 if self.upper is None else float(self.upper)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("lower and upper must be finite real values or None")
            if not lower < self.initial < upper:
                raise ValueError("bounds must satisfy lower < initial < upper")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        else:
            # logit: any finite initial; bounds may be deferred (None)
            if type(self.initial) not in (int, float) or not math.isfinite(self.initial):
                raise ValueError("initial must be a finite real value")
            object.__setattr__(self, "initial", float(self.initial))
            if self.lower is not None and not math.isfinite(self.lower):
                raise ValueError("lower must be a finite real value or None")
            if self.upper is not None and not math.isfinite(self.upper):
                raise ValueError("upper must be a finite real value or None")
            if self.lower is not None and self.upper is not None and not self.lower < self.upper:
                raise ValueError("lower must be below upper")

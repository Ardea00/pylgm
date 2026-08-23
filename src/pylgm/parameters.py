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
    """A named, positive model parameter and its optional prior.

    A declared ``Hyperparameter`` is estimated by type-II maximum likelihood
    (empirical Bayes) in ``LGM.fit``, optimizing over ``[lower, upper]`` starting
    from ``initial``. The optimizer searches in log space, and the MAP penalty
    (when a ``prior`` is declared) is evaluated on the native scale with no
    Jacobian correction; the ``transform`` field is reserved for future use and
    not yet wired into inference. This applies to both the exact-Gaussian and
    Laplace engines; the estimate is reported on ``result.hyperparameters``,
    keyed by ``name``. When ``prior`` is declared, it is added as a native-scale
    log penalty to that objective, turning the estimate into a MAP-II estimate;
    ``result.diagnostics`` records whether any declared hyperparameter was
    penalized this way. Full posterior *integration* over hyperparameters
    (marginals) remains future INLA work.
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
            object.__setattr__(self, "initial", _positive_real(self.initial, "initial"))
            lower = self.initial * 1e-3 if self.lower is None else _positive_real(self.lower, "lower")
            upper = self.initial * 1e3 if self.upper is None else _positive_real(self.upper, "upper")
            if not lower < self.initial:
                raise ValueError("lower bound must be below initial")
            if not self.initial < upper:
                raise ValueError("upper bound must be above initial")
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)
        else:
            # identity / logit: any finite initial; bounds may be deferred (None)
            if type(self.initial) not in (int, float) or not math.isfinite(self.initial):
                raise ValueError("initial must be a finite real value")
            object.__setattr__(self, "initial", float(self.initial))
            if self.lower is not None and not math.isfinite(self.lower):
                raise ValueError("lower must be a finite real value or None")
            if self.upper is not None and not math.isfinite(self.upper):
                raise ValueError("upper must be a finite real value or None")
            if self.lower is not None and self.upper is not None and not self.lower < self.upper:
                raise ValueError("lower must be below upper")

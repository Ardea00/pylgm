"""Per-parameter transforms between the natural and unconstrained scales."""

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import expit, log_expit


@runtime_checkable
class Transform(Protocol):
    """Maps a natural parameter theta to an unconstrained internal value u."""

    def to_internal(self, theta: float) -> float: ...
    def from_internal(self, u: float) -> float: ...
    def log_abs_jacobian(self, u: float) -> float: ...  # log|d theta / d u|
    def contains(self, theta: float) -> bool: ...
    def domain_description(self) -> str: ...  # human-readable natural domain


class LogTransform:
    """Positive parameters: u = log(theta), theta = exp(u)."""

    def to_internal(self, theta: float) -> float:
        return float(np.log(theta))

    def from_internal(self, u: float) -> float:
        return float(np.exp(u))

    def log_abs_jacobian(self, u: float) -> float:
        return float(u)

    def contains(self, theta: float) -> bool:
        return np.isfinite(theta) and theta > 0.0

    def domain_description(self) -> str:
        return "the positive reals"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogTransform)

    def __hash__(self) -> int:
        return hash(LogTransform)


class LogitTransform:
    """Bounded parameters on (lower, upper): theta = lower + (upper-lower)*sigmoid(u)."""

    def __init__(self, lower: float, upper: float) -> None:
        if not (np.isfinite(lower) and np.isfinite(upper)) or lower >= upper:
            raise ValueError("LogitTransform requires finite lower < upper")
        self.lower = float(lower)
        self.upper = float(upper)

    @property
    def _width(self) -> float:
        return self.upper - self.lower

    def to_internal(self, theta: float) -> float:
        z = (theta - self.lower) / self._width
        return float(np.log(z) - np.log1p(-z))

    def from_internal(self, u: float) -> float:
        return float(self.lower + self._width * expit(u))

    def log_abs_jacobian(self, u: float) -> float:
        # d theta/d u = width * sigmoid(u) * (1 - sigmoid(u))
        return float(np.log(self._width) + log_expit(u) + log_expit(-u))

    def contains(self, theta: float) -> bool:
        return np.isfinite(theta) and self.lower < theta < self.upper

    def domain_description(self) -> str:
        return f"the open interval ({self.lower:.6g}, {self.upper:.6g})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LogitTransform)
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def __hash__(self) -> int:
        return hash((LogitTransform, self.lower, self.upper))


class IdentityTransform:
    """Unconstrained parameters on the whole real line: u = theta, theta = u."""

    def to_internal(self, theta: float) -> float:
        return float(theta)

    def from_internal(self, u: float) -> float:
        return float(u)

    def log_abs_jacobian(self, u: float) -> float:
        return 0.0

    def contains(self, theta: float) -> bool:
        return bool(np.isfinite(theta))

    def domain_description(self) -> str:
        return "the real line"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdentityTransform)

    def __hash__(self) -> int:
        return hash(IdentityTransform)

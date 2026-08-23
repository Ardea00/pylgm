"""Prior distributions used by model hyperparameters."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq


_LOG_FLOAT_MAX = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
_LOG_TWO = math.log(2.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)


def _finite_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real value")
    return float(value)


def _positive_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _log_absolute_difference(left: float, right: float) -> float:
    if left == right:
        return -math.inf
    if (left < 0.0 < right) or (right < 0.0 < left):
        left_magnitude = abs(left)
        right_magnitude = abs(right)
        scale = max(left_magnitude, right_magnitude)
        return math.log(scale) + math.log(
            left_magnitude / scale + right_magnitude / scale
        )
    return math.log(abs(left - right))


def _exp_or_infinity(log_value: float) -> float:
    if log_value > _LOG_FLOAT_MAX:
        return math.inf
    try:
        return math.exp(log_value)
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class GaussianPrior:
    """A Gaussian prior parameterized by mean and precision."""

    mean: float
    precision: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _finite_real(self.mean, "mean"))
        object.__setattr__(self, "precision", _positive_real(self.precision, "precision"))

    def logpdf(self, value: float) -> float:
        value = _finite_real(value, "value")
        normalization = 0.5 * (math.log(self.precision) - _LOG_TWO_PI)
        log_difference = _log_absolute_difference(value, self.mean)
        if log_difference == -math.inf:
            return normalization
        log_quadratic = (
            -_LOG_TWO + math.log(self.precision) + 2.0 * log_difference
        )
        quadratic = _exp_or_infinity(log_quadratic)
        if not math.isfinite(quadratic):
            return -math.inf
        result = normalization - quadratic
        return result if math.isfinite(result) else -math.inf


@dataclass(frozen=True)
class PCPrecision:
    """Penalised-complexity prior for a Gaussian precision."""

    upper_sd: float
    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "upper_sd", _positive_real(self.upper_sd, "upper_sd"))
        alpha = _finite_real(self.alpha, "alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        object.__setattr__(self, "alpha", alpha)

    def logpdf(self, value: float) -> float:
        value = _positive_real(value, "value")
        log_value = math.log(value)
        log_rate = math.log(-math.log(self.alpha)) - math.log(self.upper_sd)
        penalty = _exp_or_infinity(log_rate - 0.5 * log_value)
        if not math.isfinite(penalty):
            return -math.inf
        result = log_rate - _LOG_TWO - 1.5 * log_value - penalty
        return result if math.isfinite(result) else -math.inf


@dataclass(frozen=True)
class PCBYM2Phi:
    """PC prior for the BYM2 mixing parameter (Riebler et al. 2016, §3.3).

    Declared with a calibration ``P(phi < upper) = alpha`` and *bound* to a
    graph's spectrum at compile time, because the distance scale depends on the
    eigenvalues of the scaled structure's generalized inverse.
    """

    upper: float = 0.5
    alpha: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        if type(self.upper) not in (int, float) or not 0.0 < self.upper < 1.0:
            raise ValueError("upper must lie strictly inside (0, 1)")
        if type(self.alpha) not in (int, float) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        object.__setattr__(self, "upper", float(self.upper))
        object.__setattr__(self, "alpha", float(self.alpha))

    def bind(self, gamma: "np.ndarray") -> "_BoundPCBYM2Phi":
        """Bind to the positive eigenvalues of ``pinv(R*)`` for a graph."""
        return _BoundPCBYM2Phi(np.asarray(gamma, dtype=float), self.upper, self.alpha)

    def logpdf(self, value: float) -> float:
        raise ValueError(
            "PCBYM2Phi must be bound to a graph before use; attach it to a BYM2 "
            "effect's phi hyperparameter so the compiler can bind it"
        )


class _BoundPCBYM2Phi:
    """A ``PCBYM2Phi`` bound to one graph's spectrum."""

    def __init__(self, gamma: "np.ndarray", upper: float, alpha: float) -> None:
        gamma = np.asarray(gamma, dtype=float)
        if gamma.ndim != 1 or gamma.size == 0 or not np.all(gamma > 0):
            raise ValueError("gamma must be a non-empty vector of positive eigenvalues")
        self._gamma = gamma
        self._shift = float(np.sum((gamma - 1.0) ** 2))
        self.upper = float(upper)
        self.alpha = float(alpha)
        d_upper = self.distance(self.upper)
        d_one = self.distance(1.0)
        if not np.isfinite(d_one) or d_one <= 0.0:
            raise ValueError("BYM2 PC prior distance scale is degenerate for this graph")
        floor = d_upper / d_one
        if not floor < self.alpha < 1.0:
            raise ValueError(
                f"alpha={self.alpha} is not attainable for this graph; the PC prior "
                f"admits alpha in ({floor:.6g}, 1)"
            )
        self._d_one = d_one
        self._rate = _solve_pc_rate(d_upper, d_one, self.alpha)
        # log(-expm1(x)) rather than log1p(-exp(x)): stable when rate*d_one is
        # tiny, which happens for an alpha just above the attainable floor.
        self._log_norm = float(np.log(-np.expm1(-self._rate * d_one)))

    def _offsets(self, phi: float) -> "np.ndarray":
        """``t - 1`` where ``t = 1 - phi + phi*gamma``, computed without cancellation."""
        return phi * (self._gamma - 1.0)

    def _kld(self, phi: float) -> float:
        # log1p(t - 1) rather than log(t): forming t and taking its log subtracts
        # two nearly equal O(1) numbers for small phi, which destroys the KLD (and
        # hence the density) near the phi = 0 base model.
        offsets = self._offsets(phi)
        return 0.5 * float(np.sum(offsets - np.log1p(offsets)))

    def distance(self, phi: float) -> float:
        return float(np.sqrt(max(2.0 * self._kld(phi), 0.0)))

    def distance_derivative(self, phi: float) -> float:
        distance = self.distance(phi)
        if distance < 1e-8:
            return float(np.sqrt(self._shift / 2.0))
        offsets = self._offsets(phi)
        # (gamma - 1) * (1 - 1/t) == (gamma - 1) * (t - 1)/t, stable near t = 1.
        derivative = 0.5 * float(
            np.sum((self._gamma - 1.0) * offsets / (1.0 + offsets))
        )
        return derivative / distance

    def logpdf(self, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("BYM2 phi must lie in [0, 1]")
        derivative = self.distance_derivative(value)
        if derivative <= 0.0 or not np.isfinite(derivative):
            raise ValueError("BYM2 PC prior derivative must be finite and positive")
        return float(
            np.log(self._rate)
            - self._rate * self.distance(value)
            + np.log(derivative)
            - self._log_norm
        )


def _solve_pc_rate(d_upper: float, d_one: float, alpha: float) -> float:
    """Solve (1 - exp(-rate*d_upper)) / (1 - exp(-rate*d_one)) = alpha."""

    def objective(rate: float) -> float:
        return (
            -np.expm1(-rate * d_upper) / -np.expm1(-rate * d_one)
        ) - alpha

    # The objective rises from d_upper/d_one (as rate -> 0) to 1 (as rate -> inf),
    # so bracket adaptively at BOTH ends: an alpha just above the attainable floor
    # calibrates to an arbitrarily small rate, which a fixed lower bound misses.
    low, high = 1e-9, 1.0
    for _ in range(200):
        if objective(high) > 0.0:
            break
        high *= 2.0
    else:  # pragma: no cover - unreachable for an attainable alpha
        raise ValueError("could not bracket the BYM2 PC prior rate")
    for _ in range(400):
        if objective(low) < 0.0:
            break
        low /= 2.0
    else:  # pragma: no cover - unreachable for an attainable alpha
        raise ValueError("could not bracket the BYM2 PC prior rate")
    return float(brentq(objective, low, high, xtol=1e-300, rtol=1e-14))

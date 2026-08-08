"""Prior distributions used by model hyperparameters."""

from dataclasses import dataclass
import math


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

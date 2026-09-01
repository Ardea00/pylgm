"""Likelihood definitions for latent Gaussian models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.special import gammaln, erf, gammaincc, gammainc, betainc, digamma, polygamma, gamma as gamma_fn

from pylgm.links import IdentityLink, LogLink, LogitLink
from pylgm.exceptions import DataContractError, ModelValidationError
from pylgm.parameters import Hyperparameter


def _positive_real(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real value")
    return float(value)


def _resolve_phi(phi: object, values: Mapping[str, float]) -> float:
    """Pick a family's own phi out of ``values`` (Hyperparameter) or use the fixed float.

    Lenient — unlike ``Gaussian.materialize`` which requires an exact key set —
    because the optimiser resolves phi jointly with the block precisions and so
    passes them all together in one mapping.
    """
    if isinstance(phi, Hyperparameter):
        if not isinstance(values, Mapping) or phi.name not in values:
            raise ValueError(f"values must contain {phi.name!r}")
        return values[phi.name]
    return phi


class _CompiledLikelihood:
    """Shared per-observation binding hook for compiled likelihoods.

    Default is a no-op: one compiled likelihood serves both the observed-row fit
    calls and the all-row prediction call unchanged. Families needing per-row
    data (Binomial trials; survival event/entry) override this to bind the
    vectors named in ``aux``.
    """

    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "_CompiledLikelihood":
        return self

    def restrict(self, observed: np.ndarray) -> "_CompiledLikelihood":
        """Re-index any row-indexed internal state into the observed-row subspace.

        Default is a no-op: ordinary likelihoods carry no row masks, and their
        per-row aux vectors are bound by ``for_observations`` instead. Only
        :class:`CompiledMixture` overrides this.
        """
        return self


@dataclass(frozen=True)
class CompiledGaussian(_CompiledLikelihood):
    """A Gaussian likelihood with a resolved standard deviation."""

    sigma: float
    link: IdentityLink = field(default_factory=IdentityLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _positive_real(self.sigma, "sigma"))

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        residual = np.asarray(y, dtype=float) - np.asarray(eta, dtype=float)
        n = residual.size
        return float(
            -0.5 * (n * np.log(2 * np.pi * self.variance) + residual @ residual / self.variance)
        )

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - np.asarray(eta, dtype=float)) / self.variance

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(eta).shape, 1.0 / self.variance, dtype=float)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(eta, dtype=float).shape)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta, dtype=float)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return np.asarray(eta_mean, dtype=float)

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return -0.5 * (np.log(2 * np.pi * self.variance) + (y - eta) ** 2 / self.variance)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return 0.5 * (1.0 + erf((y - eta) / (np.sqrt(self.variance) * np.sqrt(2.0))))

    def validate_response(self, y: np.ndarray) -> None:
        return None


@dataclass(frozen=True)
class Gaussian:
    """A Gaussian likelihood with a fixed or optimisable standard deviation."""

    sigma: float | Hyperparameter = 1.0
    link: IdentityLink = field(default_factory=IdentityLink, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sigma, Hyperparameter):
            return
        object.__setattr__(self, "sigma", _positive_real(self.sigma, "sigma"))

    def materialize(self, values: Mapping[str, float]) -> CompiledGaussian:
        if isinstance(self.sigma, Hyperparameter):
            if not isinstance(values, Mapping):
                raise ValueError("values must be a mapping")
            expected = {self.sigma.name}
            if set(values) != expected:
                raise ValueError(
                    f"values must contain exactly {self.sigma.name!r}"
                )
            return CompiledGaussian(values[self.sigma.name])
        return CompiledGaussian(self.sigma)


@dataclass(frozen=True)
class CompiledPoisson(_CompiledLikelihood):
    """A Poisson likelihood with a canonical log link."""

    link: LogLink = field(default_factory=LogLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)

        return float(np.sum(y * eta - np.exp(eta) - gammaln(y + 1.0)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self.link.inverse(eta)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -np.exp(np.asarray(eta, dtype=float))

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float) + 0.5 * np.asarray(eta_variance, dtype=float))

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return y * eta - np.exp(eta) - gammaln(y + 1.0)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return gammaincc(np.floor(y) + 1.0, np.exp(eta))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.round(y)):
            raise DataContractError("Poisson responses must be non-negative integers")


@dataclass(frozen=True)
class CompiledBernoulli(_CompiledLikelihood):
    """A Bernoulli likelihood with a canonical logit link."""

    link: LogitLink = field(default_factory=LogitLink, init=False)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        # log(1+exp(eta)) computed stably
        softplus = np.logaddexp(0.0, eta)
        return float(np.sum(y * eta - softplus))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self.link.inverse(eta)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(eta)
        return p * (1.0 - p)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(np.asarray(eta, dtype=float))
        return -p * (1.0 - p) * (1.0 - 2.0 * p)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float))  # point estimate; variance ignored

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return y * eta - np.logaddexp(0.0, eta)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.where(y >= 1.0, 1.0, 1.0 - self.link.inverse(eta))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isin(y, (0.0, 1.0))):
            raise DataContractError("Bernoulli responses must be 0 or 1")


@dataclass(frozen=True)
class CompiledBinomial(_CompiledLikelihood):
    """Aggregated Bernoulli trials with a logit link.

    ``trials`` is the per-row number of trials n, bound via ``for_observations``;
    the response y is the success count and the fitted mean is n*p. The n=1 case
    reduces exactly to :class:`CompiledBernoulli`.
    """

    trials: object = None
    link: LogitLink = field(default_factory=LogitLink, init=False)

    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "CompiledBinomial":
        if aux is None:
            return self
        return CompiledBinomial(aux["trials"])

    def _n(self) -> np.ndarray:
        return np.asarray(self.trials, dtype=float)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        # Full density incl. the log-binomial-coefficient constant, so the sum
        # matches pointwise_log_density and the LML is comparable across families.
        # The constant is independent of eta, so the mode and gradient are unchanged.
        return float(np.sum(self.pointwise_log_density(eta, y)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self._n() * self.link.inverse(eta)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(eta)
        return self._n() * p * (1.0 - p)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.link.inverse(np.asarray(eta, dtype=float))
        return -self._n() * p * (1.0 - p) * (1.0 - 2.0 * p)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self._n() * self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self._n() * self.link.inverse(np.asarray(eta_mean, dtype=float))  # counts; variance ignored

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        n = self._n()
        log_choose = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
        return log_choose + y * eta - n * np.logaddexp(0.0, eta)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        k = np.floor(np.asarray(y, dtype=float))
        n = self._n()
        p = self.link.inverse(eta)
        # P(Y<=k) = I_{1-p}(n-k, k+1); guard a=0 at k=n (where the cdf is 1).
        return np.where(k >= n, 1.0, betainc(np.maximum(n - k, 1.0), k + 1.0, 1.0 - p))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y < 0.0) or np.any(y != np.floor(y)):
            raise DataContractError("binomial responses must be non-negative integers")
        if np.any(y > self._n()):
            raise DataContractError("binomial responses must not exceed the number of trials")


@dataclass(frozen=True)
class CompiledNegativeBinomial(_CompiledLikelihood):
    """A negative-binomial (NB2) likelihood with a log link and dispersion phi.

    Mean ``mu = exp(eta)``; ``Var = mu + mu^2/phi``. ``phi -> inf`` recovers the
    Poisson (its derivatives below reduce to ``CompiledPoisson``'s in that limit).
    """

    phi: float
    link: LogLink = field(default_factory=LogLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", _positive_real(self.phi, "phi"))

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(self.pointwise_log_density(eta, y)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        y = np.asarray(y, dtype=float)
        return self.phi * (y - mu) / (self.phi + mu)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        return mu * self.phi / (mu + self.phi)  # Fisher information in eta, > 0

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        y = np.asarray(y, dtype=float)
        return -self.phi * (self.phi + y) * mu * (self.phi - mu) / (self.phi + mu) ** 3

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float) + 0.5 * np.asarray(eta_variance, dtype=float))

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = np.exp(eta)
        phi = self.phi
        return (
            gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0)
            + phi * np.log(phi / (phi + mu))
            + y * np.log(mu / (phi + mu))
        )

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = np.exp(eta)
        phi = self.phi
        return betainc(phi, np.floor(y) + 1.0, phi / (phi + mu))

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y != np.round(y)):
            raise DataContractError("negative-binomial responses must be non-negative integers")


@dataclass(frozen=True)
class CompiledGamma(_CompiledLikelihood):
    """A gamma likelihood with a log link and precision phi (shape ``a = phi``).

    Mean ``mu = exp(eta)``; ``Var = mu^2/phi``.
    """

    phi: float
    link: LogLink = field(default_factory=LogLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", _positive_real(self.phi, "phi"))

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(self.pointwise_log_density(eta, y)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        y = np.asarray(y, dtype=float)
        return self.phi * (y - mu) / mu

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(eta).shape, float(self.phi), dtype=float)  # Fisher info = phi

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        y = np.asarray(y, dtype=float)
        return self.phi * y / mu

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float) + 0.5 * np.asarray(eta_variance, dtype=float))

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = np.exp(eta)
        phi = self.phi
        return phi * np.log(phi / mu) - gammaln(phi) + (phi - 1.0) * np.log(y) - (phi / mu) * y

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = np.exp(eta)
        return gammainc(self.phi, (self.phi / mu) * y)

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise DataContractError("gamma responses must be positive")


@dataclass(frozen=True)
class CompiledBeta(_CompiledLikelihood):
    """A beta likelihood with a logit link and precision phi.

    Responses ``y in (0, 1)``; mean ``mu = sigmoid(eta)``; distribution
    ``Beta(mu*phi, (1-mu)*phi)``, so ``Var = mu(1-mu)/(1+phi)``.
    """

    phi: float
    link: LogitLink = field(default_factory=LogitLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", _positive_real(self.phi, "phi"))

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(self.pointwise_log_density(eta, y)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        y = np.asarray(y, dtype=float)
        phi = self.phi
        dlog_dmu = phi * ((np.log(y) - np.log1p(-y)) - (digamma(mu * phi) - digamma((1.0 - mu) * phi)))
        return dlog_dmu * mu * (1.0 - mu)  # chain through dmu/deta = mu(1-mu)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        phi = self.phi
        # Fisher information in eta: I_mu * (dmu/deta)^2, > 0 since trigamma > 0.
        return (mu * (1.0 - mu)) ** 2 * phi ** 2 * (polygamma(1, mu * phi) + polygamma(1, (1.0 - mu) * phi))

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        mu = self.link.inverse(eta)
        phi = self.phi
        t = np.log(y) - np.log1p(-np.asarray(y, dtype=float))
        L1 = phi * (t - (digamma(mu * phi) - digamma((1.0 - mu) * phi)))
        L2 = -phi ** 2 * (polygamma(1, mu * phi) + polygamma(1, (1.0 - mu) * phi))
        L3 = -phi ** 3 * (polygamma(2, mu * phi) - polygamma(2, (1.0 - mu) * phi))
        mp = mu * (1.0 - mu)                       # dmu/deta
        mpp = mp * (1.0 - 2.0 * mu)                # d2mu/deta2
        mppp = mp * (1.0 - 6.0 * mu + 6.0 * mu ** 2)  # d3mu/deta3
        return L3 * mp ** 3 + 3.0 * L2 * mp * mpp + L1 * mppp

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self.link.inverse(eta)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.link.inverse(np.asarray(eta_mean, dtype=float))  # point estimate; variance ignored

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = self.link.inverse(eta)
        phi = self.phi
        return (
            gammaln(phi) - gammaln(mu * phi) - gammaln((1.0 - mu) * phi)
            + (mu * phi - 1.0) * np.log(y)
            + ((1.0 - mu) * phi - 1.0) * np.log1p(-y)
        )

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        y = np.asarray(y, dtype=float)
        mu = self.link.inverse(eta)
        phi = self.phi
        return betainc(mu * phi, (1.0 - mu) * phi, y)

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y <= 0.0) or np.any(y >= 1.0):
            raise DataContractError("beta responses must lie strictly in (0, 1)")


@dataclass(frozen=True)
class CompiledWeibullSurv(_CompiledLikelihood):
    """Weibull proportional-hazards survival with shape ``alpha`` and a log link.

    ``eta`` is the log relative risk: hazard ``h(t)=alpha*t**(alpha-1)*exp(eta)``,
    cumulative ``H(t)=t**alpha*exp(eta)``, survival ``S(t)=exp(-H(t))``. The event
    indicator ``delta`` and the left-truncation entry time ``v`` are bound per
    observation via ``for_observations``; ``entry=None`` means no truncation
    (``v=0``). Exponential survival is this class with ``alpha=1``.
    """

    alpha: float
    event: object = None          # delta, bound via for_observations
    entry: object = None          # v (left truncation), bound via for_observations
    link: LogLink = field(default_factory=LogLink, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _positive_real(self.alpha, "alpha"))

    def for_observations(self, aux: "Mapping[str, np.ndarray] | None") -> "CompiledWeibullSurv":
        if aux is None:
            return self
        return CompiledWeibullSurv(self.alpha, aux.get("event"), aux.get("entry"))

    def _delta(self) -> np.ndarray:
        return np.asarray(self.event, dtype=float)

    def _at_risk(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A = (t**alpha - v**alpha) * exp(eta), the at-risk cumulative hazard."""
        eta = np.asarray(eta, dtype=float)
        t = np.asarray(y, dtype=float)
        v = np.zeros_like(t) if self.entry is None else np.asarray(self.entry, dtype=float)
        return (t ** self.alpha - v ** self.alpha) * np.exp(eta)

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(self.pointwise_log_density(eta, y)))

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._delta() - self._at_risk(eta, y)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._at_risk(eta, y)          # -d2/deta2 = A > 0 (posterior precision PD)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -self._at_risk(eta, y)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        # E[T] = scale * Gamma(1 + 1/alpha), scale = exp(-eta/alpha)
        return np.exp(-np.asarray(eta, dtype=float) / self.alpha) * gamma_fn(1.0 + 1.0 / self.alpha)

    def response_prediction(self, eta_mean: np.ndarray, eta_variance: np.ndarray) -> np.ndarray:
        return self.response_mean(np.asarray(eta_mean, dtype=float))  # point estimate; variance ignored

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        t = np.asarray(y, dtype=float)
        d = self._delta()
        a = self.alpha
        return d * (math.log(a) + (a - 1.0) * np.log(t) + eta) - self._at_risk(eta, y)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        t = np.asarray(y, dtype=float)
        return 1.0 - np.exp(-(t ** self.alpha) * np.exp(eta))    # marginal event CDF (entry ignored)

    def validate_response(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
            raise DataContractError("survival response (follow-up time) must be positive")


@dataclass(frozen=True, init=False)
class CompiledMixture(_CompiledLikelihood):
    """A likelihood that dispatches per row to one of several sub-likelihoods.

    ``parts`` pairs a boolean row mask with the likelihood governing those rows.
    The masks must be disjoint and together cover every row: an overlap would
    double-count an observation and a gap would silently drop one from the
    likelihood, both of which produce a plausible-looking wrong answer.

    Every method of the likelihood interface is row-separable, so this is a
    scatter over the parts and needs no change to the inference engines.
    """

    parts: tuple[tuple[np.ndarray, object], ...]
    n_rows: int

    def __init__(self, parts, n_rows: int) -> None:
        parts = tuple((np.asarray(mask, dtype=bool), lk) for mask, lk in parts)
        if not parts:
            raise ModelValidationError("mixture must have at least one part")
        n_rows = int(n_rows)
        for mask, _ in parts:
            if mask.ndim != 1 or mask.size != n_rows:
                raise ModelValidationError(
                    f"mixture masks must be 1-D boolean arrays of length {n_rows}"
                )
        counts = np.sum([mask.astype(np.int64) for mask, _ in parts], axis=0)
        if np.any(counts > 1):
            raise ModelValidationError("mixture masks must be disjoint")
        if np.any(counts < 1):
            raise ModelValidationError("mixture masks must cover every row")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "n_rows", n_rows)

    def restrict(self, observed: np.ndarray) -> "CompiledMixture":
        observed = np.asarray(observed, dtype=bool)
        return CompiledMixture(
            tuple((mask[observed], lk) for mask, lk in self.parts), int(observed.sum())
        )

    def for_observations(self, aux) -> "CompiledMixture":
        if aux is None:
            return self
        return CompiledMixture(
            tuple(
                (mask, lk.for_observations(_mask_aux(aux, mask))) for mask, lk in self.parts
            ),
            self.n_rows,
        )

    def _scatter(self, method: str, *arrays: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=float)
        for mask, likelihood in self.parts:
            out[mask] = getattr(likelihood, method)(*(a[mask] for a in arrays))
        return out

    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float:
        return float(
            sum(lk.log_likelihood(eta[mask], y[mask]) for mask, lk in self.parts)
        )

    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("gradient", eta, y)

    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("working_weights", eta, y)

    def third_derivative(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("third_derivative", eta, y)

    def pointwise_log_density(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("pointwise_log_density", eta, y)

    def cdf(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._scatter("cdf", eta, y)

    def response_mean(self, eta: np.ndarray) -> np.ndarray:
        return self._scatter("response_mean", eta)

    def response_prediction(
        self, eta_mean: np.ndarray, eta_variance: np.ndarray
    ) -> np.ndarray:
        return self._scatter("response_prediction", eta_mean, eta_variance)

    def validate_response(self, y: np.ndarray) -> None:
        for mask, likelihood in self.parts:
            likelihood.validate_response(y[mask])


def _mask_aux(aux, mask: np.ndarray):
    """Slice each aux vector down to one part's rows; ``None`` entries pass through."""
    if aux is None:
        return None
    sliced = {
        key: (None if value is None else np.asarray(value)[mask])
        for key, value in aux.items()
    }
    return sliced if any(v is not None for v in sliced.values()) else None


@dataclass(frozen=True)
class Poisson:
    """A Poisson likelihood with a canonical log link."""

    def materialize(self, values: Mapping[str, float]) -> CompiledPoisson:
        return CompiledPoisson()


@dataclass(frozen=True)
class Bernoulli:
    """A Bernoulli likelihood with a canonical logit link."""

    def materialize(self, values: Mapping[str, float]) -> CompiledBernoulli:
        return CompiledBernoulli()


@dataclass(frozen=True)
class Binomial:
    """A binomial likelihood; ``trials`` names the per-row trial-count column."""

    trials: str

    def __post_init__(self) -> None:
        if not isinstance(self.trials, str) or not self.trials:
            raise ValueError("Binomial trials must be a non-empty column name")

    def materialize(self, values: Mapping[str, float]) -> CompiledBinomial:
        # The trials vector is data; the compiler binds it via for_observations.
        return CompiledBinomial()


@dataclass(frozen=True)
class NegativeBinomial:
    """A negative-binomial (NB2) likelihood with a fixed or optimisable dispersion phi."""

    phi: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.phi, Hyperparameter):
            return
        object.__setattr__(self, "phi", _positive_real(self.phi, "phi"))

    def materialize(self, values: Mapping[str, float]) -> CompiledNegativeBinomial:
        return CompiledNegativeBinomial(_resolve_phi(self.phi, values))


@dataclass(frozen=True)
class Gamma:
    """A gamma likelihood with a fixed or optimisable precision phi."""

    phi: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.phi, Hyperparameter):
            return
        object.__setattr__(self, "phi", _positive_real(self.phi, "phi"))

    def materialize(self, values: Mapping[str, float]) -> CompiledGamma:
        return CompiledGamma(_resolve_phi(self.phi, values))


@dataclass(frozen=True)
class Beta:
    """A beta likelihood with a fixed or optimisable precision phi."""

    phi: float | Hyperparameter = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.phi, Hyperparameter):
            return
        object.__setattr__(self, "phi", _positive_real(self.phi, "phi"))

    def materialize(self, values: Mapping[str, float]) -> CompiledBeta:
        return CompiledBeta(_resolve_phi(self.phi, values))


def _survival_columns(event: object, entry: object) -> None:
    if not isinstance(event, str) or not event:
        raise ValueError("survival event must be a non-empty column name")
    if entry is not None and (not isinstance(entry, str) or not entry):
        raise ValueError("survival entry must be a non-empty column name or None")


@dataclass(frozen=True)
class WeibullSurv:
    """Weibull proportional-hazards survival; ``event`` names the 0/1 indicator column.

    ``shape`` is the Weibull shape alpha (fixed float or optimisable Hyperparameter).
    ``entry`` optionally names a left-truncation (delayed-entry) time column.
    """

    event: str
    shape: float | Hyperparameter = 1.0
    entry: str | None = None

    def __post_init__(self) -> None:
        _survival_columns(self.event, self.entry)
        if isinstance(self.shape, Hyperparameter):
            return
        object.__setattr__(self, "shape", _positive_real(self.shape, "shape"))

    def materialize(self, values: Mapping[str, float]) -> CompiledWeibullSurv:
        return CompiledWeibullSurv(_resolve_phi(self.shape, values))


@dataclass(frozen=True)
class ExponentialSurv:
    """Exponential proportional-hazards survival (Weibull with shape alpha = 1)."""

    event: str
    entry: str | None = None

    def __post_init__(self) -> None:
        _survival_columns(self.event, self.entry)

    def materialize(self, values: Mapping[str, float]) -> CompiledWeibullSurv:
        return CompiledWeibullSurv(1.0)

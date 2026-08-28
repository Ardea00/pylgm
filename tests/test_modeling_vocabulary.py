import math
import warnings

import numpy as np
import pytest

from pylgm import Bernoulli, Poisson
from pylgm.likelihoods import CompiledGaussian, Gaussian
from pylgm.links import IdentityLink
from pylgm.parameters import Hyperparameter
from pylgm.priors import GaussianPrior, PCPrecision
from pylgm.exceptions import DataContractError


def test_gaussian_vocabulary_is_immutable_and_validated():
    parameter = Hyperparameter("sigma", 2.0)
    assert Gaussian(parameter).link == IdentityLink()
    assert CompiledGaussian(2.0).variance == 4.0
    with pytest.raises(ValueError, match="positive"):
        CompiledGaussian(0.0)


def test_pc_precision_matches_change_of_variables_density():
    prior = PCPrecision(upper_sd=1.0, alpha=0.01)
    rate = -np.log(0.01)
    expected = np.log(rate / 2.0) - 1.5 * np.log(4.0) - rate / 2.0
    assert prior.logpdf(4.0) == pytest.approx(expected)


def test_prior_and_parameter_reject_invalid_values():
    with pytest.raises(ValueError, match="alpha"):
        PCPrecision(1.0, alpha=1.0)
    with pytest.raises(ValueError, match="precision"):
        GaussianPrior(0.0, precision=-1.0)
    with pytest.raises(ValueError, match="name"):
        Hyperparameter("", 1.0)


@pytest.mark.parametrize("initial", [0.0, -1.0, True, np.nan, np.inf])
def test_hyperparameter_rejects_invalid_initial_values(initial):
    with pytest.raises(ValueError, match="initial"):
        Hyperparameter("precision", initial)


def test_identity_link_cannot_be_constructed_with_another_name():
    with pytest.raises(TypeError):
        IdentityLink("log")
    with pytest.raises(TypeError):
        IdentityLink(name="log")


def test_compiled_gaussian_variance_handles_finite_float_extremes():
    maximum = float(np.finfo(float).max)
    minimum = float(np.nextafter(0.0, 1.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert CompiledGaussian(maximum).variance == math.inf
        assert CompiledGaussian(minimum).variance == 0.0


def test_gaussian_prior_extreme_tails_are_stable():
    maximum = float(np.finfo(float).max)
    minimum = float(np.nextafter(0.0, 1.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        representable_tail = GaussianPrior(-maximum, minimum).logpdf(maximum)
        underflowed_density = GaussianPrior(0.0, 1.0).logpdf(maximum)

    assert math.isfinite(representable_tail)
    assert representable_tail < 0.0
    assert underflowed_density == -math.inf


def test_gaussian_prior_matches_ordinary_log_density():
    expected = 0.5 * math.log(1.0 / math.pi) - 4.0

    assert GaussianPrior(mean=1.0, precision=2.0).logpdf(3.0) == pytest.approx(
        expected
    )


def test_pc_precision_extreme_scales_never_warn_or_return_nan():
    maximum = float(np.finfo(float).max)
    minimum = float(np.nextafter(0.0, 1.0))
    prior = PCPrecision(upper_sd=minimum, alpha=0.01)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        representable_tail = prior.logpdf(maximum)
        underflowed_density = prior.logpdf(minimum)

    assert math.isfinite(representable_tail)
    assert representable_tail < 0.0
    assert underflowed_density == -math.inf


# Tests for Poisson and Bernoulli likelihoods with GLM protocol


def test_poisson_glm_pieces():
    like = Poisson().materialize({})
    assert like.link.name == "log"
    eta = np.array([0.0, np.log(2.0)])
    y = np.array([1.0, 3.0])
    np.testing.assert_allclose(like.response_mean(eta), [1.0, 2.0])
    np.testing.assert_allclose(like.gradient(eta, y), y - np.exp(eta))   # y - mu
    np.testing.assert_allclose(like.working_weights(eta, y), np.exp(eta))  # mu
    # log-normal predictive mean for the log link
    np.testing.assert_allclose(
        like.response_prediction(np.array([0.0]), np.array([2.0])), [np.exp(1.0)]
    )


def test_poisson_rejects_non_count_response():
    like = Poisson().materialize({})
    with pytest.raises(DataContractError, match="non-negative integer"):
        like.validate_response(np.array([1.0, -2.0]))
    with pytest.raises(DataContractError, match="non-negative integer"):
        like.validate_response(np.array([1.5]))


def test_bernoulli_glm_pieces():
    like = Bernoulli().materialize({})
    assert like.link.name == "logit"
    eta = np.array([0.0])
    y = np.array([1.0])
    np.testing.assert_allclose(like.response_mean(eta), [0.5])
    np.testing.assert_allclose(like.gradient(eta, y), [0.5])       # y - p
    np.testing.assert_allclose(like.working_weights(eta, y), [0.25])  # p(1-p)


def test_bernoulli_rejects_non_binary_response():
    like = Bernoulli().materialize({})
    with pytest.raises(DataContractError, match="0 or 1"):
        like.validate_response(np.array([0.0, 2.0]))


def test_compiled_gaussian_satisfies_glm_protocol():
    from pylgm.likelihoods import CompiledGaussian

    like = CompiledGaussian(2.0)  # sigma=2 -> variance 4
    assert like.link.name == "identity"
    eta = np.array([1.0, 2.0])
    y = np.array([3.0, 2.0])
    np.testing.assert_allclose(like.gradient(eta, y), (y - eta) / 4.0)
    np.testing.assert_allclose(like.working_weights(eta, y), [0.25, 0.25])
    np.testing.assert_allclose(like.response_mean(eta), eta)


def test_hyperparameter_default_bounds_bracket_initial():
    hp = Hyperparameter("p", initial=2.0)
    assert hp.lower == pytest.approx(2.0e-3)
    assert hp.upper == pytest.approx(2.0e3)


def test_hyperparameter_explicit_bounds():
    hp = Hyperparameter("p", initial=1.0, lower=0.1, upper=10.0)
    assert (hp.lower, hp.upper) == (0.1, 10.0)


def test_hyperparameter_rejects_bounds_not_bracketing_initial():
    with pytest.raises(ValueError, match="lower"):
        Hyperparameter("p", initial=1.0, lower=2.0)
    with pytest.raises(ValueError, match="upper"):
        Hyperparameter("p", initial=1.0, upper=0.5)
    with pytest.raises(ValueError):
        Hyperparameter("p", initial=1.0, lower=-1.0)


# Tests for per-observation likelihood methods


def test_pointwise_log_density_sums_to_log_likelihood():
    from pylgm.likelihoods import CompiledPoisson, CompiledBernoulli
    eta = np.array([0.3, -0.5, 1.2])
    y = np.array([1.0, 0.0, 2.0])
    for like in (CompiledGaussian(1.3), CompiledPoisson(), CompiledBernoulli()):
        yy = y if not isinstance(like, CompiledBernoulli) else np.array([1.0, 0.0, 1.0])
        pw = like.pointwise_log_density(eta, yy)
        np.testing.assert_allclose(pw.sum(), like.log_likelihood(eta, yy))


def test_cdf_matches_scipy():
    from scipy.stats import norm, poisson
    from pylgm.likelihoods import CompiledPoisson, CompiledBernoulli
    eta = np.array([0.2, -1.0, 0.5])
    g = CompiledGaussian(2.0)
    np.testing.assert_allclose(g.cdf(eta, np.array([0.0, -1.0, 1.5])),
                               norm.cdf(np.array([0.0, -1.0, 1.5]), loc=eta, scale=2.0))
    p = CompiledPoisson()
    y = np.array([0.0, 2.0, 5.0])
    np.testing.assert_allclose(p.cdf(eta, y), poisson.cdf(y, np.exp(eta)), atol=1e-12)
    b = CompiledBernoulli()
    yb = np.array([1.0, 0.0, 1.0])
    expected = np.where(yb >= 1, 1.0, 1.0 - 1.0 / (1.0 + np.exp(-eta)))
    np.testing.assert_allclose(b.cdf(eta, yb), expected)


def test_third_derivative_values_and_finite_difference():
    from pylgm.likelihoods import CompiledPoisson, CompiledBernoulli
    eta = np.array([0.2, -0.6, 1.1])
    y = np.zeros(3)
    np.testing.assert_allclose(CompiledGaussian(1.7).third_derivative(eta, y), np.zeros(3))
    np.testing.assert_allclose(CompiledPoisson().third_derivative(eta, y), -np.exp(eta))
    p = 1.0 / (1.0 + np.exp(-eta))
    np.testing.assert_allclose(CompiledBernoulli().third_derivative(eta, y),
                               -p * (1 - p) * (1 - 2 * p))
    # cross-check: d3 log p = -d/deta working_weights, via central finite difference
    for like in (CompiledPoisson(), CompiledBernoulli()):
        h = 1e-4
        fd = -(like.working_weights(eta + h, y) - like.working_weights(eta - h, y)) / (2 * h)
        np.testing.assert_allclose(like.third_derivative(eta, y), fd, atol=1e-5)


# PyINLA GLM families: negative-binomial, gamma, beta (dispersion/precision phi)


def test_negative_binomial_glm_pieces():
    from pylgm import NegativeBinomial

    like = NegativeBinomial(3.0).materialize({})
    assert like.link.name == "log"
    eta = np.array([0.0, np.log(2.0)])
    mu = np.exp(eta)
    y = np.array([1.0, 4.0])
    np.testing.assert_allclose(like.response_mean(eta), mu)
    np.testing.assert_allclose(like.gradient(eta, y), 3.0 * (y - mu) / (3.0 + mu))
    np.testing.assert_allclose(like.working_weights(eta, y), mu * 3.0 / (mu + 3.0))
    assert np.all(like.working_weights(eta, y) > 0)


def test_gamma_glm_pieces():
    from pylgm import Gamma

    like = Gamma(2.0).materialize({})
    assert like.link.name == "log"
    eta = np.array([0.0, np.log(3.0)])
    mu = np.exp(eta)
    y = np.array([1.5, 2.0])
    np.testing.assert_allclose(like.response_mean(eta), mu)
    np.testing.assert_allclose(like.gradient(eta, y), 2.0 * (y - mu) / mu)
    np.testing.assert_allclose(like.working_weights(eta, y), [2.0, 2.0])  # constant shape


def test_beta_glm_pieces():
    from pylgm import Beta

    like = Beta(5.0).materialize({})
    assert like.link.name == "logit"
    eta = np.array([0.0, 0.5])
    mu = 1.0 / (1.0 + np.exp(-eta))
    y = np.array([0.3, 0.7])
    np.testing.assert_allclose(like.response_mean(eta), mu)
    assert np.all(like.working_weights(eta, y) > 0)
    # response prediction ignores latent variance (mirrors Bernoulli)
    np.testing.assert_allclose(like.response_prediction(eta, np.array([2.0, 2.0])), mu)


def test_new_families_reject_out_of_support_response():
    from pylgm import Beta, Gamma, NegativeBinomial

    with pytest.raises(DataContractError, match="non-negative integer"):
        NegativeBinomial(1.0).materialize({}).validate_response(np.array([1.5]))
    with pytest.raises(DataContractError, match="non-negative integer"):
        NegativeBinomial(1.0).materialize({}).validate_response(np.array([-1.0]))
    with pytest.raises(DataContractError, match="positive"):
        Gamma(1.0).materialize({}).validate_response(np.array([1.0, 0.0]))
    with pytest.raises(DataContractError, match=r"\(0, 1\)"):
        Beta(1.0).materialize({}).validate_response(np.array([0.5, 1.0]))


def test_new_families_density_and_cdf_match_scipy():
    from scipy.stats import beta as beta_dist, gamma as gamma_dist, nbinom
    from pylgm import Beta, Gamma, NegativeBinomial

    phi = 2.5
    # negative-binomial: NB2 with n=phi, p=phi/(phi+mu)
    nb = NegativeBinomial(phi).materialize({})
    eta = np.array([0.2, -0.4, 0.7])
    mu = np.exp(eta)
    y = np.array([0.0, 2.0, 5.0])
    nb.validate_response(y)
    np.testing.assert_allclose(nb.pointwise_log_density(eta, y).sum(), nb.log_likelihood(eta, y))
    np.testing.assert_allclose(nb.pointwise_log_density(eta, y),
                               nbinom.logpmf(y, phi, phi / (phi + mu)), atol=1e-10)
    np.testing.assert_allclose(nb.cdf(eta, y), nbinom.cdf(y, phi, phi / (phi + mu)), atol=1e-10)

    # gamma: shape a=phi, scale=mu/phi
    gm = Gamma(phi).materialize({})
    yg = np.array([0.5, 1.5, 3.0])
    np.testing.assert_allclose(gm.pointwise_log_density(eta, yg).sum(), gm.log_likelihood(eta, yg))
    np.testing.assert_allclose(gm.pointwise_log_density(eta, yg),
                               gamma_dist.logpdf(yg, phi, scale=mu / phi), atol=1e-10)
    np.testing.assert_allclose(gm.cdf(eta, yg), gamma_dist.cdf(yg, phi, scale=mu / phi), atol=1e-10)

    # beta: Beta(mu*phi, (1-mu)*phi)
    bt = Beta(phi).materialize({})
    p = 1.0 / (1.0 + np.exp(-eta))
    yb = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(bt.pointwise_log_density(eta, yb).sum(), bt.log_likelihood(eta, yb))
    np.testing.assert_allclose(bt.pointwise_log_density(eta, yb),
                               beta_dist.logpdf(yb, p * phi, (1 - p) * phi), atol=1e-10)
    np.testing.assert_allclose(bt.cdf(eta, yb), beta_dist.cdf(yb, p * phi, (1 - p) * phi), atol=1e-10)


def test_new_families_gradient_and_third_derivative_finite_difference():
    from pylgm import Beta, Gamma, NegativeBinomial

    eta = np.array([0.2, -0.6, 0.4])
    cases = [
        (NegativeBinomial(2.0).materialize({}), np.array([0.0, 3.0, 1.0])),
        (Gamma(2.0).materialize({}), np.array([0.5, 1.5, 2.0])),
        (Beta(4.0).materialize({}), np.array([0.3, 0.6, 0.4])),
    ]
    h = 1e-5
    for like, y in cases:
        # gradient = d logL / d eta
        num_g = (like.log_likelihood(eta + h, y) - like.log_likelihood(eta - h, y)) / (2 * h)
        np.testing.assert_allclose(like.gradient(eta, y).sum(), num_g, atol=1e-5)
        # third_derivative = d/deta of the observed second derivative (per point)
        d2 = lambda e: (like.gradient(e + h, y) - like.gradient(e - h, y)) / (2 * h)
        fd3 = (d2(eta + h) - d2(eta - h)) / (2 * h)
        np.testing.assert_allclose(like.third_derivative(eta, y), fd3, atol=1e-3)

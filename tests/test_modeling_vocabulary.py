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

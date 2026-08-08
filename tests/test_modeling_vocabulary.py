import math
import warnings

import numpy as np
import pytest

from pylgm.likelihoods import CompiledGaussian, Gaussian
from pylgm.links import IdentityLink
from pylgm.parameters import Hyperparameter
from pylgm.priors import GaussianPrior, PCPrecision


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

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

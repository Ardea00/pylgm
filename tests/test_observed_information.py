"""The Laplace Hessian is the observed information at the mode, not Fisher's.

For a non-canonical link (negative binomial and gamma with a log link) the two
differ away from ``y = mu``; the Laplace approximation to the marginal
likelihood is defined by the observed curvature.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from pylgm import LGM, Fixed, Gamma, NegativeBinomial, Poisson, ZeroInflated

PRIOR_PRECISION = 0.5


def _data(name, n=10):
    rng = np.random.default_rng(0)
    if name == "nb":
        return NegativeBinomial(phi=2.0), rng.negative_binomial(2, 2 / 7.0, size=n).astype(float)
    return Gamma(phi=3.0), rng.gamma(3.0, 5.0 / 3.0, size=n)


@pytest.mark.parametrize("name", ["nb", "gamma"])
def test_working_weights_are_the_observed_information(name):
    family, y = _data(name)
    likelihood = family.materialize({})
    eta, step = np.linspace(-1.0, 3.0, y.size), 1e-5
    observed = -(likelihood.gradient(eta + step, y) - likelihood.gradient(eta - step, y)) / (2 * step)
    np.testing.assert_allclose(likelihood.working_weights(eta, y), observed, rtol=1e-6)


@pytest.mark.parametrize("name", ["nb", "gamma"])
def test_laplace_lml_uses_the_observed_curvature_and_improves_on_fisher(name):
    family, y = _data(name)
    likelihood = family.materialize({})
    model = LGM(response="y", likelihood=family,
                predictor=Fixed("1", prior_precision=PRIOR_PRECISION))
    result = model.fit(pd.DataFrame({"y": y}), engine="laplace")

    def loglik(b):
        return likelihood.log_likelihood(np.full(y.size, b), y)

    mode, step = result.mean[0], 1e-4
    curvature = -(loglik(mode + step) - 2 * loglik(mode) + loglik(mode - step)) / step**2
    laplace = (loglik(mode) - 0.5 * PRIOR_PRECISION * mode**2 + 0.5 * np.log(PRIOR_PRECISION)
               - 0.5 * np.log(PRIOR_PRECISION + curvature))
    assert result.log_marginal_likelihood == pytest.approx(laplace, abs=1e-6)

    shift = loglik(mode)
    exact = np.log(quad(
        lambda b: np.exp(loglik(b) - shift - 0.5 * PRIOR_PRECISION * b * b), -30, 30,
        points=[mode], limit=400,
    )[0]) + shift + 0.5 * np.log(PRIOR_PRECISION / (2 * np.pi))
    fisher_error = {"nb": 0.0105, "gamma": 0.0080}[name]  # the expected-information error
    assert abs(result.log_marginal_likelihood - exact) < 0.6 * fisher_error


def test_zero_inflation_reduces_to_the_observed_base_and_keeps_its_fisher_form():
    eta = np.linspace(-2.0, 3.0, 11)
    y = np.array([0, 5, 0, 2, 9, 0, 1, 0, 7, 3, 0], dtype=float)
    base = NegativeBinomial(phi=2.0).materialize({})
    at_zero_pi = ZeroInflated(NegativeBinomial(phi=2.0), pi=0.0).materialize({})
    np.testing.assert_allclose(at_zero_pi.working_weights(eta, y), base.working_weights(eta, y))

    zinb = ZeroInflated(NegativeBinomial(phi=2.0), pi=0.3).materialize({})
    mu, phi = np.exp(eta), 2.0
    s0 = phi * (0 - mu) / (phi + mu)
    _, _, _, f0_gap = zinb._at_zero(eta)
    expected = 0.7 * (phi * mu / (phi + mu) - f0_gap * s0**2)
    np.testing.assert_allclose(zinb.fisher_information(eta), expected, rtol=1e-12)


def test_canonical_links_are_unchanged():
    likelihood = Poisson().materialize({})
    eta = np.linspace(-1.0, 2.0, 5)
    np.testing.assert_allclose(likelihood.working_weights(eta, np.ones(5)), np.exp(eta))

"""Joint draws from the Laplace (Gaussian-at-the-mode) posterior."""

import numpy as np
import pandas as pd

from pylgm import RW1, Fixed, Hyperparameter, LGM, Poisson


def _model(precision=2.0):
    return LGM(response="y", likelihood=Poisson(),
               predictor=Fixed("1") + RW1("trend", "t", precision=precision))


def _frame():
    rng = np.random.default_rng(0)
    t = np.arange(25)
    return pd.DataFrame({"t": t, "y": rng.poisson(np.exp(1.0 + np.sin(t / 4.0)))})


def _assert_moments_match(draws, mean, variance):
    se = np.sqrt(variance / draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - mean) <= 5 * se + 1e-12)
    np.testing.assert_allclose(draws.var(axis=0), variance, rtol=0.06)


def test_laplace_draws_match_the_gaussian_approximation():
    result = _model().fit(_frame(), engine="laplace")
    draws = result.sample(20_000, np.random.default_rng(0))
    assert draws.shape == (20_000, 25)
    _assert_moments_match(draws, result.predictive_mean, result.predictive_variance)


def test_integrated_laplace_draws_mix_the_grid():
    precision = Hyperparameter("trend.precision", initial=2.0)
    result = _model(precision).fit(_frame(), engine="laplace", hyperparameters="integrate")
    draws = result.sample(20_000, np.random.default_rng(2))
    _assert_moments_match(draws, result.predictive_mean, result.predictive_variance)

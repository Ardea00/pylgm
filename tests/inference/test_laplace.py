import numpy as np
import pytest

from pylgm import Bernoulli, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import InferenceConvergenceError
from pylgm.inference.gaussian import fit_gaussian
from pylgm.inference.laplace import fit_laplace
import pandas as pd


def _panel(frame, panel=(), time="t"):
    return CanonicalPanel.from_frame(frame, DataConfig(time=time, response="y", panel=panel))


def test_laplace_reproduces_exact_gaussian_on_gaussian_likelihood():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 2.5, 4.0]})
    model = LGM("y", Gaussian(0.7), Fixed("1 + x", prior_precision=0.5) + IID("g", index="t", precision=1.5), time="t")
    compiled = compile_lgm(model, _panel(frame))
    exact = fit_gaussian(compiled)
    laplace = fit_laplace(compiled)
    np.testing.assert_allclose(laplace.mean, exact.mean, atol=1e-8)
    np.testing.assert_allclose(laplace.covariance, exact.covariance, atol=1e-8)
    np.testing.assert_allclose(
        laplace.log_marginal_likelihood, exact.log_marginal_likelihood, atol=1e-8
    )


def test_laplace_poisson_mode_solves_the_score_equation():
    # single fixed intercept, flat prior -> mode is log(mean(y))
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [2.0, 3.0, 5.0, 6.0]})
    model = LGM("y", Poisson(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    np.testing.assert_allclose(result.mean, [np.log(4.0)], atol=1e-6)
    # response-scale fitted mean uses log-normal correction, so > exp(mean)
    assert result.fitted_mean[0] > np.exp(result.mean[0]) - 1e-9
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_bernoulli_intercept_matches_logit_of_rate():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [1.0, 1.0, 0.0, 1.0]})
    model = LGM("y", Bernoulli(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    p = 0.75
    np.testing.assert_allclose(result.mean, [np.log(p / (1 - p))], atol=1e-6)


def test_laplace_predicts_unobserved_rows():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, np.nan]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    assert result.predictive_mean.shape == (3,)
    assert np.isfinite(result.fitted_mean).all()


def test_laplace_raises_on_non_convergence():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 5.0, 10.0], "y": [0.0, 1.0, 20.0]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    with pytest.raises(InferenceConvergenceError):
        fit_laplace(compile_lgm(model, _panel(frame)), max_iterations=1, tolerance=1e-12)

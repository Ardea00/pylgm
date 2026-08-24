import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Fixed, Gaussian, Hyperparameter, LGM, Poisson
from pylgm.priors import PCPrecision


def _ar1_frame(n=40, rho_true=0.9, seed=0, tau=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = rng.normal(scale=np.sqrt(1.0 / tau))
    innovation = np.sqrt((1.0 - rho_true**2) / tau)
    for i in range(1, n):
        x[i] = rho_true * x[i - 1] + rng.normal(scale=innovation)
    return pd.DataFrame({"t": list(range(n)), "y": x + rng.normal(scale=0.2, size=n)})


def _model(rho, precision, likelihood=None):
    return LGM(
        response="y",
        predictor=Fixed("1") + AR1("trend", index="t", precision=precision, rho=rho),
        likelihood=likelihood or Gaussian(sigma=0.2),
    )


def test_gaussian_plugin_fit():
    frame = _ar1_frame()
    result = _model(0.9, 1.0).fit(frame)
    marginals = result.latent_marginals("trend")
    assert marginals.mean.shape == (40,)
    assert np.all(np.isfinite(marginals.mean))


def test_poisson_plugin_fit():
    frame = pd.DataFrame({"t": list(range(6)), "y": [2.0, 3.0, 5.0, 6.0, 4.0, 2.0]})
    result = _model(0.8, 1.0, likelihood=Poisson()).fit(frame, engine="laplace")
    assert np.all(np.isfinite(result.latent_marginals("trend").mean))


def _estimate_rho(frame):
    rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("trend.precision", initial=1.0,
                         prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    return _model(rho, tau).fit(frame).hyperparameters["trend.rho"]


def test_rho_estimated_and_tracks_the_simulated_truth():
    # A single realization is noisy, so average seeds and require that strongly
    # autocorrelated data separates clearly from independent data.
    seeds = (0, 1, 2, 3, 4)
    strong = [_estimate_rho(_ar1_frame(rho_true=0.95, seed=s)) for s in seeds]
    weak = [_estimate_rho(_ar1_frame(rho_true=0.0, seed=s)) for s in seeds]
    assert all(-1.0 < value < 1.0 for value in strong + weak)
    assert np.mean(strong) > np.mean(weak) + 0.3


def test_rho_integrated_by_inla():
    rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("trend.precision", initial=1.0,
                         prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    result = _model(rho, tau).fit(_ar1_frame(), hyperparameters="integrate")
    marginals = result.hyperparameter_marginals()
    assert "trend.rho" in marginals and "trend.precision" in marginals
    assert -1.0 < float(marginals["trend.rho"].mean[0]) < 1.0


@pytest.mark.parametrize("strategy", ["gaussian", "simplified_laplace", "laplace"])
def test_all_latent_strategies_accept_ar1(strategy):
    # AR1 is proper/unconstrained, so unlike RW1 every strategy must succeed.
    rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("trend.precision", initial=1.0,
                         prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    result = _model(rho, tau).fit(
        _ar1_frame(n=12), engine="exact_gaussian",
        hyperparameters="integrate", latent_strategy=strategy,
    )
    assert np.all(np.isfinite(result.latent_marginals("trend").mean))


def test_rho_hyperparameter_requires_logit_transform():
    from pylgm.exceptions import CompilationError

    rho = Hyperparameter("trend.rho", initial=0.5)  # defaults to "log"
    with pytest.raises(CompilationError, match="transform='logit'"):
        _model(rho, 1.0).fit(_ar1_frame())


def test_predict_works_and_unseen_level_points_at_the_nan_workflow():
    frame = _ar1_frame(n=10)
    result = _model(0.9, 1.0).fit(frame)
    prediction = result.predict(frame)
    assert np.allclose(prediction.predictive_mean, result.predictive_mean)
    with pytest.raises(ValueError, match="NaN response"):
        result.predict(pd.DataFrame({"t": [999]}))


def test_poisson_integrate_converges():
    """Regression for the Laplace Newton stall rescue.

    The dense Newton loop used to stall just above its absolute gradient
    tolerance at some grid point, and one bad point aborted the whole
    integration (8/10 seeds for this model). The decrement rescue fixes it.
    """
    rng = np.random.default_rng(0)
    n = 14
    signal = np.zeros(n)
    for i in range(1, n):
        signal[i] = 0.8 * signal[i - 1] + rng.normal(scale=0.3)
    frame = pd.DataFrame({
        "t": list(range(n)),
        "y": rng.poisson(np.exp(1.5 + signal)).astype(float),
    })
    rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("trend.precision", initial=1.0,
                         prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    result = _model(rho, tau, likelihood=Poisson()).fit(
        frame, engine="laplace", hyperparameters="integrate"
    )
    assert np.all(np.isfinite(result.latent_marginals("trend").mean))

import numpy as np
import pandas as pd
import pytest

from pylgm import BYM2, Fixed, Gaussian, Hyperparameter, LGM, PCBYM2Phi, Poisson
from pylgm.priors import PCPrecision


def _chain(n):
    return {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}


def _spatial_frame(n=12, rho=0.95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(scale=0.3)
    x -= x.mean()
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": x + rng.normal(scale=0.1, size=n)})


def _iid_frame(n=12, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": y - y.mean()})


def _model(graph, phi, precision, likelihood=None):
    return LGM(
        response="y",
        predictor=Fixed("1") + BYM2("region", index="region", graph=graph, phi=phi, precision=precision),
        likelihood=likelihood or Gaussian(sigma=0.1),
    )


def test_gaussian_plugin_fit():
    result = _model(_chain(12), 0.7, 1.0).fit(_spatial_frame())
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (12,)
    assert np.all(np.isfinite(marginals.mean))


def test_poisson_plugin_fit():
    frame = pd.DataFrame({
        "region": [str(i) for i in range(6)],
        "y": np.array([2, 3, 5, 6, 4, 2], dtype=float),
    })
    result = _model(_chain(6), 0.7, 1.0, likelihood=Poisson()).fit(frame, engine="laplace")
    assert np.all(np.isfinite(result.latent_marginals("region").mean))


def _estimate_phi(frame, seed_graph=None):
    phi = Hyperparameter("region.phi", initial=0.5, transform="logit", prior=PCBYM2Phi())
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    result = _model(seed_graph or _chain(12), phi, tau).fit(frame)
    return result.hyperparameters["region.phi"]


# phi is weakly identified in a single small realization -- hence the
# averaging over seeds below -- so individual fits sit near the interval edge.
@pytest.mark.filterwarnings("ignore:empirical-Bayes estimate")
def test_phi_estimated_and_tracks_the_simulated_structure():
    # Averaged over seeds, spatially-dependent data must yield a clearly larger
    # phi than independent data (a single small realization is weakly identified).
    seeds = (0, 1, 2, 3, 4)
    spatial = [_estimate_phi(_spatial_frame(seed=s)) for s in seeds]
    independent = [_estimate_phi(_iid_frame(seed=s)) for s in seeds]
    assert all(0.0 < value < 1.0 for value in spatial + independent)
    assert np.mean(spatial) > np.mean(independent)


def test_phi_integrated_by_inla():
    phi = Hyperparameter("region.phi", initial=0.5, transform="logit", prior=PCBYM2Phi())
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    result = _model(_chain(12), phi, tau).fit(_spatial_frame(), hyperparameters="integrate")
    marginals = result.hyperparameter_marginals()
    assert "region.phi" in marginals and "region.precision" in marginals
    assert 0.0 < float(marginals["region.phi"].mean[0]) < 1.0


@pytest.mark.parametrize("strategy", ["gaussian", "simplified_laplace", "laplace"])
def test_all_latent_strategies_accept_bym2(strategy):
    # BYM2 is unconstrained, so unlike Besag every strategy must succeed.
    phi = Hyperparameter("region.phi", initial=0.5, transform="logit", prior=PCBYM2Phi())
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    result = _model(_chain(8), phi, tau).fit(
        _spatial_frame(n=8), engine="exact_gaussian",
        hyperparameters="integrate", latent_strategy=strategy,
    )
    assert np.all(np.isfinite(result.latent_marginals("region").mean))


def test_phi_hyperparameter_requires_logit_transform():
    from pylgm.exceptions import CompilationError

    phi = Hyperparameter("region.phi", initial=0.5)  # defaults to "log"
    with pytest.raises(CompilationError, match="transform='logit'"):
        _model(_chain(12), phi, 1.0).fit(_spatial_frame())

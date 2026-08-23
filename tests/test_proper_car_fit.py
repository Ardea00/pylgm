import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson, ProperCAR
from pylgm.priors import PCPrecision

GRAPH = {
    str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5]
    for i in range(6)
}


def _gaussian_frame():
    regions = [str(i) for i in range(6)]
    truth = np.sin(np.arange(6) / 2.0)
    return pd.DataFrame({"region": regions, "y": truth + 0.01 * np.arange(6)}), truth


def test_gaussian_plugin_fit():
    frame, truth = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + ProperCAR("region", index="region", graph=GRAPH, rho=0.9),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_poisson_plugin_fit():
    regions = [str(i) for i in range(6)]
    counts = np.array([2, 3, 5, 6, 4, 2], dtype=float)
    frame = pd.DataFrame({"region": regions, "y": counts})
    model = LGM(
        response="y",
        predictor=Fixed("1") + ProperCAR("region", index="region", graph=GRAPH, rho=0.9),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_integrate_over_tau_with_fixed_rho():
    frame, _ = _gaussian_frame()
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + ProperCAR("region", index="region", graph=GRAPH, rho=0.9, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    assert result.engine == "inla"
    assert "region.precision" in result.hyperparameter_marginals()
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)


def test_full_laplace_accepts_proper_car():
    # Proper CAR is unconstrained, so (unlike Besag) full Laplace must SUCCEED.
    frame, _ = _gaussian_frame()
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + ProperCAR("region", index="region", graph=GRAPH, rho=0.9, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(
        frame, engine="exact_gaussian", hyperparameters="integrate", latent_strategy="laplace"
    )
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_composes_with_iid():
    frame, _ = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + ProperCAR("region", index="region", graph=GRAPH, rho=0.9)
        + IID("noise", index="region"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)
    assert result.latent_marginals("region").mean.shape == (6,)
    assert result.latent_marginals("noise").mean.shape == (6,)

import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, IID, LGM, Poisson
from pylgm.exceptions import UnsupportedEngineError
from pylgm.priors import PCPrecision

# Small connected chain graph over regions "0".."5": i <-> i-1, i <-> i+1.
GRAPH = {
    str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 5]
    for i in range(6)
}


def _gaussian_frame():
    regions = [str(i) for i in range(6)]
    truth = np.sin(np.arange(6) / 2.0)
    return pd.DataFrame({"region": regions, "y": truth + 0.01 * np.arange(6)}), truth


def test_gaussian_plugin_fit_recovers_smooth_pattern():
    frame, truth = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=1.0),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)

    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))
    # ICAR smoothing should track the underlying smooth pattern reasonably well.
    assert np.corrcoef(marginals.mean, truth)[0, 1] > 0.8


def test_poisson_plugin_fit():
    regions = [str(i) for i in range(6)]
    counts = np.array([2, 3, 5, 6, 4, 2], dtype=float)
    frame = pd.DataFrame({"region": regions, "y": counts})
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH),
        likelihood=Poisson(),
    )
    result = model.fit(frame, engine="laplace")

    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_integrate_fit_with_hyperparameter_precision():
    frame, _ = _gaussian_frame()
    # A PC-prior regularizes the precision posterior, matching the pattern used
    # by the existing IID/RW integrate tests (see tests/test_model.py's
    # test_map_ii_sets_penalized_flag_and_estimate_both_engines) so empirical
    # Bayes has a well-posed interior mode instead of diverging.
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")

    assert result.engine == "inla"
    assert "region.precision" in result.hyperparameter_marginals()
    marginals = result.latent_marginals("region")
    assert marginals.mean.shape == (6,)
    assert np.all(np.isfinite(marginals.mean))


def test_full_laplace_rejects_besag():
    frame, _ = _gaussian_frame()
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=GRAPH, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )

    with pytest.raises(UnsupportedEngineError):
        model.fit(
            frame,
            engine="exact_gaussian",
            hyperparameters="integrate",
            latent_strategy="laplace",
        )


def test_composes_with_iid():
    frame, _ = _gaussian_frame()
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + Besag("region", index="region", graph=GRAPH)
        + IID("noise", index="region"),
        likelihood=Gaussian(sigma=0.1),
    )
    result = model.fit(frame)

    besag_marginals = result.latent_marginals("region")
    noise_marginals = result.latent_marginals("noise")
    assert besag_marginals.mean.shape == (6,)
    assert noise_marginals.mean.shape == (6,)
    assert np.all(np.isfinite(besag_marginals.mean))
    assert np.all(np.isfinite(noise_marginals.mean))

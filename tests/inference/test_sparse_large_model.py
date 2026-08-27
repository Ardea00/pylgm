import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM
from pylgm.priors import PCPrecision

# A connected ring over N regions -> intrinsic Besag (single sum-to-zero
# constraint). N chosen so latent dim = N + 1 (intercept) crosses
# _MAX_DENSE_LATENT_DIMENSION (4096), forcing the sparse route.
_N = 5000


@pytest.fixture
def large_besag_frame_and_model():
    graph = {str(i): [str((i - 1) % _N), str((i + 1) % _N)] for i in range(_N)}
    rng = np.random.default_rng(0)
    regions = [str(i) for i in range(_N)]
    y = np.sin(np.arange(_N) / 50.0) + 0.1 * rng.standard_normal(_N)
    frame = pd.DataFrame({"region": regions, "y": y})
    # PC-prior on the precision keeps empirical Bayes well-posed (same pattern as
    # tests/test_besag_fit.py::test_integrate_fit_with_hyperparameter_precision).
    hp = Hyperparameter(
        "region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)
    )
    model = LGM(
        response="y",
        predictor=Fixed("1") + Besag("region", index="region", graph=graph, precision=hp),
        likelihood=Gaussian(sigma=0.1),
    )
    return model, frame


def test_large_spatial_model_fits_mean_and_lml(large_besag_frame_and_model):
    model, frame = large_besag_frame_and_model
    assert model  # latent dim = _N + 1 > 4096
    result = model.fit(frame)  # default: optimize (empirical Bayes)
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.log_marginal_likelihood)
    assert result.hyperparameters  # estimated, non-empty
    assert all(np.isfinite(v) for v in result.hyperparameters.values())
    assert np.isfinite(result.predictive_mean).all()
    # Uncertainty is pending E-sparse-C.
    for accessor in (
        lambda: result.covariance,
        lambda: result.predictive_variance,
        lambda: result.latent_marginals(),
        lambda: result.predict(frame),
    ):
        with pytest.raises(NotImplementedError, match="E-sparse-C"):
            accessor()


def test_large_model_integrate_is_unsupported(large_besag_frame_and_model):
    model, frame = large_besag_frame_and_model
    with pytest.raises(NotImplementedError, match="E-sparse-C"):
        model.fit(frame, hyperparameters="integrate")

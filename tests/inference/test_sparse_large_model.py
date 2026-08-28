import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM
from pylgm.exceptions import DenseReferenceLimitError
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

    # E-sparse-C: the uncertainty surface now works past the dense guard --
    # marginal + predictive variances and predict() are finite (no dense oracle
    # at this size; small-model dense-equivalence is Tasks 3-6's job).
    marg = result.latent_marginals()
    assert marg.variance.shape[0] == result.mean.shape[0]
    assert np.isfinite(marg.variance).all() and (marg.variance >= 0).all()
    assert np.isfinite(result.predictive_variance).all()
    predicted = result.predict(frame)
    assert np.isfinite(predicted.predictive_mean).all()
    assert np.isfinite(predicted.predictive_variance).all()
    # The full covariance is a scale limit, not pending-C: it must not be
    # materialised at 5001 dim, and the error must say so (not "E-sparse-C").
    with pytest.raises(DenseReferenceLimitError) as exc:
        _ = result.covariance
    assert "E-sparse-C" not in str(exc.value)


def test_large_model_integrate_returns_finite_diagonal_uncertainty(large_besag_frame_and_model):
    """INLA integration past the dense guard: diagonal latent/predictive
    uncertainty is finite; the full covariance stays a scale limit."""
    model, frame = large_besag_frame_and_model
    result = model.fit(frame, hyperparameters="integrate")
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.log_marginal_likelihood)
    marg = result.latent_marginals()
    assert marg.variance.shape[0] == result.mean.shape[0]
    assert np.isfinite(marg.variance).all() and (marg.variance >= 0).all()
    assert np.isfinite(result.predictive_variance).all()
    with pytest.raises(DenseReferenceLimitError):
        _ = result.covariance

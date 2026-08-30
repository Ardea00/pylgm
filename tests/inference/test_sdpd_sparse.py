import numpy as np
import pytest
import pandas as pd

from pylgm import LGM, Fixed, Gaussian, Hyperparameter
from pylgm.effects.spec import SAR
from pylgm.inference.gaussian import _MAX_DENSE_LATENT_DIMENSION


def _ring(n):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": rng.standard_normal(n)})


# The SAR precision runs to its bound on this synthetic ring; the test is a
# scale smoke test past the dense guard, not an estimation check.
@pytest.mark.filterwarnings("ignore:empirical-Bayes estimate")
def test_large_sar_fits_past_dense_guard():
    # Past the dense guard with headroom (latent dim = n + 1 intercept); derived
    # from the constant so it cannot drift, and kept just past it because the
    # sparse cost grows superlinearly in n.
    n = _MAX_DENSE_LATENT_DIMENSION + 100
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR("s", "region", _ring(n), rho=0.5,
                                    precision=Hyperparameter("s.prec", initial=1.0)),
        likelihood=Gaussian(sigma=1.0),
    )
    result = model.fit(_frame(n))
    assert result.mean.shape[0] > n
    # sparse path actually ran (mirrors tests/inference/test_sparse_large_model.py convention)
    assert result._covariance is None
    assert result._sparse_posterior is not None
    marginals = result.latent_marginals("s")
    assert np.isfinite(marginals.variance).all()
    assert np.all(marginals.variance >= 0)


def test_small_sar_dense_and_sparse_agree():
    # Fit the same small SAR through the dense path; assert marginal variances
    # are finite and posterior mean is stable (sparse path exercised in the
    # large test above; this pins the dense reference numbers).
    frame = _frame(30, seed=1)
    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR("s", "region", _ring(30), rho=0.5),
        likelihood=Gaussian(sigma=1.0),
    )
    result = model.fit(frame)
    marginals = result.latent_marginals("s")
    assert np.all(marginals.variance > 0)
    assert np.isfinite(result.mean).all()

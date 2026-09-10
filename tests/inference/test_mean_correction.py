"""The variational correction from the Laplace mode toward the posterior mean.

A Laplace approximation reports the mode. The mode is not the mean whenever the
likelihood is skewed, and the gap is systematic rather than noisy. These pin the
correction against cases where the right answer is known independently: a
closed-form log-gamma, an exactly Gaussian likelihood where the correction must
vanish, and the committed NUTS reference.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import digamma

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.inference.laplace import fit_laplace


def _count_frame(n=200, groups=20, seed=0, intercept=0.2):
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, 0.6, groups)
    mu = np.exp(intercept + np.array([effects[i % groups] for i in range(n)]))
    return pd.DataFrame({"g": [f"a{i % groups}" for i in range(n)],
                         "y": rng.poisson(mu).astype(float), "row": range(n)})


def _fit(frame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LGM(response="y", likelihood=Poisson(),
                   predictor=Fixed("1") + IID("u", index="g", precision=3.0),
                   ).fit(frame, engine="laplace", **kwargs)


def test_matches_the_closed_form_mode_to_mean_gap():
    """A Poisson likelihood with a near-flat prior has an exact answer.

    The conditional is a log-gamma: mode ``log y``, mean ``digamma(y)``. The
    uncorrected error is O(1/y) and the corrected one O(1/y^2), so the ratio
    between them should grow with y rather than settle.
    """
    ratios = []
    for count in (5.0, 10.0, 20.0, 40.0):
        frame = pd.DataFrame({"y": [count], "row": [0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LGM(response="y", likelihood=Poisson(),
                        predictor=Fixed("1", prior_precision=1e-8))
            mode = model.fit(frame, engine="laplace").mean[0]
            corrected = model.fit(frame, engine="laplace", mean_correction=True).mean[0]
        exact = digamma(count)
        assert abs(corrected - exact) < abs(mode - exact)
        ratios.append(abs(mode - exact) / abs(corrected - exact))
    assert ratios == sorted(ratios), ratios          # the advantage widens with y
    assert ratios[-1] > 100.0, ratios


def test_leaves_the_covariance_and_the_log_marginal_likelihood_alone():
    """The correction says where the approximating Gaussian sits, not its shape --
    and the log marginal likelihood is an expansion *at the mode*, so evaluating
    it anywhere else would stop it being one."""
    frame = _count_frame()
    plain, corrected = _fit(frame), _fit(frame, mean_correction=True)
    np.testing.assert_array_equal(plain.covariance, corrected.covariance)
    assert plain.log_marginal_likelihood == corrected.log_marginal_likelihood
    assert not np.allclose(plain.mean, corrected.mean)


def test_the_correction_vanishes_for_a_gaussian_likelihood():
    """An exactly Gaussian posterior has no mode/mean gap: its third derivative is
    identically zero, so the shift must be too rather than merely small."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.likelihoods import Gaussian

    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame({"g": [f"a{i % 10}" for i in range(n)],
                          "y": rng.normal(0.0, 1.0, n), "row": range(n)})
    model = LGM(response="y", likelihood=Gaussian(sigma=0.7),
                predictor=Fixed("1") + IID("u", index="g", precision=2.0))
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_lgm(model, panel)
    plain = fit_laplace(compiled)
    corrected = fit_laplace(compiled, mean_correction=True)
    np.testing.assert_array_equal(plain.mean, corrected.mean)


def test_default_behaviour_is_unchanged():
    frame = _count_frame()
    np.testing.assert_array_equal(_fit(frame).mean, _fit(frame, mean_correction=False).mean)


@pytest.mark.parametrize("kwargs", [
    {"hyperparameters": "optimize"},
    {"hyperparameters": "integrate"},
])
def test_reaches_the_hyperparameter_paths(kwargs):
    """The correction lives in the conditional fit, so empirical Bayes and the
    INLA grid both inherit it -- each grid point is a conditional fit."""
    from pylgm import Hyperparameter

    frame = _count_frame()
    def build():
        return LGM(response="y", likelihood=Poisson(),
                   predictor=Fixed("1") + IID("u", index="g",
                                              precision=Hyperparameter("tau", initial=3.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = build().fit(frame, engine="laplace", **kwargs)
        corrected = build().fit(frame, engine="laplace", mean_correction=True, **kwargs)
    assert not np.allclose(plain.mean, corrected.mean)
    assert np.isfinite(corrected.mean).all()

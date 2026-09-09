"""The posterior marginal of a hyperparameter, tabulated from the INLA grid.

A precision's posterior is right-skewed; summarising it by its first two moments
gets the credible interval badly wrong. These pin the tabulated marginal against
a brute-force reference: the exact log posterior `log p(y|theta) + log pi(theta)`
evaluated on a dense theta grid, with no INLA in the loop.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM
from pylgm.compiler import compile_family
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.inference import fit_gaussian
from pylgm.inference.result import GaussianMarginals, TabulatedMarginals
from pylgm.priors import PCPrecision

PRIOR = PCPrecision(upper_sd=1.0, alpha=0.01)


def _frame(n=60, groups=10, seed=3):
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, 0.7, groups)
    y = np.array([1.2 + effects[i % groups] for i in range(n)]) + rng.normal(0.0, 0.7, n)
    return pd.DataFrame({"g": [f"a{i % groups}" for i in range(n)], "y": y, "row": range(n)})


def _model(extra=False):
    predictor = Fixed("1", prior_precision=1.0) + IID(
        "u", index="g", precision=Hyperparameter("tau", initial=2.0, prior=PRIOR)
    )
    if extra:
        predictor = predictor + IID(
            "v", index="row", precision=Hyperparameter("kappa", initial=2.0, prior=PRIOR)
        )
    return LGM(response="y", likelihood=Gaussian(sigma=0.7), predictor=predictor)


def _reference(model, frame, points=160):
    """The exact theta posterior, by direct evaluation on a dense grid."""
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    family = compile_family(model, panel)
    taus = np.exp(np.linspace(np.log(0.05), np.log(300.0), points))
    log_density = np.array([
        fit_gaussian(family.materialize({"tau": float(t)})).log_marginal_likelihood
        + PRIOR.logpdf(float(t))
        for t in taus
    ])
    return TabulatedMarginals(taus[None, :], np.exp(log_density - log_density.max())[None, :])


@pytest.fixture(scope="module")
def fitted():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = _frame()
        result = _model().fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    return frame, result


def test_single_hyperparameter_marginal_is_tabulated(fitted):
    _, result = fitted
    assert isinstance(result.hyperparameter_marginals()["tau"], TabulatedMarginals)


def test_tabulated_marginal_tracks_the_exact_posterior(fitted):
    frame, result = fitted
    reference = _reference(_model(), frame)
    reported = result.hyperparameter_marginals()["tau"]
    for probability in (0.025, 0.5, 0.975):
        expected = reference.quantile(probability)[0]
        assert reported.quantile(probability)[0] == pytest.approx(expected, rel=0.10), probability


def test_tabulated_marginal_beats_the_moment_match_it_replaced(fitted):
    """The reason for the change: a moment-matched Gaussian has the right mean
    and variance and the wrong shape, which shows up in the credible interval."""
    frame, result = fitted
    reference = _reference(_model(), frame)
    reported = result.hyperparameter_marginals()["tau"]
    collapsed = GaussianMarginals(reported.mean, reported.variance)

    def error(marginals):
        return np.mean([
            abs(marginals.quantile(q)[0] - reference.quantile(q)[0]) / reference.quantile(q)[0]
            for q in (0.025, 0.5, 0.975)
        ])

    assert error(reported) < error(collapsed) / 3.0
    assert reference.skewness[0] > 0.5      # the skew the collapse assumes away


def test_two_hyperparameters_keep_the_moment_match():
    """The grid is a lattice rotated onto the whitened Hessian's directions, so a
    single axis's marginal cannot be read off it; that case is documented as
    moment-matched rather than silently wrong."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _model(extra=True).fit(
            _frame(), engine="exact_gaussian", hyperparameters="integrate"
        )
    marginals = result.hyperparameter_marginals()
    assert set(marginals) == {"tau", "kappa"}
    assert all(isinstance(m, GaussianMarginals) for m in marginals.values())

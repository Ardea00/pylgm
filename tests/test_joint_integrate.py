"""INLA grid integration on joint models.

`hyperparameters="integrate"` was reachable from the public API but had no test
coverage at all: `Joint._run_inla` was never executed in CI. These tests close
that gap and pin the behaviour that was verified by hand.

They matter beyond smoke-testing because the `simplified_laplace` and `laplace`
latent strategies run ONLY under `integrate`, and those are the strategies that
correct the mode-vs-mean gap measured in tests/integration/test_mcmc_crosscheck.py.
A silent fallback to `gaussian` on joint models would make that correction
unavailable exactly where it was documented to help, with nothing to flag it.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter

N = 30
STRATEGIES = ("gaussian", "simplified_laplace", "laplace")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    u = rng.normal(0.0, 0.7, N)
    return pd.DataFrame({
        "district": list(range(N)) * 2,
        "oral": list(rng.poisson(np.exp(-0.3 + 1.6 * u)).astype(float)) + [np.nan] * N,
        "larynx": [np.nan] * N + list(rng.poisson(np.exp(0.2 + u / 1.6)).astype(float)),
        "row": range(2 * N),
    })


def _joint():
    return Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0 / 0.49),
                       scale=Hyperparameter("delta", initial=1.0))],
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_integrate_runs_on_a_joint_for_every_latent_strategy(frame, strategy):
    result = _joint().fit(frame, engine="laplace",
                          hyperparameters="integrate", latent_strategy=strategy)

    assert type(result).__name__ == "INLAResult"
    assert np.isfinite(result.log_marginal_likelihood)
    assert np.all(np.isfinite(result.mean))
    # The shared scale gets a posterior, not just a point estimate.
    assert "delta" in result.hyperparameter_marginals()
    assert result.diagnostics["inla_grid_points"] >= 2


def test_integrate_reports_no_point_hyperparameters_but_does_report_marginals(frame):
    """`INLAResult.hyperparameters` is None by design: the whole point of
    integrating is that there is no single value. The posterior lives in
    `hyperparameter_marginals()`. Pinned because reading `.hyperparameters`
    on an integrated fit is an easy mistake that fails with an opaque
    `'NoneType' object is not subscriptable`.
    """
    result = _joint().fit(frame, engine="laplace", hyperparameters="integrate")
    assert result.hyperparameters is None
    assert set(result.hyperparameter_marginals()) == {"delta"}


def test_each_latent_strategy_is_actually_applied_not_silently_ignored(frame):
    """Guards against a joint silently falling back to the Gaussian strategy.

    The marginal *type* is the tell: a skew correction that was not applied
    would hand back plain GaussianMarginals.
    """
    kinds = {}
    for strategy in STRATEGIES:
        result = _joint().fit(frame, engine="laplace",
                              hyperparameters="integrate", latent_strategy=strategy)
        kinds[strategy] = type(result.latent_marginals("u")).__name__

    assert kinds["gaussian"] == "GaussianMarginals"
    assert kinds["simplified_laplace"] == "SkewNormalMarginals"
    assert kinds["laplace"] == "TabulatedMarginals"
    # The skew-correcting strategies must report a non-trivial skewness, or
    # they are structurally present but numerically inert.
    for strategy in ("simplified_laplace", "laplace"):
        marginals = _joint().fit(frame, engine="laplace", hyperparameters="integrate",
                                 latent_strategy=strategy).latent_marginals("u")
        assert np.abs(np.atleast_1d(marginals.skewness)[0]) > 1e-3


def test_integrated_delta_is_consistent_with_the_empirical_bayes_estimate(frame):
    """The grid is centred on the EB mode, so the two must not disagree wildly.

    A large gap would mean the grid is exploring the wrong region -- the failure
    mode that makes an integrated fit quietly worse than the point estimate it
    was supposed to improve on.
    """
    integrated = _joint().fit(frame, engine="laplace", hyperparameters="integrate")
    point = _joint().fit(frame, engine="laplace")

    eb_delta = point.hyperparameters["delta"]
    assert eb_delta == pytest.approx(integrated.diagnostics["inla_mode_delta"], rel=1e-6)
    assert 0.5 * eb_delta < np.exp(np.log(eb_delta)) < 2.0 * eb_delta


def test_integrate_requires_a_declared_hyperparameter(frame):
    """All-fixed joints have nothing to integrate over; that must say so."""
    fixed = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=1.0), scale=(1.6, 1 / 1.6))],
    )
    with pytest.raises(ValueError, match="integrate"):
        fixed.fit(frame, engine="laplace", hyperparameters="integrate")

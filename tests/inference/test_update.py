"""result.update(new_rows) conditions a fitted exact-Gaussian posterior on new
observations at fixed hyperparameters. The oracle is a refit on all the data:
Gaussian conditioning is exact, so the two must agree to round-off."""

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import IID, RW1, Fixed, Gaussian, LGM

REGIONS, PERIODS = 4, 15


@pytest.fixture(params=["dense", "sparse"])
def engine(request, monkeypatch):
    if request.param == "sparse":
        monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    return request.param


def _panel():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        [(f"r{r}", t) for r in range(REGIONS) for t in range(PERIODS)], columns=["region", "t"]
    )
    frame["x"] = rng.normal(size=len(frame))
    trend = np.cumsum(rng.normal(scale=0.3, size=PERIODS))
    level = rng.normal(size=REGIONS)
    frame["y"] = (1.0 + 0.5 * frame["x"] + trend[frame["t"]]
                  + level[frame["region"].str[1:].astype(int)] + rng.normal(scale=0.4, size=len(frame)))
    return frame


MODEL = LGM(
    response="y",
    likelihood=Gaussian(0.4),
    predictor=Fixed("1 + x") + RW1("trend", index="t", precision=5.0)  # RW1: sum-to-zero constraint
    + IID("level", index="region", precision=1.0),
)


def test_update_matches_refit_on_all_rows(engine):
    full = _panel()
    first = full.copy()
    # vintage 1 sees periods < 12; later periods sit on the grid with a NaN response
    first.loc[first["t"] >= 12, "y"] = np.nan
    vintage2 = full[full["t"].between(12, 13)]
    vintage3 = full[full["t"] == 14]

    updated = MODEL.fit(first).update(vintage2).update(vintage3)
    refit = MODEL.fit(full)

    np.testing.assert_allclose(updated.mean, refit.mean, atol=1e-9)
    np.testing.assert_allclose(updated.log_marginal_likelihood, refit.log_marginal_likelihood, atol=1e-8)
    for block in ("fixed", "trend", "level"):
        np.testing.assert_allclose(
            updated.latent_marginals(block).std, refit.latent_marginals(block).std, rtol=1e-6
        )  # the sparse path's selected-inverse variances carry ~1e-7 relative round-off
    # the fitted grid is the same rows in the same (caller) order in both
    np.testing.assert_allclose(updated.predictive_mean, refit.predictive_mean, atol=1e-9)
    np.testing.assert_allclose(updated.predictive_variance, refit.predictive_variance, atol=1e-9)
    new = updated.predict(full.head(7))
    old = refit.predict(full.head(7))
    np.testing.assert_allclose(new.predictive_variance, old.predictive_variance, atol=1e-9)

    draws = updated.sample(40_000, rng=0)
    np.testing.assert_allclose(draws.mean(axis=0), refit.predictive_mean, atol=0.02)
    np.testing.assert_allclose(draws.std(axis=0), np.sqrt(refit.predictive_variance), rtol=0.03)


def test_update_ignores_nan_rows_and_rejects_new_levels(engine):
    full = _panel()
    first = full.copy()
    first.loc[first["t"] >= 12, "y"] = np.nan
    result = MODEL.fit(first)
    same = result.update(first[first["t"] == 13])  # still NaN: nothing to condition on
    np.testing.assert_allclose(same.mean, result.mean)
    unseen = full.head(1).assign(t=99)
    with pytest.raises(ValueError, match="NaN response"):
        result.update(unseen)


def test_update_respects_caller_row_order(engine):
    full = _panel().sample(frac=1.0, random_state=0).reset_index(drop=True)
    first = full.copy()
    first.loc[first["t"] >= 12, "y"] = np.nan
    updated = MODEL.fit(first).update(full[full["t"] >= 12].sample(frac=1.0, random_state=1))
    refit = MODEL.fit(full)
    np.testing.assert_allclose(updated.predictive_mean, refit.predictive_mean, atol=1e-9)
    np.testing.assert_allclose(updated.predictive_variance, refit.predictive_variance, atol=1e-9)


def test_sparse_predictive_variances_match_dense_covariance(monkeypatch):
    """Selected-inverse variances vs the full covariance, for covered rows,
    rows pairing unlinked levels (fallback), and arbitrary dense rows."""
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    frame = _panel()
    frame = frame[~((frame["region"] == "r0") & (frame["t"] > 3))]  # r0 x late t never observed
    result = MODEL.fit(frame)
    posterior = result._sparse_posterior
    covariance = posterior.covariance_dense()
    probe = _panel()                                                 # includes the unlinked cells
    design = result._sampler.design.toarray()
    from pylgm.inference.prediction import _design_for
    rows = np.vstack([design, _design_for(result.prediction_context, probe),
                      np.random.default_rng(0).normal(size=(5, covariance.shape[0]))])
    np.testing.assert_allclose(
        posterior.predictive_variances(rows), np.einsum("ij,jk,ik->i", rows, covariance, rows),
        rtol=1e-6, atol=1e-10,
    )  # the random dense rows cancel a ~1e6 vague-fixed-effect variance: ~3e-7 round-off


def test_one_large_update_matches_refit(engine):
    """k = 48 new rows against p = 21 latents: the Delta = V S^-1 V^T route."""
    full = _panel()
    first = full.assign(y=np.where(full["t"] < 3, full["y"], np.nan))
    updated = MODEL.fit(first).update(full[full["t"] >= 3])
    refit = MODEL.fit(full)
    np.testing.assert_allclose(updated.mean, refit.mean, atol=1e-9)
    np.testing.assert_allclose(updated.predictive_variance, refit.predictive_variance, atol=1e-9)

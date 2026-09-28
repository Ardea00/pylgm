"""result.update(new_rows) conditions a fitted posterior on new observations.
For an exact-Gaussian fit the oracle is a refit on all the data: Gaussian
conditioning is exact, so the two must agree to round-off. Laplace updates are
checked against the eta-space Laplace approximation and track a refit; an
integrated update is checked as Bayes' rule on the fitted grid."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

import pylgm.inference.gaussian as gaussian_engine
from pylgm import IID, RW1, Binomial, Fixed, Gaussian, Hyperparameter, LGM, Poisson
from pylgm.exceptions import UnsupportedEngineError
from pylgm.inference.update import _rows_mode
from pylgm.likelihoods import CompiledPoisson

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


# --- non-Gaussian rows and integrated hyperparameters -----------------------


def test_rows_mode_is_the_laplace_approximation_in_eta_space():
    rng = np.random.default_rng(0)
    root = rng.normal(size=(3, 3))
    s0, m, y = root @ root.T + 0.1 * np.eye(3), rng.normal(size=3), np.array([0.0, 3.0, 7.0])
    likelihood = CompiledPoisson()
    a, sqrt_w, _, log_evidence = _rows_mode(m, s0, y, likelihood)

    precision = np.linalg.inv(s0)
    def negative_log_joint(eta):
        return -likelihood.log_likelihood(eta, y) + 0.5 * (eta - m) @ precision @ (eta - m)
    mode = minimize(negative_log_joint, m, method="BFGS", options={"gtol": 1e-10}).x
    hessian = precision + np.diag(np.exp(mode))
    laplace = (-negative_log_joint(mode) - 0.5 * np.linalg.slogdet(s0)[1]
               - 0.5 * np.linalg.slogdet(hessian)[1])
    np.testing.assert_allclose(m + s0 @ a, mode, atol=1e-6)
    np.testing.assert_allclose(sqrt_w ** 2, np.exp(mode), rtol=1e-6)
    np.testing.assert_allclose(log_evidence, laplace, atol=1e-8)


def _counts(shuffle=False):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        [(f"r{r}", t) for r in range(6) for t in range(30)], columns=["region", "t"]
    )
    frame["x"] = rng.normal(size=len(frame))
    trend = np.cumsum(rng.normal(scale=0.1, size=30))
    level = rng.normal(scale=0.3, size=6)
    frame["n"] = 40.0
    eta = 0.3 * frame["x"] + trend[frame["t"]] + level[frame["region"].str[1:].astype(int)]
    frame["y"] = rng.poisson(np.exp(1.5 + eta)).astype(float)
    frame["k"] = rng.binomial(40, 1 / (1 + np.exp(-eta))).astype(float)
    return frame.sample(frac=1.0, random_state=0).reset_index(drop=True) if shuffle else frame


@pytest.mark.parametrize("likelihood, response", [(Poisson(), "y"), (Binomial(trials="n"), "k")])
def test_laplace_update_tracks_refit(engine, likelihood, response):
    """Not exact: the old rows' curvature stays at the old mode. The gap is
    second order in the mode shift, so small next to the posterior spread."""
    model = LGM(response=response, likelihood=likelihood,
                predictor=Fixed("1 + x") + RW1("trend", index="t", precision=50.0)
                + IID("level", index="region", precision=5.0))
    full = _counts(shuffle=True)
    first = full.assign(**{response: np.where(full["t"] < 25, full[response], np.nan)})
    updated = (model.fit(first, engine="laplace")
               .update(full[full["t"].between(25, 27)]).update(full[full["t"] >= 28]))
    refit = model.fit(full, engine="laplace")

    spread = refit.latent_marginals().std
    assert np.max(np.abs(updated.mean - refit.mean) / spread) < 0.05
    np.testing.assert_allclose(updated.latent_marginals().std, spread, rtol=0.03)
    np.testing.assert_allclose(updated.fitted_mean, refit.fitted_mean, rtol=0.02)
    assert abs(updated.log_marginal_likelihood - refit.log_marginal_likelihood) < 0.2
    assert updated.diagnostics["updated_rows"] == int((full["t"] >= 25).sum())


GAUSSIAN_INTEGRATED = LGM(
    response="y", likelihood=Gaussian(Hyperparameter("sigma", 0.5)),
    predictor=Fixed("1 + x") + RW1("trend", index="t", precision=Hyperparameter("tau", 5.0))
    + IID("level", index="region", precision=1.0),
)


def _fixed_theta(theta):
    return LGM(response="y", likelihood=Gaussian(theta["sigma"]),
               predictor=Fixed("1 + x") + RW1("trend", index="t", precision=theta["tau"])
               + IID("level", index="region", precision=1.0))


def test_integrated_update_is_bayes_rule_on_the_grid():
    """Each grid point's Gaussian update is exact, so the updated INLA result is
    the fitted grid reweighted by p(y_new | y_old, theta) -- checked against
    fixed-theta fits on the old and on all rows."""
    full = _panel().sample(frac=1.0, random_state=0).reset_index(drop=True)
    first = full.assign(y=np.where(full["t"] < 10, full["y"], np.nan))
    base = GAUSSIAN_INTEGRATED.fit(first, hyperparameters="integrate")
    updated = base.update(full[full["t"] >= 10])

    thetas = base._grid.thetas
    old = [_fixed_theta(theta).fit(first) for theta in thetas]
    new = [_fixed_theta(theta).fit(full) for theta in thetas]
    evidence = np.array([n.log_marginal_likelihood - o.log_marginal_likelihood
                         for o, n in zip(old, new)])
    weights = base._grid.weights * np.exp(evidence - logsumexp(evidence, b=base._grid.weights))
    np.testing.assert_allclose(updated._grid.weights, weights, atol=1e-9)
    np.testing.assert_allclose(updated.mean, sum(w * n.mean for w, n in zip(weights, new)), atol=1e-8)
    np.testing.assert_allclose(
        updated.predictive_mean, sum(w * n.predictive_mean for w, n in zip(weights, new)), atol=1e-8
    )
    np.testing.assert_allclose(
        updated.log_marginal_likelihood - base.log_marginal_likelihood,
        logsumexp(evidence, b=base._grid.weights), atol=1e-8,
    )
    for name in ("sigma", "tau"):
        np.testing.assert_allclose(
            updated.hyperparameter_marginals()[name].mean,
            sum(w * theta[name] for w, theta in zip(weights, thetas)), rtol=1e-9,
        )
    assert updated.sample(10, rng=0).shape == (10, len(full))


def test_integrated_update_tracks_an_integrated_refit():
    full = _panel()
    first = full.assign(y=np.where(full["t"] < 10, full["y"], np.nan))
    updated = (GAUSSIAN_INTEGRATED.fit(first, hyperparameters="integrate")
               .update(full[full["t"].between(10, 12)]).update(full[full["t"] > 12]))
    refit = GAUSSIAN_INTEGRATED.fit(full, hyperparameters="integrate")
    for name in ("sigma", "tau"):
        mine, theirs = (r.hyperparameter_marginals()[name] for r in (updated, refit))
        assert abs(mine.mean[0] - theirs.mean[0]) < 0.1 * theirs.std[0]
    np.testing.assert_allclose(updated.mean, refit.mean, atol=0.02)
    assert abs(updated.log_marginal_likelihood - refit.log_marginal_likelihood) < 0.05
    new = updated.predict(full.head(5))
    np.testing.assert_allclose(new.predictive_mean, refit.predict(full.head(5)).predictive_mean,
                               atol=0.02)


def test_integrated_update_rejects_tabulated_latent_marginals():
    first = _panel().assign(y=lambda f: np.where(f["t"] < 10, f["y"], np.nan))
    result = GAUSSIAN_INTEGRATED.fit(first, hyperparameters="integrate",
                                     latent_strategy="simplified_laplace")
    with pytest.raises(UnsupportedEngineError, match="latent_strategy"):
        result.update(_panel()[lambda f: f["t"] >= 10])

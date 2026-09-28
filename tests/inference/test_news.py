"""result.news(release) splits the revision a release causes into one column per
released row. The oracle is the update itself: the columns must add up to
update()'s revision of every latent effect and of the linear predictor."""

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import IID, RW1, Fixed, Gaussian, Hyperparameter, LGM, Poisson
from pylgm.exceptions import ModelValidationError, UnsupportedEngineError


@pytest.fixture(params=["dense", "sparse"])
def engine(request, monkeypatch):
    if request.param == "sparse":
        monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    return request.param


def _panel():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        [(f"r{r}", t) for r in range(5) for t in range(30)], columns=["region", "t"]
    )
    frame["x"] = rng.normal(size=len(frame))
    eta = (0.3 * frame["x"] + np.cumsum(rng.normal(scale=0.15, size=30))[frame["t"]]
           + rng.normal(scale=0.6, size=5)[frame["region"].str[1:].astype(int)])
    frame["yg"] = 1.0 + eta + rng.normal(scale=0.4, size=len(frame))
    frame["yp"] = rng.poisson(np.exp(1.5 + eta)).astype(float)
    return frame


def _model(response, likelihood, tau=50.0, prec=Hyperparameter("prec", 5.0)):
    return LGM(
        response=response, likelihood=likelihood,
        predictor=Fixed("1 + x") + RW1("trend", index="t", precision=tau)
        + IID("level", index="region", precision=prec),
    )


def _vintage(response):
    frame = _panel()
    first = frame.assign(**{response: np.where(frame["t"] < 27, frame[response], np.nan)})
    return first, frame[frame["t"] == 27], frame[frame["t"] == 29]


@pytest.mark.parametrize("response, likelihood, fit_engine", [
    ("yg", Gaussian(0.4), "exact_gaussian"),
    ("yp", Poisson(), "laplace"),
])
def test_revisions_add_up_to_the_update(engine, response, likelihood, fit_engine):
    first, release, target = _vintage(response)
    result = _model(response, likelihood).fit(first, engine=fit_engine)
    news = result.news(release, at=target)
    updated = result.update(release)

    np.testing.assert_allclose(news.updated.mean, updated.mean, atol=1e-12)
    np.testing.assert_allclose(news.latent.sum(axis=1), updated.mean - result.mean, atol=1e-10)
    np.testing.assert_allclose(
        news.prediction.sum(axis=1),
        updated.predict(target).predictive_mean - result.predict(target).predictive_mean,
        atol=1e-10,
    )
    assert list(news.latent.columns) == list(release.index)
    assert list(news.prediction.index) == list(target.index)
    assert news.latent.index.names == ["block", "label"]
    np.testing.assert_allclose(news.releases["actual"], release[response])
    grid = result.news(release)                                  # default: the fitted grid
    np.testing.assert_allclose(
        grid.prediction.sum(axis=1), updated.predictive_mean - result.predictive_mean, atol=1e-10
    )


def test_gaussian_news_is_the_forecast_error_and_acts_linearly(engine):
    """Gaussian rows: news = y - E[y | old], and a release's column depends on
    its own news only, linearly -- change one value and only its column moves."""
    first, release, target = _vintage("yg")
    result = _model("yg", Gaussian(0.4)).fit(first)
    news = result.news(release, at=target)
    np.testing.assert_allclose(
        news.releases["news"], news.releases["actual"] - news.releases["expected"], atol=1e-12
    )
    np.testing.assert_allclose(
        news.releases["expected"], result.predict(release).predictive_mean, atol=1e-12
    )

    bumped = release.copy()
    bumped.iloc[1, bumped.columns.get_loc("yg")] += 1.0
    moved = result.news(bumped, at=target)
    changed = release.index[1]
    others = [label for label in release.index if label != changed]
    pd.testing.assert_frame_equal(moved.prediction[others], news.prediction[others])
    np.testing.assert_allclose(
        moved.prediction[changed],
        news.prediction[changed] * moved.releases.loc[changed, "news"]
        / news.releases.loc[changed, "news"],
        rtol=1e-10,
    )


def test_integrated_revision_adds_the_hyperparameter_part():
    first, release, target = _vintage("yg")
    model = _model("yg", Gaussian(Hyperparameter("sigma", 0.5)), tau=Hyperparameter("tau", 50.0))
    result = model.fit(first, hyperparameters="integrate")
    news = result.news(release, at=target)
    updated = result.update(release)

    assert news.latent.columns[-1] == "hyperparameters"
    np.testing.assert_allclose(news.updated.mean, updated.mean, atol=1e-12)
    np.testing.assert_allclose(news.latent.sum(axis=1), updated.mean - result.mean, atol=1e-10)
    np.testing.assert_allclose(
        news.prediction.sum(axis=1),
        updated.predict(target).predictive_mean - result.predict(target).predictive_mean,
        atol=1e-10,
    )
    assert np.abs(news.prediction["hyperparameters"]).max() > 0
    np.testing.assert_allclose(
        news.by_block.groupby(level=0, sort=False).sum().to_numpy(), news.prediction.to_numpy(),
        atol=1e-12,
    )


def test_news_rejects_ambiguous_or_empty_releases():
    first, release, _ = _vintage("yg")
    result = _model("yg", Gaussian(0.4)).fit(first)
    with pytest.raises(ValueError, match="unique"):
        result.news(pd.concat([release, release]))
    with pytest.raises(ValueError, match="observed response"):
        result.news(release.assign(yg=np.nan))


def test_by_block_splits_each_target_revision_without_cancelling():
    """A block's own latent revisions sum to ~0 under a sum-to-zero constraint;
    its contribution to a target, G[:, block] @ dmu[block], does not, and the
    blocks add up to the target's revision."""
    first, release, target = _vintage("yg")
    result = _model("yg", Gaussian(0.4)).fit(first)
    target = target.iloc[::-1]                                # order is kept, not sorted
    news = result.news(release, at=target)
    np.testing.assert_allclose(
        news.by_block.groupby(level=0, sort=False).sum().to_numpy(), news.prediction.to_numpy(),
        atol=1e-12,
    )
    assert list(news.by_block.index.get_level_values(0).unique()) == list(target.index)
    assert set(news.by_block.index.get_level_values("block")) == {"fixed", "trend", "level"}
    trend = news.by_block.xs("trend", level="block").sum(axis=1)
    assert abs(news.latent.loc["trend"].to_numpy().sum()) < 1e-10   # constrained: cancels
    assert np.abs(trend).max() > 1e-3                              # contribution: does not


# --- revisions, aggregates, uncertainty, joint models ------------------------


def _revised(response="yg"):
    first, release, target = _vintage(response)
    previous = first[first["t"].between(20, 21)]
    revised = previous.assign(**{response: previous[response] + np.linspace(0.5, -0.3, len(previous))})
    truth = first.copy()
    truth.loc[revised.index, response] = revised[response]
    truth.loc[release.index, response] = release[response]
    return first, release, target, (previous, revised), truth


def test_revisions_and_release_equal_a_refit(engine):
    """Revising Gaussian rows moves the mean linearly and re-scores the log
    marginal likelihood exactly; with the release, the result is the refit."""
    first, release, target, revisions, truth = _revised()
    model = _model("yg", Gaussian(0.4), prec=5.0)   # fixed: the refit must not re-estimate
    result = model.fit(first)
    news = result.news(release, at=target, revisions=revisions)
    refit = model.fit(truth)

    np.testing.assert_allclose(news.updated.mean, refit.mean, atol=1e-8)
    np.testing.assert_allclose(news.updated.log_marginal_likelihood,
                               refit.log_marginal_likelihood, atol=1e-8)
    np.testing.assert_allclose(result.update(release, revisions).mean, refit.mean, atol=1e-8)
    np.testing.assert_allclose(news.latent.sum(axis=1), refit.mean - result.mean, atol=1e-8)
    revised = [c for c in news.prediction.columns if str(c).startswith("revision ")]
    assert len(revised) == len(revisions[0])
    pd.testing.assert_frame_equal(
        news.revisions, pd.DataFrame({"previous": revisions[0]["yg"].to_numpy(),
                                      "revised": revisions[1]["yg"].to_numpy()},
                                     index=revisions[0].index)
    )
    only = result.news(None, at=target, revisions=revisions)       # revisions without a release
    assert only.releases is None
    np.testing.assert_allclose(only.prediction.to_numpy(),
                               news.prediction[revised].to_numpy(), atol=1e-12)


def test_revisions_of_non_gaussian_rows_are_refused():
    first, _, _, (previous, revised), _ = _revised("yp")
    result = _model("yp", Poisson()).fit(first, engine="laplace")
    with pytest.raises(UnsupportedEngineError, match="Gaussian rows only"):
        result.news(None, revisions=(previous, revised.assign(yp=revised["yp"].round())))


def test_aggregate_targets_and_sequential_uncertainty(engine):
    """weights turn target rows into an aggregate; each release's share of the
    variance drop is its reduction given the rows released before it."""
    first, release, target = _vintage("yg")
    result = _model("yg", Gaussian(0.4)).fit(first)
    quarter = pd.DataFrame([np.full(len(target), 1 / len(target))], index=["Q"],
                           columns=target.index)
    news = result.news(release, at=target, weights=quarter)
    rows = result.news(release, at=target)
    np.testing.assert_allclose(news.prediction.to_numpy(),
                               quarter.to_numpy() @ rows.prediction.to_numpy(), atol=1e-12)

    from pylgm.inference.prediction import _design_for
    aggregate = quarter.to_numpy() @ _design_for(result.prediction_context, target)
    drop = (result.linear_combinations(aggregate).variance
            - news.updated.linear_combinations(aggregate).variance)
    np.testing.assert_allclose(news.uncertainty.sum(axis=1), drop, rtol=1e-8, atol=1e-12)
    first_only = (result.linear_combinations(aggregate).variance
                  - result.update(release.iloc[:1]).linear_combinations(aggregate).variance)
    np.testing.assert_allclose(news.uncertainty.iloc[:, 0], first_only, rtol=1e-8, atol=1e-12)


def test_integrated_uncertainty_and_revisions_add_up():
    first, release, target, revisions, _ = _revised()
    model = _model("yg", Gaussian(Hyperparameter("sigma", 0.5)), tau=Hyperparameter("tau", 50.0))
    result = model.fit(first, hyperparameters="integrate")
    news = result.news(release, at=target, revisions=revisions)
    updated = result.update(release, revisions)
    np.testing.assert_allclose(news.updated.mean, updated.mean, atol=1e-12)
    np.testing.assert_allclose(news.latent.sum(axis=1), updated.mean - result.mean, atol=1e-10)
    np.testing.assert_allclose(
        news.uncertainty.sum(axis=1),
        result.predict(target).predictive_variance - updated.predict(target).predictive_variance,
        rtol=1e-6, atol=1e-12,
    )
    assert news.uncertainty.columns[-1] == "hyperparameters"


def _joint_panel():
    from pylgm import Joint, Shared

    rng = np.random.default_rng(1)
    t = np.arange(30)
    trend = np.cumsum(rng.normal(scale=0.3, size=30))
    frame = pd.concat([
        pd.DataFrame({"t": t, "a": trend + rng.normal(scale=0.3, size=30), "b": np.nan}),
        pd.DataFrame({"t": t, "a": np.nan, "b": rng.poisson(np.exp(1 + 0.8 * trend)).astype(float)}),
    ], ignore_index=True)
    joint = Joint(
        [LGM(response="a", likelihood=Gaussian(0.3), predictor=Fixed("1")),
         LGM(response="b", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(RW1("trend", index="t", precision=10.0), scale=(1.0, 0.8))],
    )
    return frame, joint


def test_joint_news_split_by_outcome():
    frame, joint = _joint_panel()
    future = (frame["t"] >= 27).to_numpy()
    first = frame.copy()
    first.loc[future, ["a", "b"]] = np.nan
    outcome_a = np.arange(len(frame)) < 30
    hold_out = {"a": future & outcome_a, "b": future & ~outcome_a}
    result = joint.fit(first, hold_out=hold_out)
    release = frame[frame["t"] == 27]
    targets = {"a": frame[(frame["t"] >= 27) & outcome_a]}
    news = result.news(release, at=targets)
    updated = result.update(release)

    assert list(news.releases.index) == [("a", release.index[0]), ("b", release.index[1])]
    np.testing.assert_allclose(news.latent.sum(axis=1), updated.mean - result.mean, atol=1e-10)
    np.testing.assert_allclose(
        news.prediction.sum(axis=1),
        updated.predict(targets["a"], outcome="a").predictive_mean
        - result.predict(targets["a"], outcome="a").predictive_mean,
        atol=1e-10,
    )
    second = first.copy()
    second.loc[release.index, ["a", "b"]] = release[["a", "b"]]
    later = (frame["t"] > 27).to_numpy()
    refit = joint.fit(second, hold_out={k: v & later for k, v in hold_out.items()})
    np.testing.assert_allclose(updated.mean, refit.mean, atol=1e-3)   # Poisson rows: Laplace


def test_joint_hold_out_is_validated():
    frame, joint = _joint_panel()
    with pytest.raises(ModelValidationError, match="unknown outcome"):
        joint.fit(frame, hold_out={"c": np.zeros(len(frame), bool)})
    with pytest.raises(ValueError, match="boolean mask"):
        joint.fit(frame, hold_out={"a": np.zeros(3, bool)})


def test_joint_integrated_news_uses_estimated_shared_scales():
    """The integrated joint result's contexts carry the estimated shared scale
    (its posterior mean), not the Hyperparameter's initial value."""
    from pylgm import Joint, Shared

    frame, _ = _joint_panel()
    joint = Joint(
        [LGM(response="a", likelihood=Gaussian(0.3), predictor=Fixed("1")),
         LGM(response="b", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(RW1("trend", index="t", precision=10.0),
                       scale=Hyperparameter("delta", 1.0))],
    )
    future = (frame["t"] >= 27).to_numpy()
    outcome_a = np.arange(len(frame)) < 30
    first = frame.copy()
    first.loc[future, ["a", "b"]] = np.nan
    result = joint.fit(first, hyperparameters="integrate",
                       hold_out={"a": future & outcome_a, "b": future & ~outcome_a})
    delta = result.hyperparameter_marginals()["delta"].mean[0]
    scales = {entry[1][3]: entry[1][4] for context in result.prediction_context.contexts.values()
              for entry in context.entries if entry[0] == "shared"}
    assert scales["delta"] in (pytest.approx(delta), pytest.approx(1 / delta))
    assert abs(delta - 1.0) > 0.05

    release = frame[frame["t"] == 27]
    news = result.news(release, at={"b": frame[future & ~outcome_a]})
    updated = result.update(release)
    np.testing.assert_allclose(news.latent.sum(axis=1), updated.mean - result.mean, atol=1e-10)
    assert news.prediction.columns[-1] == "hyperparameters"

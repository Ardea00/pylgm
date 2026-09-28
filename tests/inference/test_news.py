"""result.news(release) splits the revision a release causes into one column per
released row. The oracle is the update itself: the columns must add up to
update()'s revision of every latent effect and of the linear predictor."""

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import IID, RW1, Fixed, Gaussian, Hyperparameter, LGM, Poisson


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


def _model(response, likelihood, tau=50.0):
    return LGM(
        response=response, likelihood=likelihood,
        predictor=Fixed("1 + x") + RW1("trend", index="t", precision=tau)
        + IID("level", index="region", precision=Hyperparameter("prec", 5.0)),
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


def test_news_rejects_ambiguous_or_empty_releases():
    first, release, _ = _vintage("yg")
    result = _model("yg", Gaussian(0.4)).fit(first)
    with pytest.raises(ValueError, match="unique"):
        result.news(pd.concat([release, release]))
    with pytest.raises(ValueError, match="no row with an observed response"):
        result.news(release.assign(yg=np.nan))

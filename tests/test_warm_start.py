"""fit(..., warm_start=previous) starts the search at an earlier window's
estimates. The oracle is a cold fit on the same window: a warm start moves
where the search begins, never what it converges to."""

import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from pylgm import IID, RW1, Fixed, Gaussian, Hyperparameter, LGM, Poisson


def _panel(regions=8, trend_sd=0.15, level_sd=0.6):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        [(f"r{r}", t) for r in range(regions) for t in range(30)], columns=["region", "t"]
    )
    frame["x"] = rng.normal(size=len(frame))
    eta = (0.3 * frame["x"] + np.cumsum(rng.normal(scale=trend_sd, size=30))[frame["t"]]
           + rng.normal(scale=level_sd, size=regions)[frame["region"].str[1:].astype(int)])
    frame["yg"] = 1.0 + eta + rng.normal(scale=0.4, size=len(frame))
    frame["yp"] = rng.poisson(np.exp(1.5 + eta)).astype(float)
    return frame


def _model(response, likelihood):
    return LGM(
        response=response, likelihood=likelihood,
        predictor=Fixed("1 + x") + RW1("trend", index="t", precision=Hyperparameter("tau", 50.0))
        + IID("level", index="region", precision=Hyperparameter("prec", 5.0)),
    )


CASES = {
    "gaussian": (_model("yg", Gaussian(Hyperparameter("sigma", 0.5))), "exact_gaussian"),
    "poisson": (_model("yp", Poisson()), "laplace"),
}


@pytest.mark.parametrize("hyperparameters", ["optimize", "integrate"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_warm_start_on_a_sliding_window_matches_a_cold_fit(case, hyperparameters):
    model, engine = CASES[case]
    frame = _panel()
    first = frame[frame["t"] < 20]
    second = frame[frame["t"].between(5, 24)]   # periods leave, new ones arrive
    previous = model.fit(first, engine=engine, hyperparameters=hyperparameters)
    cold = model.fit(second, engine=engine, hyperparameters=hyperparameters)
    warm = model.fit(second, engine=engine, hyperparameters=hyperparameters, warm_start=previous)

    spread = np.sqrt(cold.predictive_variance)
    assert np.max(np.abs(warm.predictive_mean - cold.predictive_mean) / spread) < 0.05
    assert abs(warm.log_marginal_likelihood - cold.log_marginal_likelihood) < 1e-2
    if hyperparameters == "optimize":
        for name, value in cold.hyperparameters.items():
            assert warm.hyperparameters[name] == pytest.approx(value, rel=0.05)


def test_warm_start_shortens_the_search():
    model, engine = CASES["gaussian"]
    frame = _panel()
    previous = model.fit(frame[frame["t"] < 25], engine=engine)
    cold = model.fit(frame, engine=engine)
    warm = model.fit(frame, engine=engine, warm_start=previous)
    assert (warm.diagnostics["empirical_bayes_evaluations"]
            < cold.diagnostics["empirical_bayes_evaluations"])


def test_warm_start_skips_an_estimate_pinned_at_a_bound():
    """On a nearly flat trend the first window runs tau to its upper bound,
    where the objective is a plateau: a search started there stalls on it
    (0.6 nats short of the optimum here) instead of coming back inside."""
    model, engine = CASES["gaussian"]
    frame = _panel(regions=6, trend_sd=0.05, level_sd=0.3)
    with pytest.warns(UserWarning, match="edge of the declared interval"):
        previous = model.fit(frame[frame["t"] < 20], engine=engine)
    assert previous.diagnostics["hyperparameters_at_bound"] == "tau"
    second = frame[frame["t"].between(5, 24)]
    cold = model.fit(second, engine=engine)
    warm = model.fit(second, engine=engine, warm_start=previous)
    assert warm.log_marginal_likelihood == pytest.approx(cold.log_marginal_likelihood, abs=1e-3)


def test_warm_start_keeps_no_reference_to_the_previous_result():
    """A rolling experiment must not chain every earlier window into memory."""
    model, engine = CASES["poisson"]
    frame = _panel()
    previous = model.fit(frame[frame["t"] < 20], engine=engine, hyperparameters="integrate")
    alive = weakref.ref(previous)
    current = model.fit(frame, engine=engine, hyperparameters="integrate", warm_start=previous)
    del previous
    gc.collect()
    assert alive() is None
    assert current.sample(5, rng=0).shape == (5, len(frame))


def test_warm_start_must_be_a_fit_result():
    model, engine = CASES["gaussian"]
    with pytest.raises(TypeError, match="warm_start"):
        model.fit(_panel(), engine=engine, warm_start={"tau": 1.0})


def test_joint_warm_start_matches_a_cold_fit():
    from pylgm import Joint, Shared

    rng = np.random.default_rng(0)
    districts = np.repeat([f"d{i}" for i in range(15)], 4)
    shared = rng.normal(scale=0.5, size=15)[np.repeat(np.arange(15), 4)]
    own = rng.normal(scale=0.4, size=15)[np.repeat(np.arange(15), 4)]
    frame = pd.DataFrame({
        "district": districts, "t": np.tile(np.arange(4), 15),
        "a": rng.poisson(np.exp(1.5 + shared + own)).astype(float),
        "b": rng.poisson(np.exp(1.0 + 0.7 * shared)).astype(float),
    })
    joint = Joint(
        [LGM(response="a", likelihood=Poisson(),
             predictor=Fixed("1") + IID("v", index="district", precision=Hyperparameter("prec", 2.0))),
         LGM(response="b", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=4.0),
                       scale=Hyperparameter("delta", 1.0))],
    )
    previous = joint.fit(frame[frame["t"] < 3])
    cold = joint.fit(frame)
    warm = joint.fit(frame, warm_start=previous)
    assert warm.log_marginal_likelihood == pytest.approx(cold.log_marginal_likelihood, abs=1e-3)
    np.testing.assert_allclose(warm.mean, cold.mean, atol=0.02)

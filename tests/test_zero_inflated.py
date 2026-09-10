"""Zero-inflated counts: ZIP, ZINB and ZIB through one wrapper.

The derivative identities are checked against numerical truth rather than
restated, because the wrapper derives everything from the base likelihood's own
surface and an algebra slip there would be invisible: the fit would still run,
just on a different model.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from pylgm import (
    Binomial, Fixed, Gaussian, Hyperparameter, IID, LGM,
    NegativeBinomial, Poisson, ZeroInflated,
)
from pylgm.exceptions import CompilationError

ETAS = np.array([-3.0, -1.0, 0.0, 1.0, 2.0])


def _bound(spec, n, trials=None):
    compiled = spec.materialize({})
    if trials is not None:
        compiled = compiled.for_observations({"trials": np.full(n, float(trials))})
    return compiled


def _support(trials=None):
    return np.arange(0, (trials if trials is not None else 200) + 1, dtype=float)


CASES = [
    ("ZIP", ZeroInflated(Poisson(), pi=0.35), None),
    ("ZIP-heavy", ZeroInflated(Poisson(), pi=0.9), None),
    ("ZINB", ZeroInflated(NegativeBinomial(phi=2.0), pi=0.4), None),
    ("ZIB", ZeroInflated(Binomial(trials="n"), pi=0.3), 10),
]


@pytest.mark.parametrize("name, spec, trials", CASES)
def test_density_is_a_probability_distribution(name, spec, trials):
    lk = _bound(spec, 1, trials)
    support = _support(trials)
    for eta in ETAS:
        one = np.array([eta])
        total = sum(float(np.exp(lk.pointwise_log_density(one, np.array([y])))[0])
                    for y in support)
        assert total == pytest.approx(1.0, abs=1e-8), (name, eta)


@pytest.mark.parametrize("name, spec, trials", CASES)
def test_gradient_matches_the_log_densitys_slope(name, spec, trials):
    lk = _bound(spec, 1, trials)
    step = 1e-6
    for eta in ETAS:
        one = np.array([eta])
        for y in (0.0, 3.0):
            target = np.array([y])
            finite = float(
                lk.pointwise_log_density(one + step, target)[0]
                - lk.pointwise_log_density(one - step, target)[0]
            ) / (2 * step)
            assert lk.gradient(one, target)[0] == pytest.approx(finite, abs=1e-5), (name, eta, y)


@pytest.mark.parametrize("name, spec, trials", CASES)
def test_working_weights_are_the_expected_information(name, spec, trials):
    """The load-bearing derivation.

    ``I = (1 - pi)[I_base - f0 (1 - w) s0^2]`` is what the wrapper returns; here it
    is recomputed the slow, obvious way -- sum ``-p(y) d2 log p / d eta^2`` over
    the whole support -- so the closed form has to earn it.
    """
    lk = _bound(spec, 1, trials)
    support = _support(trials)
    step = 1e-4
    for eta in ETAS:
        one = np.array([eta])
        density, curvature = [], []
        for y in support:
            target = np.array([y])
            density.append(float(np.exp(lk.pointwise_log_density(one, target))[0]))
            curvature.append(float(
                lk.pointwise_log_density(one + step, target)[0]
                - 2 * lk.pointwise_log_density(one, target)[0]
                + lk.pointwise_log_density(one - step, target)[0]
            ) / step ** 2)
        expected = -float(np.dot(density, curvature))
        got = lk.working_weights(one, np.array([0.0]))[0]
        assert got == pytest.approx(expected, rel=1e-4), (name, eta)


@pytest.mark.parametrize("name, spec, trials", CASES)
def test_working_weights_stay_positive(name, spec, trials):
    """What makes the Newton step factor at all -- see the test below for why it
    is not free."""
    lk = _bound(spec, 400, trials)
    eta = np.linspace(-12.0, 12.0, 400)
    weights = lk.working_weights(eta, np.zeros_like(eta))
    assert np.isfinite(weights).all(), name
    assert (weights > 0).all(), (name, float(weights.min()))


def test_the_observed_information_really_does_go_negative():
    """Why the weights are the *expected* information and not the observed one.

    A zero-inflated density is not log-concave, so the observed curvature at a
    zero turns the wrong way over a wide band of eta -- a Newton step built on it
    would fail to factor. If this ever stops holding, the expected-information
    detour has stopped paying for itself.
    """
    lk = ZeroInflated(Poisson(), pi=0.4).materialize({})
    step, zero, negative = 1e-4, np.array([0.0]), 0
    grid = np.linspace(-2.0, 4.0, 200)
    for eta in grid:
        one = np.array([eta])
        curvature = float(
            lk.pointwise_log_density(one + step, zero)[0]
            - 2 * lk.pointwise_log_density(one, zero)[0]
            + lk.pointwise_log_density(one - step, zero)[0]
        ) / step ** 2
        negative += -curvature < 0
    assert negative > len(grid) // 4, negative


@pytest.mark.parametrize("base", [Poisson(), NegativeBinomial(phi=2.0)])
def test_third_derivative_matches_the_curvatures_slope(base):
    lk = ZeroInflated(base, pi=0.35).materialize({})
    step = 1e-4

    def curvature(eta, target):
        return float(
            lk.pointwise_log_density(eta + step, target)[0]
            - 2 * lk.pointwise_log_density(eta, target)[0]
            + lk.pointwise_log_density(eta - step, target)[0]
        ) / step ** 2

    for eta in (-1.5, -0.5, 0.5, 1.5):
        one = np.array([eta])
        for y in (0.0, 2.0):
            target = np.array([y])
            finite = (curvature(one + step, target) - curvature(one - step, target)) / (2 * step)
            assert lk.third_derivative(one, target)[0] == pytest.approx(
                finite, rel=2e-3, abs=2e-3
            ), (eta, y)


@pytest.mark.parametrize("base", [Poisson(), NegativeBinomial(phi=1.5), Binomial(trials="n")])
def test_pi_zero_is_exactly_the_base_likelihood(base):
    """Including where f(0 | eta) underflows: forming w as a ratio of underflowed
    densities gives 0/0 there, and silently drops to the wrong branch."""
    n = 9
    wrapped = _bound(ZeroInflated(base, pi=0.0), n, 10 if base.__class__ is Binomial else None)
    plain = base.materialize({})
    if base.__class__ is Binomial:
        plain = plain.for_observations({"trials": np.full(n, 10.0)})
    eta = np.array([-40.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 40.0, 80.0])
    y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0])
    for method in ("pointwise_log_density", "gradient", "working_weights", "third_derivative"):
        np.testing.assert_allclose(
            getattr(wrapped, method)(eta, y), getattr(plain, method)(eta, y), err_msg=method
        )


def test_survives_the_engines_error_state_at_extreme_eta():
    """The engines run under ``over/invalid/divide="raise"`` (underflow ignored --
    see ``inference/laplace.py``), so an intermediate ``0/0`` aborts a fit even
    where the final value would have been right. A density that underflows to
    zero is expected and fine; a ratio of two that have is not.
    """
    eta = np.linspace(-40.0, 40.0, 200)
    zeros = np.zeros_like(eta)
    for pi in (0.0, 0.5, 0.99):
        lk = ZeroInflated(Poisson(), pi=pi).materialize({})
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            assert np.isfinite(lk.working_weights(eta, zeros)).all()
            assert np.isfinite(lk.gradient(eta, zeros)).all()
            assert np.isfinite(lk.pointwise_log_density(eta, zeros)).all()


def test_cdf_adds_the_point_mass():
    lk = ZeroInflated(Poisson(), pi=0.3).materialize({})
    eta = np.array([0.5, 0.5])
    base = Poisson().materialize({})
    y = np.array([0.0, 4.0])
    np.testing.assert_allclose(lk.cdf(eta, y), 0.3 + 0.7 * base.cdf(eta, y))
    # and it is a genuine CDF: reaching one over the support
    support = np.arange(0, 100.0)
    values = lk.cdf(np.full_like(support, 0.5), support)
    assert values[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(values) >= -1e-12)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def _zero_inflated_frame(pi=0.45, n=600, groups=30, seed=1, kind="poisson", phi=2.0):
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, 0.6, groups)
    mu = np.exp(1.1 + np.array([effects[i % groups] for i in range(n)]))
    counts = rng.poisson(mu) if kind == "poisson" else rng.negative_binomial(phi, phi / (phi + mu))
    y = np.where(rng.random(n) < pi, 0.0, counts.astype(float))
    return pd.DataFrame({"g": [f"a{i % groups}" for i in range(n)], "y": y, "row": range(n)})


def _predictor():
    return Fixed("1") + IID("u", index="g", precision=2.0)


@pytest.mark.parametrize("base, kind", [(Poisson(), "poisson"),
                                        (NegativeBinomial(phi=2.0), "nb")])
def test_recovers_the_zero_inflation_probability(base, kind):
    frame = _zero_inflated_frame(pi=0.45, kind=kind)
    model = LGM(
        response="y",
        likelihood=ZeroInflated(base, pi=Hyperparameter("pi", initial=0.2, transform="logit")),
        predictor=_predictor(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(frame, engine="laplace")
    assert result.hyperparameters["pi"] == pytest.approx(0.45, abs=0.06)


def test_zero_inflation_beats_the_plain_count_model_on_inflated_data():
    frame = _zero_inflated_frame(pi=0.45)
    plain = LGM(response="y", likelihood=Poisson(), predictor=_predictor()).fit(frame, engine="laplace")
    inflated = LGM(response="y", likelihood=ZeroInflated(Poisson(), pi=0.45),
                   predictor=_predictor()).fit(frame, engine="laplace")
    assert inflated.log_marginal_likelihood > plain.log_marginal_likelihood + 50.0


def test_fitted_mean_is_the_base_mean_thinned_by_pi():
    from pylgm.likelihoods import CompiledPoisson

    frame = _zero_inflated_frame(pi=0.4)
    result = LGM(response="y", likelihood=ZeroInflated(Poisson(), pi=0.4),
                 predictor=_predictor()).fit(frame, engine="laplace")
    prediction = result.predict(frame)
    np.testing.assert_allclose(
        prediction.fitted_mean,
        0.6 * CompiledPoisson().response_prediction(
            prediction.predictive_mean, prediction.predictive_variance
        ),
    )


def test_zero_inflated_binomial_binds_its_trials_column():
    """A wrapper hides the base's type from every ``isinstance`` check that reads
    per-row data, and unbound trials give a NaN log density rather than an error."""
    rng = np.random.default_rng(1)
    n, groups = 400, 20
    y = np.where(rng.random(n) < 0.3, 0.0, rng.binomial(10, 0.4, n).astype(float))
    frame = pd.DataFrame({"g": [f"a{i % groups}" for i in range(n)], "y": y,
                          "n": 10, "row": range(n)})
    model = LGM(
        response="y",
        likelihood=ZeroInflated(Binomial(trials="n"),
                                pi=Hyperparameter("pi", initial=0.15, transform="logit")),
        predictor=_predictor(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(frame, engine="laplace")
    assert result.hyperparameters["pi"] == pytest.approx(0.3, abs=0.08)
    assert np.isfinite(result.predict(frame).fitted_mean).all()


def test_integrates_over_pi_on_the_logit_scale():
    frame = _zero_inflated_frame(pi=0.4, n=400, groups=20)
    model = LGM(
        response="y",
        likelihood=ZeroInflated(Poisson(), pi=Hyperparameter("pi", initial=0.25, transform="logit")),
        predictor=_predictor(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(frame, engine="laplace", hyperparameters="integrate")
    marginal = result.hyperparameter_marginals()["pi"]
    assert 0.0 < marginal.quantile(0.025)[0] < 0.4 < marginal.quantile(0.975)[0] < 1.0
    assert marginal.std[0] > 0.0


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def test_rejects_a_non_count_base():
    with pytest.raises(ValueError, match="count likelihood"):
        ZeroInflated(Gaussian(1.0))


@pytest.mark.parametrize("pi", [1.0, -0.1, float("nan")])
def test_rejects_an_impossible_probability(pi):
    with pytest.raises(ValueError, match="pi must be"):
        ZeroInflated(Poisson(), pi=pi)


def test_rejects_a_pi_declared_on_the_log_scale():
    """The default transform is log, which would let a probability exceed one."""
    with pytest.raises(ValueError, match="logit"):
        ZeroInflated(Poisson(), pi=Hyperparameter("pi", initial=0.2))


def test_rejects_estimating_both_pi_and_the_base_dispersion():
    frame = _zero_inflated_frame()
    model = LGM(
        response="y",
        likelihood=ZeroInflated(
            NegativeBinomial(phi=Hyperparameter("phi", initial=2.0)),
            pi=Hyperparameter("pi", initial=0.2, transform="logit"),
        ),
        predictor=_predictor(),
    )
    with pytest.raises(CompilationError, match="only one optimisable scalar"):
        model.fit(frame, engine="laplace")

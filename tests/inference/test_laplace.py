import numpy as np
import pytest

from pylgm import Bernoulli, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import InferenceConvergenceError
from pylgm.inference.gaussian import fit_gaussian
from pylgm.inference.laplace import fit_laplace
import pandas as pd


def _panel(frame, panel=(), time="t"):
    return CanonicalPanel.from_frame(frame, DataConfig(time=time, response="y", panel=panel))


def test_laplace_reproduces_exact_gaussian_on_gaussian_likelihood():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 2.5, 4.0]})
    model = LGM("y", Gaussian(0.7), Fixed("1 + x", prior_precision=0.5) + IID("g", index="t", precision=1.5), time="t")
    compiled = compile_lgm(model, _panel(frame))
    exact = fit_gaussian(compiled)
    laplace = fit_laplace(compiled)
    np.testing.assert_allclose(laplace.mean, exact.mean, atol=1e-8)
    np.testing.assert_allclose(laplace.covariance, exact.covariance, atol=1e-8)
    np.testing.assert_allclose(
        laplace.log_marginal_likelihood, exact.log_marginal_likelihood, atol=1e-8
    )


def test_laplace_poisson_mode_solves_the_score_equation():
    # single fixed intercept, flat prior -> mode is log(mean(y))
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [2.0, 3.0, 5.0, 6.0]})
    model = LGM("y", Poisson(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    np.testing.assert_allclose(result.mean, [np.log(4.0)], atol=1e-6)
    # response-scale fitted mean uses log-normal correction, so > exp(mean)
    assert result.fitted_mean[0] > np.exp(result.mean[0]) - 1e-9
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_bernoulli_intercept_matches_logit_of_rate():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [1.0, 1.0, 0.0, 1.0]})
    model = LGM("y", Bernoulli(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    p = 0.75
    np.testing.assert_allclose(result.mean, [np.log(p / (1 - p))], atol=1e-6)


def test_laplace_predicts_unobserved_rows():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, np.nan]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    assert result.predictive_mean.shape == (3,)
    assert np.isfinite(result.fitted_mean).all()


def test_laplace_raises_on_non_convergence():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 5.0, 10.0], "y": [0.0, 1.0, 20.0]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    with pytest.raises(InferenceConvergenceError):
        fit_laplace(compile_lgm(model, _panel(frame)), max_iterations=1, tolerance=1e-12)


def test_laplace_poisson_converges_for_large_counts():
    # A high count drives the first Newton step's eta past the exp-overflow
    # threshold; the line search must backtrack rather than raise NumericalError.
    frame = pd.DataFrame({"t": [1], "y": [1000.0]})
    model = LGM("y", Poisson(), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    np.testing.assert_allclose(result.mean, [np.log(1000.0)], atol=1e-6)


def test_laplace_succeeds_within_its_actual_iteration_budget():
    # Fitting with exactly the number of Newton iterations the model needs must
    # NOT spuriously raise (guards the off-by-one in the convergence check).
    frame = pd.DataFrame({"t": [1, 2, 3, 4, 5], "x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 5.0, 8.0]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    compiled = compile_lgm(model, _panel(frame))
    full = fit_laplace(compiled)
    needed = full.diagnostics["newton_iterations"]
    tight = fit_laplace(compiled, max_iterations=needed)
    np.testing.assert_allclose(tight.mean, full.mean, atol=1e-8)


def _stalling_poisson_model(seed=0, n=14):
    """A Poisson model whose Newton loop stalls above the absolute tolerance."""
    from pylgm import AR1, Fixed, Hyperparameter, LGM, Poisson
    from pylgm.priors import PCPrecision

    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    for i in range(1, n):
        signal[i] = 0.8 * signal[i - 1] + rng.normal(scale=0.3)
    frame = pd.DataFrame({
        "t": list(range(n)),
        "y": rng.poisson(np.exp(1.5 + signal)).astype(float),
    })
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + AR1("trend", index="t",
              precision=Hyperparameter("trend.precision", initial=1.0,
                                       prior=PCPrecision(upper_sd=1.0, alpha=0.01)),
              rho=Hyperparameter("trend.rho", initial=0.0, transform="logit")),
        likelihood=Poisson(),
    )
    return model, frame


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stalling_poisson_integrate_now_converges(seed):
    # Before the Newton decrement rescue these raised InferenceConvergenceError
    # on 8/10 seeds: the loop reached the mode but could not push max|grad|
    # below the absolute 1e-8 tolerance, and one bad grid point aborted the
    # whole integration.
    model, frame = _stalling_poisson_model(seed)
    result = model.fit(frame, engine="laplace", hyperparameters="integrate")
    assert np.all(np.isfinite(result.latent_marginals("trend").mean))


def test_rescued_fit_is_actually_at_the_mode():
    # The rescue must accept a genuinely optimal point, not merely relabel a
    # failure. max_iterations=5 is chosen to land IN the rescue branch: the
    # gradient test has not fired, so the decrement is what accepts the point.
    # The iteration assertion pins that -- without it this test silently stops
    # exercising the branch it exists to guard.
    model, frame = _stalling_poisson_model(0)
    panel = CanonicalPanel(frame, np.ones(len(frame), dtype=bool), ("t",), "y")
    compiled = compile_lgm(model, panel)

    rescued = fit_laplace(compiled, max_iterations=5)
    assert rescued.diagnostics["newton_iterations"] == 5
    assert rescued.diagnostics["final_gradient_norm"] > 1e-8  # gradient test did NOT fire

    # Compare against a well-converged reference rather than a raw score
    # threshold: the decrement scales the gradient by H^-1, so a stiff
    # direction tolerates a large raw score for a negligible positional error.
    reference = fit_laplace(compiled, max_iterations=200)
    spread = np.sqrt(np.diag(reference.covariance))
    assert np.max(np.abs(rescued.mean - reference.mean) / spread) < 1e-4
    assert abs(rescued.log_marginal_likelihood - reference.log_marginal_likelihood) < 1e-4


def test_genuine_non_convergence_still_raises():
    # The rescue must not swallow a fit that really has not converged.

    model, frame = _stalling_poisson_model(0)
    panel = CanonicalPanel(frame, np.ones(len(frame), dtype=bool), ("t",), "y")
    compiled = compile_lgm(model, panel)
    with pytest.raises(InferenceConvergenceError):
        fit_laplace(compiled, max_iterations=1)


@pytest.mark.parametrize("family_name", ["nbinomial", "gamma"])
def test_laplace_log_link_families_intercept_matches_log_mean(family_name):
    # NB & Gamma both use a log link; the intercept-only score equation is
    # solved by mu = mean(y), i.e. eta = log(ybar), independent of phi.
    from pylgm import Gamma, NegativeBinomial

    family = {"nbinomial": NegativeBinomial(2.5), "gamma": Gamma(2.5)}[family_name]
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [2.0, 3.0, 5.0, 6.0]})
    model = LGM("y", family, Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    np.testing.assert_allclose(result.mean, [np.log(4.0)], atol=1e-6)
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_beta_intercept_matches_logit_link_mode():
    # Beta uses a logit link; for data symmetric about 0.5 the mean(logit y)
    # term vanishes, so the intercept-only mode is logit(0.5) = 0.
    from pylgm import Beta

    frame = pd.DataFrame({"t": [1, 2, 3, 4], "y": [0.2, 0.8, 0.3, 0.7]})
    model = LGM("y", Beta(5.0), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    np.testing.assert_allclose(result.mean, [0.0], atol=1e-6)


def test_empirical_bayes_recovers_planted_nbinomial_dispersion():
    # A phi Hyperparameter must flow through the generalized likelihood hook and
    # be estimated: fit EB on data drawn from a known NB2 dispersion.
    from pylgm import Hyperparameter, NegativeBinomial

    rng = np.random.default_rng(0)
    n, mu, phi_true = 4000, np.exp(1.0), 3.0
    y = rng.negative_binomial(phi_true, phi_true / (phi_true + mu), size=n).astype(float)
    frame = pd.DataFrame({"t": np.arange(n), "y": y})
    model = LGM(
        "y",
        NegativeBinomial(Hyperparameter("phi", initial=1.0, transform="log")),
        Fixed("1", prior_precision=1e-8),
        time="t",
    )
    result = model.fit(frame, engine="laplace", hyperparameters="optimize")
    assert result.hyperparameters["phi"] == pytest.approx(phi_true, rel=0.1)


def test_laplace_binomial_intercept_matches_aggregate_logit():
    # Aggregated binomial, logit link: the intercept-only mode is logit of the
    # pooled success rate, logit(sum y / sum n), independent of the split.
    from pylgm import Binomial

    frame = pd.DataFrame(
        {"t": [1, 2, 3, 4], "y": [3.0, 4.0, 7.0, 2.0], "n": [10.0, 8.0, 12.0, 5.0]}
    )
    model = LGM("y", Binomial("n"), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    total_y, total_n = 16.0, 35.0
    np.testing.assert_allclose(result.mean, [np.log(total_y / (total_n - total_y))], atol=1e-6)
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_binomial_predicts_counts_with_new_trials():
    # Fit with a trials column, then predict on new rows whose trials column
    # differs: fitted_mean must be n_new * p (counts), not the probability p.
    from pylgm import Binomial

    frame = pd.DataFrame(
        {"t": [1, 2, 3, 4], "y": [3.0, 4.0, 7.0, 2.0], "n": [10.0, 8.0, 12.0, 5.0]}
    )
    model = LGM("y", Binomial("n"), Fixed("1", prior_precision=1e-8), time="t")
    result = model.fit(frame, engine="laplace")
    p = 1.0 / (1.0 + np.exp(-float(result.mean[0])))

    new_data = pd.DataFrame({"t": [5, 6, 7], "n": [100.0, 50.0, 7.0]})
    prediction = result.predict(new_data)
    np.testing.assert_allclose(prediction.fitted_mean, np.array([100.0, 50.0, 7.0]) * p)

    with pytest.raises(ValueError, match="trials column"):
        result.predict(pd.DataFrame({"t": [5], "z": [1.0]}))


def test_laplace_exponential_surv_intercept_solves_score_equation():
    # Intercept-only exponential PH: the MLE of eta solves sum(delta) = sum(t*exp(eta)),
    # i.e. eta_hat = log( sum(delta) / sum(t) ). With a near-flat prior the mode matches.
    from pylgm import ExponentialSurv

    frame = pd.DataFrame({
        "t": np.arange(1, 6, dtype=float),
        "y": [1.0, 2.0, 1.5, 3.0, 2.5],          # follow-up times
        "d": [1.0, 0.0, 1.0, 1.0, 0.0],          # events
    })
    model = LGM("y", ExponentialSurv("d"), Fixed("1", prior_precision=1e-8), time="t")
    result = fit_laplace(compile_lgm(model, _panel(frame)))
    expected = np.log(frame["d"].sum() / frame["y"].sum())
    np.testing.assert_allclose(result.mean, [expected], atol=1e-6)
    assert result.diagnostics["final_gradient_norm"] < 1e-8


def test_laplace_weibull_surv_fixed_shape_recovers_slope():
    # Fixed-shape Weibull PH: recover a planted slope from simulated event times.
    from pylgm import WeibullSurv

    rng = np.random.default_rng(0)
    n, alpha, b0, b1 = 6000, 1.5, 0.3, -0.8
    x = rng.normal(size=n)
    eta = b0 + b1 * x
    scale = np.exp(-eta / alpha)                  # S(t)=exp(-(t/scale)^alpha)
    t = rng.weibull(alpha, size=n) * scale        # all observed (no censoring)
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "x": x, "d": np.ones(n)})
    model = LGM("y", WeibullSurv("d", shape=alpha), Fixed("1 + x"), time="t")
    result = model.fit(frame, engine="laplace")
    idx = {lab: i for i, lab in enumerate(result.labels)}
    assert result.mean[idx["fixed:x"]] == pytest.approx(b1, abs=0.05)


def test_survival_missing_event_column_raises():
    from pylgm import ExponentialSurv
    from pylgm.exceptions import DataContractError

    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})   # no "d" column
    model = LGM("y", ExponentialSurv("d"), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="event column not found"):
        compile_lgm(model, _panel(frame))


def test_survival_entry_after_time_raises():
    from pylgm import WeibullSurv
    from pylgm.exceptions import DataContractError

    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0], "d": [1.0, 1.0], "v": [0.5, 3.0]})
    model = LGM("y", WeibullSurv("d", entry="v"), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="entry"):
        compile_lgm(model, _panel(frame))


def test_empirical_bayes_recovers_planted_weibull_shape():
    from pylgm import Hyperparameter, WeibullSurv

    rng = np.random.default_rng(1)
    n, alpha_true, b0 = 6000, 1.8, 0.5
    scale = np.exp(-b0 / alpha_true)
    t = rng.weibull(alpha_true, size=n) * scale
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "d": np.ones(n)})
    model = LGM(
        "y",
        WeibullSurv("d", shape=Hyperparameter("alpha", initial=1.0, transform="log")),
        Fixed("1", prior_precision=1e-8),
        time="t",
    )
    result = model.fit(frame, engine="laplace", hyperparameters="optimize")
    assert result.hyperparameters["alpha"] == pytest.approx(alpha_true, rel=0.1)


def test_weibull_surv_predict_uses_fitted_shape():
    # Fit with an estimated shape, predict on new rows: the predicted E[T] must use
    # the FITTED alpha, not the Hyperparameter's initial guess.
    from pylgm import Hyperparameter, WeibullSurv
    from scipy.special import gamma as gamma_fn

    rng = np.random.default_rng(3)
    n, alpha_true, b0 = 4000, 1.9, 0.4
    scale = np.exp(-b0 / alpha_true)
    t = rng.weibull(alpha_true, size=n) * scale
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "d": np.ones(n)})
    model = LGM(
        "y",
        WeibullSurv("d", shape=Hyperparameter("alpha", initial=1.0, transform="log")),
        Fixed("1", prior_precision=1e-8),
        time="t",
    )
    result = model.fit(frame, engine="laplace", hyperparameters="optimize")
    alpha_hat = result.hyperparameters["alpha"]
    new = pd.DataFrame({"t": [n], "y": [1.0], "d": [1.0]})
    pred = result.predict(new)
    eta_hat = float(result.mean[0])
    expected = np.exp(-eta_hat / alpha_hat) * gamma_fn(1.0 + 1.0 / alpha_hat)
    # would differ by >5% if the initial alpha=1.0 were used instead of alpha_hat
    np.testing.assert_allclose(pred.fitted_mean, [expected], rtol=1e-3)


def test_weibull_surv_left_truncation_fits_end_to_end():
    # Delayed entry: rows enter the risk set at v>0. Fit must run and the slope
    # estimate must be finite and close to the planted value.
    from pylgm import WeibullSurv

    rng = np.random.default_rng(2)
    n, alpha, b1 = 5000, 1.2, -0.6
    x = rng.normal(size=n)
    eta = b1 * x
    scale = np.exp(-eta / alpha)
    t = rng.weibull(alpha, size=n) * scale
    v = np.minimum(0.2 * rng.random(n), 0.99 * t)      # entry strictly before t
    frame = pd.DataFrame({"t": np.arange(n), "y": t, "x": x, "d": np.ones(n), "v": v})
    model = LGM("y", WeibullSurv("d", shape=alpha, entry="v"), Fixed("1 + x"), time="t")
    result = model.fit(frame, engine="laplace")
    idx = {lab: i for i, lab in enumerate(result.labels)}
    assert np.isfinite(result.mean[idx["fixed:x"]])
    assert result.mean[idx["fixed:x"]] == pytest.approx(b1, abs=0.08)

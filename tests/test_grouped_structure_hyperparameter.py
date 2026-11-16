"""An estimated ``AR1Structure(rho=Hyperparameter(...))`` on ``Grouped``.

R-INLA's ``control.group=list(model="ar1")`` with the group rho estimated,
jointly with the inner effect's own hyperparameters.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from scipy.sparse import identity, kron

from pylgm import AR1, AR1Structure, Fixed, Gaussian, Grouped, IID, LGM, Poisson, Replicated
from pylgm.compiler import compile_family
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.effects.ar1 import ar1_structure
from pylgm.inference import fit_gaussian
from pylgm.optimization import empirical_bayes, optimize_empirical_bayes
from pylgm.parameters import Hyperparameter


def _tau(initial=2.0):
    return Hyperparameter("u.tau", initial=initial)


def _rho(initial=0.3):
    return Hyperparameter("t.rho", initial=initial, transform="logit")


def _simulate(seed=0, units=30, periods=20, rho=0.7, tau=2.0, sigma=0.3):
    """Gaussian data from Grouped(IID over unit, AR1 over period).

    Each unit is one stationary AR1 series in time with variance 1/tau, so the
    latent precision is Q_AR1(rho) (x) tau * I_units.
    """
    rng = np.random.default_rng(seed)
    sd = 1.0 / np.sqrt(tau)
    x = np.empty((units, periods))
    x[:, 0] = rng.normal(0.0, sd, units)
    for t in range(1, periods):
        x[:, t] = rho * x[:, t - 1] + rng.normal(0.0, sd * np.sqrt(1 - rho**2), units)
    unit, period = np.meshgrid(np.arange(units), np.arange(periods), indexing="ij")
    frame = pd.DataFrame({
        "i": [f"u{k}" for k in unit.ravel()],
        "t": period.ravel(),
        "y": x.ravel() + rng.normal(0.0, sigma, x.size),
    })
    frame["row"] = range(len(frame))
    return frame


def _grouped_model(tau, rho, sigma=0.3):
    return LGM(
        response="y", likelihood=Gaussian(sigma=sigma),
        predictor=Grouped(IID("u", index="i", precision=tau), over="t",
                          structure=AR1Structure(rho)),
    )


def _panel(frame, response="y"):
    return CanonicalPanel.from_frame(frame, DataConfig(time="row", response=response, panel=()))


def _block(compiled, name):
    return next(block for block in compiled.blocks if block.name == name)


@pytest.mark.parametrize("rho", [-0.4, 0.65])
def test_materialised_precision_is_the_kronecker_product_at_each_rho(rho):
    frame = _simulate(units=4, periods=5)
    family = compile_family(_grouped_model(_tau(), _rho()), _panel(frame))
    assert set(family.parameter_names) == {"u.tau", "t.rho"}
    compiled = family.materialize({"u.tau": 1.7, "t.rho": rho})
    expected = kron(ar1_structure(5, rho), 1.7 * identity(4)).toarray()
    np.testing.assert_array_equal(_block(compiled, "u").precision.toarray(), expected)


def test_structural_inner_rebuild_composes_against_the_structure_precision():
    """Both factors estimated and both structural (inner AR1 rho, group rho):
    the rebuild must be Q_S(rho_g) (x) tau Q_E(rho_e), never identity (x) ..."""
    frame = pd.DataFrame({"s": [0, 1, 2] * 4, "t": np.repeat([0, 1, 2, 3], 3), "y": 0.0})
    frame["row"] = range(len(frame))
    model = LGM(response="y", likelihood=Poisson(), predictor=Grouped(
        AR1("u", index="s", precision=_tau(), rho=Hyperparameter("s.rho", 0.2, transform="logit")),
        over="t", structure=AR1Structure(_rho()),
    ))
    family = compile_family(model, _panel(frame))
    values = {"u.tau": 0.8, "s.rho": -0.3, "t.rho": 0.55}
    expected = kron(ar1_structure(4, 0.55), 0.8 * ar1_structure(3, -0.3)).toarray()
    np.testing.assert_allclose(
        _block(family.materialize(values), "u").precision.toarray(), expected, rtol=0, atol=1e-14
    )


def test_matches_its_replicated_ar1_oracle():
    """Grouped(IID over i, AR1Structure over t) is the same model as
    Replicated(AR1 over t, over=i) up to a permutation of the latent: same LML
    and the same estimates of tau and rho."""
    frame = _simulate(units=8, periods=10)
    grouped = _grouped_model(_tau(), _rho()).fit(frame, engine="exact_gaussian")
    replicated = LGM(
        response="y", likelihood=Gaussian(sigma=0.3),
        predictor=Replicated(AR1("u", index="t", precision=_tau(), rho=_rho()), over="i"),
    ).fit(frame, engine="exact_gaussian")
    assert grouped.log_marginal_likelihood == pytest.approx(
        replicated.log_marginal_likelihood, abs=1e-8
    )
    for name in ("u.tau", "t.rho"):
        assert grouped.hyperparameters[name] == pytest.approx(
            replicated.hyperparameters[name], rel=1e-6
        )


def test_fixing_rho_at_the_optimum_reproduces_the_estimated_fit():
    """rho fixed at its EB optimum, tau still estimated: the same optimum.
    Measured: dLML ~ 1e-13, tau rel ~ 8e-8, mean ~ 3e-8 (optimizer tolerance)."""
    frame = _simulate(units=8, periods=10)
    estimated = _grouped_model(_tau(), _rho()).fit(frame, engine="exact_gaussian")
    refit = _grouped_model(_tau(), estimated.hyperparameters["t.rho"]).fit(
        frame, engine="exact_gaussian"
    )
    assert refit.log_marginal_likelihood == pytest.approx(
        estimated.log_marginal_likelihood, abs=1e-9
    )
    assert refit.hyperparameters["u.tau"] == pytest.approx(
        estimated.hyperparameters["u.tau"], rel=1e-5
    )
    np.testing.assert_allclose(refit.mean, estimated.mean, rtol=0, atol=1e-6)


def test_fixing_both_at_the_optimum_reproduces_the_estimated_fit_exactly():
    frame = _simulate(units=8, periods=10)
    estimated = _grouped_model(_tau(), _rho()).fit(frame, engine="exact_gaussian")
    tau_hat, rho_hat = estimated.hyperparameters["u.tau"], estimated.hyperparameters["t.rho"]
    fixed = _grouped_model(tau_hat, rho_hat).fit(frame, engine="exact_gaussian")
    assert fixed.log_marginal_likelihood == pytest.approx(
        estimated.log_marginal_likelihood, abs=1e-9
    )
    np.testing.assert_allclose(fixed.mean, estimated.mean, rtol=0, atol=1e-9)


def test_recovers_known_rho_and_tau_from_simulated_data():
    # Truth: rho = 0.7, tau = 2.0, 30 units x 20 periods, sigma = 0.3 known.
    frame = _simulate(seed=1, units=30, periods=20, rho=0.7, tau=2.0)
    result = _grouped_model(_tau(1.0), _rho(0.0)).fit(frame, engine="exact_gaussian")
    assert result.hyperparameters["t.rho"] == pytest.approx(0.7, abs=0.06)
    assert result.hyperparameters["u.tau"] == pytest.approx(2.0, rel=0.2)


def test_integrated_rho_marginal_covers_the_truth():
    frame = _simulate(seed=2, units=20, periods=12, rho=0.7, tau=2.0)
    result = _grouped_model(_tau(1.0), _rho(0.0)).fit(
        frame, engine="exact_gaussian", hyperparameters="integrate"
    )
    rho = result.hyperparameter_marginals()["t.rho"]
    low, high = float(rho.quantile(0.025)[0]), float(rho.quantile(0.975)[0])
    assert low < 0.7 < high
    assert high - low < 0.5


def test_logit_transform_is_required():
    from pylgm.exceptions import CompilationError

    frame = _simulate(units=3, periods=4)
    with pytest.raises(CompilationError, match="AR1Structure rho.*logit"):
        compile_family(_grouped_model(1.0, Hyperparameter("t.rho", 0.5)), _panel(frame))


def _shuffled_panel_frame(seed=5):
    """Group and index columns whose first-seen order differs from their sorted
    order (t arrives 3, 1, 4, 0, 2; i arrives c, a, d, b), so a level-order bug
    would misalign cells."""
    rng = np.random.default_rng(seed)
    rows = [(i, t) for t in (3, 1, 4, 0, 2) for i in ("c", "a", "d", "b")]
    frame = pd.DataFrame(rows * 2, columns=["i", "t"])
    frame["firm"] = np.repeat(["f2", "f1"], len(rows))
    frame["y"] = rng.poisson(3.0, len(frame)).astype(float)
    frame["row"] = range(len(frame))
    return frame


def test_replicated_grouped_with_both_estimated_composes_and_round_trips():
    frame = _shuffled_panel_frame()
    model = LGM(response="y", likelihood=Poisson(), predictor=Fixed("1") + Replicated(
        Grouped(IID("u", index="i", precision=_tau()), over="t", structure=AR1Structure(_rho())),
        over="firm",
    ))
    family = compile_family(model, _panel(frame))
    assert set(family.parameter_names) == {"u.tau", "t.rho"}
    precision = _block(family.materialize({"u.tau": 1.3, "t.rho": 0.6}), "u").precision
    expected = kron(identity(2), kron(ar1_structure(5, 0.6), 1.3 * identity(4))).toarray()
    np.testing.assert_array_equal(precision.toarray(), expected)

    result = model.fit(frame, engine="laplace")
    assert result.hyperparameters["t.rho"] != pytest.approx(0.3, abs=1e-4)
    np.testing.assert_allclose(
        result.predict(frame).predictive_mean, result.predictive_mean, rtol=0, atol=1e-10
    )


def test_grouped_prediction_round_trips_on_a_shuffled_level_order():
    frame = _shuffled_panel_frame()
    model = LGM(response="y", likelihood=Poisson(), predictor=Fixed("1") + Grouped(
        IID("u", index="i", precision=_tau()), over="t", structure=AR1Structure(_rho()),
    ))
    result = model.fit(frame, engine="laplace")
    np.testing.assert_allclose(
        result.predict(frame).predictive_mean, result.predictive_mean, rtol=0, atol=1e-10
    )


def test_eb_gradient_is_analytic_for_rho_and_matches_central_differences(monkeypatch):
    frame = _simulate(units=6, periods=8)
    family = compile_family(_grouped_model(_tau(), _rho()), _panel(frame))
    counts, captured = [], {}

    def fit(model, *, predictive_variances=True, lml_directions=None):
        if lml_directions is not None:
            counts.append(len(lml_directions))
        return fit_gaussian(
            model, predictive_variances=predictive_variances, lml_directions=lml_directions
        )

    def fake_minimize(objective, start, *, jac, **_):
        point = start + np.array([0.2, -0.3])
        objective(point)
        captured["g"] = jac(point)
        h = 1e-4
        captured["fd"] = np.array([
            (objective(point + h * e) - objective(point - h * e)) / (2 * h) for e in np.eye(2)
        ])
        return OptimizeResult(x=start, fun=0.0, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)
    optimize_empirical_bayes(family, dict(family.parameter_bounds), fit=fit)
    assert counts and set(counts) == {2}  # tau AND rho analytic: precision-only directions
    np.testing.assert_allclose(captured["g"], captured["fd"], rtol=1e-4, atol=1e-7)

"""A ``Shared`` effect's own hyperparameters, estimated jointly with its scales.

The missing half of R-INLA's ``copy``: the shared field's precision (and a
structural parameter such as AR1 rho) is estimated together with the
per-outcome ``Shared`` scales.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from pylgm import AR1, Fixed, IID, LGM, Poisson
from pylgm.compiler import compile_joint_family
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.effects.ar1 import ar1_structure
from pylgm.exceptions import CompilationError
from pylgm.inference.laplace import fit_laplace
from pylgm.joint import Joint, Shared
from pylgm.optimization import empirical_bayes, optimize_empirical_bayes
from pylgm.parameters import Hyperparameter

OUTCOMES = ("oral", "larynx")


def _tau(initial=2.0, name="u.tau"):
    return Hyperparameter(name, initial=initial)


def _delta(initial=1.2):
    return Hyperparameter("delta", initial=initial)


def _simulate(seed=0, n=40, tau=1.0 / 0.7**2, delta=1.6, intercepts=(-0.3, 0.2)):
    """Two Poisson outcomes over one district field: oral gets delta*u,
    larynx gets u/delta, u ~ N(0, 1/tau). Districts arrive in a shuffled
    order (d17, d3, ...), so first-seen and sorted level order differ."""
    rng = np.random.default_rng(seed)
    districts = [f"d{k}" for k in rng.permutation(n)]
    u = rng.normal(0.0, 1.0 / np.sqrt(tau), n)
    eta = (intercepts[0] + delta * u, intercepts[1] + u / delta)
    return pd.DataFrame({
        "district": districts * 2,
        "oral": list(rng.poisson(np.exp(eta[0])).astype(float)) + [np.nan] * n,
        "larynx": [np.nan] * n + list(rng.poisson(np.exp(eta[1])).astype(float)),
        "row": range(2 * n),
    })


def _joint(effect, scale):
    return Joint(
        [LGM(response=name, likelihood=Poisson(), predictor=Fixed("1")) for name in OUTCOMES],
        shared=[Shared(effect, scale=scale)],
    )


def _panels(frame):
    panels = {}
    for name in OUTCOMES:
        sub = frame[frame[name].notna()].reset_index(drop=True)
        panels[name] = CanonicalPanel.from_frame(
            sub.assign(row=range(len(sub))), DataConfig(time="row", response=name, panel=()),
        )
    return panels


def test_one_outcome_joint_reduces_to_the_plain_lgm():
    frame = _simulate()
    oral = frame[frame["oral"].notna()].reset_index(drop=True)
    joint = Joint._unchecked(
        (LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),),
        (Shared(IID("u", index="district", precision=_tau()), scale=1.0),),
    ).fit(oral, engine="laplace")
    plain = LGM(
        response="oral", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="district", precision=_tau()),
    ).fit(oral, engine="laplace")
    assert joint.log_marginal_likelihood == pytest.approx(plain.log_marginal_likelihood, abs=1e-9)
    assert joint.hyperparameters["u.tau"] == pytest.approx(
        plain.hyperparameters["u.tau"], rel=1e-9
    )
    assert joint.hyperparameters["u.tau"] != pytest.approx(2.0, rel=1e-3)  # actually moved


def test_recovers_shared_precision_and_delta_from_simulated_data():
    # Truth: tau = 1/0.7^2 = 2.04, delta = 1.6; 300 districts, rates ~ e^2.
    frame = _simulate(seed=3, n=300, intercepts=(2.0, 2.0))
    result = _joint(
        IID("u", index="district", precision=_tau(1.0)), _delta(1.0)
    ).fit(frame, engine="laplace")
    assert result.hyperparameters["u.tau"] == pytest.approx(1.0 / 0.7**2, rel=0.25)
    assert result.hyperparameters["delta"] == pytest.approx(1.6, rel=0.1)


def test_integrated_marginals_cover_the_true_precision_and_delta():
    frame = _simulate(seed=3, n=150, intercepts=(2.0, 2.0))
    result = _joint(
        IID("u", index="district", precision=_tau(1.0)), _delta(1.0)
    ).fit(frame, engine="laplace", hyperparameters="integrate")
    marginals = result.hyperparameter_marginals()
    for name, truth in (("u.tau", 1.0 / 0.7**2), ("delta", 1.6)):
        low, high = (float(marginals[name].quantile(p)[0]) for p in (0.025, 0.975))
        assert low < truth < high, (name, low, high)


def test_prediction_round_trips_on_the_fit_rows_with_shuffled_levels():
    frame = _simulate(seed=4)
    result = _joint(IID("u", index="district", precision=_tau()), _delta()).fit(
        frame, engine="laplace"
    )
    start = 0
    for name in OUTCOMES:
        rows = frame[frame[name].notna()]
        predicted = result.predict(rows, outcome=name).predictive_mean
        np.testing.assert_allclose(
            predicted, result.predictive_mean[start:start + len(rows)], rtol=0, atol=1e-10
        )
        start += len(rows)


@pytest.mark.parametrize("scale", [
    (Hyperparameter("s_oral", 1.0), Hyperparameter("s_larynx", 1.0)),
    (_delta(), _delta()),
])
def test_free_scale_on_every_outcome_and_free_precision_is_rejected(scale):
    frame = _simulate()
    joint = _joint(IID("u", index="district", precision=_tau()), scale)
    with pytest.raises(CompilationError, match="not identified.*Fix the precision"):
        joint.fit(frame, engine="laplace")


def test_one_fixed_scale_identifies_the_precision():
    frame = _simulate()
    joint = _joint(
        IID("u", index="district", precision=_tau()), (1.0, Hyperparameter("s_larynx", 1.0))
    )
    result = joint.fit(frame, engine="laplace")
    assert {"u.tau", "s_larynx"} <= set(result.hyperparameters)


def test_existing_fixed_precision_joint_is_numerically_unchanged(shared_component_frame):
    """Pinned before this change (trunk 9cfc3fe): the delta-recovery joint of
    tests/test_joint_fit.py with the shared precision fixed."""
    frame, _ = shared_component_frame()
    result = _joint(IID("u", index="district", precision=1.0 / 0.7**2),
                    Hyperparameter("delta", initial=1.0)).fit(frame, engine="laplace")
    assert result.log_marginal_likelihood == pytest.approx(-138.6228549937102, rel=1e-12)
    assert result.hyperparameters["delta"] == pytest.approx(1.1465776901562956, rel=1e-10)
    assert float(np.sum(result.mean)) == pytest.approx(0.4281344461992429, rel=1e-9)


def _ar1_joint():
    frame = pd.DataFrame({
        "t": [3, 0, 2, 1, 4] * 2,
        "oral": [1.0, 2.0, 0.0, 3.0, 1.0] + [np.nan] * 5,
        "larynx": [np.nan] * 5 + [2.0, 1.0, 1.0, 4.0, 0.0],
    })
    frame["row"] = range(len(frame))
    effect = AR1("u", index="t", precision=_tau(1.5),
                 rho=Hyperparameter("u.rho", initial=0.3, transform="logit"))
    return frame, _joint(effect, _delta())


@pytest.mark.filterwarnings("ignore:empirical-Bayes estimate")  # 10 rows: rho hits its bound
def test_structural_rho_and_scale_rebuild_both_precision_and_design():
    frame, joint = _ar1_joint()
    family = compile_joint_family(joint, _panels(frame))
    assert set(family.parameter_names) == {"u.tau", "u.rho", "delta"}
    base = {name: family.parameter_bounds[name].initial for name in family.parameter_names}
    at = family.materialize({**base, "u.tau": 0.7, "u.rho": -0.4, "delta": 2.0})
    block = next(b for b in at.blocks if b.name == "u")
    np.testing.assert_allclose(
        block.precision.toarray(), (0.7 * ar1_structure(5, -0.4)).toarray(), rtol=0, atol=1e-14
    )
    design = block.design.toarray()
    assert set(np.unique(design[:5][design[:5] != 0])) == {2.0}
    assert set(np.unique(design[5:][design[5:] != 0])) == {0.5}
    result = joint.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)


def test_eb_gradient_is_analytic_for_own_parameters_and_fd_for_the_scale(monkeypatch):
    frame, joint = _ar1_joint()
    family = compile_joint_family(joint, _panels(frame))
    counts, captured = [], {}
    n = len(family.parameter_names)

    def fit(model, *, predictive_variances=True, initial_mode=None, lml_directions=None):
        if lml_directions is not None:
            counts.append(len(lml_directions))
        return fit_laplace(model, predictive_variances=predictive_variances,
                           initial_mode=initial_mode, lml_directions=lml_directions)

    def fake_minimize(objective, start, *, jac, **_):
        point = start + 0.2
        objective(point)
        captured["g"] = jac(point)
        h = 1e-4
        captured["fd"] = np.array([
            (objective(point + h * e) - objective(point - h * e)) / (2 * h) for e in np.eye(n)
        ])
        return OptimizeResult(x=start, fun=0.0, success=True, message="ok")

    monkeypatch.setattr(empirical_bayes.scipy.optimize, "minimize", fake_minimize)
    optimize_empirical_bayes(family, dict(family.parameter_bounds), fit=fit)
    # tau and rho move only the precision -> analytic; delta moves the design -> FD
    assert counts and set(counts) == {n - 1}
    np.testing.assert_allclose(captured["g"], captured["fd"], rtol=1e-3, atol=1e-6)


def test_name_collision_with_a_submodel_hyperparameter_is_rejected():
    frame = _simulate()
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(),
             predictor=Fixed("1") + IID("v", index="district", precision=_tau())),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district", precision=_tau()), scale=_delta())],
    )
    with pytest.raises(CompilationError, match="u.tau"):
        joint.fit(frame, engine="laplace")

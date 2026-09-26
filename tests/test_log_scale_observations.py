import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, LinearConstraint, LinearObservation, Poisson
from pylgm.exceptions import ModelValidationError
from pylgm.joint import Joint, Shared
from pylgm.observations import linearize, project_gaussian_model, reorder_linear_inputs, replace_sigma


def test_scale_validation():
    with pytest.raises(ValueError, match="LinearObservation scale"):
        LinearObservation([1.0], [[1.0]], sigma=1.0, scale="exp")
    with pytest.raises(ValueError, match="LinearConstraint scale"):
        LinearConstraint([[1.0]], [1.0], scale="exp")

    observation = LinearObservation([1.0], [[1.0]], sigma=1.0)
    constraint = LinearConstraint([[1.0]], [1.0])
    assert observation.scale == "identity"
    assert constraint.scale == "identity"

    log_observation = LinearObservation([1.0], [[1.0]], sigma=1.0, scale="log")
    log_constraint = LinearConstraint([[1.0]], [1.0], scale="log")

    replaced = replace_sigma(log_observation, 2.0)
    assert replaced.scale == "log"

    reordered_obs, reordered_con = reorder_linear_inputs(
        (log_observation,), (log_constraint,), [0]
    )
    assert reordered_obs[0].scale == "log"
    assert reordered_con[0].scale == "log"


def test_linearize_is_tangent():
    rng = np.random.default_rng(0)
    C = rng.uniform(0.1, 1.0, size=(3, 5))
    eta0 = rng.normal(size=5)

    observation = LinearObservation(np.zeros(3), C, sigma=1.0, scale="log")
    (linear_obs,), _ = linearize((observation,), (), eta0)

    lhs = linear_obs.operator @ eta0 + (observation.values - linear_obs.values)
    rhs = C @ np.exp(eta0)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

    def error(h, d):
        eta = eta0 + h * d
        (item,), _ = linearize((observation,), (), eta0)
        approx = item.operator @ eta + (observation.values - item.values)
        exact = C @ np.exp(eta)
        return np.max(np.abs(approx - exact))

    d = rng.normal(size=5)
    h = 1e-3
    err_h = error(h, d)
    err_2h = error(2 * h, d)
    ratio = err_2h / err_h
    assert 3.5 <= ratio <= 4.5


def _fixed_point_model():
    return LGM(
        response="y",
        likelihood=Gaussian(0.3),
        predictor=IID("u", index="cell", precision=2.0),
        panel=("cell",),
    )


def _fixed_point_frame():
    cells = [f"c{i}" for i in range(6)]
    y = [0.8, 1.1, 0.9, 1.2, np.nan, np.nan]
    return pd.DataFrame({"cell": cells, "y": y})


def _fixed_point_operator_and_values():
    C = np.array([
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ])
    values = np.array([12.0, 5.0])
    return C, values


def _true_objective(x, y_obs, C, values):
    obs_term = sum(0.5 * ((y_obs[i] - x[i]) / 0.3) ** 2 for i in range(4))
    agg_term = 0.5 * np.sum(((values - C @ np.exp(x)) / 0.5) ** 2)
    prior_term = 0.5 * 2.0 * float(x @ x)
    return prior_term + obs_term + agg_term


def _true_gradient(x, y_obs, C, values):
    # d/dx of 0.5*((y-x)/s)^2 is -(y-x)/s^2
    grad = 2.0 * x
    for i in range(4):
        grad[i] += -(y_obs[i] - x[i]) / (0.3 ** 2)
    residual = values - C @ np.exp(x)
    weighted = residual / (0.5 ** 2)
    grad += -(weighted @ C) * np.exp(x)
    return grad


def _minimize_true_mode(y_obs, C, values):
    x0 = np.zeros(6)
    result = minimize(
        _true_objective, x0, args=(y_obs, C, values),
        jac=_true_gradient, method="BFGS", options={"gtol": 1e-10},
    )
    return result.x


def test_fixed_point_is_the_true_mode():
    frame = _fixed_point_frame()
    C, values = _fixed_point_operator_and_values()
    observation = LinearObservation(values, C, sigma=0.5, scale="log")
    result = _fixed_point_model().fit(
        frame, observations=[observation], engine="exact_gaussian"
    )

    y_obs = frame["y"].to_numpy()[:4]
    expected = _minimize_true_mode(y_obs, C, values)
    np.testing.assert_allclose(result.predictive_mean, expected, atol=1e-6)


def test_log_constraint_holds_exactly():
    frame = _fixed_point_frame()
    C, _values = _fixed_point_operator_and_values()
    constraint = LinearConstraint(C[:1], [12.0], scale="log")
    result = _fixed_point_model().fit(
        frame, constraints=[constraint], engine="exact_gaussian"
    )

    assert C[:1] @ np.exp(result.predictive_mean) == pytest.approx(12.0, rel=1e-8)


def test_identity_scale_is_unchanged():
    frame = _fixed_point_frame()
    observation_implicit = LinearObservation([9.0], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], sigma=0.5)
    observation_explicit = LinearObservation(
        [9.0], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], sigma=0.5, scale="identity"
    )

    result_implicit = _fixed_point_model().fit(
        frame, observations=[observation_implicit], engine="exact_gaussian"
    )
    result_explicit = _fixed_point_model().fit(
        frame, observations=[observation_explicit], engine="exact_gaussian"
    )

    np.testing.assert_array_equal(result_implicit.predictive_mean, result_explicit.predictive_mean)
    assert result_implicit.log_marginal_likelihood == result_explicit.log_marginal_likelihood


def _hyperparameter_model():
    return LGM(
        response="y",
        likelihood=Gaussian(0.3),
        predictor=IID(
            "u", index="cell",
            precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2),
        ),
        panel=("cell",),
    )


def test_estimated_sigma_with_log_observation():
    frame = _fixed_point_frame()
    C, values = _fixed_point_operator_and_values()
    sigma_hp = Hyperparameter("s", initial=1.0, lower=1e-3, upper=1e3)
    observation = LinearObservation(values, C, sigma=sigma_hp, scale="log")

    result = _hyperparameter_model().fit(
        frame, observations=[observation], engine="exact_gaussian"
    )

    assert np.isfinite(result.log_marginal_likelihood)
    assert np.isfinite(result.hyperparameters["s"])
    aggregate = C @ np.exp(result.predictive_mean)
    np.testing.assert_allclose(aggregate, values, rtol=0.25)


def test_integrate_with_log_observation():
    frame = _fixed_point_frame()
    C, values = _fixed_point_operator_and_values()
    sigma_hp = Hyperparameter("s", initial=1.0, lower=1e-3, upper=1e3)
    observation = LinearObservation(values, C, sigma=sigma_hp, scale="log")

    result = _hyperparameter_model().fit(
        frame, observations=[observation], engine="exact_gaussian",
        hyperparameters="integrate",
    )

    assert np.isfinite(result.log_marginal_likelihood)
    assert np.isfinite(result.predictive_mean).all()
    assert np.isfinite(result.predictive_variance).all()
    assert np.isfinite(result.criteria.waic)
    draws = result.sample(10, rng=np.random.default_rng(0))
    assert draws.shape[0] == 10


def test_joint_log_observation():
    rng = np.random.default_rng(0)
    n = 24
    u = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "cell": [f"c{i:02d}" for i in range(n)],
        "target": 1 + u + rng.normal(0, 0.3, n),
        "indicator": rng.poisson(np.exp(0.5 + 0.5 * u)).astype(float),
    })
    frame.loc[18:, "target"] = np.nan

    target_model = LGM(
        response="target", likelihood=Gaussian(0.3),
        predictor=(
            Fixed("1")
            + IID(
                "v", index="cell",
                precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2),
            )
        ),
        panel=("cell",),
    )
    indicator_model = LGM(
        response="indicator", likelihood=Poisson(),
        predictor=Fixed("1"), panel=("cell",),
    )

    operator = np.zeros((6, n))
    for i in range(6):
        operator[i, 12 + 2 * i] = 1.0
        operator[i, 12 + 2 * i + 1] = 1.0
    noise = rng.normal(0, 0.05, 6)
    values = operator @ np.exp(1 + u) * np.exp(noise)
    sigma_hp = Hyperparameter("sa", initial=1.0, lower=1e-2, upper=1e2)
    obs = LinearObservation(values, operator, sigma_hp, scale="log")

    joint = Joint(
        [target_model, indicator_model],
        shared=[Shared(
            IID("s", index="cell", precision=1.0),
            scale=(1.0, Hyperparameter("lam", initial=0.5, lower=0.05, upper=5)),
        )],
    )
    result = joint.fit(frame, observations={"target": [obs]})

    assert np.isfinite(result.log_marginal_likelihood)
    prediction = result.predict(frame, outcome="target").predictive_mean
    aggregate = operator @ np.exp(prediction)
    np.testing.assert_allclose(aggregate, values, rtol=0.15)

    result_integrate = joint.fit(
        frame, observations={"target": [obs]},
        hyperparameters="integrate", latent_strategy="gaussian",
    )
    assert np.isfinite(result_integrate.log_marginal_likelihood)
    assert np.isfinite(result_integrate.criteria.waic)


def test_log_without_hyperparameters_rejects_integrate():
    frame = _fixed_point_frame()
    C, values = _fixed_point_operator_and_values()
    observation = LinearObservation(values, C, sigma=0.5, scale="log")
    with pytest.raises(ValueError, match="hyperparameters='integrate'"):
        _fixed_point_model().fit(
            frame, observations=[observation], engine="exact_gaussian",
            hyperparameters="integrate",
        )


def test_projection_rejects_unlinearized_log_items():
    from pylgm.compiler import compile_lgm

    frame = _fixed_point_frame()
    compiled = compile_lgm(_fixed_point_model(), _compile_panel(frame))

    C, values = _fixed_point_operator_and_values()
    observation = LinearObservation(values, C, sigma=0.5, scale="log")
    with pytest.raises(ModelValidationError, match="linearized before projection"):
        project_gaussian_model(compiled, (observation,), ())


def _compile_panel(frame):
    from pylgm.config.schema import DataConfig
    from pylgm.data import CanonicalPanel

    prepared = frame.copy(deep=True)
    prepared["__pylgm_row__"] = np.arange(len(prepared), dtype=np.int64)
    data = DataConfig(time="__pylgm_row__", response="y", panel=("cell",))
    return CanonicalPanel.from_frame(prepared, data, require_observed=False)


def test_caller_order():
    frame = _fixed_point_frame()
    C, values = _fixed_point_operator_and_values()
    observation = LinearObservation(values, C, sigma=0.5, scale="log")
    baseline = _fixed_point_model().fit(
        frame, observations=[observation], engine="exact_gaussian"
    )

    permutation = [3, 1, 4, 0, 5, 2]
    shuffled_frame = frame.iloc[permutation].reset_index(drop=True)
    shuffled_operator = C[:, permutation]
    shuffled_observation = LinearObservation(values, shuffled_operator, sigma=0.5, scale="log")
    shuffled_result = _fixed_point_model().fit(
        shuffled_frame, observations=[shuffled_observation], engine="exact_gaussian"
    )

    order = np.argsort(shuffled_frame["cell"].to_numpy())
    reordered = shuffled_result.predictive_mean[order]
    baseline_order = np.argsort(frame["cell"].to_numpy())
    np.testing.assert_allclose(
        reordered, baseline.predictive_mean[baseline_order], atol=1e-8
    )

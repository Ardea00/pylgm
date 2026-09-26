import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, LinearConstraint, LinearObservation, Poisson
from pylgm.exceptions import ModelValidationError, UnsupportedEngineError
from pylgm.joint import Joint, Shared
from pylgm.parameters import Hyperparameter


def _wide_frame():
    return pd.DataFrame({
        "cell": ["a", "b", "c", "d"],
        "target": [1.0, 2.0, np.nan, 4.0],
        "indicator": [5.0, 6.0, 7.0, 8.0],
    })


def _target_model():
    return LGM(
        response="target", likelihood=Gaussian(1.7),
        predictor=Fixed("1") + IID("u", index="cell", precision=2.3),
        panel=("cell",),
    )


def _indicator_model():
    return LGM(response="indicator", likelihood=Gaussian(1.0), predictor=Fixed("1"))


def _joint():
    return Joint([_target_model(), _indicator_model()])


def test_observation_anchors_to_lgm_lml_and_predictions():
    frame = _wide_frame()
    observation = LinearObservation([9.5], [[1.0, 1.0, 1.0, 0.0]], sigma=0.4)
    joint_result = _joint().fit(frame, observations={"target": [observation]})

    lgm_result = _target_model().fit(frame, engine="exact_gaussian", observations=[observation])
    indicator_result = _indicator_model().fit(frame, engine="exact_gaussian")

    expected_lml = lgm_result.log_marginal_likelihood + indicator_result.log_marginal_likelihood
    assert joint_result.log_marginal_likelihood == pytest.approx(expected_lml, rel=1e-8)

    prediction = joint_result.predict(frame, outcome="target").predictive_mean
    np.testing.assert_allclose(prediction, lgm_result.predictive_mean, rtol=1e-7)


def test_constraint_matches_lgm_and_sums_to_target():
    frame = _wide_frame()
    constraint = LinearConstraint([[1.0, 1.0, 1.0, 0.0]], [10.0])
    joint_result = _joint().fit(frame, constraints={"target": [constraint]})

    lgm_result = _target_model().fit(frame, engine="exact_gaussian", constraints=[constraint])
    indicator_result = _indicator_model().fit(frame, engine="exact_gaussian")

    expected_lml = lgm_result.log_marginal_likelihood + indicator_result.log_marginal_likelihood
    assert joint_result.log_marginal_likelihood == pytest.approx(expected_lml, rel=1e-8)

    prediction = joint_result.predict(frame, outcome="target").predictive_mean
    np.testing.assert_allclose(prediction, lgm_result.predictive_mean, rtol=1e-7)
    assert prediction[:3].sum() == pytest.approx(10.0, abs=1e-8)


def test_row_rule_keeps_referenced_nan_rows_and_drops_unreferenced_ones():
    frame = pd.DataFrame({
        "cell": ["a", "b", "c", "d"],
        "target": [1.0, 2.0, np.nan, np.nan],
        "indicator": [5.0, 6.0, 7.0, 8.0],
    })
    result_without = _joint().fit(frame)
    assert len(result_without.predictive_mean) == 2 + 4

    observation = LinearObservation([9.5], [[1.0, 1.0, 1.0, 0.0]], sigma=0.4)
    result_with = _joint().fit(frame, observations={"target": [observation]})
    assert len(result_with.predictive_mean) == 3 + 4


def test_caller_order_is_respected_by_the_operator_columns():
    frame = _wide_frame()
    observation = LinearObservation([9.5], [[1.0, 1.0, 1.0, 0.0]], sigma=0.4)
    baseline = _joint().fit(frame, observations={"target": [observation]})
    baseline_prediction = baseline.predict(frame, outcome="target").predictive_mean

    permutation = [2, 0, 3, 1]
    shuffled = frame.iloc[permutation].reset_index(drop=True)
    operator = np.asarray([[1.0, 1.0, 1.0, 0.0]])[:, permutation]
    shuffled_observation = LinearObservation([9.5], operator, sigma=0.4)
    shuffled_result = _joint().fit(shuffled, observations={"target": [shuffled_observation]})

    sorted_frame = shuffled.sort_values("cell").reset_index(drop=True)
    prediction = shuffled_result.predict(sorted_frame, outcome="target").predictive_mean
    np.testing.assert_allclose(prediction, baseline_prediction, rtol=1e-8)


def _long_two_outcome_frame(n=6):
    units = [f"u{i}" for i in range(n)]
    return pd.DataFrame({
        "unit": units * 2,
        "outcome_a": list(np.linspace(1.0, 2.0, n)) + [np.nan] * n,
        "outcome_b": [np.nan] * n + list(np.linspace(3.0, 4.0, n)),
    })


def test_observation_on_one_outcome_shifts_the_shared_outcome():
    long_frame = _long_two_outcome_frame()
    joint = Joint(
        [LGM(response="outcome_a", likelihood=Gaussian(1.0), predictor=Fixed("1")),
         LGM(response="outcome_b", likelihood=Gaussian(1.0), predictor=Fixed("1"))],
        shared=[Shared(IID("s", index="unit", precision=1.0), scale=(1.0, 1.0))],
    )
    baseline = joint.fit(long_frame)
    baseline_b = baseline.predict(long_frame, outcome="outcome_b").predictive_mean

    n = len(long_frame)
    a_positions = np.flatnonzero(long_frame["outcome_a"].notna().to_numpy())
    operator = np.zeros(n)
    operator[a_positions[: len(a_positions) // 2]] = 1.0
    assert operator.shape[0] == len(long_frame)
    observation = LinearObservation([500.0], operator[None, :], sigma=0.1)
    shifted = joint.fit(long_frame, observations={"outcome_a": [observation]})
    shifted_b = shifted.predict(long_frame, outcome="outcome_b").predictive_mean

    assert np.max(np.abs(shifted_b - baseline_b)) > 1e-3


def test_nongaussian_shared_model_converges_with_an_observation():
    n = 6
    units = [f"u{i}" for i in range(n)]
    long_frame = pd.DataFrame({
        "unit": units * 2,
        "counts": list(np.arange(1, n + 1, dtype=float)) + [np.nan] * n,
        "target": [np.nan] * n + list(np.linspace(2.0, 3.0, n)),
    })
    joint = Joint(
        [LGM(response="counts", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="target", likelihood=Gaussian(1.0), predictor=Fixed("1"))],
        shared=[Shared(IID("s", index="unit", precision=1.0), scale=(1.0, 1.0))],
    )
    target_sum = float(long_frame["target"].dropna().sum())
    operator = np.zeros(len(long_frame))
    operator[long_frame["target"].notna().to_numpy()] = 1.0
    observation = LinearObservation([target_sum], operator[None, :], sigma=1e-3)
    result = joint.fit(long_frame, observations={"target": [observation]})

    target_rows = long_frame[long_frame["target"].notna()].reset_index(drop=True)
    target_prediction = result.predict(target_rows, outcome="target").predictive_mean
    assert target_prediction.sum() == pytest.approx(target_sum, abs=1e-2)
    assert len(result.fitted_mean) == len(result.predictive_mean)
    assert np.all(np.isfinite(result.fitted_mean))
    assert np.all(np.isfinite(result.predictive_mean))


def test_estimated_sigma_recovers_the_true_noise_level():
    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(24)]
    frame = pd.DataFrame({
        "cell": cells,
        "target": [np.nan] * 24,
        "indicator": rng.normal(size=24),
    })
    target_model = LGM(
        response="target", likelihood=Gaussian(1.0),
        predictor=Fixed("1") + IID("u", index="cell", precision=4.0),
        panel=("cell",),
    )
    true_latent = rng.normal(scale=1.0 / np.sqrt(4.0), size=24)

    operators = np.zeros((12, 24))
    for i in range(12):
        operators[i, 2 * i] = 1.0
        operators[i, 2 * i + 1] = 1.0
    values = operators @ true_latent + rng.normal(scale=0.5, size=12)

    joint = Joint([target_model, _indicator_model()])
    sigma_hp = Hyperparameter("sigma_agg", initial=1.0, lower=1e-3, upper=100)
    observation = LinearObservation(values, operators, sigma=sigma_hp)
    result = joint.fit(frame, observations={"target": [observation]})

    assert np.isfinite(result.hyperparameters["sigma_agg"])
    assert 0.1 <= result.hyperparameters["sigma_agg"] <= 2.5


def test_validation_errors():
    frame = _wide_frame()
    observation = LinearObservation([9.5], [[1.0, 1.0, 1.0, 0.0]], sigma=0.4)

    with pytest.raises(TypeError, match="mapping"):
        _joint().fit(frame, observations=[observation])

    with pytest.raises(ModelValidationError, match="unknown outcome"):
        _joint().fit(frame, observations={"nonexistent": [observation]})

    constraint = LinearConstraint([[1.0, 1.0, 1.0, 0.0]], [10.0])
    with pytest.raises(TypeError, match="LinearObservation"):
        _joint().fit(frame, observations={"target": [constraint]})

    bad_operator = LinearObservation([9.5], [[1.0, 1.0, 1.0]], sigma=0.4)
    with pytest.raises(ModelValidationError, match="columns"):
        _joint().fit(frame, observations={"target": [bad_operator]})

    baseline = _joint().fit(frame)
    empty_observations = _joint().fit(frame, observations={})
    empty_constraints = _joint().fit(frame, constraints={})
    assert empty_observations.log_marginal_likelihood == baseline.log_marginal_likelihood
    assert empty_constraints.log_marginal_likelihood == baseline.log_marginal_likelihood
    assert np.array_equal(empty_observations.predictive_mean, baseline.predictive_mean)
    assert np.array_equal(empty_constraints.predictive_mean, baseline.predictive_mean)


def _integrate_setup():
    rng = np.random.default_rng(3)
    n = 16
    u = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "cell": [f"c{i:02d}" for i in range(n)],
        "target": 1 + u + rng.normal(0, 0.5, n),
        "indicator": rng.normal(0, 1, n),
    })
    frame.loc[12:, "target"] = np.nan
    operator = np.zeros((2, n))
    operator[0, 12:14] = 1.0
    operator[1, 14:16] = 1.0
    return frame, operator, operator @ (1 + u)


def _integrated_target():
    return LGM(
        response="target", likelihood=Gaussian(0.5),
        predictor=Fixed("1") + IID(
            "v", index="cell",
            precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2),
        ),
        panel=("cell",),
    )


def test_integrate_anchors_to_lgm_with_fixed_sigma():
    frame, operator, values = _integrate_setup()
    obs = LinearObservation(values, operator, 0.3)
    indicator = LGM(
        response="indicator", likelihood=Gaussian(1.0), predictor=Fixed("1"), panel=("cell",),
    )

    lgm = _integrated_target().fit(
        frame[["cell", "target"]], engine="exact_gaussian",
        hyperparameters="integrate", observations=[obs],
    )
    joint = Joint([_integrated_target(), indicator]).fit(
        frame, hyperparameters="integrate", observations={"target": [obs]},
    )
    ind_lml = indicator.fit(frame[["cell", "indicator"]], engine="exact_gaussian").log_marginal_likelihood

    assert joint.log_marginal_likelihood == pytest.approx(
        lgm.log_marginal_likelihood + ind_lml, rel=1e-8
    )

    joint_prediction = joint.predict(frame, outcome="target")
    np.testing.assert_allclose(
        joint_prediction.predictive_mean, lgm.predictive_mean, rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(
        joint_prediction.predictive_variance, lgm.predictive_variance, rtol=1e-7, atol=1e-9
    )
    assert joint.hyperparameter_marginals()["tau"].mean == pytest.approx(
        lgm.hyperparameter_marginals()["tau"].mean, rel=1e-7
    )

    print("joint.criteria.cpo:", joint.criteria.cpo)
    print("lgm.criteria.cpo:", lgm.criteria.cpo)
    np.testing.assert_allclose(joint.criteria.cpo[:12], lgm.criteria.cpo[:12], rtol=1e-6)
    np.testing.assert_allclose(joint.criteria.cpo[-2:], lgm.criteria.cpo[-2:], rtol=1e-6)


def test_integrate_estimates_observation_sigma():
    frame, operator, values = _integrate_setup()
    sigma_hp = Hyperparameter("sa", initial=1.0, lower=1e-2, upper=1e2)
    obs = LinearObservation(values, operator, sigma_hp)
    indicator = LGM(
        response="indicator", likelihood=Gaussian(1.0), predictor=Fixed("1"), panel=("cell",),
    )

    lgm = _integrated_target().fit(
        frame[["cell", "target"]], engine="exact_gaussian",
        hyperparameters="integrate", observations=[obs],
    )
    joint = Joint([_integrated_target(), indicator]).fit(
        frame, hyperparameters="integrate", observations={"target": [obs]},
    )

    assert joint.hyperparameter_marginals()["sa"].mean == pytest.approx(
        lgm.hyperparameter_marginals()["sa"].mean, rel=1e-2
    )
    joint_prediction = joint.predict(frame, outcome="target")
    np.testing.assert_allclose(joint_prediction.predictive_mean, lgm.predictive_mean, atol=1e-2)
    assert np.isfinite(joint.criteria.waic)
    assert abs(joint.criteria.waic) < 1e4


@pytest.mark.parametrize("latent_strategy", ["gaussian", "simplified_laplace", "laplace"])
def test_integrate_nongaussian_joint_with_observation(latent_strategy):
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
        predictor=Fixed("1") + IID(
            "v", index="cell",
            precision=Hyperparameter("tau", initial=1.0, lower=1e-2, upper=1e2),
        ),
        panel=("cell",),
    )
    indicator_model = LGM(
        response="indicator", likelihood=Poisson(), predictor=Fixed("1"), panel=("cell",),
    )

    operator = np.zeros((6, n))
    for i in range(6):
        operator[i, 12 + 2 * i] = 1.0
        operator[i, 12 + 2 * i + 1] = 1.0
    values = operator @ (1 + u) + rng.normal(0, 0.4, 6)
    sigma_hp = Hyperparameter("sa", initial=1.0, lower=1e-2, upper=1e2)
    obs = LinearObservation(values, operator, sigma_hp)

    joint = Joint(
        [target_model, indicator_model],
        shared=[Shared(
            IID("s", index="cell", precision=1.0),
            scale=(1.0, Hyperparameter("lam", initial=0.5, lower=0.05, upper=5)),
        )],
    )
    result = joint.fit(
        frame, hyperparameters="integrate", observations={"target": [obs]},
        latent_strategy=latent_strategy,
    )

    assert np.isfinite(result.log_marginal_likelihood)
    prediction = result.predict(frame, outcome="target")
    assert np.isfinite(prediction.predictive_mean).all()
    assert np.isfinite(prediction.predictive_variance).all()
    assert np.isfinite(result.criteria.waic)
    assert result.sample(20, rng=np.random.default_rng(1)).shape[0] == 20


def test_integrate_with_constraint():
    frame, operator, values = _integrate_setup()
    constraint = LinearConstraint(
        np.r_[np.zeros(12), np.ones(4)][None], [values.sum()]
    )
    indicator = LGM(
        response="indicator", likelihood=Gaussian(1.0), predictor=Fixed("1"), panel=("cell",),
    )
    joint = Joint([_integrated_target(), indicator])

    result = joint.fit(frame, hyperparameters="integrate", constraints={"target": [constraint]})
    prediction = result.predict(frame, outcome="target").predictive_mean
    assert prediction[12:].sum() == pytest.approx(values.sum(), abs=1e-6)

    with pytest.raises(UnsupportedEngineError):
        joint.fit(
            frame, hyperparameters="integrate", constraints={"target": [constraint]},
            latent_strategy="laplace",
        )

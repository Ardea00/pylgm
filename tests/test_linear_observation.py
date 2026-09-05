import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Fixed, Gaussian, Hyperparameter, IID, LGM, LinearConstraint
from pylgm import LinearObservation
from pylgm.exceptions import ModelValidationError


def _grid(with_response=False):
    data = {"cell": ["b", "a"], "time": [1, 1]}
    if with_response:
        data["y"] = [1.0, 2.0]
    return pd.DataFrame(data)


def _model(precision=1.0, sigma=3.0):
    return LGM(
        response="y",
        likelihood=Gaussian(sigma),
        predictor=IID("cell", "cell", precision=precision),
        panel=("cell",),
        time="time",
    )


def test_linear_observation_needs_no_pseudo_response_and_keeps_caller_order():
    # Operator columns follow the caller's [b, a] order, while compilation sorts [a, b].
    observation = LinearObservation([9.0], [[2.0, 1.0]], sigma=2.0)
    result = _model().fit(_grid(), observations=[observation])

    np.testing.assert_allclose(result.predictive_mean, [2.0, 1.0], atol=1e-12)
    assert result.observation_variance == pytest.approx(9.0)
    expected_lml = -0.5 * (np.log(2 * np.pi * 9.0) + 9.0)
    assert result.log_marginal_likelihood == pytest.approx(expected_lml)


def test_row_and_linear_observations_combine_with_their_own_noise():
    observation = LinearObservation([4.0], [[1.0, 1.0]], sigma=1.0)
    result = _model(sigma=2.0).fit(_grid(with_response=True), observations=[observation])

    # Work in caller [b, a] order: Q_post = I + I/4 + 11', score=y/4 + 4*1.
    precision = 1.25 * np.eye(2) + np.ones((2, 2))
    score = np.array([1.0, 2.0]) / 4.0 + 4.0
    np.testing.assert_allclose(result.predictive_mean, np.linalg.solve(precision, score))


def test_predictor_constraint_is_exact_and_redundant_rows_are_removed():
    observation = LinearObservation([3.0], [[1.0, 0.0]], sigma=0.2)
    constraint = LinearConstraint([[1.0, 1.0], [2.0, 2.0]], [10.0, 20.0])
    result = _model().fit(
        _grid(), observations=[observation], constraints=[constraint]
    )

    assert result.predictive_mean.sum() == pytest.approx(10.0, abs=1e-10)
    assert result.diagnostics["constraint_count"] == 1


def test_incompatible_predictor_constraints_fail_before_inference():
    constraint = LinearConstraint([[1.0, 1.0], [2.0, 2.0]], [10.0, 21.0])
    with pytest.raises(ModelValidationError, match="mutually inconsistent"):
        _model().fit(_grid(), constraints=[constraint])


@pytest.mark.parametrize("mode", ["optimize", "integrate"])
def test_linear_observations_follow_hyperparameter_paths(mode):
    precision = Hyperparameter("cell.precision", 1.0, lower=0.5, upper=2.0)
    observation = LinearObservation([2.0, 1.0], np.eye(2), sigma=[0.5, 1.0])
    if mode == "optimize":
        with pytest.warns(UserWarning, match="edge of the declared interval"):
            result = _model(precision=precision).fit(
                _grid(), observations=[observation], hyperparameters=mode
            )
    else:
        result = _model(precision=precision).fit(
            _grid(), observations=[observation], hyperparameters=mode
        )

    assert result.predictive_mean.shape == (2,)
    assert np.isfinite(result.predictive_mean).all()
    if mode == "integrate":
        assert "cell.precision" in result.hyperparameter_marginals()
    else:
        assert "cell.precision" in result.hyperparameters


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinearObservation([1.0], [[1.0]], sigma=0.0),
        lambda: LinearObservation([1.0, 2.0], [[1.0]], sigma=1.0),
        lambda: LinearConstraint([[1.0]], [1.0, 2.0]),
    ],
)
def test_linear_input_validation(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_operator_width_is_checked_against_the_grid():
    observation = LinearObservation([1.0], [[1.0, 1.0, 1.0]], sigma=1.0)
    with pytest.raises(ModelValidationError, match="one per grid row"):
        _model().fit(_grid(), observations=[observation])


def test_regional_temporal_aggregation_pattern_needs_no_adapter():
    regions, quarters = ("a", "b", "c"), (1, 2, 3, 4)
    grid = pd.DataFrame(
        [(region, quarter) for region in regions for quarter in quarters],
        columns=["region", "quarter"],
    )
    hidden = np.arange(1.0, len(grid) + 1.0).reshape(len(regions), len(quarters))
    geographic = np.zeros((len(quarters), len(grid)))
    temporal = np.zeros((len(regions), len(grid)))
    for region in range(len(regions)):
        for quarter in range(len(quarters)):
            column = region * len(quarters) + quarter
            geographic[quarter, column] = 1.0
            temporal[region, column] = 1.0

    model = LGM(
        response="gdp",
        likelihood=Gaussian(10.0),
        predictor=(
            Fixed("1", prior_precision=1.0)
            + IID("space", "region", precision=1.0)
            + AR1("common", "quarter", precision=1.0, rho=0.5)
            + AR1(
                "regional", "quarter", replicate="region", precision=1.0, rho=0.5
            )
        ),
        panel=("region",),
        time="quarter",
    )
    result = model.fit(
        grid,
        observations=[
            LinearObservation(hidden.sum(axis=0), geographic, sigma=0.5),
            LinearObservation(hidden.sum(axis=1), temporal, sigma=1.0),
        ],
        constraints=[LinearConstraint([np.ones(len(grid))], [hidden.sum()])],
    )

    assert result.predictive_mean.shape == (len(grid),)
    assert result.predictive_mean.sum() == pytest.approx(hidden.sum(), abs=1e-9)

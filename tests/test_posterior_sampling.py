"""Joint posterior draws of the linear predictor on the prediction grid."""

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import AR1, IID, Fixed, Gaussian, Hyperparameter, LGM, LinearConstraint

REGIONS, QUARTERS = 3, 12
N_DRAWS = 20_000


@pytest.fixture(params=["dense", "sparse"])
def engine(request, monkeypatch):
    if request.param == "sparse":
        monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    return request.param


def _panel(shuffle=False):
    grid = pd.DataFrame(
        [(f"r{r}", q) for r in range(REGIONS) for q in range(QUARTERS)],
        columns=["region", "q"],
    )
    rng = np.random.default_rng(1)
    truth = (5.0 + rng.normal(size=(REGIONS, 1)) + np.cumsum(
        rng.normal(size=(REGIONS, QUARTERS)), axis=1)).ravel()
    grid["y"] = truth + rng.normal(scale=0.5, size=truth.size)
    grid.loc[grid["q"] >= 8, "y"] = np.nan  # the last year is known only through aggregates
    n = len(grid)
    annual = np.zeros((REGIONS * QUARTERS // 4, n))
    national = np.zeros((QUARTERS, n))
    for column, (r, q) in enumerate(zip(grid["region"].str[1:].astype(int), grid["q"])):
        annual[r * (QUARTERS // 4) + q // 4, column] = 1.0
        national[q, column] = 1.0
    if shuffle:
        order = rng.permutation(n)
        grid, annual, national, truth = (
            grid.iloc[order].reset_index(drop=True), annual[:, order],
            national[:, order], truth[order],
        )
    constraints = [LinearConstraint(annual, annual @ truth),
                   LinearConstraint(national, national @ truth)]
    return grid, constraints


def _model(rho=0.8):
    return LGM(
        response="y", likelihood=Gaussian(0.5),
        predictor=Fixed("1") + IID("reg", "region", precision=1.0)
        + AR1("common", "q", precision=1.0, rho=rho)
        + AR1("regional", "q", replicate="region", precision=2.0, rho=rho),
        panel=("region",), time="q",
    )


def _assert_constraints_hold(draws, constraints):
    for constraint in constraints:
        residual = draws @ constraint.operator.T.toarray() - constraint.rhs
        scale = np.abs(constraint.rhs).max()
        assert np.abs(residual).max() <= 1e-9 * scale


def _assert_moments_match(draws, mean, variance):
    se = np.sqrt(variance / draws.shape[0])
    assert np.all(np.abs(draws.mean(axis=0) - mean) <= 5 * se + 1e-12)
    np.testing.assert_allclose(draws.var(axis=0), variance, rtol=0.06, atol=1e-10)


def test_every_draw_satisfies_all_constraints(engine):
    grid, constraints = _panel()
    result = _model().fit(grid, constraints=constraints)
    draws = result.sample(2_000, np.random.default_rng(0))

    assert draws.shape == (2_000, len(grid))
    _assert_constraints_hold(draws, constraints)


def test_draws_reproduce_the_posterior_moments_of_the_predictor(engine):
    grid, constraints = _panel(shuffle=True)
    result = _model().fit(grid, constraints=constraints)
    draws = result.sample(N_DRAWS, np.random.default_rng(1))

    # Caller row order: draws align with predictive_mean / predictive_variance.
    _assert_moments_match(draws, result.predictive_mean, result.predictive_variance)


def test_linear_combinations_of_draws_match_linear_combinations(engine):
    frame = pd.DataFrame({"u": ["a"] * 16, "t": range(16)})
    frame["y"] = np.sin(np.arange(16.0))
    frame.loc[12:, "y"] = np.nan
    model = LGM(response="y", likelihood=Gaussian(0.3),
                predictor=AR1("ar", "t", precision=1.0, rho=0.7), panel=("u",), time="t")
    total = LinearConstraint(np.ones((1, 16)), [3.0])
    result = model.fit(frame, constraints=[total])  # eta = x: one latent per grid row
    weights = np.random.default_rng(2).normal(size=(4, 16))
    # The summed row checks the off-diagonal covariance, which a per-row variance cannot.
    weights = np.vstack([weights, weights.sum(axis=0)])
    expected = result.linear_combinations(weights)

    combined = result.sample(N_DRAWS, np.random.default_rng(3)) @ weights.T
    _assert_moments_match(combined, expected.mean, expected.variance)


def test_a_fixed_rng_reproduces_the_draws(engine):
    grid, constraints = _panel()
    result = _model().fit(grid, constraints=constraints)
    np.testing.assert_array_equal(result.sample(5, 7), result.sample(5, 7))
    np.testing.assert_array_equal(
        result.sample(5, np.random.default_rng(7)), result.sample(5, np.random.default_rng(7))
    )
    assert not np.array_equal(result.sample(5, 7), result.sample(5, 8))


def test_integrated_draws_mix_the_hyperparameter_grid(engine):
    grid, constraints = _panel()
    rho = Hyperparameter("rho", initial=0.5, transform="logit")
    model = LGM(
        response="y", likelihood=Gaussian(0.5),
        predictor=Fixed("1") + AR1("regional", "q", replicate="region", precision=1.0, rho=rho),
        panel=("region",), time="q",
    )
    result = model.fit(grid, constraints=constraints, hyperparameters="integrate")
    assert result.diagnostics["inla_grid_points"] > 1
    draws = result.sample(N_DRAWS, np.random.default_rng(4))

    _assert_constraints_hold(draws, constraints)
    _assert_moments_match(draws, result.predictive_mean, result.predictive_variance)


@pytest.mark.parametrize("n", [0, -1, 2.5])
def test_sample_size_must_be_a_positive_integer(n):
    grid, constraints = _panel()
    result = _model().fit(grid, constraints=constraints)
    with pytest.raises(ValueError, match="positive integer"):
        result.sample(n)

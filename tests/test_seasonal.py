"""Seasonal effect: a slowly-drifting periodic pattern."""

import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, RW1, Seasonal
from pylgm.effects.seasonal import (
    build_seasonal,
    seasonal_null_basis,
    seasonal_operator,
    seasonal_penalty,
)


def _quarterly(n, period, pattern, sd, seed=7):
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n) * 0.08)
    seasonal = np.tile(pattern, n // period)[:n]
    y = trend + seasonal + sd * rng.standard_normal(n)
    return pd.DataFrame({"t": range(n), "y": y})


def test_operator_is_a_sliding_sum():
    s = seasonal_operator(6, 3).toarray()
    assert s.shape == (4, 6)
    assert np.array_equal(s[0], [1, 1, 1, 0, 0, 0])
    assert np.array_equal(s[3], [0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize(("n", "period"), [(24, 4), (23, 4), (36, 12), (30, 7)])
def test_null_basis_spans_the_fixed_seasonal_patterns(n, period):
    s = seasonal_operator(n, period).toarray()
    basis = seasonal_null_basis(n, period)
    assert basis.shape == (n, period - 1)
    assert np.allclose(s @ basis, 0.0, atol=1e-10)
    assert np.linalg.matrix_rank(basis) == period - 1
    # and it is the WHOLE null space, not part of it
    assert np.linalg.matrix_rank(s.T @ s) == n - (period - 1)


@pytest.mark.parametrize(("n", "period"), [(24, 4), (23, 4), (36, 12)])
def test_penalty_is_positive_definite_without_constraints(n, period):
    sts, projector = seasonal_penalty(n, period)
    q = (1.0 * sts + 1e-6 * projector).toarray()
    assert np.linalg.eigvalsh(q).min() > 0.0
    # projector is a genuine orthogonal projector onto the null space
    p = projector.toarray()
    assert np.allclose(p @ p, p)
    assert np.allclose(p, p.T)
    assert np.linalg.matrix_rank(p) == period - 1


def test_a_fixed_seasonal_pattern_is_unpenalized_by_precision():
    """The regression guard: the null space is the signal, not a nuisance.

    Constraining it away (the RW1/RW2 convention) would annihilate a stable
    seasonal pattern, so the effect must leave it free of the precision term
    and hold it only by the fixed ridge.
    """
    n, period = 24, 4
    pattern = np.array([3.0, -1.0, -1.5, -0.5])
    pattern -= pattern.mean()
    signal = np.tile(pattern, n // period)
    sts, projector = seasonal_penalty(n, period)
    assert signal @ (sts @ signal) == pytest.approx(0.0, abs=1e-10)
    assert signal @ (projector @ signal) > 0.0


def test_seasonal_block_has_no_constraints():
    frame = pd.DataFrame({"t": range(12), "y": np.zeros(12)})
    block = build_seasonal(frame, "s", "t", precision=1.0, period=4, ridge=1e-6)
    assert block.constraints.shape == (0, 12)
    assert block.labels == tuple(str(i) for i in range(12))


def test_recovers_a_seasonal_pattern_alongside_a_trend():
    period = 4
    pattern = np.array([1.6, -0.4, -0.9, -0.3])
    pattern -= pattern.mean()
    frame = _quarterly(60, period, pattern, sd=0.25)
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + RW1("trend", index="t", precision=Hyperparameter("trend.precision", initial=10.0))
        + Seasonal("seas", index="t", period=period,
                   precision=Hyperparameter("seas.precision", initial=10.0)),
        likelihood=Gaussian(sigma=0.25),
    )
    result = model.fit(frame)
    estimated = result.latent_marginals("seas").mean
    by_period = np.array([
        estimated[np.arange(60) % period == k].mean() for k in range(period)
    ])
    by_period -= by_period.mean()
    assert np.corrcoef(pattern, by_period)[0, 1] > 0.95
    # amplitude is recovered, not shrunk to nothing
    assert np.abs(by_period).max() > 0.5 * np.abs(pattern).max()


def test_fixed_precision_path_fits():
    frame = _quarterly(40, 4, np.array([1.0, -0.5, -0.2, -0.3]), sd=0.3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + Seasonal("seas", index="t", period=4, precision=2.0),
        likelihood=Gaussian(sigma=0.3),
    )
    result = model.fit(frame)
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.log_marginal_likelihood)


def test_predict_scores_fitted_levels():
    frame = _quarterly(40, 4, np.array([1.0, -0.5, -0.2, -0.3]), sd=0.3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + Seasonal("seas", index="t", period=4, precision=2.0),
        likelihood=Gaussian(sigma=0.3),
    )
    result = model.fit(frame)
    predicted = result.predict(frame).predictive_mean
    assert np.isfinite(predicted).all()
    assert np.corrcoef(predicted, frame["y"])[0, 1] > 0.5


def test_period_must_be_at_least_two():
    with pytest.raises(ValueError, match="period must be at least 2"):
        Seasonal("s", "t", period=1)


def test_period_must_be_an_integer():
    with pytest.raises(TypeError, match="period must be an integer"):
        Seasonal("s", "t", period=4.0)


def test_requires_more_levels_than_the_period():
    frame = pd.DataFrame({"t": range(4), "y": np.zeros(4)})
    with pytest.raises(ValueError, match="requires more than 4 ordered levels"):
        build_seasonal(frame, "s", "t", precision=1.0, period=4, ridge=1e-6)


def test_ridge_must_be_positive():
    with pytest.raises(ValueError, match="ridge"):
        Seasonal("s", "t", period=4, ridge=0.0)

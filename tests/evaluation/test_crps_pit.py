"""CRPS and PIT: closed form for Gaussian predictions, and from joint draws."""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pylgm.evaluation import aggregate_metrics, crps_from_draws, score_predictions


def _predictions(actual, mean, variance):
    n = len(actual)
    return pd.DataFrame({
        "actual": actual, "mean": mean, "variance": variance, "candidate": ["a"] * n,
        "origin": [1] * n, "horizon": [1] * n, "evaluation_mode": ["latest"] * n,
    })


def _crps_by_integration(actual, mean, sd):
    """CRPS = integral of (F(z) - 1{z >= y})^2 dz, the definition."""
    below = quad(lambda z: norm.cdf(z, mean, sd) ** 2, -np.inf, actual)[0]
    above = quad(lambda z: norm.sf(z, mean, sd) ** 2, actual, np.inf)[0]
    return below + above


def test_gaussian_crps_and_pit_match_their_definitions():
    actual, mean, variance = [1.0, -2.0, 0.3], [0.5, 0.0, 0.3], [2.0, 0.25, 1.0]
    scored = score_predictions(_predictions(actual, mean, variance), interval_levels=(0.9,))

    for i in range(3):
        sd = np.sqrt(variance[i])
        assert scored.loc[i, "crps"] == pytest.approx(
            _crps_by_integration(actual[i], mean[i], sd), rel=1e-7)
        assert scored.loc[i, "pit"] == pytest.approx(norm.cdf(actual[i], mean[i], sd))


def test_crps_aggregates_as_a_row_weighted_mean():
    scored = score_predictions(
        _predictions([1.0, 2.0, 3.0], [0.0, 2.0, 2.5], [1.0, 1.0, 4.0]), interval_levels=(0.9,)
    )
    overall = aggregate_metrics(scored).query("origin.isna() and horizon.isna()")
    assert overall["crps"].item() == pytest.approx(scored["crps"].mean())
    assert overall["crps_sum"].item() == pytest.approx(scored["crps"].sum())


def test_crps_from_draws_is_the_energy_form_and_converges_to_the_closed_form():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.5, np.sqrt(2.0), size=(40_000, 1))
    small = draws[:300, 0]
    energy = np.abs(small - 1.0).mean() - 0.5 * np.abs(small[:, None] - small[None, :]).mean()
    assert crps_from_draws(draws[:300], [1.0])[0] == pytest.approx(energy, rel=1e-12)
    assert crps_from_draws(draws, [1.0])[0] == pytest.approx(
        _crps_by_integration(1.0, 0.5, np.sqrt(2.0)), rel=0.02)


def test_crps_from_draws_is_per_column():
    draws = np.column_stack([np.zeros(10), np.ones(10)])
    np.testing.assert_allclose(crps_from_draws(draws, [0.0, 3.0]), [0.0, 2.0])

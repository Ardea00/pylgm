import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pylgm.evaluation import aggregate_metrics, score_predictions
from pylgm.exceptions import DataContractError


def test_gaussian_log_score_matches_scipy() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [0.5],
            "variance": [2.0],
            "candidate": ["a"],
            "origin": [1],
            "horizon": [1],
        }
    )

    scored = score_predictions(predictions, interval_levels=(0.95,))

    assert scored.loc[0, "log_predictive_density"] == pytest.approx(
        norm.logpdf(1.0, loc=0.5, scale=np.sqrt(2.0))
    )


@pytest.mark.parametrize("variance", [0.0, -1.0, np.inf, np.nan])
def test_scoring_rejects_nonpositive_or_nonfinite_variance(variance: float) -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [0.0],
            "variance": [variance],
            "candidate": ["a"],
            "origin": [1],
            "horizon": [1],
        }
    )

    with pytest.raises(DataContractError, match="variance"):
        score_predictions(predictions, interval_levels=(0.95,))


@pytest.mark.parametrize("level", [None, "0.95", True])
def test_scoring_rejects_nonreal_or_boolean_interval_levels(level: object) -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [0.0],
            "variance": [1.0],
            "candidate": ["a"],
            "origin": [1],
            "horizon": [1],
        }
    )

    with pytest.raises(DataContractError, match="interval levels"):
        score_predictions(predictions, interval_levels=(level,))  # type: ignore[arg-type]


def test_scoring_adds_intervals_coverage_and_benchmark_identity() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [0.0],
            "variance": [1.0],
            "candidate": ["a"],
            "origin": [1],
            "horizon": [1],
        }
    )

    scored = score_predictions(predictions, interval_levels=(0.95,))
    quantile = norm.ppf(0.975)

    assert scored.loc[0, "lower_0_95"] == pytest.approx(-quantile)
    assert scored.loc[0, "upper_0_95"] == pytest.approx(quantile)
    assert bool(scored.loc[0, "covered_0_95"])
    assert bool(scored.loc[0, "is_benchmark"]) is False


def test_aggregation_sums_prediction_rows_before_deriving_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0, 3.0, 10.0],
            "mean": [0.0, 1.0, 8.0],
            "variance": [1.0, 1.0, 4.0],
            "candidate": ["a", "a", "a"],
            "origin": [1, 1, 2],
            "horizon": [1, 1, 2],
        }
    )

    metrics = aggregate_metrics(score_predictions(predictions, interval_levels=(0.5,)))
    horizon_one = metrics.loc[metrics["horizon"].eq(1)].iloc[0]
    overall = metrics.loc[metrics["horizon"].isna()].iloc[0]

    assert horizon_one["prediction_count"] == 2
    assert horizon_one["rmse"] == pytest.approx(np.sqrt(2.5))
    assert horizon_one["mae"] == pytest.approx(1.5)
    assert horizon_one["bias"] == pytest.approx(1.5)
    assert overall["prediction_count"] == 3
    assert overall["rmse"] == pytest.approx(np.sqrt(3.0))
    assert overall["mae"] == pytest.approx(5.0 / 3.0)
    assert overall["mean_log_predictive_density"] == pytest.approx(
        np.mean(
            [
                norm.logpdf(1.0, 0.0, 1.0),
                norm.logpdf(3.0, 1.0, 1.0),
                norm.logpdf(10.0, 8.0, 2.0),
            ]
        )
    )
    assert overall["coverage_0_5"] == pytest.approx(0.0)
    assert overall["average_width_0_5"] == pytest.approx(
        (2 * norm.ppf(0.75) + 2 * norm.ppf(0.75) + 4 * norm.ppf(0.75)) / 3
    )


def test_aggregation_reports_benchmark_rows_at_every_level() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0, 1.0],
            "mean": [1.0, 0.0],
            "variance": [1.0, 1.0],
            "candidate": ["model", "persistence"],
            "origin": [1, 1],
            "horizon": [1, 1],
            "is_benchmark": [False, True],
        }
    )

    metrics = aggregate_metrics(score_predictions(predictions, interval_levels=()))

    assert metrics.loc[metrics["candidate"].eq("persistence"), "is_benchmark"].all()
    assert len(metrics.loc[metrics["candidate"].eq("persistence")]) == 2

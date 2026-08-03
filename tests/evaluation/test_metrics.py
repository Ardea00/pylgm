from datetime import date
from pathlib import Path

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
            "actual": [1.0, 3.0, 10.0, 6.0],
            "mean": [0.0, 1.0, 8.0, 2.0],
            "variance": [1.0, 1.0, 4.0, 1.0],
            "candidate": ["a", "a", "a", "a"],
            "origin": [1, 1, 2, 2],
            "horizon": [1, 1, 2, 1],
        }
    )

    metrics = aggregate_metrics(score_predictions(predictions, interval_levels=(0.5,)))
    fold_one = metrics.loc[metrics["origin"].eq(1) & metrics["horizon"].eq(1)].iloc[0]
    horizon_one = metrics.loc[metrics["origin"].isna() & metrics["horizon"].eq(1)].iloc[0]
    overall = metrics.loc[metrics["origin"].isna() & metrics["horizon"].isna()].iloc[0]

    assert fold_one["prediction_count"] == 2
    assert horizon_one["prediction_count"] == 3
    assert horizon_one["rmse"] == pytest.approx(np.sqrt(7.0))
    assert horizon_one["mae"] == pytest.approx(7.0 / 3.0)
    assert horizon_one["bias"] == pytest.approx(7.0 / 3.0)
    assert overall["prediction_count"] == 4
    assert overall["rmse"] == pytest.approx(2.5)
    assert overall["mae"] == pytest.approx(2.25)
    assert overall["mean_log_predictive_density"] == pytest.approx(
        np.mean(
            [
                norm.logpdf(1.0, 0.0, 1.0),
                norm.logpdf(3.0, 1.0, 1.0),
                norm.logpdf(10.0, 8.0, 2.0),
                norm.logpdf(6.0, 2.0, 1.0),
            ]
        )
    )
    assert overall["coverage_0_5"] == pytest.approx(0.0)
    assert overall["average_width_0_5"] == pytest.approx(2.5 * norm.ppf(0.75))
    assert set(zip(metrics["origin"].notna(), metrics["horizon"].notna())) == {
        (True, True),
        (False, True),
        (False, False),
    }


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
    assert len(metrics.loc[metrics["candidate"].eq("persistence")]) == 3


def test_aggregation_requires_nonnull_origins() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [1.0],
            "variance": [1.0],
            "candidate": ["a"],
            "horizon": [1],
        }
    )

    scored = score_predictions(predictions, interval_levels=())

    with pytest.raises(DataContractError, match="origin"):
        aggregate_metrics(scored)


def test_aggregation_rejects_incomparable_mixed_origin_scalars() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0, 1.0],
            "mean": [1.0, 1.0],
            "variance": [1.0, 1.0],
            "candidate": ["a", "a"],
            "origin": pd.Series(
                [date(2025, 1, 1), pd.Timestamp("2025-01-02T00:00:00+01:00")],
                dtype=object,
            ),
            "horizon": [1, 1],
        }
    )

    with pytest.raises(DataContractError, match="origins.*coherent total order"):
        aggregate_metrics(score_predictions(predictions, interval_levels=()))


def test_aggregation_requires_exact_integer_horizons() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [1.0],
            "variance": [1.0],
            "candidate": ["a"],
            "origin": [1],
            "horizon": [1.0],
        }
    )

    with pytest.raises(DataContractError, match="horizon.*integer"):
        aggregate_metrics(score_predictions(predictions, interval_levels=()))


def test_metric_coordinates_preserve_large_integers_through_parquet(tmp_path: Path) -> None:
    origin = 2**60 + 1
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [1.0],
            "variance": [1.0],
            "candidate": ["a"],
            "origin": [origin],
            "horizon": [1],
        }
    )

    metrics = aggregate_metrics(score_predictions(predictions, interval_levels=()))
    fold = metrics.loc[metrics["origin"].notna() & metrics["horizon"].notna()].iloc[0]

    assert fold["origin"] == origin
    assert fold["horizon"] == 1
    assert str(metrics["origin"].dtype) == "Int64"
    assert str(metrics["horizon"].dtype) == "Int64"
    path = tmp_path / "metrics.parquet"
    metrics.to_parquet(path, index=False)
    persisted = pd.read_parquet(path)
    persisted_fold = persisted.loc[persisted["origin"].notna() & persisted["horizon"].notna()].iloc[
        0
    ]
    assert persisted_fold["origin"] == origin
    assert persisted_fold["horizon"] == 1
    assert str(persisted["origin"].dtype) == "Int64"
    assert str(persisted["horizon"].dtype) == "Int64"


@pytest.mark.parametrize(
    "origin",
    [date(2025, 1, 1), pd.Timestamp("2025-01-01T00:00:00+01:00"), "late"],
)
def test_metric_origins_preserve_supported_scalars_through_parquet(
    tmp_path: Path, origin: object
) -> None:
    origins = (
        pd.Categorical([origin], categories=["early", "late"], ordered=True)
        if origin == "late"
        else [origin]
    )
    predictions = pd.DataFrame(
        {
            "actual": [1.0],
            "mean": [1.0],
            "variance": [1.0],
            "candidate": ["a"],
            "origin": origins,
            "horizon": [1],
        }
    )

    metrics = aggregate_metrics(score_predictions(predictions, interval_levels=()))
    fold = metrics.loc[metrics["origin"].notna() & metrics["horizon"].notna()].iloc[0]
    path = tmp_path / "metrics.parquet"
    metrics.to_parquet(path, index=False)
    persisted = pd.read_parquet(path)
    persisted_fold = persisted.loc[persisted["origin"].notna() & persisted["horizon"].notna()].iloc[
        0
    ]

    assert fold["origin"] == origin
    assert persisted_fold["origin"] == origin

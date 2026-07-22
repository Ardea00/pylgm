from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from pylgm.config.experiment import EvaluationConfig, OriginConfig
from pylgm.evaluation import CandidateDecision, select_candidate
from pylgm.exceptions import SelectionError


def selection_config(**updates: object) -> EvaluationConfig:
    return EvaluationConfig(
        horizons=(1,),
        origins=OriginConfig(last=1),
        interval_levels=(0.95,),
        max_abs_coverage_error=0.1,
        rmse_tie_tolerance=0.01,
        **updates,
    )


def metrics_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "horizon": None,
        "is_benchmark": False,
        "mean_log_predictive_density": -1.0,
        "rmse": 1.0,
        "coverage_0_95": 0.95,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_selection_uses_log_score_then_rmse_tie_break() -> None:
    metrics = metrics_table(
        [
            {"candidate": "lower_rmse", "mean_log_predictive_density": -2.0, "rmse": 0.1},
            {"candidate": "better_log_score", "mean_log_predictive_density": -1.0, "rmse": 9.0},
        ]
    )

    result = select_candidate(metrics, failures={}, config=selection_config())

    assert result.selected == "better_log_score"
    assert result.ranking == ("better_log_score", "lower_rmse")


def test_selection_uses_rmse_tolerance_then_name_for_equal_log_scores() -> None:
    metrics = metrics_table(
        [
            {"candidate": "zeta", "rmse": 1.0},
            {"candidate": "alpha", "rmse": 1.005},
        ]
    )

    result = select_candidate(metrics, failures={}, config=selection_config())

    assert result.selected == "alpha"
    assert result.ranking == ("alpha", "zeta")


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"candidate": "zeta", "rmse": 1.000},
            {"candidate": "beta", "rmse": 1.008},
            {"candidate": "alpha", "rmse": 1.016},
        ],
        [
            {"candidate": "alpha", "rmse": 1.016},
            {"candidate": "zeta", "rmse": 1.000},
            {"candidate": "beta", "rmse": 1.008},
        ],
    ],
)
def test_selection_anchors_rmse_tolerance_to_minimum_for_three_candidate_chain(
    rows: list[dict[str, object]],
) -> None:
    result = select_candidate(metrics_table(rows), failures={}, config=selection_config())

    assert result.selected == "beta"
    assert result.ranking == ("beta", "zeta", "alpha")


def test_failures_and_bad_calibration_are_ineligible_with_reasons() -> None:
    metrics = metrics_table(
        [
            {"candidate": "failed"},
            {"candidate": "uncalibrated", "coverage_0_95": 0.5},
            {"candidate": "good", "mean_log_predictive_density": -2.0},
        ]
    )

    result = select_candidate(
        metrics, failures={"failed": ("optimizer exploded",)}, config=selection_config()
    )

    assert result.selected == "good"
    assert result.eligible == {"failed": False, "good": True, "uncalibrated": False}
    assert result.reasons["failed"] == ("failure: optimizer exploded",)
    assert result.reasons["uncalibrated"] == ("coverage_0_95 exceeds tolerance",)


def test_benchmark_is_ranked_and_reported_but_never_selected() -> None:
    metrics = metrics_table(
        [
            {"candidate": "model", "mean_log_predictive_density": -2.0},
            {
                "candidate": "persistence",
                "is_benchmark": True,
                "mean_log_predictive_density": 100.0,
            },
        ]
    )

    result = select_candidate(metrics, failures={}, config=selection_config())

    assert result.selected == "model"
    assert result.eligible["persistence"] is False
    assert result.reasons["persistence"] == ("benchmark",)
    assert result.ranking == ("model", "persistence")


def test_nonfinite_metrics_are_ineligible_and_all_ineligible_raises() -> None:
    metrics = metrics_table(
        [{"candidate": "bad", "mean_log_predictive_density": float("nan")}]
    )

    with pytest.raises(SelectionError, match="no eligible"):
        select_candidate(metrics, failures={}, config=selection_config())


def test_decision_is_immutable_isolated_and_json_ready() -> None:
    eligible = {"a": True}
    reasons = {"a": ()}
    decision = CandidateDecision("a", ("a",), eligible, reasons)
    eligible["a"] = False
    reasons["a"] = ("changed",)

    assert decision.eligible["a"] is True
    assert decision.reasons["a"] == ()
    with pytest.raises((FrozenInstanceError, TypeError)):
        decision.eligible["a"] = False  # type: ignore[index]
    assert decision.to_dict() == {
        "selected": "a",
        "ranking": ["a"],
        "eligible": {"a": True},
        "reasons": {"a": []},
    }

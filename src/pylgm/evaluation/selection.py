"""Deterministic, eligibility-first predictive candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pylgm.config.experiment import EvaluationConfig
from pylgm.exceptions import DataContractError, SelectionError


@dataclass(frozen=True)
class CandidateDecision:
    """An immutable, JSON-serializable explanation of a selection decision."""

    selected: str
    ranking: tuple[str, ...]
    eligible: Mapping[str, bool]
    reasons: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranking", tuple(self.ranking))
        object.__setattr__(
            self,
            "eligible",
            MappingProxyType({name: bool(value) for name, value in self.eligible.items()}),
        )
        object.__setattr__(
            self,
            "reasons",
            MappingProxyType(
                {name: tuple(reason) for name, reason in self.reasons.items()}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a defensive JSON-ready representation."""
        return {
            "selected": self.selected,
            "ranking": list(self.ranking),
            "eligible": dict(self.eligible),
            "reasons": {name: list(reasons) for name, reasons in self.reasons.items()},
        }


def _coverage_column(level: float) -> str:
    return f"coverage_{str(level).replace('.', '_')}"


def _failure_reasons(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        details = (value,)
    else:
        try:
            details = tuple(str(item) for item in value)  # type: ignore[arg-type]
        except TypeError:
            details = (str(value),)
    return tuple(f"failure: {detail}" for detail in details) or ("failure",)


def _overall_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate", "horizon", "is_benchmark", "mean_log_predictive_density", "rmse"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise DataContractError(f"metric rows lack required columns: {missing}")
    if not metrics["is_benchmark"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise DataContractError("metric is_benchmark values must be booleans")
    return metrics.loc[metrics["horizon"].isna()].copy(deep=True)


def _metric_values(row: pd.Series, config: EvaluationConfig) -> tuple[float, float]:
    values = [row["mean_log_predictive_density"], row["rmse"]]
    coverage_columns = [_coverage_column(level) for level in config.interval_levels]
    missing = [column for column in coverage_columns if column not in row.index]
    if missing:
        raise DataContractError(f"overall metric row lacks coverage columns: {missing}")
    values.extend(row[column] for column in coverage_columns)
    try:
        numeric = tuple(float(value) for value in values)
    except (TypeError, ValueError) as cause:
        raise DataContractError("selection metrics must be numeric") from cause
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite selection metric")
    return numeric[0], numeric[1]


def select_candidate(
    metrics: pd.DataFrame,
    failures: Mapping[str, object],
    config: EvaluationConfig,
) -> CandidateDecision:
    """Select a non-benchmark candidate with deterministic eligibility reasons."""
    overall = _overall_rows(metrics)
    names = set(str(name) for name in overall["candidate"].dropna()) | set(failures)
    if not names:
        raise SelectionError("no candidates available for selection")

    rows_by_name = {
        name: overall.loc[overall["candidate"].astype(str).eq(name)] for name in names
    }
    eligible: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    ranking_values: dict[str, tuple[float, float]] = {}
    for name in sorted(names):
        candidate_rows = rows_by_name[name]
        candidate_reasons: list[str] = []
        if name in failures:
            candidate_reasons.extend(_failure_reasons(failures[name]))
        if candidate_rows.empty:
            candidate_reasons.append("missing overall metrics")
        elif len(candidate_rows) != 1:
            candidate_reasons.append("inconsistent benchmark identity")
        else:
            row = candidate_rows.iloc[0]
            if bool(row["is_benchmark"]):
                candidate_reasons.append("benchmark")
            try:
                log_density, rmse = _metric_values(row, config)
            except ValueError:
                candidate_reasons.append("non-finite metric")
            else:
                ranking_values[name] = (log_density, rmse)
                for level in config.interval_levels:
                    coverage = float(row[_coverage_column(level)])
                    if abs(coverage - level) > config.max_abs_coverage_error:
                        candidate_reasons.append(
                            f"{_coverage_column(level)} exceeds tolerance"
                        )
        eligible[name] = not candidate_reasons
        reasons[name] = tuple(candidate_reasons)

    def compare(left: str, right: str) -> int:
        if eligible[left] != eligible[right]:
            return -1 if eligible[left] else 1
        left_values = ranking_values.get(left, (-np.inf, np.inf))
        right_values = ranking_values.get(right, (-np.inf, np.inf))
        if left_values[0] != right_values[0]:
            return -1 if left_values[0] > right_values[0] else 1
        if abs(left_values[1] - right_values[1]) > config.rmse_tie_tolerance:
            return -1 if left_values[1] < right_values[1] else 1
        return -1 if left < right else (1 if left > right else 0)

    ranking = tuple(sorted(names, key=cmp_to_key(compare)))
    selected = next((name for name in ranking if eligible[name]), None)
    if selected is None:
        raise SelectionError("no eligible candidates available for selection")
    return CandidateDecision(selected, ranking, eligible, reasons)

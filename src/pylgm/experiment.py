"""Explicit orchestration for leakage-safe predictive comparisons."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from pandas.api.types import is_object_dtype

from pylgm.compiler import compile_gaussian_family
from pylgm.config import ExperimentConfig, load_experiment_config, resolve_candidates
from pylgm.config.experiment import ResolvedCandidate
from pylgm.data import CanonicalPanel
from pylgm.evaluation import (
    CandidateDecision,
    FoldData,
    aggregate_metrics,
    build_fold_definitions,
    materialize_fold,
    persistence_predictions,
    score_predictions,
    select_candidate,
)
from pylgm.exceptions import PyLGMError
from pylgm.inference import fit_gaussian
from pylgm.optimization import OptimizationBounds, optimize_empirical_bayes
from pylgm.optimization.result import OptimizationDiagnostics


def _copy_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for position, dtype in enumerate(result.dtypes):
        if is_object_dtype(dtype):
            result.iloc[:, position] = result.iloc[:, position].map(copy.deepcopy)
    result.attrs = copy.deepcopy(frame.attrs)
    return result


@dataclass(frozen=True, init=False)
class ComparisonResult:
    """Immutable comparison metadata with defensive tabular views."""

    candidates: tuple[str, ...]
    selected: str
    _predictions: pd.DataFrame = field(repr=False)
    _metrics: pd.DataFrame = field(repr=False)
    _diagnostics: pd.DataFrame = field(repr=False)
    failures: Mapping[str, tuple[str, ...]]
    decision: CandidateDecision

    def __init__(
        self,
        candidates: tuple[str, ...],
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
        diagnostics: pd.DataFrame,
        failures: Mapping[str, tuple[str, ...]],
        decision: CandidateDecision,
    ) -> None:
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "selected", decision.selected)
        object.__setattr__(self, "_predictions", _copy_table(predictions))
        object.__setattr__(self, "_metrics", _copy_table(metrics))
        object.__setattr__(self, "_diagnostics", _copy_table(diagnostics))
        object.__setattr__(
            self,
            "failures",
            MappingProxyType({name: tuple(messages) for name, messages in failures.items()}),
        )
        object.__setattr__(self, "decision", decision)

    @property
    def predictions(self) -> pd.DataFrame:
        return _copy_table(self._predictions)

    @property
    def metrics(self) -> pd.DataFrame:
        return _copy_table(self._metrics)

    @property
    def diagnostics(self) -> pd.DataFrame:
        return _copy_table(self._diagnostics)


def _bounds(candidate: ResolvedCandidate) -> dict[str, OptimizationBounds]:
    return {
        name: OptimizationBounds(value.initial, value.lower, value.upper)
        for name, value in candidate.optimize.items()
    }


def _prediction_rows(
    config: ExperimentConfig,
    candidate: ResolvedCandidate,
    fold: FoldData,
    parameters: Mapping[str, float],
) -> pd.DataFrame:
    panel = CanonicalPanel.from_frame(fold.model_frame, config.data)
    family = compile_gaussian_family(config.data, candidate.model, panel, tuple(candidate.optimize))
    fit = fit_gaussian(family.materialize(parameters))
    keys = [*config.data.panel, config.data.time]
    predicted = panel.frame.loc[:, keys]
    predicted["mean"] = fit.predictive_mean
    predicted["variance"] = fit.predictive_variance
    target = fold.target_frame.loc[:, [*keys, config.data.response]].rename(
        columns={config.data.response: "actual"}
    )
    result = target.merge(predicted, on=keys, how="left", validate="one_to_one")
    result = result.rename(columns={config.data.time: "target_time"})
    result["origin"] = fold.definition.origin
    result["horizon"] = fold.definition.horizon
    result["candidate"] = candidate.name
    result["is_benchmark"] = False
    return result


def _diagnostic_rows(
    candidate: str,
    origin: object,
    bounds: Mapping[str, OptimizationBounds],
    diagnostics: OptimizationDiagnostics,
) -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "origin": origin,
            "parameter": parameter,
            "initial": diagnostics.initial[parameter],
            "optimum": diagnostics.optimum[parameter],
            "lower": bounds[parameter].lower,
            "upper": bounds[parameter].upper,
            "active_bound": parameter in diagnostics.active_bounds,
            "converged": diagnostics.converged,
            "objective": diagnostics.objective,
            "evaluations": diagnostics.evaluations,
            "cache_hits": diagnostics.cache_hits,
            "elapsed_seconds": diagnostics.elapsed_seconds,
            "numerical_failures": diagnostics.numerical_failures,
        }
        for parameter in bounds
    ]


def _folds_by_origin(
    folds: tuple[FoldData, ...],
) -> tuple[tuple[object, tuple[FoldData, ...]], ...]:
    grouped: dict[object, list[FoldData]] = {}
    for fold in folds:
        grouped.setdefault(fold.definition.origin, []).append(fold)
    return tuple((origin, tuple(values)) for origin, values in grouped.items())


def _run_candidates(
    config: ExperimentConfig,
    candidates: tuple[ResolvedCandidate, ...],
    folds: tuple[FoldData, ...],
) -> tuple[list[pd.DataFrame], pd.DataFrame, dict[str, tuple[str, ...]]]:
    predictions: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, object]] = []
    failures: dict[str, tuple[str, ...]] = {}
    grouped_folds = _folds_by_origin(folds)
    for candidate in candidates:
        candidate_predictions: list[pd.DataFrame] = []
        candidate_diagnostics: list[dict[str, object]] = []
        warm_start: Mapping[str, float] | None = None
        for origin, origin_folds in grouped_folds:
            try:
                training_panel = CanonicalPanel.from_frame(
                    origin_folds[0].training_frame, config.data
                )
                family = compile_gaussian_family(
                    config.data,
                    candidate.model,
                    training_panel,
                    tuple(candidate.optimize),
                )
                bounds = _bounds(candidate)
                optimized = optimize_empirical_bayes(family, bounds, initial=warm_start)
                candidate_diagnostics.extend(
                    _diagnostic_rows(candidate.name, origin, bounds, optimized.diagnostics)
                )
                candidate_predictions.extend(
                    _prediction_rows(config, candidate, fold, optimized.parameters)
                    for fold in origin_folds
                )
                warm_start = optimized.parameters
            except PyLGMError as error:
                failures[candidate.name] = (f"origin={origin!r}: {type(error).__name__}: {error}",)
                break
        else:
            predictions.extend(candidate_predictions)
        diagnostic_records.extend(candidate_diagnostics)
    return predictions, pd.DataFrame.from_records(diagnostic_records), failures


def _run_persistence(config: ExperimentConfig, folds: tuple[FoldData, ...]) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for fold in folds:
        result = persistence_predictions(fold).rename(columns={config.data.time: "target_time"})
        result["origin"] = fold.definition.origin
        result["horizon"] = fold.definition.horizon
        result["candidate"] = "persistence"
        result["is_benchmark"] = True
        predictions.append(result)
    return pd.concat(predictions, ignore_index=True)


@dataclass(frozen=True)
class Experiment:
    """A configured predictive candidate comparison."""

    config: ExperimentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        return cls(load_experiment_config(Path(path)))

    def compare(self, frame: pd.DataFrame, output: str | Path | None = None) -> ComparisonResult:
        if output is not None:
            raise PyLGMError(
                "experiment artifact output is not available in this release; use output=None"
            )
        definitions = build_fold_definitions(frame, self.config.data, self.config.evaluation)
        folds = tuple(
            materialize_fold(frame, self.config.data, self.config.evaluation, definition)
            for definition in definitions
        )
        candidates = resolve_candidates(self.config)
        candidate_predictions, diagnostics, failures = _run_candidates(
            self.config, candidates, folds
        )
        predictions = pd.concat(
            [*candidate_predictions, _run_persistence(self.config, folds)],
            ignore_index=True,
        )
        scored = score_predictions(predictions, self.config.evaluation.interval_levels)
        metrics = aggregate_metrics(scored)
        decision = select_candidate(metrics, failures, self.config.evaluation)
        return ComparisonResult(
            tuple(candidate.name for candidate in candidates),
            scored,
            metrics,
            diagnostics,
            failures,
            decision,
        )

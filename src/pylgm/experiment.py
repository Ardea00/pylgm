"""Explicit orchestration for leakage-safe predictive comparisons."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from pylgm.artifacts import write_experiment
from pylgm.compiler import compile_gaussian_family
from pylgm.config import ExperimentConfig, load_experiment_config, resolve_candidates
from pylgm.config.experiment import ResolvedCandidate
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.data.fingerprint import panel_fingerprint
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
from pylgm.exceptions import OptimizationError, PyLGMError
from pylgm.inference import fit_gaussian, preflight_dense_reference
from pylgm.optimization import OptimizationBounds, optimize_empirical_bayes
from pylgm.optimization.result import OptimizationDiagnostics


def _copy_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for position, dtype in enumerate(result.dtypes):
        if is_object_dtype(dtype):
            result.iloc[:, position] = result.iloc[:, position].map(copy.deepcopy)
    result.attrs = copy.deepcopy(frame.attrs)
    return result


def _json_ready_coordinate(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_ready_coordinate(value.item())
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return {"type": type(value).__name__, "value": str(value)}


@dataclass(frozen=True)
class FailureCause:
    """Immutable root cause retained for a structured candidate failure."""

    type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "message": self.message}


@dataclass(frozen=True)
class CandidateFailure:
    """Immutable, JSON-ready details for one failed candidate stage or fold."""

    candidate: str
    origin: object
    horizon: int | None
    stage: str
    type: str
    message: str
    evaluations: int = 0
    cache_hits: int = 0
    numerical_failures: tuple[str, ...] = ()
    root_cause: FailureCause | None = None

    @classmethod
    def from_error(
        cls,
        candidate: str,
        origin: object,
        error: PyLGMError,
        *,
        stage: str,
        horizon: int | None = None,
    ) -> CandidateFailure:
        cause = error.__cause__
        root_cause = None if cause is None else FailureCause(type(cause).__name__, str(cause))
        if isinstance(error, OptimizationError):
            evaluations = error.evaluations
            cache_hits = error.cache_hits
            numerical_failures = error.numerical_failures
        else:
            evaluations = 0
            cache_hits = 0
            numerical_failures = ()
        return cls(
            candidate=candidate,
            origin=origin,
            horizon=horizon,
            stage=stage,
            type=type(error).__name__,
            message=str(error),
            evaluations=evaluations,
            cache_hits=cache_hits,
            numerical_failures=tuple(numerical_failures),
            root_cause=root_cause,
        )

    def __str__(self) -> str:
        location = f"origin={self.origin!r}"
        if self.horizon is not None:
            location += f" horizon={self.horizon}"
        return f"{self.stage} {location}: {self.type}: {self.message}"

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "origin": _json_ready_coordinate(self.origin),
            "horizon": self.horizon,
            "stage": self.stage,
            "type": self.type,
            "message": self.message,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "numerical_failures": list(self.numerical_failures),
            "root_cause": (None if self.root_cause is None else self.root_cause.to_dict()),
        }


@dataclass(frozen=True, init=False)
class ComparisonResult:
    """Immutable comparison metadata with defensive tabular views."""

    candidates: tuple[str, ...]
    selected: str
    _predictions: pd.DataFrame = field(repr=False)
    _metrics: pd.DataFrame = field(repr=False)
    _diagnostics: pd.DataFrame = field(repr=False)
    failures: Mapping[str, tuple[CandidateFailure, ...]]
    decision: CandidateDecision

    def __init__(
        self,
        candidates: tuple[str, ...],
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
        diagnostics: pd.DataFrame,
        failures: Mapping[str, tuple[CandidateFailure, ...]],
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
    fit = fit_gaussian(
        family.materialize(parameters),
        allow_large_dense=config.inference.allow_large_dense,
    )
    keys = [*config.data.panel, config.data.time]
    predicted = panel.frame.loc[:, keys]
    predicted["mean"] = fit.predictive_mean
    # Backtest scoring is on the RESPONSE scale: evaluation.metrics feeds this
    # column to norm.logpdf(actual, ...) and to the interval bounds, so it needs
    # the observation noise that predictive_variance (the linear-predictor
    # variance) deliberately excludes.
    predicted["variance"] = fit.predictive_variance + fit.observation_variance
    target = fold.target_frame.loc[:, [*keys, config.data.response]].rename(
        columns={config.data.response: "actual"}
    )
    result = target.merge(predicted, on=keys, how="left", validate="one_to_one")
    result = result.rename(columns={config.data.time: "target_time"})
    result["origin"] = fold.definition.origin
    result["horizon"] = fold.definition.horizon
    result["candidate"] = candidate.name
    result["engine"] = "exact_gaussian"
    result["is_benchmark"] = False
    result["evaluation_mode"] = config.evaluation.mode
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


def _initial_parameters(candidate: ResolvedCandidate) -> dict[str, float]:
    return {name: float(bounds.initial) for name, bounds in candidate.optimize.items()}


def _preflight_family(
    config: ExperimentConfig,
    candidate: ResolvedCandidate,
    frame: pd.DataFrame,
) -> None:
    panel = CanonicalPanel.from_frame(frame, config.data)
    family = compile_gaussian_family(
        config.data,
        candidate.model,
        panel,
        tuple(candidate.optimize),
    )
    preflight_dense_reference(
        family.materialize(_initial_parameters(candidate)),
        allow_large_dense=config.inference.allow_large_dense,
    )


def _preflight_candidates(
    config: ExperimentConfig,
    candidates: tuple[ResolvedCandidate, ...],
    folds: tuple[FoldData, ...],
) -> dict[str, tuple[CandidateFailure, ...]]:
    """Preflight every candidate training/prediction family before optimization."""
    failures: dict[str, list[CandidateFailure]] = {}
    grouped_folds = _folds_by_origin(folds)
    for candidate in candidates:
        for origin, origin_folds in grouped_folds:
            try:
                _preflight_family(config, candidate, origin_folds[0].training_frame)
            except PyLGMError as error:
                failures.setdefault(candidate.name, []).append(
                    CandidateFailure.from_error(
                        candidate.name,
                        origin,
                        error,
                        stage="preflight_training",
                    )
                )
            for fold in origin_folds:
                try:
                    _preflight_family(config, candidate, fold.model_frame)
                except PyLGMError as error:
                    failures.setdefault(candidate.name, []).append(
                        CandidateFailure.from_error(
                            candidate.name,
                            origin,
                            error,
                            stage="preflight_prediction",
                            horizon=fold.definition.horizon,
                        )
                    )
    return {name: tuple(values) for name, values in failures.items()}


def _run_candidates(
    config: ExperimentConfig,
    candidates: tuple[ResolvedCandidate, ...],
    folds: tuple[FoldData, ...],
    preflight_failures: Mapping[str, tuple[CandidateFailure, ...]],
) -> tuple[
    list[pd.DataFrame],
    pd.DataFrame,
    dict[str, tuple[CandidateFailure, ...]],
]:
    predictions: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, object]] = []
    failures = dict(preflight_failures)
    grouped_folds = _folds_by_origin(folds)
    for candidate in candidates:
        if candidate.name in failures:
            continue
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
            except PyLGMError as error:
                failures[candidate.name] = (
                    CandidateFailure.from_error(
                        candidate.name,
                        origin,
                        error,
                        stage="training_compile",
                    ),
                )
                break
            try:
                optimized = optimize_empirical_bayes(
                    family,
                    bounds,
                    initial=warm_start,
                    allow_large_dense=config.inference.allow_large_dense,
                )
                candidate_diagnostics.extend(
                    _diagnostic_rows(candidate.name, origin, bounds, optimized.diagnostics)
                )
            except PyLGMError as error:
                failures[candidate.name] = (
                    CandidateFailure.from_error(
                        candidate.name,
                        origin,
                        error,
                        stage="optimization",
                    ),
                )
                break
            for fold in origin_folds:
                try:
                    prediction = _prediction_rows(config, candidate, fold, optimized.parameters)
                except PyLGMError as error:
                    failures[candidate.name] = (
                        CandidateFailure.from_error(
                            candidate.name,
                            origin,
                            error,
                            stage="prediction",
                            horizon=fold.definition.horizon,
                        ),
                    )
                    break
                candidate_predictions.append(prediction)
            if candidate.name in failures:
                break
            warm_start = optimized.parameters
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
        result["engine"] = "persistence"
        result["is_benchmark"] = True
        predictions.append(result)
    return pd.concat(predictions, ignore_index=True)


def _data_fingerprint(frame: pd.DataFrame, config: ExperimentConfig) -> str:
    data = config.data
    fingerprint_config: DataConfig = data
    if data.vintage is not None:
        fingerprint_config = DataConfig(
            time=data.time,
            response=data.response,
            panel=(*data.panel, data.vintage),
        )
    return panel_fingerprint(CanonicalPanel.from_frame(frame, fingerprint_config))


@dataclass(frozen=True)
class Experiment:
    """A configured predictive candidate comparison."""

    config: ExperimentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        return cls(load_experiment_config(Path(path)))

    def compare(self, frame: pd.DataFrame, output: str | Path | None = None) -> ComparisonResult:
        definitions = build_fold_definitions(frame, self.config.data, self.config.evaluation)
        folds = tuple(
            materialize_fold(frame, self.config.data, self.config.evaluation, definition)
            for definition in definitions
        )
        candidates = resolve_candidates(self.config)
        preflight_failures = _preflight_candidates(self.config, candidates, folds)
        candidate_predictions, diagnostics, failures = _run_candidates(
            self.config,
            candidates,
            folds,
            preflight_failures,
        )
        predictions = pd.concat(
            [*candidate_predictions, _run_persistence(self.config, folds)],
            ignore_index=True,
        )
        scored = score_predictions(predictions, self.config.evaluation.interval_levels)
        metrics = aggregate_metrics(scored)
        decision = select_candidate(metrics, failures, self.config.evaluation)
        result = ComparisonResult(
            tuple(candidate.name for candidate in candidates),
            scored,
            metrics,
            diagnostics,
            failures,
            decision,
        )
        if output is not None:
            write_experiment(
                Path(output),
                self.config,
                candidates,
                definitions,
                result,
                _data_fingerprint(frame, self.config),
            )
        return result

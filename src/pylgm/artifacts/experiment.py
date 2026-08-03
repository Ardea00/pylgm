"""Versioned, transactional predictive-comparison artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pylgm.artifacts.run import write_transactionally
from pylgm.config import ExperimentConfig, ResolvedCandidate
from pylgm.evaluation.folds import FoldDefinition

if TYPE_CHECKING:
    from pylgm.experiment import ComparisonResult


_DIRECT_DEPENDENCIES = (
    "pylgm",
    "formulaic",
    "numpy",
    "pandas",
    "pydantic",
    "PyYAML",
    "scipy",
    "typer",
    "pyarrow",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact JSON values must be finite")
        return value
    if isinstance(value, np.datetime64):
        return {"type": "timestamp", "value": pd.Timestamp(value).isoformat()}
    if isinstance(value, np.timedelta64):
        return {"type": "timedelta_ns", "value": int(pd.Timedelta(value).value)}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, (pd.Timedelta, timedelta)):
        return {"type": "timedelta_ns", "value": int(pd.Timedelta(value).value)}
    if isinstance(value, pd.Period):
        return {
            "type": "period",
            "ordinal": int(value.ordinal),
            "frequency": value.freqstr,
        }
    if isinstance(value, pd.Interval):
        return {
            "type": "interval",
            "left": _json_scalar(value.left),
            "right": _json_scalar(value.right),
            "closed": value.closed,
        }
    if isinstance(value, Decimal):
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "bytes", "value_hex": bytes(value).hex()}
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, (np.generic, complex)):
            return _json_scalar(item)
        dtype = value.dtype.newbyteorder(">")
        encoded = np.asarray(value, dtype=value.dtype).astype(dtype, copy=False).tobytes().hex()
        return {"type": "numpy_scalar", "dtype": dtype.str, "value_hex": encoded}
    raise TypeError(f"unsupported artifact JSON value: {type(value).__qualname__}")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("artifact JSON mappings require string keys")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return _json_scalar(value)


def _environment() -> dict[str, object]:
    from pylgm import __version__

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: __version__ if name == "pylgm" else version(name) for name in _DIRECT_DEPENDENCIES
        },
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, engine="pyarrow", index=False)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(_json_ready(value)))


def _sha256(path: Path) -> str:
    with path.open("rb") as payload:
        return hashlib.file_digest(payload, "sha256").hexdigest()


def _fold_payload(folds: tuple[FoldDefinition, ...]) -> list[dict[str, object]]:
    return [
        {
            "origin": _json_ready(fold.origin),
            "target": _json_ready(fold.target),
            "horizon": fold.horizon,
        }
        for fold in folds
    ]


def write_experiment(
    output: Path,
    config: ExperimentConfig,
    candidates: tuple[ResolvedCandidate, ...],
    folds: tuple[FoldDefinition, ...],
    result: ComparisonResult,
    data_fingerprint: str,
) -> None:
    """Persist a complete version-2 comparison without replacing existing output."""
    resolved_config = config.model_dump(mode="json")
    resolved_candidates = [candidate.to_dict() for candidate in candidates]
    fold_definitions = _fold_payload(folds)
    decision = result.decision.to_dict()
    failures = {
        name: [failure.to_dict() for failure in candidate_failures]
        for name, candidate_failures in sorted(result.failures.items())
    }
    environment = _environment()
    experiment_fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "artifact_schema_version": 2,
                "data_fingerprint": data_fingerprint,
                "resolved_config": resolved_config,
                "resolved_candidates": resolved_candidates,
                "folds": fold_definitions,
            }
        )
    ).hexdigest()

    def writer(temporary: Path) -> None:
        json_payloads = {
            "decision.json": decision,
            "environment.json": environment,
            "failures.json": failures,
            "folds.json": fold_definitions,
            "resolved_candidates.json": resolved_candidates,
            "resolved_config.json": resolved_config,
        }
        for name, payload in json_payloads.items():
            _write_json(temporary / name, payload)
        tables = {
            "metrics.parquet": result.metrics,
            "optimization.parquet": result.diagnostics,
            "predictions.parquet": result.predictions,
        }
        for name, table in tables.items():
            _write_parquet(temporary / name, table)
        payload_sha256 = {
            path.name: _sha256(path)
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
        }
        summary_core = {
            "artifact_schema_version": 2,
            "engine": "exact_gaussian",
            "selected": result.selected,
            "candidates": list(result.candidates),
            "decision": decision,
            "resolved_candidates": resolved_candidates,
            "data_fingerprint": data_fingerprint,
            "experiment_fingerprint": experiment_fingerprint,
            "payload_sha256": payload_sha256,
        }
        artifact_fingerprint = hashlib.sha256(_canonical_json(summary_core)).hexdigest()
        summary = {**summary_core, "artifact_fingerprint": artifact_fingerprint}
        _write_json(temporary / "summary.json", summary)

    write_transactionally(output, writer)


__all__ = ["write_experiment"]

"""Persistence for local Gaussian model runs."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from pylgm.config import RunConfig
from pylgm.inference import GaussianResult


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _environment() -> dict[str, Any]:
    dependencies = ["pylgm", "numpy", "scipy", "pandas", "formulaic", "pydantic"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in dependencies},
    }


def _array_payload(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def write_run(
    output: Path,
    config: RunConfig,
    result: GaussianResult,
    data_fingerprint: str,
) -> None:
    """Write a complete run once; pre-existing outputs are never overwritten."""
    resolved_config = config.model_dump(mode="json")
    environment = _environment()
    artifact_fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "data_fingerprint": data_fingerprint,
                "resolved_config": resolved_config,
                "result": {
                    "labels": list(result.labels),
                    "log_marginal_likelihood": result.log_marginal_likelihood,
                    "mean": _array_payload(result.mean),
                    "covariance": _array_payload(result.covariance),
                    "predictive_mean": _array_payload(result.predictive_mean),
                    "predictive_variance": _array_payload(result.predictive_variance),
                },
                "environment": environment,
            }
        )
    ).hexdigest()
    summary = {
        "engine": "exact_gaussian",
        "conditional_on_fixed_hyperparameters": True,
        "data_fingerprint": data_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "log_marginal_likelihood": result.log_marginal_likelihood,
        "labels": list(result.labels),
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.json").write_bytes(_canonical_json(resolved_config))
    np.savez_compressed(output / "posterior.npz", mean=result.mean, covariance=result.covariance)
    (output / "summary.json").write_bytes(_canonical_json(summary))
    (output / "environment.json").write_bytes(_canonical_json(environment))

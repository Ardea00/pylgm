"""Persistence for local Gaussian model runs."""

import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from pylgm.config import RunConfig
from pylgm.inference import GaussianResult


_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4
_AT_FDCWD = -100
_RENAMEAT2_SYSCALLS = {"aarch64": 276, "x86_64": 316}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _environment() -> dict[str, Any]:
    dependencies = [
        "pylgm",
        "numpy",
        "scipy",
        "pandas",
        "formulaic",
        "pydantic",
        "PyYAML",
        "typer",
    ]
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


class _OutputLock(AbstractContextManager["_OutputLock"]):
    """Crash-released advisory lock shared by cooperating pyLGM writers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "_OutputLock":
        self.handle = self.path.open("a+b")
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self.handle.close()
                raise FileExistsError(f"run output is being created: {self.path}") from error
            return self
        if os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            if self.handle.read(1) == b"":
                self.handle.seek(0)
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                self.handle.close()
                raise FileExistsError(f"run output is being created: {self.path}") from error
            return self
        self.handle.close()
        raise OSError(errno.ENOTSUP, "advisory locks are unsupported on this platform")

    def __exit__(self, *args: object) -> None:
        assert self.handle is not None
        if os.name == "posix":
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        elif os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()


def _destination_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _raise_publish_error(destination: Path) -> None:
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(errno.EEXIST, "File exists", destination)
    raise OSError(error, os.strerror(error), destination)


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing any destination."""
    if _destination_exists(destination):
        raise FileExistsError(errno.EEXIST, "File exists", destination)
    if sys.platform == "darwin":
        rename = ctypes.CDLL(None, use_errno=True).renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL) != 0:
            _raise_publish_error(destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is not None:
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            status = rename(
                _AT_FDCWD,
                os.fsencode(source),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
        else:
            syscall_number = _RENAMEAT2_SYSCALLS.get(platform.machine().lower())
            if syscall_number is None:
                raise OSError(
                    errno.ENOTSUP,
                    "atomic no-replace renameat2 is unsupported on this Linux platform",
                )
            syscall = libc.syscall
            status = syscall(
                syscall_number,
                _AT_FDCWD,
                os.fsencode(source),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
        if status != 0:
            _raise_publish_error(destination)
        return
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        move_file.restype = ctypes.c_int
        if not move_file(str(source), str(destination), 0x8):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(errno.EEXIST, "File exists", destination)
            raise OSError(error, "MoveFileW failed", destination)
        return
    raise OSError(errno.ENOTSUP, "atomic no-replace publication is unsupported")


def write_transactionally(output: Path, writer: Callable[[Path], None]) -> None:
    """Write in a sibling temporary directory and atomically publish it once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    with _OutputLock(output.parent / f".{output.name}.lock"):
        try:
            if _destination_exists(output):
                raise FileExistsError(errno.EEXIST, "File exists", output)
            temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
            writer(temporary)
            _publish_no_replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)


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
                "artifact_schema_version": 2,
                "data_fingerprint": data_fingerprint,
                "resolved_config": resolved_config,
                "result": {
                    "labels": list(result.labels),
                    "log_marginal_likelihood": result.log_marginal_likelihood,
                    "mean": _array_payload(result.mean),
                    "covariance": _array_payload(result.covariance),
                    "predictive_mean": _array_payload(result.predictive_mean),
                    # v2: predictive_variance is now the linear-predictor
                    # variance; the observation variance is recorded separately
                    # so a persisted artifact still determines the response-scale
                    # variance.
                    "predictive_variance": _array_payload(result.predictive_variance),
                    "observation_variance": getattr(result, "observation_variance", None),
                },
                "environment": environment,
            }
        )
    ).hexdigest()
    summary = {
        "artifact_schema_version": 2,
        "engine": "exact_gaussian",
        "conditional_on_fixed_hyperparameters": True,
        "data_fingerprint": data_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "log_marginal_likelihood": result.log_marginal_likelihood,
        "labels": list(result.labels),
    }

    def writer(temporary: Path) -> None:
        (temporary / "resolved_config.json").write_bytes(_canonical_json(resolved_config))
        np.savez_compressed(
            temporary / "posterior.npz", mean=result.mean, covariance=result.covariance
        )
        (temporary / "summary.json").write_bytes(_canonical_json(summary))
        (temporary / "environment.json").write_bytes(_canonical_json(environment))

    write_transactionally(output, writer)

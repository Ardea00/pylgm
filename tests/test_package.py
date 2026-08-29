from importlib.metadata import entry_points
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

import pylgm
from pylgm.cli import app


def test_general_lgm_api_is_exported_without_removing_legacy_api() -> None:
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    expected = {
        "LGM",
        "Gaussian",
        "Fixed",
        "IID",
        "RW1",
        "RW2",
        "Hyperparameter",
        "GaussianPrior",
        "PCPrecision",
        "Pipeline",
        "Experiment",
        "ComparisonResult",
        "CandidateFailure",
        "FailureCause",
        "WeibullSurv",
        "ExponentialSurv",
    }

    assert expected.issubset(set(pylgm.__all__))
    assert pylgm.__version__ == "0.4.0"
    assert metadata["project"]["version"] == pylgm.__version__


def test_spark_extra_is_optional_and_declared() -> None:
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert metadata["project"]["optional-dependencies"]["spark"] == ["pyspark>=3.5"]
    assert "pyspark" not in metadata["project"]["dependencies"]


def test_spark_example_runs_and_reports_prediction_keys() -> None:
    pytest.importorskip("pyspark")
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/general_lgm/run_spark.py")],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )
    if completed.returncode != 0 and "JAVA_GATEWAY_EXITED" in completed.stderr:
        pytest.skip("Spark example needs a JVM (no Java installed)")
    assert completed.returncode == 0, completed.stderr
    assert "prediction_keys" in completed.stdout


def test_empirical_bayes_example_reports_estimates():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/empirical_bayes/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hyperparameters" in completed.stdout


def test_map_ii_example_reports_penalized_estimate():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/map_ii/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hyperparameters" in completed.stdout


def test_inla_example_reports_integrated_marginals():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/inla/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hyperparameter_marginals" in completed.stdout


def test_inla_sla_example_reports_skewness():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/inla_sla/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "skewness" in completed.stdout.lower()


def test_inla_full_laplace_example_reports_quantile():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/inla_full_laplace/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "quantile" in completed.stdout.lower()


def test_inla_criteria_example_reports_dic_waic():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/inla_criteria/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "waic" in completed.stdout.lower()


def test_hybrid_nowcast_example_reports_correlation():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/hybrid_nowcast/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "corr(pred,y)=" in completed.stdout


def test_installed_console_entrypoint_loads_and_reports_help() -> None:
    matches = tuple(entry_points(group="console_scripts", name="pylgm"))

    assert len(matches) == 1
    assert matches[0].load() is app
    executable = shutil.which("pylgm")
    assert executable is not None
    completed = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0
    assert "fit" in completed.stdout
    assert "compare" in completed.stdout

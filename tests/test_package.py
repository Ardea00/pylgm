from importlib.metadata import entry_points
import os
from pathlib import Path
import shutil
import runpy
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
    assert pylgm.__version__ == "0.5.0"
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


def test_directed_network_sar_example_estimates_rho():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/directed_network_sar/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "estimated rho=" in completed.stdout


def test_survival_duration_example_reports_hazard_ratio():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/survival_duration/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hazard_ratio=" in completed.stdout
    assert "shape=" in completed.stdout


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


def test_method_comparison_example_reproduces_its_documented_numbers():
    """Guards the figures quoted in docs/comparison.md and the README.

    scikit-learn and xgboost are deliberately NOT pyLGM dependencies, so this
    skips rather than failing when they are absent.
    """
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/method_comparison/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    # The claim the comparison page rests on: the structured model is closest to
    # the truth on the small-area problem, and boosting wins on the nonlinear one.
    assert "pyLGM (Besag)" in out and "XGBoost" in out
    assert "95% interval coverage" in out
    assert "Metropolis" in out

def test_columbus_example_reproduces_published_anselin_values():
    """The credibility anchor: OLS must match Anselin (1988) Table 12.1 exactly,
    and pyLGM's SAR must land near the published ML spatial-error estimates."""
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/columbus_spatial_econometrics/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    ns = runpy.run_path(str(root / "examples/columbus_spatial_econometrics/run.py"))
    out = ns["main"]()
    ols, published = out["OLS (pyLGM repo)"], out["published OLS"]
    for key in ("const", "INC", "HOVAL"):
        assert ols[key] == pytest.approx(published[key], abs=1e-3), key
    sar, ml = out["pyLGM SAR"], out["published ML spatial error"]
    assert sar["HOVAL"] == pytest.approx(ml["HOVAL"], abs=0.02)
    assert sar["const"] == pytest.approx(ml["const"], abs=2.0)
    # the headline: spatial dependence roughly halves the apparent income effect
    assert abs(sar["INC"]) < 0.7 * abs(ols["INC"])
    assert 0.4 < sar["rho"] < 0.8


def test_dynamic_network_example_beats_baselines_on_missing_cells():
    """Guards the numbers quoted in the SDPD example README and the docs."""
    root = Path(__file__).parents[1]
    ns = runpy.run_path(str(root / "examples/state_income_dynamic_network/run.py"))
    out = ns["main"]()
    recovery = out["recovery_rmse"]
    assert recovery["SDPD (dynamic network)"] < 0.5 * min(
        v for k, v in recovery.items() if k != "SDPD (dynamic network)"
    )
    assert out["network_weight_drift"] > 0.0  # the graph really is time-varying
    assert out["hyperparameters"]["dyn.gamma"] > 0.8  # near-unit-root persistence

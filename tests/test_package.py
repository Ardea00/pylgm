from importlib.metadata import entry_points
from pathlib import Path
import shutil
import subprocess
import tomllib

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
    }

    assert expected.issubset(set(pylgm.__all__))
    assert pylgm.__version__ == "0.3.0"
    assert metadata["project"]["version"] == pylgm.__version__


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

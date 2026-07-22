from importlib.metadata import entry_points
import shutil
import subprocess

import pylgm
from pylgm.cli import app


def test_package_exports_version() -> None:
    assert pylgm.__version__ == "0.1.0"


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

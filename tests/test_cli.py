from pathlib import Path

from typer.testing import CliRunner

from pylgm.cli import app


def test_cli_fits_csv(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.csv"
    output = tmp_path / "run"
    config.write_text(
        "schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n"
    )
    data.write_text("month,y\n1,1.0\n2,2.0\n")

    response = CliRunner().invoke(app, ["fit", str(config), str(data), "--output", str(output)])

    assert response.exit_code == 0
    assert "exact_gaussian" in response.stdout


def test_cli_reports_existing_output_failure(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.csv"
    output = tmp_path / "run"
    config.write_text(
        "schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n"
    )
    data.write_text("month,y\n1,1.0\n2,2.0\n")
    output.mkdir()

    response = CliRunner().invoke(app, ["fit", str(config), str(data), "--output", str(output)])

    assert response.exit_code != 0
    assert "File exists" in response.stdout or "File exists" in str(response.exception)

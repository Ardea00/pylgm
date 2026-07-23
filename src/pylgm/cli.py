"""Command-line interface for local pyLGM fits."""

from pathlib import Path

import pandas as pd
import typer

from pylgm.experiment import Experiment
from pylgm.pipeline import Pipeline

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run local pyLGM commands."""


@app.command()
def fit(
    config: Path,
    data: Path,
    output: Path = typer.Option(...),
) -> None:
    """Fit a configured model to a CSV file and persist a run artifact."""
    result = Pipeline.from_yaml(config).run(pd.read_csv(data), output)
    typer.echo(
        f"engine=exact_gaussian log_marginal_likelihood={result.log_marginal_likelihood:.6f}"
    )


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise typer.BadParameter("data must be CSV or Parquet")


@app.command()
def compare(
    config: Path,
    data: Path,
    output: Path = typer.Option(...),
) -> None:
    """Compare configured candidates using a local CSV or Parquet frame."""
    result = Experiment.from_yaml(config).compare(_read_frame(data), output)
    typer.echo(f"selected={result.selected} candidates={len(result.candidates)}")

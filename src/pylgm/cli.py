"""Command-line interface for local pyLGM fits."""

from pathlib import Path

import pandas as pd
import typer

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

"""Command-line interface for local pyLGM fits."""

import argparse
from pathlib import Path

import pandas as pd

from pylgm.experiment import Experiment
from pylgm.pipeline import Pipeline


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("data must be CSV or Parquet")


def fit(config: Path, data: Path, output: Path) -> None:
    """Fit a configured model to a CSV file and persist a run artifact."""
    result = Pipeline.from_yaml(config).run(pd.read_csv(data), output)
    print(f"engine=exact_gaussian log_marginal_likelihood={result.log_marginal_likelihood:.6f}")


def compare(config: Path, data: Path, output: Path) -> None:
    """Compare configured candidates using a local CSV or Parquet frame."""
    result = Experiment.from_yaml(config).compare(_read_frame(data), output)
    print(f"selected={result.selected} candidates={len(result.candidates)}")


def app(argv: list[str] | None = None) -> None:
    """Run local pyLGM commands."""
    parser = argparse.ArgumentParser(prog="pylgm", description=app.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (fit, compare):
        sub = commands.add_parser(command.__name__, help=command.__doc__)
        sub.add_argument("config", type=Path)
        sub.add_argument("data", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        sub.set_defaults(run=command)
    args = parser.parse_args(argv)
    args.run(args.config, args.data, args.output)

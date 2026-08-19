"""Fit a Poisson count model with an offset through the YAML frontend and Laplace engine."""

from pathlib import Path

import pandas as pd

from pylgm.config import load_model


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(EXAMPLE_DIRECTORY / "data.csv")
    model = load_model(EXAMPLE_DIRECTORY / "config.yaml")
    result = model.fit(frame, engine="laplace")
    print(f"fitted_mean: {result.fitted_mean.round(4).tolist()}")
    print(f"newton_iterations: {result.diagnostics['newton_iterations']}")


if __name__ == "__main__":
    main()

"""Fit a Gaussian region panel with a PC-prior-penalized (MAP-II) precision."""

from pathlib import Path

import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM, PCPrecision


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(EXAMPLE_DIRECTORY / "data.csv")
    model = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed("1")
        + IID(
            "region",
            index="region",
            precision=Hyperparameter(
                "region_precision",
                initial=1.0,
                prior=PCPrecision(upper_sd=1.0, alpha=0.01),
            ),
        ),
        panel=("region",),
        time="time",
    )
    result = model.fit(frame, engine="exact_gaussian")
    print(f"hyperparameters: {result.hyperparameters}")
    print(f"hyperparameter_penalized: {result.diagnostics['hyperparameter_penalized']}")


if __name__ == "__main__":
    main()

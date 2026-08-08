"""Fit the general LGM example through the Python and YAML frontends."""

from pathlib import Path

import pandas as pd

from pylgm import Fixed, Gaussian, IID, LGM, RW1
from pylgm.config import load_model


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def _python_model() -> LGM:
    return LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=(
            Fixed("1 + x")
            + IID("region_effect", index="region", precision=2.0)
            + RW1("trend", index="time", precision=3.0)
        ),
        panel=("region",),
        time="time",
    )


def main() -> None:
    frame = pd.read_csv(EXAMPLE_DIRECTORY / "data.csv")
    models = {
        "python": _python_model(),
        "yaml": load_model(EXAMPLE_DIRECTORY / "config.yaml"),
    }
    for frontend, model in models.items():
        result = model.fit(frame)
        print(f"{frontend}: {result.predictive_mean.round(4).tolist()}")


if __name__ == "__main__":
    main()

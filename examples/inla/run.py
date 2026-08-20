"""Integrate over a declared precision hyperparameter with INLA-style grid quadrature."""

from pathlib import Path

import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, IID, LGM


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
            precision=Hyperparameter("region_precision", initial=1.0),
        ),
        panel=("region",),
        time="time",
    )
    result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")

    marginals = result.hyperparameter_marginals()
    region_precision = marginals["region_precision"]
    print(f"hyperparameter_marginals: region_precision "
          f"mean={region_precision.mean[0]:.4f} sd={region_precision.std[0]:.4f}")

    region = result.latent_marginals("region")
    print(f"latent_marginals(region): mean min={region.mean.min():.4f} "
          f"max={region.mean.max():.4f}, sd mean={region.std.mean():.4f}")

    print(f"inla_grid_points: {result.diagnostics['inla_grid_points']}")
    print(f"inla_active_bounds: {result.diagnostics['inla_active_bounds']!r}")


if __name__ == "__main__":
    main()

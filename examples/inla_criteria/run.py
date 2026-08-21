"""Report DIC/WAIC/CPO/PIT model-assessment criteria for an integrated INLA fit."""

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

    criteria = result.criteria
    print(f"DIC: {criteria.dic:.4f} (effective parameters: "
          f"{criteria.dic_effective_parameters:.4f})")
    print(f"WAIC: {criteria.waic:.4f} (effective parameters: "
          f"{criteria.waic_effective_parameters:.4f})")
    print(f"cpo_failures: {criteria.cpo_failures}")
    print(f"log_cpo_sum: {criteria.log_cpo_sum:.4f}")
    print(f"pit[:5]: {criteria.pit[:5]}")


if __name__ == "__main__":
    main()

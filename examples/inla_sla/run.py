"""Skew-aware latent marginals via INLA's simplified-Laplace strategy."""

from pathlib import Path

import pandas as pd

from pylgm import Fixed, Hyperparameter, IID, LGM, Poisson


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(EXAMPLE_DIRECTORY / "data.csv")
    model = LGM(
        response="y",
        likelihood=Poisson(),
        predictor=Fixed("1")
        + IID(
            "region",
            index="region",
            precision=Hyperparameter("region_precision", initial=1.0),
        ),
        panel=("region",),
        time="time",
    )

    sla = model.fit(
        frame, engine="laplace", hyperparameters="integrate",
        latent_strategy="simplified_laplace",
    )
    region_sla = sla.latent_marginals("region")
    print(f"sla mean[:3]: {region_sla.mean[:3].round(4).tolist()}")
    print(f"sla std[:3]: {region_sla.std[:3].round(4).tolist()}")
    print(f"sla skewness[:3]: {region_sla.skewness[:3].round(4).tolist()}")
    sla_lo = region_sla.quantile(0.025)
    sla_hi = region_sla.quantile(0.975)
    print(f"sla 95% interval[:3]: lo={sla_lo[:3].round(4).tolist()} "
          f"hi={sla_hi[:3].round(4).tolist()}")

    gauss = model.fit(frame, engine="laplace", hyperparameters="integrate")
    region_gauss = gauss.latent_marginals("region")
    gauss_lo = region_gauss.mean - 1.959964 * region_gauss.std
    gauss_hi = region_gauss.mean + 1.959964 * region_gauss.std
    print(f"gaussian 95% interval[:3]: lo={gauss_lo[:3].round(4).tolist()} "
          f"hi={gauss_hi[:3].round(4).tolist()}")
    print("gaussian interval is symmetric about the mean; the SLA interval, "
          "skewed by skewness[:3] above, is not.")


if __name__ == "__main__":
    main()

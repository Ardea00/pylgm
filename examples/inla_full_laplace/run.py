"""Full-Laplace latent marginals via INLA's most accurate latent strategy."""

from pathlib import Path

import pandas as pd

from pylgm import Fixed, Hyperparameter, IID, LGM, Poisson


EXAMPLE_DIRECTORY = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(EXAMPLE_DIRECTORY / "data.csv")
    # Fixed + IID only (no RW): full Laplace rejects constrained effects.
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

    full = model.fit(
        frame, engine="laplace", hyperparameters="integrate",
        latent_strategy="laplace",
    )
    m = full.latent_marginals("region")
    print(f"full-laplace mean[:3]: {m.mean[:3].round(4).tolist()}")
    print(f"full-laplace std[:3]: {m.std[:3].round(4).tolist()}")
    print(f"full-laplace skewness[:3]: {m.skewness[:3].round(4).tolist()}")
    full_lo = m.quantile(0.025)
    full_hi = m.quantile(0.975)
    print(f"full-laplace quantile(0.025)[:3]: {full_lo[:3].round(4).tolist()}")
    print(f"full-laplace quantile(0.975)[:3]: {full_hi[:3].round(4).tolist()}")

    gauss = model.fit(frame, engine="laplace", hyperparameters="integrate")
    g = gauss.latent_marginals("region")
    gauss_lo = g.mean - 1.959964 * g.std
    gauss_hi = g.mean + 1.959964 * g.std

    sla = model.fit(
        frame, engine="laplace", hyperparameters="integrate",
        latent_strategy="simplified_laplace",
    )
    s = sla.latent_marginals("region")
    sla_lo = s.quantile(0.025)
    sla_hi = s.quantile(0.975)

    print("\nstrategy       skewness[:3]                  95% interval[:3]")
    print(f"gaussian       {[0.0, 0.0, 0.0]}   lo={gauss_lo[:3].round(4).tolist()} "
          f"hi={gauss_hi[:3].round(4).tolist()}")
    print(f"simplified-LA  {s.skewness[:3].round(4).tolist()}   "
          f"lo={sla_lo[:3].round(4).tolist()} hi={sla_hi[:3].round(4).tolist()}")
    print(f"full-laplace   {m.skewness[:3].round(4).tolist()}   "
          f"lo={full_lo[:3].round(4).tolist()} hi={full_hi[:3].round(4).tolist()}")
    print("\nThe gaussian interval is symmetric about its mean by construction; "
          "the simplified- and full-Laplace intervals are skewed, with full "
          "Laplace applying the additional denominator correction the "
          "simplified strategy omits (RMC 2009 section 3.2.2/3.2.3).")


if __name__ == "__main__":
    main()

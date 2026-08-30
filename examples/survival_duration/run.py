"""Unemployment-duration example: Weibull proportional-hazards survival with
right-censoring, left-truncation, an estimated shape, and unobserved
heterogeneity (frailty) across individuals via an IID effect.

Run from the repo root:
    PYTHONPATH=src python examples/survival_duration/run.py
"""
import numpy as np
import pandas as pd

from pylgm import Fixed, Hyperparameter, IID, LGM, WeibullSurv

N = 400           # kept small: an IID frailty term adds one latent dim per individual,
                  # and the dense reference engine caps total latent dimension
SEED = 0
ALPHA_TRUE = 1.4      # duration dependence: hazard rises with elapsed time (alpha > 1)
BETA0 = -0.2
BETA_TREATED = 0.5    # treated (e.g. job-search assistance) raises the hazard -> shorter spells
FRAILTY_SD = 0.4      # unobserved heterogeneity across individuals
CENSOR_AT = 1.4       # administrative end of the observation window


def build() -> pd.DataFrame:
    """Simulate individual unemployment spells with censoring and delayed entry."""
    rng = np.random.default_rng(SEED)
    treated = rng.integers(0, 2, size=N).astype(float)
    frailty = rng.normal(scale=FRAILTY_SD, size=N)
    eta = BETA0 + BETA_TREATED * treated + frailty
    scale = np.exp(-eta / ALPHA_TRUE)                  # S(t) = exp(-(t/scale)^alpha)
    duration = rng.weibull(ALPHA_TRUE, size=N) * scale

    observed = np.minimum(duration, CENSOR_AT)         # administrative right-censoring
    event = (duration <= CENSOR_AT).astype(float)

    # A third of spells are left-truncated: already running before the window opened.
    left_truncated = rng.random(N) < 0.3
    entry = np.where(left_truncated, rng.uniform(0.0, 0.4, size=N) * observed, 0.0)

    return pd.DataFrame({
        "id": np.arange(N).astype(str),
        "spell_time": observed,
        "event": event,
        "entry": entry,
        "treated": treated,
    })


def main() -> dict:
    frame = build()
    model = LGM(
        response="spell_time",
        likelihood=WeibullSurv(
            "event",
            shape=Hyperparameter("alpha", initial=1.0, transform="log"),
            entry="entry",
        ),
        # frailty precision fixed (not estimated) to keep the empirical-Bayes
        # search one-dimensional (shape alone) and the example fast
        predictor=Fixed("1 + treated") + IID("frailty", index="id", precision=2.0),
    )
    result = model.fit(frame, engine="laplace", hyperparameters="optimize")

    idx = {label: i for i, label in enumerate(result.labels)}
    beta_treated = float(result.mean[idx["fixed:treated"]])
    hazard_ratio = float(np.exp(beta_treated))
    shape = float(result.hyperparameters["alpha"])

    fitted_mean_duration = float(np.mean(result.fitted_mean))
    observed_mean_duration = float(frame["spell_time"].mean())

    print(f"n={len(frame)} spells, censored={(frame['event'] == 0).mean():.1%}, "
          f"left_truncated={(frame['entry'] > 0).mean():.1%}")
    print(f"hazard_ratio={hazard_ratio:.3f} (planted={np.exp(BETA_TREATED):.3f})")
    print(f"shape={shape:.3f} (planted={ALPHA_TRUE:.3f})")
    print(f"fitted_mean_duration={fitted_mean_duration:.3f} "
          f"observed_mean_duration={observed_mean_duration:.3f} "
          "(observed is the raw, censoring-truncated average; fitted is the "
          "model-implied unconditional E[T])")

    return {
        "hazard_ratio": hazard_ratio,
        "shape": shape,
        "fitted_mean_duration": fitted_mean_duration,
        "observed_mean_duration": observed_mean_duration,
    }


if __name__ == "__main__":
    main()

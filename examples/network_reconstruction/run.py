"""Mask-and-reconstruct a censored bank-firm credit network: pyLGM vs RAS and dcGM."""

import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm

from pylgm import (IID, LGM, Bernoulli, CensoredHurdle, Fixed, Gaussian, Hyperparameter, Joint,
                   LinearObservation)
from pylgm.evaluation.network import reconstruction_scores

FIRMS, BANKS, NOISE = 1500, 15, 0.02


def simulate(rng):
    firm, bank = np.repeat(np.arange(FIRMS), BANKS), np.tile(np.arange(BANKS), FIRMS)
    fe, be = rng.normal(0, 1.0, FIRMS), rng.normal(0, 0.7, BANKS)
    exists = rng.random(firm.size) < expit(-0.5 + fe[firm] + be[bank])
    amount = (10 + 0.5 * rng.normal(size=FIRMS)[firm] + 0.5 * rng.normal(size=BANKS)[bank]
              + rng.normal(size=firm.size))
    log_c = np.quantile(amount[exists], 0.35)  # ~35% of existing links fall below c
    reported = exists & (amount >= log_c)
    known_absent = ~exists & (rng.random(firm.size) < 0.3)  # stands in for link-intercept information
    censored = ~reported & ~known_absent
    frame = pd.DataFrame({
        "firm": firm, "bank": bank,
        "linked": np.where(reported, 1.0, np.where(known_absent, 0.0, np.nan)),
        "log_amount": np.where(reported, amount, np.nan), "unreported": censored,
    })
    below = exists & censored  # existing edges below c
    return frame, log_c, np.where(below, np.exp(amount), 0.0)[censored]


def fit_pylgm(frame, log_c, margins):
    def effects(tag):
        return (Fixed("1") + IID(f"firm_{tag}", index="firm", precision=1.0)
                + IID(f"bank_{tag}", index="bank", precision=1.0))

    sigma = Hyperparameter("sigma", initial=1.0, lower=0.1, upper=5.0)
    link = LGM(response="linked", likelihood=Bernoulli(), panel=("firm", "bank"), predictor=effects("l"))
    amount = LGM(response="log_amount", likelihood=Gaussian(sigma), panel=("firm", "bank"),
                 predictor=effects("a"))
    joint = Joint([link, amount], censoring=CensoredHurdle(
        link="linked", amount="log_amount", censored="unreported", threshold=log_c))
    cols = np.flatnonzero(frame["unreported"].to_numpy())
    operator = np.zeros((FIRMS, len(frame)))
    operator[frame["firm"].to_numpy()[cols], cols] = 1.0
    keep = margins > 0
    obs = LinearObservation(margins[keep], operator[keep], NOISE * margins[keep], scale="below_threshold")
    result = joint.fit(frame, observations={"log_amount": [obs]})
    cen = frame[frame["unreported"]]
    p = expit(result.predict(cen, outcome="linked").predictive_mean)
    b = result.predict(cen, outcome="log_amount").predictive_mean
    s = result.hyperparameters["sigma"]
    # P(link | absent) = p Phi((log c - b)/s) / (1 - p Phi((b - log c)/s))
    prob = p * norm.cdf((log_c - b) / s) / (1 - p * norm.cdf((b - log_c) / s))
    # E[W 1{link} | absent] = p M(b) / (1 - p S(b)), the below_threshold map
    mass = np.exp(b + s**2 / 2) * norm.cdf((log_c - b - s**2) / s)
    return prob, p * mass / (1 - p * norm.cdf((b - log_c) / s))


def baselines(frame, margins, n_links):
    cen = frame[frame["unreported"]]
    f, b = cen["firm"].to_numpy(), cen["bank"].to_numpy()
    reported = frame[frame["linked"] == 1.0]
    lending = np.bincount(reported["bank"], weights=np.exp(reported["log_amount"]), minlength=BANKS)
    # (b) maximum entropy / proportional: spread each firm's margin over its censored edges by bank size.
    share = lending[b] / np.bincount(f, weights=lending[b], minlength=FIRMS)[f]
    ras = margins[f] * share
    # (c) dcGM: x_i = firm margin, y_j = bank mass implied by (b); z fixed by the link count.
    x, y = margins[f], np.bincount(b, weights=ras, minlength=BANKS)[b]
    z = brentq(lambda z: (z * x * y / (1 + z * x * y)).sum() - n_links, 1e-30, 1e30)
    p = np.clip(z * x * y / (1 + z * x * y), 1e-12, 1.0)
    return (share, ras), (p, x * y / (margins.sum() * p))


def main() -> None:
    frame, log_c, truth = simulate(np.random.default_rng(20240607))
    firm = frame.loc[frame["unreported"], "firm"].to_numpy()
    margins = np.bincount(firm, weights=truth, minlength=FIRMS)
    margins = margins * np.exp(NOISE * np.random.default_rng(1).normal(size=FIRMS))  # ~2% noise
    t0 = time.perf_counter()
    prob, weight = fit_pylgm(frame, log_c, margins)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    (ras_score, ras_w), (dc_p, dc_w) = baselines(frame, margins, int((truth > 0).sum()))
    base_seconds = time.perf_counter() - t0
    rows = {"pylgm": (prob, weight, fit_seconds), "ras": (ras_score, ras_w, base_seconds),
            "dcgm (oracle density)": (dc_p, dc_w, base_seconds)}
    print(pd.DataFrame({
        name: {**reconstruction_scores(truth, score, w), "fit_seconds": seconds}
        for name, (score, w, seconds) in rows.items()
    }).T.round(3).to_string())
    print(f"{int((truth > 0).sum())} true below-threshold links among {len(truth)} censored "
          "edges; scores are on censored edges only.")


if __name__ == "__main__":
    main()

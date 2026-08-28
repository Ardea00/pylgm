"""Hybrid nowcast: one latent field composed from a MIDAS lag term, a BYM2
spatial term, and an AR1 temporal term, fit two ways (U-MIDAS and parametric).

Run from the repo root:
    PYTHONPATH=src python examples/hybrid_nowcast/run.py
"""
import numpy as np
import pandas as pd

from pylgm import AR1, BYM2, Fixed, Gaussian, Hyperparameter, LGM, MIDAS, MIDASParametric
from pylgm.effects.midas import midas_weights

R, T, K = 8, 40, 12
SEED = 0


def build():
    """Synthetic regional-GDP panel keyed by (region, quarter).

    y = mu + sum_k w_k * x_lag_k + s[region] + a[quarter] + noise, where w_k is a
    known exp-Almon decay, s is smooth over the ring graph, a is AR1 over quarters.
    """
    rng = np.random.default_rng(SEED)
    regions = [str(i) for i in range(R)]
    graph = {str(i): [str((i - 1) % R), str((i + 1) % R)] for i in range(R)}  # ring
    w_true = midas_weights("exp_almon", K, (0.2, -0.05))
    s = 1.2 * np.sin(2 * np.pi * np.arange(R) / R)  # smooth spatial signal
    a = np.zeros(T)  # AR1 temporal signal
    for t in range(1, T):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    cols = tuple(f"ind_lag{k}" for k in range(K))
    rows = []
    for ri, reg in enumerate(regions):
        monthly = rng.normal(size=T + K)
        lagged = {name: pd.Series(monthly).shift(k) for k, name in enumerate(cols)}
        rdf = pd.DataFrame(lagged).iloc[K:].reset_index(drop=True)  # drop NaN warmup
        rdf = rdf.iloc[:T].reset_index(drop=True)
        hf = rdf[list(cols)].to_numpy() @ w_true
        rdf["region"] = reg
        rdf["quarter"] = np.arange(len(rdf))
        rdf["y"] = 0.5 + 3.0 * hf + s[ri] + a[: len(rdf)] + rng.normal(scale=0.3, size=len(rdf))
        rows.append(rdf)
    return pd.concat(rows, ignore_index=True), cols, graph, w_true


def main():
    frame, cols, graph, w_true = build()
    print(f"panel: {len(frame)} rows, {R} regions x {T} quarters, {K} lags")

    # ponytail: phi/rho fixed as floats -> deterministic, fast example. Both accept
    # Hyperparameter(...) to estimate them (weakly identified on short panels).
    common = (
        BYM2("region", index="region", graph=graph,
             precision=Hyperparameter("region.precision", initial=1.0), phi=0.5)
        + AR1("quarter", index="quarter",
              precision=Hyperparameter("quarter.precision", initial=1.0), rho=0.6)
    )

    # Fit A: U-MIDAS (smoothed coefficient per lag)
    mA = LGM(response="y", panel=("region",), time="quarter",
             predictor=Fixed("1")
             + MIDAS("lag", columns=cols, precision=Hyperparameter("lag.precision", initial=1.0))
             + common,
             likelihood=Gaussian(sigma=0.3))
    rA = mA.fit(frame)
    predA = rA.predict(frame).predictive_mean
    kernelA = rA.latent_marginals("lag").mean
    print(f"[A U-MIDAS] corr(pred,y)={np.corrcoef(predA, frame['y'])[0, 1]:.3f} "
          f"corr(kernel,true)={np.corrcoef(kernelA, w_true)[0, 1]:.3f} "
          f"hyper={ {k: round(v, 3) for k, v in rA.hyperparameters.items()} }")

    # Fit B: parametric MIDAS (two shape parameters)
    mB = LGM(response="y", panel=("region",), time="quarter",
             predictor=Fixed("1")
             + MIDASParametric("m", cols, kernel="exp_almon")
             + common,
             likelihood=Gaussian(sigma=0.3))
    rB = mB.fit(frame)
    predB = rB.predict(frame).predictive_mean
    theta_hat = (rB.hyperparameters["m.shape1"], rB.hyperparameters["m.shape2"])
    kernelB = midas_weights("exp_almon", K, theta_hat)
    print(f"[B param ] corr(pred,y)={np.corrcoef(predB, frame['y'])[0, 1]:.3f} "
          f"corr(kernel,true)={np.corrcoef(kernelB, w_true)[0, 1]:.3f} "
          f"theta_hat=({theta_hat[0]:.3f}, {theta_hat[1]:.3f})")


if __name__ == "__main__":
    main()

"""Horseshoe-crab count regression: pyLGM vs. a standard Poisson GLM.

Genuine data: Agresti/Brockmann horseshoe crabs (173 females). Each row has a
satellite count ``sat`` and carapace ``width``. Satellite counts are famously
*over-dispersed* relative to Poisson (variance far exceeds the mean), so a plain
Poisson GLM understates coefficient uncertainty.

This script fits, and compares against ``statsmodels`` (the standard Python
regression library):

  * plain Poisson GLM  ``sat ~ width``           (matches statsmodels exactly),
  * Poisson + IID overdispersion (estimated)     (honest, wider CI),
  * out-of-sample prediction on a held-out split (predictive ability).

It prints a comparison table and writes two figures into ``docs/img/``.

Run:  PYTHONPATH=src python examples/count_regression/run.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pylgm import LGM, Fixed, Hyperparameter, IID, PCPrecision, Poisson

HERE = Path(__file__).resolve().parent
IMG = HERE.parents[1] / "docs" / "img"


def load() -> pd.DataFrame:
    frame = pd.read_csv(HERE / "data.csv")
    frame["crab"] = frame["crab"].astype(str)  # IID index key
    frame["width_c"] = frame["width"] - frame["width"].mean()
    return frame


def pylgm_slope(result):
    fixed = result.latent_marginals("fixed")
    return float(fixed.mean[1]), float(fixed.std[1])  # (mean, sd) of the width coeff


def main() -> None:
    frame = load()
    print(f"{len(frame)} crabs; sat mean={frame['sat'].mean():.2f} "
          f"var={frame['sat'].var():.2f} (var >> mean => over-dispersed)")

    y = frame["sat"].to_numpy(float)
    X = sm.add_constant(frame[["width_c"]].to_numpy(float))

    # --- standard regression baseline: statsmodels Poisson GLM ---------------
    sm_pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    sm_slope, sm_se = sm_pois.params[1], sm_pois.bse[1]

    # --- pyLGM plain Poisson (should match statsmodels) ----------------------
    r_pois = LGM(response="sat", likelihood=Poisson(),
                 predictor=Fixed("1 + width_c")).fit(frame, engine="laplace")
    lg_slope, lg_se = pylgm_slope(r_pois)

    # --- pyLGM Poisson + estimated IID overdispersion ------------------------
    r_od = LGM(
        response="sat", likelihood=Poisson(),
        predictor=Fixed("1 + width_c") + IID(
            "od", index="crab",
            precision=Hyperparameter("od_prec", initial=1.0,
                                     prior=PCPrecision(upper_sd=2.0, alpha=0.01))),
    ).fit(frame, engine="laplace")
    od_slope, od_se = pylgm_slope(r_od)
    od_prec = r_od.hyperparameters["od_prec"]

    print("\nwidth coefficient (log-count per cm):")
    print(f"  statsmodels Poisson GLM      {sm_slope:+.3f}  (se {sm_se:.3f})")
    print(f"  pyLGM Poisson                {lg_slope:+.3f}  (se {lg_se:.3f})   <- matches")
    print(f"  pyLGM Poisson + overdisp.    {od_slope:+.3f}  (se {od_se:.3f})   "
          f"<- honest wider CI (od_prec={od_prec:.2f})")

    # --- predictive ability: hold out the last 23 crabs ----------------------
    train, test = frame.iloc[:150], frame.iloc[150:]
    Xtr = sm.add_constant(train[["width_c"]].to_numpy(float))
    Xte = sm.add_constant(test[["width_c"]].to_numpy(float))
    sm_tr = sm.GLM(train["sat"].to_numpy(float), Xtr,
                   family=sm.families.Poisson()).fit()
    sm_pred = sm_tr.predict(Xte)
    lg_tr = LGM(response="sat", likelihood=Poisson(),
                predictor=Fixed("1 + width_c")).fit(train, engine="laplace")
    lg_pred = np.asarray(lg_tr.predict(test).fitted_mean)
    obs = test["sat"].to_numpy(float)

    def rmse(p):
        return float(np.sqrt(np.mean((p - obs) ** 2)))

    print(f"\nout-of-sample (fit 150, predict 23) RMSE:  "
          f"statsmodels {rmse(sm_pred):.3f}   pyLGM {rmse(lg_pred):.3f}")

    _figures(frame, sm_pois, r_pois, obs, sm_pred, lg_pred,
             (sm_slope, sm_se), (lg_slope, lg_se), (od_slope, od_se))
    print(f"\nwrote figures to {IMG}")


def _figures(frame, sm_pois, r_pois, obs, sm_pred, lg_pred,
             sm_c, lg_c, od_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    blue, orange = "#2b6cb0", "#dd6b20"

    # Figure 1: mean structure -- data + fitted E[sat | width] curves ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    order = np.argsort(frame["width"].to_numpy())
    w = frame["width"].to_numpy()[order]
    ax.scatter(frame["width"], frame["sat"], s=18, alpha=0.35,
               color="#4a5568", label="observed crabs")
    ax.plot(w, sm_pois.predict(sm.add_constant(frame[["width_c"]].to_numpy(float)))[order],
            color=orange, lw=2.2, label="statsmodels Poisson GLM")
    ax.plot(w, np.asarray(r_pois.fitted_mean)[order],
            color=blue, lw=2.2, ls="--", label="pyLGM Poisson")
    ax.set_xlabel("carapace width (cm)")
    ax.set_ylabel("number of satellites")
    ax.set_title("Mean structure: pyLGM matches the standard GLM")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "crabs_fit.png", dpi=120)
    plt.close(fig)

    # Figure 2: out-of-sample predicted vs observed + coefficient CIs ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    hi = max(obs.max(), sm_pred.max(), lg_pred.max()) * 1.05
    a1.plot([0, hi], [0, hi], color="#a0aec0", lw=1, ls=":")
    a1.scatter(obs, sm_pred, s=30, color=orange, label="statsmodels")
    a1.scatter(obs, lg_pred, s=30, color=blue, marker="x", label="pyLGM")
    a1.set_xlabel("observed count (held-out)")
    a1.set_ylabel("predicted mean count")
    a1.set_title("Out-of-sample prediction")
    a1.legend(frameon=False)

    labels = ["statsmodels\nPoisson", "pyLGM\nPoisson", "pyLGM +\noverdispersion"]
    means = [sm_c[0], lg_c[0], od_c[0]]
    ses = [sm_c[1], lg_c[1], od_c[1]]
    xs = np.arange(3)
    a2.errorbar(xs, means, yerr=[1.96 * s for s in ses], fmt="o",
                color=blue, capsize=5, lw=2, ms=7)
    a2.set_xticks(xs)
    a2.set_xticklabels(labels)
    a2.set_ylabel("width coefficient  (95% CI)")
    a2.set_title("Overdispersion widens the CI honestly")
    a2.axhline(0, color="#a0aec0", lw=1, ls=":")
    fig.tight_layout()
    fig.savefig(IMG / "crabs_prediction.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

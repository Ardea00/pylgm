"""Scotland lip-cancer disease mapping: pyLGM spatial model vs. a standard GLM.

Genuine data: the canonical GeoBUGS Scotland lip-cancer example, 56 districts.
For each district we have observed cases ``observed``, the expected count
``expected`` (age-standardised), and ``aff`` = proportion employed in
agriculture/fishing/forestry (an outdoor-exposure proxy). The model is

    observed_i ~ Poisson(mu_i),
    log(mu_i)  = log(expected_i) + b0 + b1 * aff_i + s_i,

where ``s`` is a Besag (ICAR) spatial effect over the district adjacency graph:
neighbouring districts borrow strength, smoothing noisy raw rates.

This script fits, and compares against a standard non-spatial Poisson GLM
(``statsmodels``) with the same offset and covariate:

  * the spatial random effect shrinks extreme raw SMRs toward local means,
  * it tracks observed counts far better than the covariate-only GLM,
  * the ``aff`` effect stays positive under both.

It prints a comparison and writes two figures into ``docs/img/``.

Run:  PYTHONPATH=src python examples/disease_mapping/run.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pylgm import (LGM, Besag, Fixed, Hyperparameter, PCPrecision, Poisson)

HERE = Path(__file__).resolve().parent
IMG = HERE.parents[1] / "docs" / "img"


def load():
    frame = pd.read_csv(HERE / "data.csv")
    graph = json.loads((HERE / "graph.json").read_text(encoding="utf-8"))
    frame["area"] = frame["area"].astype(str)
    frame["log_E"] = np.log(frame["expected"])
    frame["smr"] = frame["observed"] / frame["expected"]
    return frame, graph


def main() -> None:
    frame, graph = load()
    expected = frame["expected"].to_numpy(float)
    obs = frame["observed"].to_numpy(float)
    print(f"{len(frame)} Scottish districts; raw SMR spread "
          f"{frame['smr'].min():.2f}..{frame['smr'].max():.2f} (sd {frame['smr'].std():.2f})")

    # --- standard regression baseline: non-spatial Poisson GLM ---------------
    X = sm.add_constant(frame[["aff"]].to_numpy(float))
    sm_glm = sm.GLM(obs, X, family=sm.families.Poisson(),
                    offset=frame["log_E"].to_numpy(float)).fit()
    glm_fitted = sm_glm.predict(X, offset=frame["log_E"].to_numpy(float))

    # --- pyLGM spatial model: covariate + Besag ICAR, estimated precision ----
    result = LGM(
        response="observed", likelihood=Poisson(), offset="log_E",
        predictor=Fixed("1 + aff") + Besag(
            "spatial", index="area", graph=graph,
            precision=Hyperparameter("spatial_prec", initial=1.0,
                                     prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
    ).fit(frame, engine="laplace")
    spatial_fitted = np.asarray(result.fitted_mean)
    spatial_rr = spatial_fitted / expected

    aff = result.latent_marginals("fixed")
    print(f"\naff coefficient:  statsmodels {sm_glm.params[1]:+.3f}   "
          f"pyLGM {aff.mean[1]:+.3f} (sd {aff.std[1]:.3f})")
    print(f"estimated spatial precision: {result.hyperparameters['spatial_prec']:.2f}")

    def corr(p):
        return float(np.corrcoef(p, obs)[0, 1])

    print(f"\nfit to observed counts (corr):  "
          f"non-spatial GLM {corr(glm_fitted):.3f}   spatial {corr(spatial_fitted):.3f}")
    print(f"fitted RR spread shrinks {frame['smr'].std():.2f} (raw SMR) "
          f"-> {spatial_rr.std():.2f} (spatial)")

    _figures(frame, obs, expected, glm_fitted, spatial_fitted, spatial_rr,
             corr(glm_fitted), corr(spatial_fitted))
    print(f"\nwrote figures to {IMG}")


def _figures(frame, obs, expected, glm_fitted, spatial_fitted, spatial_rr,
             glm_corr, spatial_corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG.mkdir(parents=True, exist_ok=True)
    blue, orange = "#2b6cb0", "#dd6b20"
    smr = frame["smr"].to_numpy(float)

    # Figure 1: shrinkage -- raw SMR vs spatially-smoothed relative risk -------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hi = max(smr.max(), spatial_rr.max()) * 1.05
    ax.plot([0, hi], [0, hi], color="#a0aec0", lw=1, ls=":", label="no shrinkage (y = x)")
    ax.axhline(1.0, color="#718096", lw=1, ls="--", label="null risk (RR = 1)")
    ax.scatter(smr, spatial_rr, s=np.clip(expected * 6, 15, 200),
               alpha=0.7, color=blue, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("raw SMR  (observed / expected)")
    ax.set_ylabel("spatial relative risk  (pyLGM)")
    ax.set_title("Besag smoothing shrinks noisy rates toward local means\n"
                 "(marker size ∝ expected count)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "scotland_shrinkage.png", dpi=120)
    plt.close(fig)

    # Figure 2: fit to observed counts -- spatial vs standard GLM --------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hi = max(obs.max(), glm_fitted.max(), spatial_fitted.max()) * 1.05
    ax.plot([0, hi], [0, hi], color="#a0aec0", lw=1, ls=":")
    ax.scatter(obs, glm_fitted, s=34, color=orange,
               label=f"non-spatial GLM  (r={glm_corr:.2f})")
    ax.scatter(obs, spatial_fitted, s=34, color=blue, marker="x",
               label=f"pyLGM spatial  (r={spatial_corr:.2f})")
    ax.set_xlabel("observed cases")
    ax.set_ylabel("fitted cases")
    ax.set_title("The spatial effect tracks observed counts;\n"
                 "the covariate-only GLM cannot")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "scotland_fit.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

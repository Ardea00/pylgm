"""Reproducing Anselin (1988) Columbus crime results with pyLGM.

The Columbus, OH neighbourhood crime data is the reference dataset of spatial
econometrics: 49 contiguous neighbourhoods, residential burglaries and vehicle
thefts per 1000 households, regressed on household income and housing value.
Anselin's point is that OLS is *biased* here, because the residuals are
spatially correlated -- neighbourhoods are not independent draws.

pyLGM's `SAR` effect puts a spatial-autoregressive field on the latent scale,
`Q = tau * (I - rho W)'(I - rho W)`, which is the **spatial error model**
structure: `y = X beta + x`, with `(I - rho W) x = eps`. Fitting it should move
the income coefficient away from the OLS value and toward the published
maximum-likelihood spatial-error estimate.

Run from the repo root:
    PYTHONPATH=src python examples/columbus_spatial_econometrics/run.py

Data: Anselin, L. (1988). *Spatial Econometrics: Methods and Models*, Table
12.1, p. 189. Redistributed with the PySAL/libpysal project (BSD-3-Clause);
`graph.json` is the first-order contiguity graph shipped as `columbus.gal`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM
from pylgm.effects.spec import SAR

HERE = Path(__file__).parent
IMG = HERE.parents[1] / "docs" / "img"

# Published reference values. OLS is Anselin's Table 12.1 and is reproduced in
# every spatial-econometrics text; the ML spatial-error row is what `spreg`
# (PySAL) returns on this same data and weights, and is what pyLGM's SAR is
# structurally comparable to.
PUBLISHED = {
    "OLS":            {"const": 68.619, "INC": -1.5973, "HOVAL": -0.2739, "rho": None},
    "ML spatial error": {"const": 60.2795, "INC": -0.9573, "HOVAL": -0.3046, "rho": 0.5468},
}


def _ols(frame):
    x = np.column_stack([np.ones(len(frame)), frame["INC"], frame["HOVAL"]])
    beta, *_ = np.linalg.lstsq(x, frame["CRIME"].to_numpy(), rcond=None)
    return {"const": beta[0], "INC": beta[1], "HOVAL": beta[2], "rho": None}


def _figures(frame, graph, sar_field, estimates):
    """Write the two figures used on the docs page.

    Skipped, not fatal, when matplotlib is absent: this example runs in CI,
    where matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    IMG.mkdir(parents=True, exist_ok=True)
    blue, orange, grey = "#2b6cb0", "#dd6b20", "#718096"

    # Figure 1: what ignoring spatial correlation does to the coefficients ----
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    models = ["OLS\n(ignores space)", "ML spatial error\n(published)", "pyLGM SAR"]
    inc = [estimates["OLS (pyLGM repo)"]["INC"],
           estimates["published ML spatial error"]["INC"],
           estimates["pyLGM SAR"]["INC"]]
    hoval = [estimates["OLS (pyLGM repo)"]["HOVAL"],
             estimates["published ML spatial error"]["HOVAL"],
             estimates["pyLGM SAR"]["HOVAL"]]
    y = np.arange(len(models))
    ax.barh(y - 0.18, inc, height=0.34, color=blue, label="income (INC)")
    ax.barh(y + 0.18, hoval, height=0.34, color=orange, label="housing value (HOVAL)")
    for i, v in enumerate(inc):
        ax.text(v - 0.06, y[i] - 0.18, f"{v:.2f}", va="center", ha="right", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.0, color=grey, lw=0.8)
    ax.set_xlabel("estimated coefficient")
    ax.set_title("Ignoring spatial correlation inflates the income effect\n"
                 "(Columbus crime, Anselin 1988)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "columbus_coefficients.png", dpi=120)
    plt.close(fig)

    # Figure 2: the spatial field OLS throws away -----------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    order = list(frame["id"])
    values = np.array([sar_field[node] for node in order])
    limit = np.abs(values).max()
    dots = ax.scatter(frame["X"], frame["Y"], c=values, cmap="RdBu_r",
                      vmin=-limit, vmax=limit, s=110, edgecolor="white", linewidth=0.6)
    position = {node: (x, y_) for node, x, y_ in zip(order, frame["X"], frame["Y"])}
    for node, neighbours in graph.items():
        for other in neighbours:
            if node < other and node in position and other in position:
                xs = [position[node][0], position[other][0]]
                ys = [position[node][1], position[other][1]]
                ax.plot(xs, ys, color=grey, lw=0.4, alpha=0.45, zorder=0)
    fig.colorbar(dots, ax=ax, label="fitted SAR field")
    ax.set_title("The spatial field a non-spatial GLM discards\n"
                 "(neighbourhood centroids, contiguity edges)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(IMG / "columbus_spatial_field.png", dpi=120)
    plt.close(fig)
    return True


def _pylgm_sar(frame, graph):
    model = LGM(
        response="CRIME",
        predictor=Fixed("1 + INC + HOVAL") + SAR(
            "nbhd", "id", graph,
            rho=Hyperparameter("nbhd.rho", initial=0.0, transform="logit"),
            precision=Hyperparameter("nbhd.precision", initial=0.01,
                                     lower=1e-6, upper=1e3),
        ),
        likelihood=Gaussian(sigma=Hyperparameter("sigma", initial=5.0,
                                                 lower=1e-2, upper=1e3)),
    )
    result = model.fit(frame)
    names = [label.split(":", 1)[1] for label in result.labels[:3]]
    beta = dict(zip(names, result.mean[:3]))
    field = {
        label.split(":", 1)[1]: value
        for label, value in zip(result.labels, result.mean)
        if label.startswith("nbhd:")
    }
    return {
        "const": beta["Intercept"], "INC": beta["INC"], "HOVAL": beta["HOVAL"],
        "rho": result.hyperparameters["nbhd.rho"],
    }, field


def main() -> dict:
    frame = pd.read_csv(HERE / "data.csv", dtype={"id": str})
    graph = json.loads((HERE / "graph.json").read_text(encoding="utf-8"))
    sar, field = _pylgm_sar(frame, graph)
    estimates = {
        "OLS (pyLGM repo)": _ols(frame),
        "pyLGM SAR": sar,
        "published OLS": PUBLISHED["OLS"],
        "published ML spatial error": PUBLISHED["ML spatial error"],
    }
    estimates["figures_written"] = _figures(frame, graph, field, estimates)
    return estimates


if __name__ == "__main__":
    out = main()
    order = ["published OLS", "OLS (pyLGM repo)", "published ML spatial error", "pyLGM SAR"]
    print(f"{'model':<28}{'const':>9}{'INC':>9}{'HOVAL':>9}{'rho':>8}")
    for name in order:
        r = out[name]
        rho = "  --  " if r["rho"] is None else f"{r['rho']:.4f}"
        print(f"{name:<28}{r['const']:>9.3f}{r['INC']:>9.4f}{r['HOVAL']:>9.4f}{rho:>8}")
    ols, sar = out["OLS (pyLGM repo)"], out["pyLGM SAR"]
    print(
        f"\nIgnoring spatial correlation overstates the income effect by "
        f"{abs(ols['INC']) / abs(sar['INC']):.2f}x "
        f"({ols['INC']:.3f} -> {sar['INC']:.3f})."
    )

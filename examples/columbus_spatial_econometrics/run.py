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
    return {
        "const": beta["Intercept"], "INC": beta["INC"], "HOVAL": beta["HOVAL"],
        "rho": result.hyperparameters["nbhd.rho"],
    }


def main() -> dict:
    frame = pd.read_csv(HERE / "data.csv", dtype={"id": str})
    graph = json.loads((HERE / "graph.json").read_text())
    return {
        "OLS (pyLGM repo)": _ols(frame),
        "pyLGM SAR": _pylgm_sar(frame, graph),
        "published OLS": PUBLISHED["OLS"],
        "published ML spatial error": PUBLISHED["ML spatial error"],
    }


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

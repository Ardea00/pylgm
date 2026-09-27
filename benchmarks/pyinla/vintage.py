"""Absorbing a new data vintage: pyLGM ``result.update`` vs refitting.

A regional panel y[r, t] = b0 + b1 x + trend[t] + level[r] + e arrives one
period at a time. Every period is on the latent grid from the start (NaN
response until observed), so each new vintage only adds R observed rows.
Hyperparameters are held at the same fixed values in every engine, so all
three compute the same conditional posterior and the timing isolates the
cost of absorbing the new rows:

* ``pyLGM update``  -- ``result.update(new_rows)``: k = R solves, no refactor
* ``pyLGM refit``   -- ``LGM.fit`` on everything observed so far
* ``pyINLA refit``  -- ``pyinla`` on everything observed so far

    PYTHONPATH=src python benchmarks/pyinla/vintage.py
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from pylgm import IID, LGM, RW1, Fixed, Gaussian

SIGMA, TREND_PREC, LEVEL_PREC = 0.5, 20.0, 1.0


def panel(regions, periods, rng):
    frame = pd.DataFrame({"region": np.repeat(np.arange(regions), periods),
                          "t": np.tile(np.arange(periods), regions)})
    frame["x"] = rng.normal(size=len(frame))
    trend = np.cumsum(rng.normal(scale=TREND_PREC ** -0.5, size=periods))
    level = rng.normal(scale=LEVEL_PREC ** -0.5, size=regions)
    frame["y"] = (1.0 + 0.5 * frame["x"] + trend[frame["t"]] + level[frame["region"]]
                  + rng.normal(scale=SIGMA, size=len(frame)))
    return frame


MODEL = LGM(
    response="y", likelihood=Gaussian(SIGMA),
    predictor=Fixed("1 + x", prior_precision=1e-3)
    + RW1("trend", index="t", precision=TREND_PREC, scale=True)
    + IID("level", index="region", precision=LEVEL_PREC),
)


def fit_pyinla(frame):
    from pyinla import pyinla

    def fixed(prec):
        return {"prec": {"initial": math.log(prec), "fixed": True}}

    model = {"response": "y", "fixed": ["1", "x"], "random": [
        {"id": "t", "model": "rw1", "scale.model": True, "hyper": fixed(TREND_PREC)},
        {"id": "region", "model": "iid", "hyper": fixed(LEVEL_PREC)},
    ]}
    result = pyinla(model=model, family="gaussian", data=frame,
                    control={"fixed": {"prec": 1e-3, "prec_intercept": 1e-3},
                             "family": {"hyper": [{"id": "prec", "initial": math.log(SIGMA ** -2), "fixed": True}]}})
    return result.summary_fixed["mean"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/pyinla/vintage_results.jsonl")
    ap.add_argument("--large", action="store_true", help="add the 3000 x 200 panel")
    args = ap.parse_args()
    open(args.out, "w").close()
    fit_pyinla(panel(5, 10, np.random.default_rng(0)))  # binary warm-up
    for regions, periods in [(100, 60), (1000, 100)] + ([(3000, 200)] if args.large else []):
        full = panel(regions, periods, np.random.default_rng(regions))
        start = periods - args.vintages
        seen = full.assign(y=np.where(full["t"] < start, full["y"], np.nan))
        result = MODEL.fit(seen)
        t_update = t_refit = t_inla = 0.0
        for t in range(start, periods):
            new = full[full["t"] == t]
            seen.loc[new.index, "y"] = new["y"]
            tic = time.perf_counter(); result = result.update(new); t_update += time.perf_counter() - tic
            tic = time.perf_counter(); refit = MODEL.fit(seen); t_refit += time.perf_counter() - tic
            tic = time.perf_counter(); inla_fixed = fit_pyinla(seen.dropna(subset=["y"])); t_inla += time.perf_counter() - tic
        lgm_fixed = result.latent_marginals("fixed").mean
        row = {"regions": regions, "periods": periods, "rows": len(full), "rows_per_vintage": regions,
               "latent": len(result.mean), "vintages": args.vintages,
               "update_s": t_update / args.vintages, "pylgm_refit_s": t_refit / args.vintages,
               "pyinla_refit_s": t_inla / args.vintages,
               "update_vs_refit_max_abs": float(np.max(np.abs(result.mean - refit.mean))),
               "fixed_pylgm": lgm_fixed.tolist(), "fixed_pyinla": inla_fixed.tolist()}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"R={regions:5d} T={periods:4d} latent={row['latent']:5d}  per vintage: "
              f"update {row['update_s']:.3f}s  pyLGM refit {row['pylgm_refit_s']:.3f}s  "
              f"pyINLA refit {row['pyinla_refit_s']:.3f}s  |update-refit|={row['update_vs_refit_max_abs']:.1e}  "
              f"beta pyLGM={np.round(lgm_fixed, 4)} pyINLA={np.round(inla_fixed, 4)}", flush=True)


if __name__ == "__main__":
    main()

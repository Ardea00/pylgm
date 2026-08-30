"""A network that changes every year: US state income with a dynamic spatial panel.

Most spatial models fix the network once. Economic influence does not sit
still -- who a state's income actually co-moves with shifts over a decade. The
`DynamicSpatialPanel` (SDPD) effect takes **one network per period**, so the
graph itself is a time series.

Here the topology is real geography (state contiguity) but the edge *weights*
are economic: neighbouring states are linked more strongly when their income
levels are close. Crucially the weight for year `t` is built from year `t-1`
income, so the network never sees the value being modelled -- which is what
makes the out-of-sample forecast at the end honest.

Panels have holes -- late revisions, small units, series that start later.
This example knocks 20% of the cells out and asks each method to put them
back, which is where a network that knows *who is next to whom, this year*
earns its keep.

The three coefficients separate effects that a static model conflates:

    rho    contemporaneous spillover -- how much of a state's position is
           explained by its neighbours' this year
    gamma  own persistence           -- how much carries over from last year
    eta    spatio-temporal diffusion -- how much of last year's *neighbours*
           reaches a state this year

Run from the repo root (takes about half a minute):
    PYTHONPATH=src python examples/state_income_dynamic_network/run.py

Data: per-capita income by US state, and the 48-state first-order contiguity
graph, both redistributed with PySAL/libpysal (BSD-3-Clause). Original source
is the US Bureau of Economic Analysis.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, forecast_dynamic_spatial_panel
from pylgm.effects.spec import DynamicSpatialPanel

HERE = Path(__file__).parent
IMG = HERE.parents[1] / "docs" / "img"
FIT_YEARS = [str(y) for y in range(1997, 2008)]   # 11 years fitted
FORECAST_YEARS = ["2008", "2009"]                 # held out


def _load():
    frame = pd.read_csv(HERE / "income.csv")
    adjacency = json.loads((HERE / "adjacency.json").read_text())
    log_income = {
        row["state"]: {c: float(np.log(row[c])) for c in frame.columns if c != "state"}
        for _, row in frame.iterrows()
    }
    return log_income, adjacency


def _network_for(year, log_income, adjacency):
    """Contiguity weighted by *previous* year's income similarity.

    Using t-1 keeps the network free of the year being modelled, so the same
    construction works for a forecast period without leaking its target.
    """
    source = str(int(year) - 1)
    weights = {}
    for state, neighbours in adjacency.items():
        weights[state] = {
            other: 1.0 / (1.0 + abs(log_income[state][source] - log_income[other][source]))
            for other in neighbours
        }
    return weights


def _figures(graphs, recovery, truth, recovered, held):
    """Write the docs figures; skipped when matplotlib is absent (as in CI)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    IMG.mkdir(parents=True, exist_ok=True)
    blue, grey = "#2b6cb0", "#718096"

    # Figure 1: the network is a time series, not a fixture --------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    edges = [(s_, o) for s_ in graphs[FIT_YEARS[0]] for o in graphs[FIT_YEARS[0]][s_]]
    series = np.array([[graphs[y][s_][o] for y in FIT_YEARS] for s_, o in edges])
    years = [int(y) for y in FIT_YEARS]
    for row in series[:: max(1, len(series) // 60)]:
        ax.plot(years, row, color=blue, alpha=0.18, lw=0.8)
    ax.plot(years, series.mean(axis=0), color="#c53030", lw=2.2, label="mean edge weight")
    ax.set_xlabel("year")
    ax.set_ylabel("edge weight (income similarity)")
    ax.set_title("One network per year: contiguity edges reweighted annually\n"
                 "(48 US states, weight built from the previous year)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "state_network_drift.png", dpi=120)
    plt.close(fig)

    # Figure 2: recovering knocked-out panel cells ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    names = list(recovery)
    scores = [recovery[n] for n in names]
    order = np.argsort(scores)
    colours = [blue if names[i].startswith("SDPD") else grey for i in order]
    axes[0].barh(range(len(order)), [scores[i] for i in order], color=colours)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([names[i].replace(" (dynamic network)", "\n(dynamic network)")
                             for i in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE on held-out cells (log income)")
    axes[0].set_title("Recovering 20% knocked-out cells", fontsize=10)

    axes[1].scatter(truth, recovered, s=26, color=blue, alpha=0.7, edgecolor="white",
                    linewidth=0.4)
    lo, hi = float(min(truth)), float(max(truth))
    axes[1].plot([lo, hi], [lo, hi], color=grey, lw=1.0, ls="--", label="perfect")
    axes[1].set_xlabel("actual log income")
    axes[1].set_ylabel("SDPD reconstruction")
    axes[1].set_title(f"{len(truth)} held-out cells", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "state_cell_recovery.png", dpi=120)
    plt.close(fig)
    return True


def main() -> dict:
    log_income, adjacency = _load()
    graphs = {y: _network_for(y, log_income, adjacency) for y in FIT_YEARS}
    states = sorted(adjacency)

    frame = pd.DataFrame([
        {"state": s, "year": y, "loginc": log_income[s][y]}
        for s in states for y in FIT_YEARS
    ])

    # Knock out 20% of the panel. Held-out rows keep their latent cell but
    # contribute no likelihood -- the NaN-response workflow pyLGM uses for any
    # unobserved cell.
    rng = np.random.default_rng(0)
    missing = rng.random(len(frame)) < 0.20
    truth = frame.loc[missing, "loginc"].to_numpy()
    training = frame.copy()
    training.loc[missing, "loginc"] = np.nan

    effect = DynamicSpatialPanel(
        "dyn", "state", "year", graphs,
        rho=Hyperparameter("dyn.rho", initial=0.0, transform="logit"),
        gamma=Hyperparameter("dyn.gamma", initial=0.9, transform="identity",
                             lower=-2.0, upper=2.0),
        eta=Hyperparameter("dyn.eta", initial=0.0, transform="identity",
                           lower=-2.0, upper=2.0),
        precision=Hyperparameter("dyn.precision", initial=100.0, lower=1e-2, upper=1e7),
    )
    model = LGM(
        response="loginc",
        predictor=Fixed("1") + effect,
        likelihood=Gaussian(sigma=0.02),
    )
    result = model.fit(training)

    def rmse(pred):
        return float(np.sqrt(np.mean((np.asarray(pred) - truth) ** 2)))

    recovered = result.predictive_mean[missing]
    state_mean = {s: np.mean([log_income[s][y] for y in FIT_YEARS]) for s in states}
    year_mean = {y: np.mean([log_income[s][y] for s in states]) for y in FIT_YEARS}
    held = frame.loc[missing]
    recovery = {
        "SDPD (dynamic network)": rmse(recovered),
        "state mean": rmse([state_mean[s] for s in held["state"]]),
        "year mean": rmse([year_mean[y] for y in held["year"]]),
        "grand mean": rmse(np.full(len(truth), truth.mean())),
    }

    figures_written = _figures(graphs, recovery, truth, recovered, held)

    # How much did the network actually move over the fitted decade?
    first, last = graphs[FIT_YEARS[0]], graphs[FIT_YEARS[-1]]
    drift = float(np.mean([abs(first[s][o] - last[s][o]) for s in first for o in first[s]]))

    # Forecast the held-out years from the same fit. The result is the LATENT
    # field, so add the fixed part to reach the response scale.
    future = {y: _network_for(y, log_income, adjacency) for y in FORECAST_YEARS}
    forecast = forecast_dynamic_spatial_panel(result, effect, future)
    beta = dict(zip(result.labels, result.mean))
    forecast["predicted"] = forecast["latent_mean"] + beta["fixed:Intercept"]
    forecast["actual"] = [log_income[r.unit][r.time] for r in forecast.itertuples()]
    forecast_rmse = {
        y: float(np.sqrt(np.mean(
            (g["predicted"] - g["actual"]) ** 2)))
        for y, g in forecast.groupby("time")
    }
    naive_rmse = {
        y: float(np.sqrt(np.mean([
            (log_income[s]["2007"] - log_income[s][y]) ** 2 for s in states
        ])))
        for y in FORECAST_YEARS
    }
    return {
        "hyperparameters": {k: float(v) for k, v in result.hyperparameters.items()},
        "latent_cells": int(result.mean.shape[0]) - 1,
        "held_out": int(missing.sum()),
        "network_weight_drift": drift,
        "recovery_rmse": recovery,
        "figures_written": figures_written,
        "forecast_rmse": forecast_rmse,
        "naive_forecast_rmse": naive_rmse,
    }


if __name__ == "__main__":
    out = main()
    h = out["hyperparameters"]
    print(f"Fitted {out['latent_cells']} latent cells "
          f"(48 states x {len(FIT_YEARS)} years), one network per year.")
    print(f"Mean edge-weight drift {FIT_YEARS[0]} -> {FIT_YEARS[-1]}: "
          f"{out['network_weight_drift']:.4f}  (the graph really does move)\n")
    print(f"  rho   (neighbour spillover, same year) = {h['dyn.rho']:+.4f}")
    print(f"  gamma (own persistence, one year back) = {h['dyn.gamma']:+.4f}")
    print(f"  eta   (neighbour spillover, lagged)    = {h['dyn.eta']:+.4f}\n")

    print(f"Recovering {out['held_out']} knocked-out panel cells -- RMSE (log income):")
    for name, score in sorted(out["recovery_rmse"].items(), key=lambda kv: kv[1]):
        print(f"     {name:<26}{score:.4f}")

    print("\nHeld-out level forecast -- RMSE (log income):")
    print(f"  {'year':<8}{'SDPD':>10}{'last value':>12}")
    for y in FORECAST_YEARS:
        print(f"  {y:<8}{out['forecast_rmse'][y]:>10.4f}{out['naive_forecast_rmse'][y]:>12.4f}")
    print("  (gamma is about 1, i.e. log income is near a random walk -- for which")
    print("   the last observed value IS the optimal forecast. Structure pays off")
    print("   on the gaps above, not on beating a random walk at its own game.)")

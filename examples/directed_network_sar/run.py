"""Directed interbank-exposure network: a spatial-autoregressive (SAR) field.

Each bank is influenced by a handful of counterparties it is exposed to — a
directed, generally asymmetric relation (row i lists *i's* counterparties, not
who is exposed to i), unlike the symmetric neighbour graphs `Besag`/`ProperCAR`
require. `SAR` builds the precision from `M = I - rho*W` on the row-standardized
`W`, so `rho` reads as the network's contagion strength: how much of a bank's
latent stress is explained by its counterparties' stress rather than its own
idiosyncratic shock.

Run from the repo root:
    PYTHONPATH=src python examples/directed_network_sar/run.py
"""
import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM
from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.spec import SAR

N_BANKS = 30
N_COUNTERPARTIES = 2
TRUE_RHO = 0.6
NOISE_SD = 0.15
SEED = 1


def _interbank_graph(rng):
    """Each bank is exposed to a few counterparties, chosen at random (seeded)."""
    banks = [f"bank_{i}" for i in range(N_BANKS)]
    graph = {}
    for bank in banks:
        others = [b for b in banks if b != bank]
        counterparties = rng.choice(others, size=N_COUNTERPARTIES, replace=False)
        graph[bank] = list(counterparties)
    return banks, graph


def _simulate(graph, rng):
    """A SAR field x = (I - rho*W)^-1 eps on the same row-standardized W the
    fitted SAR effect will build internally, plus observation noise."""
    nodes, w = normalize_directed_graph(graph)
    w = row_standardize(w).toarray()
    m = np.eye(len(nodes)) - TRUE_RHO * w
    x = np.linalg.solve(m, rng.standard_normal(len(nodes)))
    y = x + NOISE_SD * rng.standard_normal(len(nodes))
    return pd.DataFrame({"bank": list(nodes), "y": y})


def main() -> dict:
    rng = np.random.default_rng(SEED)
    banks, graph = _interbank_graph(rng)
    frame = _simulate(graph, rng)

    model = LGM(
        response="y",
        predictor=Fixed("1") + SAR(
            "influence", "bank", graph,
            rho=Hyperparameter("influence.rho", initial=0.0, transform="logit"),
            precision=Hyperparameter("influence.precision", initial=1.0),
        ),
        likelihood=Gaussian(sigma=NOISE_SD),
    )
    result = model.fit(frame)
    predicted = result.predict(frame).predictive_mean
    corr = float(np.corrcoef(predicted, frame["y"])[0, 1])

    return {
        "rho": result.hyperparameters["influence.rho"],
        "precision": result.hyperparameters["influence.precision"],
        "true_rho": TRUE_RHO,
        "corr(pred,y)": corr,
    }


if __name__ == "__main__":
    summary = main()
    print(
        f"true rho={summary['true_rho']:.2f} estimated rho={summary['rho']:.3f} "
        f"precision={summary['precision']:.3f} corr(pred,y)={summary['corr(pred,y)']:.3f}"
    )

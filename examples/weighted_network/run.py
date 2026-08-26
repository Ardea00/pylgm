"""Weighted-network BYM2: firm exposures instead of geographic adjacency.

The CAR math ``Q = τ(D−W)`` never cared whether ``W`` is geography. Give the
graph edge *weights* and the whole CAR family (Besag/ProperCAR/BYM2) becomes a
firm / bank / ownership / supply-chain model: ``W`` encodes *coupling strength*
(interbank exposure, ownership share, supply dependence) rather than 0/1
contiguity, and BYM2's ``φ`` reads as "fraction of variance explained by
network dependence vs. idiosyncratic."

Here: 6 firms in two tightly-coupled blocks {a,b,c} and {d,e,f}, joined by one
weak cross-block link. We observe a noisy performance signal per firm and fit a
weighted ``BYM2``; a firm's estimate borrows strength in proportion to exposure,
so the strongly-coupled partners pull together while the weak link barely
transmits. We contrast it with the unweighted (0/1) graph over the same edges to
show the weights actually change the smoothing.

Run:  PYTHONPATH=src python examples/weighted_network/run.py
"""
import numpy as np
import pandas as pd

from pylgm import BYM2, Fixed, Gaussian, LGM

# Two exposure blocks, coupled strongly within and weakly across (b-d link).
# W is symmetric and strictly positive, as a valid ICAR/CAR requires.
WEIGHTED = {
    "a": {"b": 8.0, "c": 8.0},
    "b": {"a": 8.0, "c": 8.0, "d": 0.2},
    "c": {"a": 8.0, "b": 8.0},
    "d": {"e": 8.0, "f": 8.0, "b": 0.2},
    "e": {"d": 8.0, "f": 8.0},
    "f": {"d": 8.0, "e": 8.0},
}
# Same topology, weights stripped to 0/1 — the "just adjacency" view.
UNWEIGHTED = {node: list(nbrs) for node, nbrs in WEIGHTED.items()}


def fit(graph, frame):
    model = LGM(
        response="y",
        likelihood=Gaussian(sigma=0.3),
        predictor=Fixed("1") + BYM2("firm", index="firm", graph=graph, phi=0.8),
    )
    return model.fit(frame).latent_marginals("firm").mean


def main() -> None:
    firms = list(WEIGHTED)
    # One firm per block gets a sharp idiosyncratic shock; the rest are quiet.
    # Strong within-block exposure should spread each shock to its partners.
    y = np.array([1.5, 0.0, 0.0, -1.5, 0.0, 0.0])
    frame = pd.DataFrame({"firm": firms, "y": y})

    w = fit(WEIGHTED, frame)
    u = fit(UNWEIGHTED, frame)

    print(f"firms:            {firms}")
    print(f"observed y:       {np.array2string(y, precision=2)}")
    print(f"weighted BYM2:    {np.array2string(w, precision=3)}")
    print(f"unweighted BYM2:  {np.array2string(u, precision=3)}")

    # a's partners (b, c) are strongly exposed to a's +1.5 shock; under the
    # weighted graph they are pulled up markedly more than under plain 0/1
    # adjacency, which treats the weak cross-block link the same as a partner.
    bc_weighted = w[1] + w[2]
    bc_unweighted = u[1] + u[2]
    print(f"\nb+c estimate — weighted {bc_weighted:+.3f} vs unweighted "
          f"{bc_unweighted:+.3f}: strong exposure borrows more strength")
    assert not np.allclose(w, u), "weights should change the smoothing"


if __name__ == "__main__":
    main()

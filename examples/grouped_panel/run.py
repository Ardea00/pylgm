"""Grouped: correlated copies of a spatial field, tied across years by an AR1.

``Replicated(effect, over=r)`` gives ``R`` *independent* copies of an effect
sharing its hyperparameters: precision ``I_R (x) Q_E``. ``Grouped`` replaces
that identity with a real between-group precision -- ``Q_S (x) Q_E`` -- so the
copies borrow strength from each other.

Here the copies are years and the effect is a spatial field over 8 regions on a
chain graph. The truth is a spatial pattern that *persists*: each year's field
is 0.9 times the previous year's plus a fresh spatial innovation, which is
exactly ``kron(AR1(0.9), Besag)``. One noisy observation per (region, year).

Two fits on identical data:

  Replicated(Besag, over="year")                       -- years independent
  Grouped(Besag, over="year", structure=AR1Structure(0.9))  -- years correlated

The independent fit can only smooth *within* a year, so each year's estimate
sees 8 observations. The correlated fit also smooths *across* years, so a
region's estimate borrows from its own past and future. With a persistent
truth, that is a large amount of extra information, and the recovered latent
field should be markedly closer to it.

Note ``AR1Structure`` is what makes this un-expressible as a ``SpaceTime``
interaction: Knorr-Held's four types pair {iid, structured} with {iid,
structured}, where "structured" means Besag or a random walk. An AR1 between
groups is outside that family.

Run:  PYTHONPATH=src python examples/grouped_panel/run.py
"""
import numpy as np
import pandas as pd

from pylgm import AR1Structure, Besag, Fixed, Gaussian, Grouped, LGM, Replicated

REGIONS = [f"r{i}" for i in range(8)]
# A chain: r0 - r1 - ... - r7. Symmetric, as any ICAR graph must be.
GRAPH = {
    r: [n for n in (REGIONS[i - 1] if i else None, REGIONS[i + 1] if i < 7 else None) if n]
    for i, r in enumerate(REGIONS)
}
YEARS = list(range(2012, 2022))
RHO, NOISE = 0.9, 0.6


def _spatial_sampler(seed):
    """Draw sum-to-zero fields from the Besag structure's pseudo-inverse.

    The ICAR precision is singular -- its null space is the constant -- so the
    draw lives on the orthogonal complement: drop the zero eigenvalue rather
    than inverting through it.
    """
    from pylgm.effects.besag import _scaled_structure
    from pylgm.effects.graph import normalize_graph

    nodes, w = normalize_graph(GRAPH)
    values, vectors = np.linalg.eigh(_scaled_structure(w, nodes, scale=True))
    keep = values > 1e-8
    root = vectors[:, keep] / np.sqrt(values[keep])
    rng = np.random.default_rng(seed)
    return nodes, lambda: root @ rng.standard_normal(root.shape[1])


def simulate(seed=0):
    """A spatial field that persists across years: u_t = rho*u_{t-1} + innovation."""
    nodes, draw = _spatial_sampler(seed)
    fields = [draw()]
    for _ in YEARS[1:]:
        fields.append(RHO * fields[-1] + np.sqrt(1 - RHO**2) * draw())
    truth = np.concatenate(fields)

    rng = np.random.default_rng(seed + 1)
    frame = pd.DataFrame([
        {"region": r, "year": y, "truth": fields[t][s]}
        for t, y in enumerate(YEARS) for s, r in enumerate(nodes)
    ])
    frame["y"] = frame["truth"] + NOISE * rng.standard_normal(len(frame))
    return frame, truth


def fit(effect, frame):
    result = LGM(
        response="y", likelihood=Gaussian(sigma=NOISE),
        predictor=Fixed("1") + effect,
    ).fit(frame)
    return result.latent_marginals("u").mean, result


def main() -> None:
    frame, truth = simulate()
    spatial = Besag("u", index="region", graph=GRAPH, precision=1.0)

    independent, r_ind = fit(Replicated(spatial, over="year"), frame)
    correlated, r_cor = fit(
        Grouped(spatial, over="year", structure=AR1Structure(rho=RHO)), frame
    )

    # Both models constrain each year's field to sum to zero, so compare against
    # a truth centred the same way -- otherwise the level, which neither model
    # can see, would dominate the error.
    centred = (truth.reshape(len(YEARS), len(REGIONS))
               - truth.reshape(len(YEARS), len(REGIONS)).mean(axis=1, keepdims=True)).ravel()

    def rmse(estimate):
        return float(np.sqrt(np.mean((estimate - centred) ** 2)))

    print(f"regions x years:            {len(REGIONS)} x {len(YEARS)}"
          f"  ({len(frame)} observations)")
    print(f"true year-to-year rho:      {RHO}")
    print(f"observation noise sd:       {NOISE}")
    print()
    print(f"Replicated (independent):   latent RMSE {rmse(independent):.4f}"
          f"   log ML {r_ind.log_marginal_likelihood:.2f}")
    print(f"Grouped    (AR1-correlated): latent RMSE {rmse(correlated):.4f}"
          f"   log ML {r_cor.log_marginal_likelihood:.2f}")
    print()
    improvement = 100 * (1 - rmse(correlated) / rmse(independent))
    print(f"borrowing strength across years cuts the latent error by {improvement:.1f}%")
    print(f"and the correlated model has the higher marginal likelihood: "
          f"{r_cor.log_marginal_likelihood > r_ind.log_marginal_likelihood}")


if __name__ == "__main__":
    main()

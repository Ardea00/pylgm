"""pyLGM vs pyINLA: same model, same priors, same data.

Needs ``pip install pyinla`` (Python 3.10-3.12; proprietary licence; downloads
the ``inla`` binary on first use). Run from the repository root:

    PYTHONPATH=src python benchmarks/pyinla/run.py [--quick]

Two cases, each at increasing size:

* ``gaussian_iid``  -- y = b0 + b1 x + u[group] + e, estimated noise and
  group precisions, 10 rows per group.
* ``poisson_besag`` -- y ~ Poisson(E exp(b0 + b1 x + s[area])), scaled ICAR on
  an L x L rook lattice.

Both packages use PC(1, 0.01) priors on every precision, a scaled Besag, and
fixed-effect prior precision 1e-3; pyLGM integrates the hyperparameters
(``hyperparameters="integrate"``) and, for the Poisson case, applies the
variational mean correction INLA applies by default. Reported: wall time (median of ``--repeat``
runs, first pyINLA call excluded as binary warm-up), fixed-effect posterior
means/sds, and the hyperparameter posterior mean.
"""

import argparse
import json
import statistics
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pylgm import IID, LGM, Besag, Fixed, Gaussian, Hyperparameter, PCPrecision, Poisson

PC = dict(upper_sd=1.0, alpha=0.01)
PC_INLA = {"prec": {"prior": "pc.prec", "param": [1.0, 0.01]}}
FIXED_PREC = 1e-3


def gaussian_iid(groups, rng):
    n = groups * 10
    g = np.repeat(np.arange(groups), 10)
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.7, size=groups)[g] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def lattice(side):
    idx = np.arange(side * side).reshape(side, side)
    pairs = [(idx[i, j], idx[i, j + 1]) for i in range(side) for j in range(side - 1)]
    pairs += [(idx[i, j], idx[i + 1, j]) for i in range(side - 1) for j in range(side)]
    a, b = np.array(pairs).T
    n = side * side
    adj = sp.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    graph = {str(i): [str(j) for j in adj[i].indices] for i in range(n)}
    return adj, graph


def poisson_besag(side, rng):
    n = side * side
    ii, jj = np.divmod(np.arange(n), side)
    s = 0.6 * np.sin(2 * np.pi * ii / side) * np.cos(2 * np.pi * jj / side)
    x = rng.normal(size=n)
    E = rng.uniform(5, 20, size=n)
    y = rng.poisson(E * np.exp(-0.2 + 0.3 * x + s))
    return pd.DataFrame({"y": y, "x": x, "E": E, "logE": np.log(E), "area": np.arange(n)})


def fit_pylgm(case, frame, graph=None):
    if case == "gaussian_iid":
        model = LGM(
            response="y",
            likelihood=Gaussian(Hyperparameter("sigma", initial=1.0)),
            predictor=Fixed("1 + x", prior_precision=FIXED_PREC)
            + IID("u", index="g", precision=Hyperparameter("u_prec", initial=1.0, prior=PCPrecision(**PC))),
        )
        result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    else:
        f = frame.assign(area=frame["area"].astype(str))
        model = LGM(
            response="y", likelihood=Poisson(), offset="logE",
            predictor=Fixed("1 + x", prior_precision=FIXED_PREC)
            + Besag("s", index="area", graph=graph,
                    precision=Hyperparameter("s_prec", initial=1.0, prior=PCPrecision(**PC))),
        )
        # INLA's default latent mean carries its variational (VB) correction;
        # pyLGM's equivalent is mean_correction=True (off by default).
        result = model.fit(f, engine="laplace", hyperparameters="integrate", mean_correction=True)
    fixed = result.latent_marginals("fixed")
    hyper = {k: float(v.mean[0]) for k, v in result.hyperparameter_marginals().items()}
    return {"fixed_mean": list(map(float, fixed.mean)), "fixed_sd": list(map(float, fixed.std)), "hyper_mean": hyper}


def fit_pyinla(case, frame, adj=None):
    from pyinla import pyinla

    control_fixed = {"prec": FIXED_PREC, "prec_intercept": FIXED_PREC}
    if case == "gaussian_iid":
        model = {"response": "y", "fixed": ["1", "x"],
                 "random": [{"id": "g", "model": "iid", "hyper": PC_INLA}]}
        r = pyinla(model=model, family="gaussian", data=frame, control={"fixed": control_fixed})
    else:
        model = {"response": "y", "fixed": ["1", "x"],
                 "random": [{"id": "area", "model": "besag", "graph": adj,
                             "scale.model": True, "hyper": PC_INLA}]}
        r = pyinla(model=model, family="poisson", data=frame, E=frame["E"].to_numpy(),
                   control={"fixed": control_fixed})
    sf, sh = r.summary_fixed, r.summary_hyperpar
    return {"fixed_mean": sf["mean"].tolist(), "fixed_sd": sf["sd"].tolist(),
            "hyper_mean": dict(zip(sh.index, sh["mean"].tolist()))}


def _append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def timed(fn, repeat):
    times, out = [], None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/pyinla/results.jsonl")
    args = ap.parse_args()
    sizes = {"gaussian_iid": [50, 500] if args.quick else [50, 500, 2000, 5000],
             "poisson_besag": [10, 20] if args.quick else [10, 30, 50, 70]}

    fit_pyinla("gaussian_iid", gaussian_iid(10, np.random.default_rng(1)))  # binary warm-up
    open(args.out, "w").close()
    for case, grid in sizes.items():
        for size in grid:
            rng = np.random.default_rng(size)
            if case == "gaussian_iid":
                frame, adj, graph = gaussian_iid(size, rng), None, None
            else:
                adj, graph = lattice(size)
                frame = poisson_besag(size, rng)
            try:
                t_lgm, lgm = timed(lambda: fit_pylgm(case, frame, graph), args.repeat)
                t_inla, inla = timed(lambda: fit_pyinla(case, frame, adj), args.repeat)
            except Exception as error:  # record and move on: one failure must not cost the run
                _append(args.out, {"case": case, "size": size, "error": repr(error)})
                print(f"{case:14s} size={size}  FAILED: {error!r}", flush=True)
                continue
            row = {"case": case, "size": size, "n_rows": len(frame),
                   "pylgm_s": t_lgm, "pyinla_s": t_inla, "pylgm": lgm, "pyinla": inla,
                   "max_abs_fixed_mean_diff": float(np.max(np.abs(np.subtract(lgm["fixed_mean"], inla["fixed_mean"])))),
                   "max_rel_fixed_sd_diff": float(np.max(np.abs(np.divide(lgm["fixed_sd"], inla["fixed_sd"]) - 1)))}
            _append(args.out, row)
            print(f"{case:14s} n={len(frame):6d}  pyLGM {t_lgm:7.2f}s  pyINLA {t_inla:7.2f}s  "
                  f"|dmean|={row['max_abs_fixed_mean_diff']:.4f}  |dsd|/sd={row['max_rel_fixed_sd_diff']:.3f}  "
                  f"hyper pyLGM={lgm['hyper_mean']}  pyINLA={inla['hyper_mean']}", flush=True)


if __name__ == "__main__":
    main()

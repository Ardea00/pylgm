"""pyLGM vs GLM vs gradient boosting, on two deliberately different problems.

Neither method wins everywhere, and the point of this script is to show where
each one does. Both problems are simulated, so the *true* latent surface is
known and can be scored directly -- which is the honest target for a smoothing
method, rather than in-sample fit to noisy observations.

    A. Small-area spatial counts. Many areas, few observations each, a smooth
       spatial signal. The regime latent Gaussian models exist for.
    B. Nonlinear covariate interactions. One pooled population, plenty of rows,
       response driven by products and thresholds of continuous features. The
       regime gradient boosting exists for.

Run from the repo root:
    PYTHONPATH=src python examples/method_comparison/run.py

Requires scikit-learn and xgboost, which are NOT pyLGM dependencies:
    pip install scikit-learn xgboost
"""
import numpy as np
import pandas as pd

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM, Poisson
from pylgm.priors import PCPrecision

N_AREAS = 200
OBS_PER_AREA = 3
SEED = 0


def _label(i):
    """Zero-padded so the graph's sorted node order matches numeric order.

    Graph nodes are sorted as strings, so bare str(i) would order them
    0, 1, 10, 100, ... and silently misalign the latent field against the
    simulated truth.
    """
    return f"{i:04d}"


def _ring_graph(n):
    return {_label(i): [_label((i - 1) % n), _label((i + 1) % n)] for i in range(n)}


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def problem_a_small_area_counts(rng):
    """Smooth spatial log-rate over a ring of areas; few Poisson counts each."""
    position = np.arange(N_AREAS)
    true_field = 0.8 * np.sin(2 * np.pi * position / N_AREAS * 3)
    rows = []
    for area in range(N_AREAS):
        for _ in range(OBS_PER_AREA):
            rate = np.exp(0.5 + true_field[area])
            rows.append({"area": _label(area), "pos": float(area), "y": rng.poisson(rate)})
    frame = pd.DataFrame(rows)

    # pyLGM: a Besag field over the adjacency graph, precision estimated.
    model = LGM(
        response="y",
        likelihood=Poisson(),
        predictor=Fixed("1") + Besag(
            "area", index="area", graph=_ring_graph(N_AREAS),
            precision=Hyperparameter("area.precision", initial=1.0,
                                     prior=PCPrecision(upper_sd=1.0, alpha=0.01)),
        ),
    )
    result = model.fit(frame, engine="laplace")
    marg = result.latent_marginals("area")
    lgm_field = np.asarray(marg.mean) - np.mean(marg.mean)

    # GLM: an unpooled fixed effect per area (the "just add a dummy" baseline).
    counts = frame.groupby("area")["y"].mean().reindex([_label(i) for i in range(N_AREAS)])
    glm_field = np.log(np.maximum(counts.to_numpy(), 0.5))
    glm_field = glm_field - glm_field.mean()

    # Gradient boosting on the same information (area id + position).
    from xgboost import XGBRegressor
    x = frame[["pos"]].to_numpy()
    gb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                      random_state=SEED, verbosity=0)
    gb.fit(x, frame["y"].to_numpy())
    gb_field = np.log(np.maximum(gb.predict(np.arange(N_AREAS).reshape(-1, 1)), 0.05))
    gb_field = gb_field - gb_field.mean()

    centred_truth = true_field - true_field.mean()
    return {
        "pyLGM (Besag)": _rmse(lgm_field, centred_truth),
        "GLM (area dummies)": _rmse(glm_field, centred_truth),
        "XGBoost": _rmse(gb_field, centred_truth),
    }, result


def problem_b_nonlinear_interactions(rng):
    """Response driven by products/thresholds of continuous covariates."""
    n = 4000
    x1, x2, x3 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    signal = 2.0 * x1 * x2 + 1.5 * np.where(x3 > 0.5, 1.0, -1.0) + 0.5 * x1**2
    y = signal + 0.3 * rng.standard_normal(n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    split = n // 2
    train, test = frame.iloc[:split], frame.iloc[split:]
    truth_test = signal[split:]

    lgm = LGM(response="y", predictor=Fixed("1 + x1 + x2 + x3"),
              likelihood=Gaussian(sigma=0.3)).fit(train)
    lgm_pred = lgm.predict(test).predictive_mean

    from xgboost import XGBRegressor
    gb = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                      random_state=SEED, verbosity=0)
    gb.fit(train[["x1", "x2", "x3"]].to_numpy(), train["y"].to_numpy())
    gb_pred = gb.predict(test[["x1", "x2", "x3"]].to_numpy())

    return {
        "pyLGM (linear predictor)": _rmse(lgm_pred, truth_test),
        "XGBoost": _rmse(gb_pred, truth_test),
    }


def problem_c_deterministic_vs_monte_carlo(rng):
    """Same posterior, two ways: one exact solve vs a random-walk sampler.

    A small Gaussian LGM has a closed-form posterior, so both pyLGM and the
    sampler are targeting something we can write down exactly. That makes the
    comparison about the *inference style*, not the model: pyLGM returns the
    exact mean in one solve, while Monte Carlo error falls as 1/sqrt(S).
    """
    n, sigma, tau = 40, 0.5, 2.0
    truth = np.cumsum(rng.standard_normal(n) * 0.3)
    y = truth + sigma * rng.standard_normal(n)

    # Exact posterior for x ~ N(0, (tau*D'D + ridge)^-1), y = x + noise.
    d = np.diff(np.eye(n), axis=0)
    q = tau * (d.T @ d) + 1e-6 * np.eye(n)
    post_precision = q + np.eye(n) / sigma**2
    exact_mean = np.linalg.solve(post_precision, y / sigma**2)

    # Random-walk Metropolis on the same target.
    def log_target(x):
        return -0.5 * x @ q @ x - 0.5 * np.sum((y - x) ** 2) / sigma**2

    results = {}
    for draws in (1_000, 10_000, 100_000):
        x = np.zeros(n)
        lp = log_target(x)
        total, accepted = np.zeros(n), 0
        step = 0.12
        for i in range(draws):
            proposal = x + step * rng.standard_normal(n)
            lp_new = log_target(proposal)
            if np.log(rng.uniform()) < lp_new - lp:
                x, lp, accepted = proposal, lp_new, accepted + 1
            total += x
        results[draws] = _rmse(total / draws, exact_mean)
    return {"mc_error_vs_exact": results}


def coverage_of_credible_intervals(result):
    """Do pyLGM's 95% intervals actually contain the truth 95% of the time?"""
    position = np.arange(N_AREAS)
    true_field = 0.8 * np.sin(2 * np.pi * position / N_AREAS * 3)
    truth = true_field - true_field.mean()
    marg = result.latent_marginals("area")
    mean = np.asarray(marg.mean) - np.mean(marg.mean)
    sd = np.sqrt(np.asarray(marg.variance))
    inside = np.abs(truth - mean) <= 1.96 * sd
    return float(inside.mean())


def main() -> dict:
    rng = np.random.default_rng(SEED)
    a_scores, a_result = problem_a_small_area_counts(rng)
    b_scores = problem_b_nonlinear_interactions(rng)
    c_scores = problem_c_deterministic_vs_monte_carlo(rng)
    return {
        "A_small_area_rmse": a_scores,
        "B_nonlinear_rmse": b_scores,
        "A_pylgm_95pct_coverage": coverage_of_credible_intervals(a_result),
        "C_monte_carlo": c_scores,
    }


if __name__ == "__main__":
    out = main()
    print("A. Small-area spatial counts -- RMSE against the TRUE latent field")
    for name, score in out["A_small_area_rmse"].items():
        print(f"     {name:24s} {score:.4f}")
    print(f"     pyLGM 95% interval coverage: {out['A_pylgm_95pct_coverage']:.2f}")
    print("\nB. Nonlinear covariate interactions -- RMSE against the TRUE signal")
    for name, score in out["B_nonlinear_rmse"].items():
        print(f"     {name:24s} {score:.4f}")
    print("\nC. Deterministic vs Monte Carlo -- distance from the EXACT posterior mean")
    print("     pyLGM (one exact solve)  0.0000")
    for draws, err in out["C_monte_carlo"]["mc_error_vs_exact"].items():
        print(f"     Metropolis, {draws:>7,} draws {err:.4f}")

"""Gradient boosting and pyLGM in one linear predictor, joined by the offset.

`examples/method_comparison` shows the two regimes separately: structured
smoothing wins on small-area spatial signal, boosting wins on nonlinear
covariate interactions. Real panels have both at once. This script simulates
exactly that -- Poisson counts whose log-rate is

    eta[i] = 0.5 + g(x1, x2, x3) + s[area[i]]

with `g` a product/threshold surface no linear predictor can represent and `s`
a smooth field over a ring of areas -- and combines the two methods in the one
place where they compose without either giving anything up:

    eta = f(z)              <- boosting, carried in pyLGM's `offset`
        + sum_k A_k x_k     <- the latent Gaussian field, with its posterior

Both orderings are shown. Boosting first (`offset=`) leaves every pyLGM
posterior exactly what it claims to be, because an offset is a known constant.
pyLGM first (`base_margin=`) does not: the boosting stage is fit on the same
rows and is not in the latent covariance.

The fourth result is the one worth the trouble. Feeding *in-sample* boosting
margins into the offset shrinks the spatial variance component -- the booster
has already fit the training rows, so the Laplace fit sees residuals that are
too small and empirical Bayes answers with a precision roughly twice what the
honest fit reports. The field loses a third of its amplitude and its 95%
intervals cover 0.64 instead of 0.96. Point predictions barely notice; the
posterior, which is the reason to use this library, is ruined. The offset must
be out-of-fold.

Run from the repo root:
    PYTHONPATH=src python examples/boosted_offset/run.py

Requires xgboost, which is NOT a pyLGM dependency:
    pip install xgboost
"""
import numpy as np
import pandas as pd

from pylgm import Besag, Fixed, Hyperparameter, LGM, Poisson
from pylgm.priors import PCPrecision

N_AREAS = 100
REPS = 20            # rows per area
TRAIN_REPS = 14      # first 14 of each area's rows train, last 6 test
N_FOLDS = 5
SEED = 0
FEATURES = ["x1", "x2", "x3"]

# One booster configuration throughout: the variants below differ only in what
# they are asked to predict, never in how hard they are allowed to fit.
BOOSTER = dict(
    objective="count:poisson", n_estimators=400, max_depth=4,
    learning_rate=0.05, random_state=SEED, verbosity=0,
)


def _label(i):
    """Zero-padded so the graph's sorted node order matches numeric order."""
    return f"{i:04d}"


def _ring_graph(n):
    return {_label(i): [_label((i - 1) % n), _label((i + 1) % n)] for i in range(n)}


GRAPH = _ring_graph(N_AREAS)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def simulate(rng):
    """Counts driven by BOTH a nonlinear covariate surface and a smooth field."""
    area = np.repeat(np.arange(N_AREAS), REPS)
    x = rng.uniform(-1.5, 1.5, (N_AREAS * REPS, 3))
    nonlinear = (
        0.7 * x[:, 0] * x[:, 1]                       # interaction
        + 0.6 * np.where(x[:, 2] > 0.5, 1.0, -1.0)    # threshold
        + 0.4 * (x[:, 0] ** 2 - 0.75)                 # curvature
    )
    field = 0.8 * np.sin(2 * np.pi * 3 * np.arange(N_AREAS) / N_AREAS)
    field = field - field.mean()                      # Besag is sum-to-zero
    eta = 0.5 + nonlinear + field[area]
    frame = pd.DataFrame({
        "area": [_label(a) for a in area],
        "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2],
        "y": rng.poisson(np.exp(eta)),
        "eta_true": eta,
    })
    # Every area appears in both halves, so the Besag field is identified on
    # train and every test row maps to a fitted level.
    rep = np.tile(np.arange(REPS), N_AREAS)
    train = frame[rep < TRAIN_REPS].reset_index(drop=True)
    test = frame[rep >= TRAIN_REPS].reset_index(drop=True)
    return train, test, field


def _fit_booster(x, y, base_margin=None):
    from xgboost import XGBRegressor
    booster = XGBRegressor(**BOOSTER)
    booster.fit(x, y, base_margin=base_margin)
    return booster


def out_of_fold_margin(x, y, seed=SEED):
    """Leakage-free training offsets: each row scored by a booster that never saw it.

    Random K-fold is right here because the panel has no time axis. On a
    time-indexed panel this must instead be a rolling-origin scheme -- the
    margin for time t may not be built from t+1 -- which is what
    `pylgm.evaluation.folds.build_fold_definitions` already generates.
    """
    fold = np.random.default_rng(seed).permutation(len(x)) % N_FOLDS
    margin = np.zeros(len(x))
    for f in range(N_FOLDS):
        held_out = fold == f
        booster = _fit_booster(x[~held_out], y[~held_out])
        margin[held_out] = booster.predict(x[held_out], output_margin=True)
    return margin


def fit_lgm(frame, *, formula="1", offset=None):
    """Fixed terms + a Besag field over the ring, precision estimated."""
    model = LGM(
        response="y",
        likelihood=Poisson(),
        offset=offset,
        predictor=Fixed(formula) + Besag(
            "area", index="area", graph=GRAPH,
            precision=Hyperparameter(
                "area.precision", initial=1.0,
                prior=PCPrecision(upper_sd=1.0, alpha=0.01),
            ),
        ),
    )
    return model.fit(frame, engine="laplace")


def _field_report(result, truth):
    """What the Besag field recovered: amplitude, accuracy, interval calibration."""
    marginals = result.latent_marginals("area")
    mean = np.asarray(marginals.mean) - np.mean(marginals.mean)
    sd = np.sqrt(np.asarray(marginals.variance))
    return {
        "field_rmse": _rmse(mean, truth),
        "field_sd": float(np.std(mean)),
        "coverage95": float(np.mean(np.abs(truth - mean) <= 1.96 * sd)),
        "area.precision": float(result.hyperparameters["area.precision"]),
    }


def main() -> dict:
    rng = np.random.default_rng(SEED)
    train, test, true_field = simulate(rng)
    x_train, y_train = train[FEATURES].to_numpy(), train["y"].to_numpy()
    x_test = test[FEATURES].to_numpy()
    eta_test = test["eta_true"].to_numpy()

    # The booster used at prediction time is always the full-training-data one;
    # only the *training* offset needs to be out-of-fold.
    deployed = _fit_booster(x_train, y_train)
    margin_test = deployed.predict(x_test, output_margin=True)
    margin_oof = out_of_fold_margin(x_train, y_train)
    margin_insample = deployed.predict(x_train, output_margin=True)

    scores, fields = {}, {}

    # 1. Boosting alone -- no way to express the spatial field.
    scores["boosting alone"] = _rmse(margin_test, eta_test)

    # 2. pyLGM alone -- linear in the covariates, so the interactions are lost.
    lgm = fit_lgm(train, formula="1 + x1 + x2 + x3")
    scores["pyLGM alone"] = _rmse(lgm.predict(test).predictive_mean, eta_test)
    fields["pyLGM alone"] = _field_report(lgm, true_field)

    # 3. Boost -> LGM, out-of-fold offset. The combination that works.
    frame = train.assign(boost=margin_oof)
    fit = fit_lgm(frame, offset="boost")
    scores["boost -> LGM (out-of-fold offset)"] = _rmse(
        fit.predict(test.assign(boost=margin_test)).predictive_mean, eta_test
    )
    fields["boost -> LGM (out-of-fold offset)"] = _field_report(fit, true_field)

    # 4. Same, with an in-sample offset. The failure this example exists to show.
    naive = fit_lgm(train.assign(boost=margin_insample), offset="boost")
    scores["boost -> LGM (in-sample offset)"] = _rmse(
        naive.predict(test.assign(boost=margin_test)).predictive_mean, eta_test
    )
    fields["boost -> LGM (in-sample offset)"] = _field_report(naive, true_field)

    # 5. LGM -> boost: the latent predictor becomes the booster's base margin.
    #    Fits as well, but the latent posterior no longer accounts for stage 2.
    eta_lgm_train = lgm.predictive_mean
    residual_booster = _fit_booster(x_train, y_train, base_margin=eta_lgm_train)
    scores["LGM -> boost (base margin)"] = _rmse(
        residual_booster.predict(
            x_test, output_margin=True,
            base_margin=lgm.predict(test).predictive_mean,
        ),
        eta_test,
    )

    return {"eta_rmse": scores, "field": fields,
            "true_field_sd": float(np.std(true_field))}


if __name__ == "__main__":
    out = main()
    print("Held-out RMSE against the TRUE log-rate eta (lower is better)")
    for name, score in out["eta_rmse"].items():
        print(f"     {name:36s} {score:.4f}")
    print(f"\nSpatial field recovery (true field sd = {out['true_field_sd']:.3f})")
    header = f"     {'':36s} {'rmse':>7s} {'sd':>7s} {'cover':>7s} {'precision':>11s}"
    print(header)
    for name, report in out["field"].items():
        print(
            f"     {name:36s} {report['field_rmse']:7.4f} {report['field_sd']:7.4f}"
            f" {report['coverage95']:7.2f} {report['area.precision']:11.2f}"
        )

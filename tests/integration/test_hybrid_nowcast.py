import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, BYM2, Fixed, Gaussian, Hyperparameter, LGM, MIDAS, MIDASParametric
from pylgm.effects.midas import midas_weights

R, T, K = 8, 40, 12
SEED = 0


def _panel():
    """Deterministic regional-GDP panel; see examples/hybrid_nowcast for the DGP."""
    rng = np.random.default_rng(SEED)
    regions = [str(i) for i in range(R)]
    graph = {str(i): [str((i - 1) % R), str((i + 1) % R)] for i in range(R)}  # ring
    w_true = midas_weights("exp_almon", K, (0.2, -0.05))
    s = 1.2 * np.sin(2 * np.pi * np.arange(R) / R)
    a = np.zeros(T)
    for t in range(1, T):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    cols = tuple(f"ind_lag{k}" for k in range(K))
    rows = []
    for ri, reg in enumerate(regions):
        monthly = rng.normal(size=T + K)
        lagged = {name: pd.Series(monthly).shift(k) for k, name in enumerate(cols)}
        rdf = pd.DataFrame(lagged).iloc[K:].reset_index(drop=True)
        rdf = rdf.iloc[:T].reset_index(drop=True)
        hf = rdf[list(cols)].to_numpy() @ w_true
        rdf["region"] = reg
        rdf["quarter"] = np.arange(len(rdf))
        rdf["y"] = 0.5 + 3.0 * hf + s[ri] + a[: len(rdf)] + rng.normal(scale=0.3, size=len(rdf))
        rows.append(rdf)
    return pd.concat(rows, ignore_index=True), cols, graph, w_true


def _common(graph):
    # ponytail: phi/rho fixed floats -> deterministic test; both accept Hyperparameter.
    return (
        BYM2("region", index="region", graph=graph,
             precision=Hyperparameter("region.precision", initial=1.0), phi=0.5)
        + AR1("quarter", index="quarter",
              precision=Hyperparameter("quarter.precision", initial=1.0), rho=0.6)
    )


def _fit_umidas(frame, cols, graph):
    model = LGM(response="y", panel=("region",), time="quarter",
                predictor=Fixed("1")
                + MIDAS("lag", columns=cols, precision=Hyperparameter("lag.precision", initial=1.0))
                + _common(graph),
                likelihood=Gaussian(sigma=0.3))
    result = model.fit(frame)
    kernel = result.latent_marginals("lag").mean
    return result, "lag", kernel


def _fit_parametric(frame, cols, graph):
    model = LGM(response="y", panel=("region",), time="quarter",
                predictor=Fixed("1")
                + MIDASParametric("m", cols, kernel="exp_almon")
                + _common(graph),
                likelihood=Gaussian(sigma=0.3))
    result = model.fit(frame)
    theta_hat = (result.hyperparameters["m.shape1"], result.hyperparameters["m.shape2"])
    kernel = midas_weights("exp_almon", K, theta_hat)
    return result, "m", kernel


@pytest.mark.parametrize("fit", [_fit_umidas, _fit_parametric])
# The MIDAS smoothness precision pins on this short synthetic series; the
# test checks composition and recovery of the other terms.
@pytest.mark.filterwarnings("ignore:empirical-Bayes estimate")
def test_hybrid_composes_predicts_and_recovers(fit):
    frame, cols, graph, w_true = _panel()

    result, midas_block, kernel = fit(frame, cols, graph)  # (1) fit succeeds

    # (2) every composed block resolves to finite latent marginals
    for block in ("fixed", midas_block, "region", "quarter"):
        assert np.isfinite(result.latent_marginals(block).mean).all()

    # (3) prediction is finite
    pred = result.predict(frame).predictive_mean
    assert np.isfinite(pred).all()

    # (4) the composed field tracks the target
    assert np.corrcoef(pred, frame["y"].to_numpy())[0, 1] > 0.8

    # (5) the recovered lag kernel matches the true kernel
    assert np.corrcoef(kernel, w_true)[0, 1] > 0.85

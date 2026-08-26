import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM, RW1, SpaceTime
from pylgm.exceptions import UnsupportedEngineError
from pylgm.priors import PCPrecision

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def _frame(seed=0):
    rows = [(s, t) for s in ["a", "b", "c"] for t in range(6)]
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": rng.normal(size=len(rows))}
    )


def test_type_i_accepted_under_full_laplace():
    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=hp),
        likelihood=Gaussian(sigma=0.5),
    )
    # Type I is proper/unconstrained -> full Laplace must not reject it.
    with pytest.warns(UserWarning, match=r"'st'"):
        result = model.fit(_frame(), hyperparameters="integrate", latent_strategy="laplace")
    assert result.latent_marginals("st").mean.shape == (3 * 6,)


@pytest.mark.parametrize("interaction,order", [("II", 1), ("III", 1), ("IV", 1)])
def test_types_ii_iii_iv_rejected_under_full_laplace(interaction, order):
    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    graph = None if interaction == "II" else GRAPH
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + SpaceTime("st", space="area", time="t", graph=graph, interaction=interaction, order=order, precision=hp),
        likelihood=Gaussian(sigma=0.5),
    )
    with pytest.raises(UnsupportedEngineError, match="constrained|Laplace"):
        model.fit(_frame(), hyperparameters="integrate", latent_strategy="laplace")


def test_type_iv_recovers_inseparable_field():
    # Simulate a smooth inseparable space-time field on the a-b-c path x 8 periods,
    # observe it with light noise, and check Type IV recovers it.
    areas, times = ["a", "b", "c"], list(range(8))
    rng = np.random.default_rng(7)
    coords = {"a": 0.0, "b": 1.0, "c": 2.0}
    rows, truth = [], []
    for s in areas:
        for t in times:
            # a travelling smooth bump: peak position moves with time
            value = np.exp(-0.6 * (coords[s] - 0.25 * t) ** 2)
            rows.append((s, t, value + rng.normal(scale=0.05)))
            truth.append(value)
    frame = pd.DataFrame(rows, columns=["area", "t", "y"])
    truth = np.array(truth)

    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    predictor = (
        Fixed("1")
        + Besag("area", index="area", graph=GRAPH)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV", order=1, precision=hp)
    )
    result = LGM(response="y", predictor=predictor, likelihood=Gaussian(sigma=0.05)).fit(frame)

    # The fitted linear predictor (intercept + area + period + interaction) should
    # track the truth. The point is a sane fit, not exact recovery.
    eta = result.predict(frame).predictive_mean
    centred_truth = truth - truth.mean()
    centred_eta = eta - eta.mean()
    assert np.corrcoef(centred_eta, centred_truth)[0, 1] > 0.9
    assert result.hyperparameters["st.precision"] > 0.0

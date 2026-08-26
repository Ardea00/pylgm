# tests/test_spacetime_compile.py
import warnings

import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, Gaussian, Hyperparameter, LGM, RW1, SpaceTime
from pylgm.priors import PCPrecision

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def _frame():
    rows = [(s, t) for s in ["a", "b", "c"] for t in range(5)]
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": rng.normal(size=len(rows))}
    )


def _full_predictor():
    return (
        Fixed("1")
        + Besag("area", index="area", graph=GRAPH)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV", order=1)
    )


def test_type_i_fixed_precision_fits():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=2.0),
        likelihood=Gaussian(sigma=0.5),
    )
    # A lone SpaceTime effect has no spatial/temporal main effects, so the
    # compiler correctly emits the missing-companion warning; capture it here
    # (rather than let it leak into pytest output) while still asserting the fit.
    with pytest.warns(UserWarning, match=r"'st'"):
        result = model.fit(_frame())
    assert result.latent_marginals("st").mean.shape == (3 * 5,)


def test_type_iv_estimates_tau():
    hp = Hyperparameter("st.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    predictor = (
        Fixed("1")
        + Besag("area", index="area", graph=GRAPH)
        + RW1("period", index="t")
        + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV", order=1, precision=hp)
    )
    result = LGM(response="y", predictor=predictor, likelihood=Gaussian(sigma=0.5)).fit(_frame())
    assert result.hyperparameters["st.precision"] > 0.0


def test_warns_when_a_main_effect_is_missing():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", graph=GRAPH, interaction="IV"),
        likelihood=Gaussian(sigma=0.5),
    )
    with pytest.warns(UserWarning, match="st"):
        model.fit(_frame())


def test_no_warning_when_both_main_effects_present():
    model = LGM(response="y", predictor=_full_predictor(), likelihood=Gaussian(sigma=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(_frame())  # must not raise a converted warning

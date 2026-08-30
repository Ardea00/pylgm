"""User-facing ergonomics: any effect is a predictor, and errors name the cause."""

import numpy as np
import pandas as pd
import pytest

from pylgm import (
    AR1, BYM2, Besag, Fixed, Gaussian, Hyperparameter, IID, LGM, MIDAS,
    ProperCAR, RW1, RW2, SAR, Seasonal, SpaceTime,
)
from pylgm.effects.spec import DynamicSpatialPanel, MIDASParametric, _ComposableEffect

_GRAPH = {"a": ["b"], "b": ["a"]}
_ALL_EFFECTS = (
    Fixed("1"),
    IID("i", "r"),
    RW1("w1", "t"),
    RW2("w2", "t"),
    AR1("a", "t"),
    Seasonal("s", "t", period=3),
    Besag("b", "r", _GRAPH),
    ProperCAR("p", "r", _GRAPH, rho=0.5),
    BYM2("y", "r", _GRAPH),
    SAR("sa", "r", _GRAPH, rho=0.5),
    DynamicSpatialPanel("d", "r", "t", {0: _GRAPH, 1: _GRAPH}, rho=0.5),
    MIDAS("m", ("x0", "x1", "x2")),
    MIDASParametric("mp", ("x0", "x1", "x2"), kernel="beta", shape1=1.5, shape2=1.5),
    SpaceTime("st", "r", "t", graph=_GRAPH, interaction="I"),
)


def test_every_effect_subclasses_the_composable_base():
    """The base class is what the predictor checks against, so drift here is
    what previously let listed-tuple checks go stale."""
    for effect in _ALL_EFFECTS:
        assert isinstance(effect, _ComposableEffect), type(effect).__name__


@pytest.mark.parametrize("effect", _ALL_EFFECTS, ids=lambda e: type(e).__name__)
def test_any_single_effect_is_a_valid_predictor(effect):
    model = LGM(response="y", predictor=effect, likelihood=Gaussian(sigma=1.0))
    assert model.predictor.effects == (effect,)


def test_predictor_rejects_a_non_effect_by_name():
    with pytest.raises(TypeError, match="got str"):
        LGM(response="y", predictor="nope", likelihood=Gaussian(sigma=1.0))


def test_missing_index_column_names_the_column_and_the_alternatives():
    frame = pd.DataFrame({"t": range(20), "y": np.zeros(20)})
    model = LGM(
        response="y", predictor=Fixed("1") + AR1("a", "tt"), likelihood=Gaussian(sigma=1.0)
    )
    with pytest.raises(Exception, match=r"column 'tt' is not in the data"):
        model.fit(frame)


def test_missing_index_column_message_hides_internal_columns():
    frame = pd.DataFrame({"t": range(20), "y": np.zeros(20)})
    model = LGM(
        response="y", predictor=Fixed("1") + AR1("a", "tt"), likelihood=Gaussian(sigma=1.0)
    )
    with pytest.raises(Exception) as error:
        model.fit(frame)
    assert "__pylgm" not in str(error.value)


def test_non_positive_initial_under_log_points_at_the_logit_transform():
    with pytest.raises(ValueError, match="transform='logit'"):
        Hyperparameter("a.rho", initial=0.0)
    with pytest.raises(ValueError, match="transform='logit'"):
        Hyperparameter("a.rho", initial=-0.4)


def test_logit_transform_accepts_a_zero_or_negative_initial():
    assert Hyperparameter("a.rho", initial=0.0, transform="logit").initial == 0.0
    assert Hyperparameter("a.rho", initial=-0.4, transform="logit").initial == -0.4


def test_duplicate_parameter_names_are_reported():
    frame = pd.DataFrame({"t": range(20), "y": np.linspace(0.0, 1.0, 20)})
    model = LGM(
        response="y",
        predictor=Fixed("1")
        + AR1("a", "t",
              precision=Hyperparameter("shared", initial=1.0),
              rho=Hyperparameter("shared", initial=0.0, transform="logit")),
        likelihood=Gaussian(sigma=0.3),
    )
    with pytest.raises(Exception, match=r"repeated: \['shared'\]"):
        model.fit(frame)

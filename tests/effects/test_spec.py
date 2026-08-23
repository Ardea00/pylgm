from dataclasses import FrozenInstanceError

import pytest

from pylgm import Fixed, IID, RW1
from pylgm.effects import RW2
from pylgm.parameters import Hyperparameter


def test_effects_compose_in_declaration_order():
    predictor = Fixed("1 + x") + IID("region", "region", 2.0) + RW1(
        "trend", "time", 3.0
    )

    assert [effect.name for effect in predictor.effects] == [
        "fixed",
        "region",
        "trend",
    ]


def test_predictor_rejects_duplicate_effect_names():
    with pytest.raises(ValueError, match="unique"):
        Fixed("1") + IID("fixed", "region")


def test_effect_declarations_are_immutable():
    effect = RW2("trend", "time", Hyperparameter("trend_precision", 3.0))

    with pytest.raises(FrozenInstanceError):
        effect.index = "other"  # type: ignore[misc]


PATH_GRAPH = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}


def test_besag_defaults():
    from pylgm.effects import Besag

    effect = Besag("region", index="region", graph=PATH_GRAPH)
    assert effect.name == "region"
    assert effect.index == "region"
    assert effect.precision == 1.0
    assert effect.scale is True


def test_besag_accepts_hyperparameter_precision():
    from pylgm.effects import Besag

    hp = Hyperparameter("region.precision", initial=1.0)
    effect = Besag("region", index="region", graph=PATH_GRAPH, precision=hp)
    assert effect.precision is hp


def test_besag_is_frozen_and_hashable():
    from pylgm.effects import Besag

    effect = Besag("region", index="region", graph=PATH_GRAPH)
    assert hash(effect) == hash(Besag("region", index="region", graph=PATH_GRAPH))
    with pytest.raises(FrozenInstanceError):
        effect.name = "x"


def test_besag_graph_roundtrips_to_dict():
    from pylgm.effects import Besag

    effect = Besag("region", index="region", graph=PATH_GRAPH)
    assert dict(effect.graph) == {"A": ("B",), "B": ("A", "C"), "C": ("B",)}


def test_besag_rejects_bad_graph():
    from pylgm.effects import Besag

    with pytest.raises(ValueError):
        Besag("region", index="region", graph={})


def test_besag_rejects_non_boolean_scale():
    from pylgm.effects import Besag

    with pytest.raises(ValueError, match="scale"):
        Besag("region", index="region", graph=PATH_GRAPH, scale="yes")


def test_besag_composes_in_predictor():
    from pylgm.effects import Besag, Predictor

    predictor = Fixed("1") + Besag("region", index="region", graph=PATH_GRAPH)
    assert isinstance(predictor, Predictor)
    assert [type(e).__name__ for e in predictor.effects] == ["Fixed", "Besag"]


def test_predictor_rejects_duplicate_besag_name():
    from pylgm.effects import Besag

    with pytest.raises(ValueError, match="unique"):
        IID("region", "region") + Besag("region", index="region", graph=PATH_GRAPH)


def test_proper_car_defaults():
    from pylgm.effects import ProperCAR

    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    assert effect.name == "region"
    assert effect.index == "region"
    assert effect.rho == 0.5
    assert effect.precision == 1.0


def test_proper_car_accepts_hyperparameter_precision():
    from pylgm.effects import ProperCAR

    hp = Hyperparameter("region.precision", initial=1.0)
    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5, precision=hp)
    assert effect.precision is hp


def test_proper_car_accepts_hyperparameter_rho():
    from pylgm.effects import ProperCAR

    hp = Hyperparameter("region.rho", initial=0.3, transform="logit")
    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=hp)
    assert effect.rho is hp


def test_proper_car_rejects_non_finite_rho():
    from pylgm.effects import ProperCAR

    with pytest.raises(ValueError, match="rho"):
        ProperCAR("region", index="region", graph=PATH_GRAPH, rho=float("inf"))


def test_proper_car_is_frozen_and_hashable():
    from pylgm.effects import ProperCAR

    effect = ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    assert hash(effect) == hash(
        ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    )
    with pytest.raises(FrozenInstanceError):
        effect.rho = 0.1


def test_proper_car_composes_in_predictor():
    from pylgm.effects import Predictor, ProperCAR

    predictor = Fixed("1") + ProperCAR("region", index="region", graph=PATH_GRAPH, rho=0.5)
    assert isinstance(predictor, Predictor)
    assert [type(e).__name__ for e in predictor.effects] == ["Fixed", "ProperCAR"]

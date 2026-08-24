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


def test_bym2_defaults():
    from pylgm.effects import BYM2

    effect = BYM2("region", index="region", graph=PATH_GRAPH)
    assert effect.name == "region"
    assert effect.phi == 0.5
    assert effect.precision == 1.0


def test_bym2_accepts_hyperparameter_phi_and_precision():
    from pylgm.effects import BYM2

    phi = Hyperparameter("region.phi", initial=0.5, transform="logit")
    tau = Hyperparameter("region.precision", initial=1.0)
    effect = BYM2("region", index="region", graph=PATH_GRAPH, phi=phi, precision=tau)
    assert effect.phi is phi
    assert effect.precision is tau


@pytest.mark.parametrize("phi", [-0.1, 0.0, 1.0, 1.5])
def test_bym2_rejects_phi_outside_open_unit_interval(phi):
    from pylgm.effects import BYM2

    with pytest.raises(ValueError, match="phi"):
        BYM2("region", index="region", graph=PATH_GRAPH, phi=phi)


def test_bym2_is_frozen_hashable_and_roundtrips_graph():
    from pylgm.effects import BYM2

    effect = BYM2("region", index="region", graph=PATH_GRAPH)
    assert hash(effect) == hash(BYM2("region", index="region", graph=PATH_GRAPH))
    assert dict(effect.graph) == {"A": ("B",), "B": ("A", "C"), "C": ("B",)}
    with pytest.raises(FrozenInstanceError):
        effect.phi = 0.1


def test_bym2_composes_in_predictor():
    from pylgm.effects import BYM2, Predictor

    predictor = Fixed("1") + BYM2("region", index="region", graph=PATH_GRAPH)
    assert isinstance(predictor, Predictor)
    assert [type(e).__name__ for e in predictor.effects] == ["Fixed", "BYM2"]


def test_ar1_defaults():
    from pylgm.effects import AR1

    effect = AR1("trend", index="t")
    assert effect.name == "trend"
    assert effect.index == "t"
    assert effect.precision == 1.0
    assert effect.rho == 0.5


def test_ar1_accepts_hyperparameter_rho_and_precision():
    from pylgm.effects import AR1

    rho = Hyperparameter("trend.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("trend.precision", initial=1.0)
    effect = AR1("trend", index="t", precision=tau, rho=rho)
    assert effect.rho is rho
    assert effect.precision is tau


@pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5, float("nan")])
def test_ar1_rejects_rho_outside_the_open_unit_interval(rho):
    from pylgm.effects import AR1

    with pytest.raises(ValueError, match="rho"):
        AR1("trend", index="t", rho=rho)


def test_ar1_is_frozen_and_hashable():
    from pylgm.effects import AR1

    effect = AR1("trend", index="t")
    assert hash(effect) == hash(AR1("trend", index="t"))
    with pytest.raises(FrozenInstanceError):
        effect.rho = 0.1


def test_ar1_composes_in_predictor():
    from pylgm.effects import AR1, Predictor

    predictor = Fixed("1") + AR1("trend", index="t")
    assert isinstance(predictor, Predictor)
    assert [type(e).__name__ for e in predictor.effects] == ["Fixed", "AR1"]

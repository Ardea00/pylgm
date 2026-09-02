import pytest

from pylgm import Besag, Fixed, IID, Weighted
from pylgm.effects.spec import Predictor


def test_weighted_delegates_its_name_to_the_inner_effect():
    inner = IID("u", index="district")
    assert Weighted(inner, by="z").name == "u"


def test_weighted_keeps_the_inner_effect_accessible():
    inner = IID("u", index="district")
    wrapped = Weighted(inner, by="z")
    assert wrapped.effect is inner
    assert wrapped.by == "z"


def test_weighted_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + Weighted(IID("u", index="district"), by="z")
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 2


def test_weighted_rejects_a_non_string_by():
    with pytest.raises((TypeError, ValueError), match="by"):
        Weighted(IID("u", index="district"), by=3)


def test_weighted_rejects_an_empty_by():
    with pytest.raises(ValueError, match="by"):
        Weighted(IID("u", index="district"), by="")


def test_weighted_rejects_an_effect_with_no_index():
    # Fixed builds its design from a formula, not an index, so weighting it is
    # meaningless -- multiply the covariate into the formula instead.
    with pytest.raises(TypeError, match="index"):
        Weighted(Fixed("1"), by="z")


def test_weighted_rejects_wrapping_a_weighted():
    # Two weight columns on one block is just their product; nesting would give
    # two ways to say one thing and complicate the prediction entry.
    inner = Weighted(IID("u", index="district"), by="z")
    with pytest.raises(TypeError, match="already weighted"):
        Weighted(inner, by="w")


def test_weighted_accepts_a_graph_effect():
    graph = {"a": ["b"], "b": ["a"]}
    wrapped = Weighted(Besag("u", index="district", graph=graph), by="z")
    assert wrapped.name == "u"

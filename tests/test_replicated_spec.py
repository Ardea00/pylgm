import pytest

from pylgm import AR1, Besag, Copy, Fixed, IID, Replicated, Weighted
from pylgm.effects.spec import Predictor


def test_replicated_delegates_its_name_to_the_inner_effect():
    assert Replicated(IID("u", index="t"), over="firm").name == "u"


def test_replicated_keeps_the_inner_effect_and_the_over_column():
    inner = IID("u", index="t")
    wrapped = Replicated(inner, over="firm")
    assert wrapped.effect is inner
    assert wrapped.over == "firm"


def test_replicated_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + Replicated(IID("u", index="t"), over="firm")
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 2


def test_replicated_rejects_a_non_string_over():
    with pytest.raises((TypeError, ValueError), match="over"):
        Replicated(IID("u", index="t"), over=3)


def test_replicated_rejects_an_empty_over():
    with pytest.raises(ValueError, match="over"):
        Replicated(IID("u", index="t"), over="")


def test_replicated_rejects_an_effect_with_no_index():
    with pytest.raises(TypeError, match="index"):
        Replicated(Fixed("1"), over="firm")


def test_replicated_rejects_wrapping_a_replicated():
    # Two replicate columns is one replicate over their cross product, so the
    # doubled form says nothing the single form cannot.
    with pytest.raises(TypeError, match="already replicated"):
        Replicated(Replicated(IID("u", index="t"), over="firm"), over="year")


def test_replicated_rejects_wrapping_a_copy():
    # A Copy is a term referencing another term, not an indexed effect of its own.
    with pytest.raises(TypeError):
        Replicated(Copy("u", index="j"), over="firm")


def test_replicated_rejects_an_ar1_that_already_replicates_itself():
    # AR1(replicate=) is the same concept; wrapping it would give two
    # replication mechanisms on one effect with no defined interaction.
    with pytest.raises(TypeError, match="replicate"):
        Replicated(AR1("t", index="year", replicate="firm"), over="country")


def test_replicated_may_wrap_a_weighted_effect():
    # Weighted touches only the design, Replicated only precision and indexing,
    # so the two compose; the spec asserts they commute.
    wrapped = Replicated(Weighted(IID("u", index="t"), by="z"), over="firm")
    assert wrapped.name == "u"


def test_replicated_accepts_a_constrained_effect():
    graph = {"a": ["b"], "b": ["a"]}
    assert Replicated(Besag("u", index="d", graph=graph), over="firm").name == "u"

def test_weighted_still_exposes_no_index_attribute():
    """Pins the invariant joint.Shared relies on.

    Shared distinguishes "a wrapper, which cannot be shared" from "no index at
    all" by `hasattr(effect, "index")`. Giving Weighted an index of its own --
    the obvious way to let Replicated wrap it -- silently turns that guard into
    dead code, and Shared(Weighted(...)) starts being accepted. Replicated
    unwraps locally instead; this stops the shortcut being reintroduced.
    """
    assert not hasattr(Weighted(IID("u", index="t"), by="z"), "index")

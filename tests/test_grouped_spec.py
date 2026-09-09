import pytest

from pylgm import (
    AR1, BesagStructure, Copy, Fixed, Grouped, IID, IIDStructure, Replicated,
    RW1Structure, Weighted,
)
from pylgm.effects.spec import Predictor
from pylgm.joint import Shared

GRAPH = {"a": ["b"], "b": ["a"]}


def test_grouped_delegates_its_name_to_the_inner_effect():
    assert Grouped(IID("u", index="t"), over="region", structure=RW1Structure()).name == "u"


def test_grouped_keeps_the_effect_the_column_and_the_structure():
    inner, structure = IID("u", index="t"), RW1Structure()
    wrapped = Grouped(inner, over="region", structure=structure)
    assert wrapped.effect is inner
    assert wrapped.over == "region"
    assert wrapped.structure is structure


def test_grouped_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + Grouped(IID("u", index="t"), over="r", structure=IIDStructure())
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 2


def test_grouped_rejects_an_empty_over():
    with pytest.raises(ValueError, match="over"):
        Grouped(IID("u", index="t"), over="", structure=IIDStructure())


def test_grouped_rejects_an_effect_with_no_index():
    with pytest.raises(TypeError, match="index"):
        Grouped(Fixed("1"), over="r", structure=IIDStructure())


def test_grouped_rejects_something_that_is_not_a_structure():
    with pytest.raises(TypeError, match="structure"):
        Grouped(IID("u", index="t"), over="r", structure="besag")


def test_grouped_rejects_wrapping_a_grouped():
    with pytest.raises(TypeError, match="already grouped"):
        Grouped(
            Grouped(IID("u", index="t"), over="r", structure=IIDStructure()),
            over="year", structure=IIDStructure(),
        )


def test_grouped_rejects_wrapping_a_replicated():
    """R-INLA permits group and replicate together; pyLGM does not.

    The labels would become r@g@level, and both _prediction_entry's
    split("@", 1) and the single-inner-index assumption would have to be
    generalised. Recorded as an f() parity gap, not half-implemented.
    """
    with pytest.raises(TypeError, match="Use one or the other"):
        Grouped(
            Replicated(IID("u", index="t"), over="firm"),
            over="year", structure=IIDStructure(),
        )


def test_replicated_rejects_wrapping_a_grouped():
    with pytest.raises(TypeError, match="Use one or the other"):
        Replicated(
            Grouped(IID("u", index="t"), over="r", structure=IIDStructure()),
            over="firm",
        )


def test_grouped_rejects_an_ar1_that_already_replicates_itself():
    with pytest.raises(TypeError, match="replicate"):
        Grouped(AR1("t", index="year", replicate="firm"), over="r", structure=IIDStructure())


def test_grouped_rejects_a_weighted_ar1_that_already_replicates_itself():
    with pytest.raises(TypeError, match="replicate"):
        Grouped(
            Weighted(AR1("t", index="year", replicate="firm"), by="z"),
            over="r", structure=IIDStructure(),
        )


def test_grouped_rejects_wrapping_a_copy():
    with pytest.raises(TypeError, match="Copy"):
        Grouped(Copy("u", index="j"), over="r", structure=IIDStructure())


def test_grouped_may_wrap_a_weighted_effect():
    wrapped = Grouped(
        Weighted(IID("u", index="t"), by="z"), over="r", structure=BesagStructure(GRAPH)
    )
    assert wrapped.name == "u"


def test_weighted_may_wrap_a_grouped_effect():
    wrapped = Weighted(
        Grouped(IID("u", index="t"), over="r", structure=IIDStructure()), by="z"
    )
    assert wrapped.name == "u"


def test_grouped_has_no_index_so_shared_s_wrapper_guard_stays_alive():
    """joint.Shared tells "wrapper" from "no index at all" by hasattr(effect, "index").

    A previous slice gave a wrapper an index as a shortcut and silently turned
    that guard into dead code. This pins the shape, not just the behaviour.
    """
    grouped = Grouped(IID("u", index="t"), over="r", structure=IIDStructure())
    assert not hasattr(grouped, "index")
    with pytest.raises(TypeError, match="index"):
        Shared(grouped)

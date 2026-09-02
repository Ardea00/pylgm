import pytest

from pylgm import Copy, Fixed, IID, Weighted
from pylgm.effects.spec import Predictor
from pylgm.parameters import Hyperparameter


def test_copy_carries_the_target_name_the_index_and_the_scale():
    copy = Copy("u", index="j", scale=2.0)
    assert copy.name == "u"
    assert copy.index == "j"
    assert copy.scale == 2.0


def test_copy_scale_defaults_to_one():
    assert Copy("u", index="j").scale == 1.0


def test_copy_accepts_a_hyperparameter_scale():
    beta = Hyperparameter("beta", initial=1.0)
    assert Copy("u", index="j", scale=beta).scale is beta


def test_copy_composes_with_plus_like_any_effect():
    predictor = Fixed("1") + IID("u", index="i") + Copy("u", index="j")
    assert isinstance(predictor, Predictor)
    assert len(predictor.effects) == 3


def test_copy_rejects_an_empty_target_name():
    with pytest.raises(ValueError, match="name"):
        Copy("", index="j")


def test_copy_rejects_an_empty_index():
    with pytest.raises(ValueError, match="index"):
        Copy("u", index="")


def test_copy_rejects_a_non_numeric_non_hyperparameter_scale():
    with pytest.raises(TypeError, match="scale"):
        Copy("u", index="j", scale="2.0")


def test_copy_cannot_be_wrapped_by_weighted():
    # Copy is a term referencing another term, not an indexed effect of its own,
    # so wrapping it has no meaning: weight the target instead.
    with pytest.raises(TypeError):
        Weighted(Copy("u", index="j"), by="z")

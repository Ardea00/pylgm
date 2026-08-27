# tests/test_spacetime_spec.py
import pytest

from pylgm import SpaceTime
from pylgm.parameters import Hyperparameter


def test_defaults_and_graph_canonicalization():
    st = SpaceTime("st", space="area", time="t", graph={"a": ["b"], "b": ["a"]})
    assert st.name == "st"
    assert st.space == "area"
    assert st.time == "t"
    assert st.interaction == "IV"
    assert st.order == 1
    assert st.scale is True
    assert dict(st.graph) == {"a": ("b",), "b": ("a",)}


def test_graph_optional_for_types_i_ii():
    st = SpaceTime("st", space="area", time="t", interaction="II")
    assert st.graph is None


def test_hyperparameter_precision_allowed():
    hp = Hyperparameter("st.precision", initial=1.0)
    st = SpaceTime("st", space="area", time="t", interaction="I", precision=hp)
    assert st.precision is hp


@pytest.mark.parametrize("bad", ["V", "iv", "", 1])
def test_rejects_bad_interaction(bad):
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", graph={"a": ["b"], "b": ["a"]}, interaction=bad)


@pytest.mark.parametrize("bad", [0, 3, 1.5])
def test_rejects_bad_order(bad):
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", graph={"a": ["b"], "b": ["a"]}, order=bad)


def test_rejects_missing_graph_for_iii_iv():
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", interaction="III")
    with pytest.raises(ValueError):
        SpaceTime("st", space="area", time="t", interaction="IV")


def test_composes_with_plus():
    from pylgm import Fixed

    predictor = Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I")
    assert predictor.effects[-1].name == "st"

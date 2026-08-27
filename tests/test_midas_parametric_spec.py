import pytest

from pylgm import Fixed, MIDASParametric
from pylgm.parameters import Hyperparameter


def test_beta_defaults_are_positive_log_hyperparameters():
    e = MIDASParametric("m", ("x0", "x1", "x2"), kernel="beta")
    assert e.name == "m" and e.kernel == "beta"
    for shape in (e.shape1, e.shape2):
        assert isinstance(shape, Hyperparameter)
        assert shape.transform == "log" and shape.initial == 2.0
    assert e.shape1.name == "m.shape1" and e.shape2.name == "m.shape2"
    assert e.prior_precision == 1e-6


def test_exp_almon_defaults_are_identity_hyperparameters():
    e = MIDASParametric("m", ("x0", "x1"), kernel="exp_almon")
    assert e.shape1.transform == "identity" and e.shape1.initial == 0.0
    assert e.shape2.transform == "identity" and e.shape2.initial == -0.1


def test_fixed_float_shapes_are_kept():
    e = MIDASParametric("m", ("x0", "x1"), kernel="exp_almon", shape1=0.2, shape2=-0.4)
    assert e.shape1 == 0.2 and e.shape2 == -0.4


def test_validation_rejects_bad_kernel_and_short_columns_and_precision():
    with pytest.raises(ValueError, match="kernel"):
        MIDASParametric("m", ("x0", "x1"), kernel="nope")
    with pytest.raises(ValueError, match="at least 2|columns"):
        MIDASParametric("m", ("x0",))
    with pytest.raises(ValueError):
        MIDASParametric("m", ("x0", "x1"), prior_precision=0.0)


def test_composes_with_other_effects():
    predictor = Fixed("1") + MIDASParametric("m", ("x0", "x1"))
    names = [e.name for e in predictor.effects]
    assert "m" in names and "fixed" in names

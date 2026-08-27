import numpy as np
import pytest

from pylgm.optimization.transforms import IdentityTransform, Transform
from pylgm.parameters import Hyperparameter


def test_identity_transform_is_a_transform_and_round_trips():
    t = IdentityTransform()
    assert isinstance(t, Transform)
    for x in (-3.5, 0.0, 2.7):
        assert t.from_internal(t.to_internal(x)) == pytest.approx(x)
        assert t.log_abs_jacobian(x) == 0.0
    assert t.contains(1.0) and not t.contains(np.inf)
    assert IdentityTransform() == IdentityTransform()
    assert hash(IdentityTransform()) == hash(IdentityTransform())


def test_identity_hyperparameter_accepts_nonpositive_initial():
    hp = Hyperparameter("k.shape2", initial=-0.1, transform="identity")
    assert hp.initial == pytest.approx(-0.1)
    assert hp.lower < hp.initial < hp.upper
    assert np.isfinite(hp.lower) and np.isfinite(hp.upper)


def test_identity_hyperparameter_accepts_zero_initial_and_custom_bounds():
    hp = Hyperparameter("k.shape1", initial=0.0, transform="identity", lower=-5.0, upper=5.0)
    assert (hp.lower, hp.initial, hp.upper) == (-5.0, 0.0, 5.0)

import numpy as np
import pytest

from pylgm.optimization.transforms import LogitTransform, LogTransform


def _fd_log_jac(transform, u, h=1e-6):
    # finite-difference of log|d theta / d u|
    dtheta = (transform.from_internal(u + h) - transform.from_internal(u - h)) / (2 * h)
    return np.log(abs(dtheta))


def test_log_roundtrip_and_jacobian():
    t = LogTransform()
    for theta in (1e-3, 1.0, 25.0, 1e3):
        assert np.isclose(t.from_internal(t.to_internal(theta)), theta)
    for u in (-5.0, 0.0, 3.0):
        assert np.isclose(t.log_abs_jacobian(u), u)              # log path anchor
        assert np.isclose(t.log_abs_jacobian(u), _fd_log_jac(t, u), atol=1e-5)


def test_log_domain():
    t = LogTransform()
    assert t.contains(0.5)
    assert not t.contains(0.0)
    assert not t.contains(-1.0)


def test_logit_roundtrip_and_range():
    t = LogitTransform(-1.0, 1.0)
    for theta in (-0.9, -0.3, 0.0, 0.4, 0.95):
        assert np.isclose(t.from_internal(t.to_internal(theta)), theta)
    for u in (-8.0, -1.0, 0.0, 2.0, 9.0):
        theta = t.from_internal(u)
        assert -1.0 < theta < 1.0


def test_logit_jacobian_matches_finite_difference():
    t = LogitTransform(0.0, 1.0)
    for u in (-3.0, -0.5, 0.0, 1.5, 4.0):
        assert np.isclose(t.log_abs_jacobian(u), _fd_log_jac(t, u), atol=1e-5)


def test_logit_domain_and_validation():
    t = LogitTransform(-1.0, 1.0)
    assert t.contains(0.0)
    assert not t.contains(-1.0)
    assert not t.contains(1.0)
    assert not t.contains(2.0)
    with pytest.raises(ValueError):
        LogitTransform(1.0, 1.0)
    with pytest.raises(ValueError):
        LogitTransform(2.0, 1.0)


def test_logit_rejects_non_finite_bounds():
    for a, b in ((float("nan"), 1.0), (0.0, float("inf")), (float("-inf"), 0.0)):
        with pytest.raises(ValueError):
            LogitTransform(a, b)


def test_logit_jacobian_stable_in_tails():
    # log_expit keeps the Jacobian finite at large |u| where a naive
    # sigmoid(u)*(1-sigmoid(u)) underflows to 0 and log -> -inf.
    t = LogitTransform(0.0, 1.0)
    for u in (-100.0, -40.0, 40.0, 100.0, 700.0):
        assert np.isfinite(t.log_abs_jacobian(u))

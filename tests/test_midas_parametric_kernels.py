# tests/test_midas_parametric_kernels.py
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.midas import build_midas_parametric, midas_weights


def test_weights_sum_to_one_both_kernels():
    for kernel, theta in (("beta", (2.0, 2.0)), ("exp_almon", (0.0, -0.5))):
        w = midas_weights(kernel, 8, theta)
        assert w.shape == (8,)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_exp_almon_decays_when_quadratic_negative():
    w = midas_weights("exp_almon", 8, (0.0, -0.3))
    assert np.all(np.diff(w) < 0)  # strictly decreasing over lag index


def test_beta_has_interior_hump_when_symmetric():
    w = midas_weights("beta", 9, (2.0, 2.0))
    assert np.argmax(w) == 4  # symmetric interior peak on a 9-point grid


def test_extreme_theta_is_overflow_safe():
    w = midas_weights("exp_almon", 12, (5.0, 0.0))  # large positive slope
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)


def test_builder_produces_aggregated_column():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(6, 3))
    frame = pd.DataFrame({f"x{k}": V[:, k] for k in range(3)})
    theta = (1.0, 2.0)
    block = build_midas_parametric(frame, "m", ("x0", "x1", "x2"), "beta", theta, 1e-6)
    w = midas_weights("beta", 3, theta)
    np.testing.assert_allclose(block.design.toarray().ravel(), V @ w)
    assert block.labels == ("m",)
    np.testing.assert_allclose(block.precision.toarray(), [[1e-6]])
    assert block.constraints.shape == (0, 1)


def test_builder_rejects_missing_and_nonfinite_columns():
    frame = pd.DataFrame({"x0": [1.0, 2.0], "x1": [np.nan, 1.0]})
    with pytest.raises(ValueError, match="missing"):
        build_midas_parametric(frame, "m", ("x0", "zzz"), "beta", (2.0, 2.0), 1e-6)
    with pytest.raises(ValueError, match="non-finite|finite"):
        build_midas_parametric(frame, "m", ("x0", "x1"), "beta", (2.0, 2.0), 1e-6)

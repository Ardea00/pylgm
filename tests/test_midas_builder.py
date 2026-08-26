import numpy as np
import pandas as pd
import pytest

from pylgm import MIDAS
from pylgm.effects.midas import build_midas, midas_penalty
from pylgm.effects.random_walk import difference_operator


def _frame(n=8, k=5):
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"lag{j}": rng.normal(size=n) for j in range(k)})


def _second_difference(k):
    d = np.zeros((k - 2, k))
    for i in range(k - 2):
        d[i, i], d[i, i + 1], d[i, i + 2] = 1.0, -2.0, 1.0
    return d


def _null_projector(order, k):
    columns = [np.ones(k)] if order == 1 else [np.ones(k), np.arange(k, dtype=float)]
    t = np.column_stack(columns)
    return t @ np.linalg.pinv(t)


def test_difference_operator_matches_hand_built_second_difference():
    got = difference_operator(6, 2).toarray()
    assert np.allclose(got, _second_difference(6))


def test_design_is_the_stacked_lag_columns():
    frame = _frame()
    columns = ("lag0", "lag1", "lag2", "lag3", "lag4")
    block = build_midas(frame, "m", columns, precision=1.0, order=2, ridge=1e-6)
    assert block.design.shape == (len(frame), len(columns))
    assert np.allclose(block.design.toarray(), frame[list(columns)].to_numpy(float))


def test_order2_precision_is_tau_curvature_plus_delta_nullspace():
    frame = _frame(k=5)
    columns = tuple(f"lag{j}" for j in range(5))
    tau, delta = 3.0, 1e-6
    block = build_midas(frame, "m", columns, precision=tau, order=2, ridge=delta)
    q = block.precision.toarray()
    d2 = _second_difference(5)
    expected = tau * (d2.T @ d2) + delta * _null_projector(2, 5)
    assert np.allclose(q, expected)
    assert np.allclose(q, q.T)
    assert np.linalg.eigvalsh(q).min() > 0  # PD thanks to the ridge


def test_smoothing_precision_never_touches_level_or_slope():
    # A linear lag curve lives in null(D2); its penalty must be delta*||b||^2,
    # independent of tau. This is the load-bearing B property.
    frame = _frame(k=6)
    columns = tuple(f"lag{j}" for j in range(6))
    beta = 0.7 + 0.3 * np.arange(6)  # a + b*k, in null(D2)
    delta = 1e-6
    energies = []
    for tau in (1.0, 1000.0):
        q = build_midas(frame, "m", columns, precision=tau, order=2, ridge=delta).precision
        energies.append(beta @ (q @ beta))
    assert np.allclose(energies[0], energies[1])
    assert np.allclose(energies[0], delta * (beta @ beta))


def test_order1_uses_first_difference_and_mean_projector():
    dtd, p0 = midas_penalty(5, 1)
    d1 = difference_operator(5, 1).toarray()
    assert np.allclose(dtd.toarray(), d1.T @ d1)
    assert np.allclose(p0.toarray() if hasattr(p0, "toarray") else p0, np.full((5, 5), 1 / 5))


def test_block_is_unconstrained():
    frame = _frame()
    columns = ("lag0", "lag1", "lag2")
    block = build_midas(frame, "m", columns, precision=1.0, order=2, ridge=1e-6)
    assert block.constraints.shape == (0, len(columns))


def test_too_few_columns_for_order_raises():
    frame = _frame(k=3)
    with pytest.raises(ValueError):
        build_midas(frame, "m", ("lag0", "lag1"), precision=1.0, order=2, ridge=1e-6)


def test_missing_column_raises():
    frame = _frame(k=3)
    with pytest.raises((KeyError, ValueError)):
        build_midas(frame, "m", ("lag0", "nope", "lag2"), precision=1.0, order=2, ridge=1e-6)


def test_nan_in_lag_column_raises_naming_it():
    frame = _frame(k=4)
    frame.loc[2, "lag1"] = np.nan
    with pytest.raises(ValueError, match="lag1"):
        build_midas(frame, "m", ("lag0", "lag1", "lag2", "lag3"), precision=1.0, order=2, ridge=1e-6)


def test_midas_spec_normalises_columns_and_composes():
    spec = MIDAS("lag", columns=["a", "b", "c"], precision=2.0)
    assert spec.columns == ("a", "b", "c")
    assert spec.order == 2 and spec.ridge == 1e-6
    predictor = spec + MIDAS("lag2", columns=("x", "y"), order=1)
    assert [effect.name for effect in predictor.effects] == ["lag", "lag2"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"columns": ("a", "b"), "order": 2},  # too few for order
        {"columns": ("a", "b", "c"), "order": 3},  # order not in {1,2}
        {"columns": ("a", "b", "c"), "ridge": 0.0},  # ridge not positive
    ],
)
def test_midas_spec_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        MIDAS("lag", **kwargs)

from math import comb

import numpy as np
import pandas as pd
import pytest

from pylgm import Gaussian, IID, LGM, LinearObservation
from pylgm.operators import (
    aggregation_operator,
    compose,
    cumulation_operator,
    difference_operator,
)


# --------------------------------------------------------------------------
# aggregation_operator
# --------------------------------------------------------------------------


def _aggregation_frame():
    # Caller order shuffled relative to sorted (g, p) key order.
    return pd.DataFrame(
        {
            "g": ["y", "x", "x", "y", "x", "y"],
            "p": [1, 2, 1, 2, 1, 1],
            "v": [60.0, 20.0, 10.0, 40.0, 50.0, 30.0],
        }
    )


def test_aggregation_operator_matches_hand_built_matrix_and_keys():
    frame = _aggregation_frame()
    operator, keys = aggregation_operator(frame, ["g", "p"])

    # frame rows: (y,1) (x,2) (x,1) (y,2) (x,1) (y,1)
    # Distinct (g, p) keys, sorted ascending: (x,1), (x,2), (y,1), (y,2).
    expected = np.array(
        [
            [0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(operator.toarray(), expected)
    pd.testing.assert_frame_equal(
        keys,
        pd.DataFrame({"g": ["x", "x", "y", "y"], "p": [1, 2, 1, 2]}),
    )


def test_aggregation_operator_weights_column():
    frame = _aggregation_frame()
    operator, keys = aggregation_operator(frame, "g", weights="v")

    expected = np.array(
        [
            [0.0, 20.0, 10.0, 0.0, 50.0, 0.0],
            [60.0, 0.0, 0.0, 40.0, 0.0, 30.0],
        ]
    )
    np.testing.assert_array_equal(operator.toarray(), expected)
    pd.testing.assert_frame_equal(keys, pd.DataFrame({"g": ["x", "y"]}))


def test_aggregation_operator_rows_mask_drops_a_whole_group():
    frame = _aggregation_frame()
    rows = (frame["g"] != "y").to_numpy()
    operator, keys = aggregation_operator(frame, "g", rows=rows)

    expected = np.array([[0, 1, 1, 0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(operator.toarray(), expected)
    pd.testing.assert_frame_equal(keys, pd.DataFrame({"g": ["x"]}))


def test_aggregation_operator_missing_column_raises():
    with pytest.raises(ValueError, match="not found"):
        aggregation_operator(_aggregation_frame(), "missing")


def test_aggregation_operator_nan_in_by_column_raises():
    frame = _aggregation_frame()
    frame.loc[0, "g"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        aggregation_operator(frame, "g")


def test_aggregation_operator_wrong_length_weights_raises():
    with pytest.raises(ValueError, match="weights"):
        aggregation_operator(_aggregation_frame(), "g", weights=[1.0, 2.0])


def test_aggregation_operator_wrong_length_rows_raises():
    with pytest.raises(ValueError, match="rows"):
        aggregation_operator(_aggregation_frame(), "g", rows=[True, False])


def test_aggregation_operator_non_finite_weights_raises():
    with pytest.raises(ValueError, match="finite"):
        aggregation_operator(_aggregation_frame(), "g", weights=[1.0, np.nan, 1.0, 1.0, 1.0, 1.0])


def test_aggregation_operator_zero_selected_rows_raises():
    frame = _aggregation_frame()
    with pytest.raises(ValueError, match="zero rows"):
        aggregation_operator(frame, "g", rows=np.zeros(len(frame), dtype=bool))


# --------------------------------------------------------------------------
# difference_operator / cumulation_operator shared fixture
# --------------------------------------------------------------------------


def _panel_frame():
    # 2 units x 5 periods, caller order shuffled.
    rows = [
        {"unit": unit, "t": t, "x": base + t}
        for unit, base in (("a", 10.0), ("b", 100.0))
        for t in range(5)
    ]
    frame = pd.DataFrame(rows)
    shuffle = [7, 0, 4, 2, 9, 1, 8, 3, 6, 5]
    return frame.iloc[shuffle].reset_index(drop=True)


def _expected_difference(frame, lag, order):
    """Hand-built dense (operator, keys) by looping per-unit in numpy."""
    units = sorted(frame["unit"].unique())
    n = len(frame)
    op_rows = []
    key_rows = []
    for unit in units:
        positions = frame.index[frame["unit"] == unit].to_numpy()
        order_by_time = positions[np.argsort(frame.loc[positions, "t"].to_numpy(), kind="stable")]
        m = len(order_by_time)
        for i in range(lag * order, m):
            row = np.zeros(n)
            for k in range(order + 1):
                coeff = ((-1) ** k) * comb(order, k)
                row[order_by_time[i - k * lag]] = coeff
            op_rows.append(row)
            key_rows.append((unit, frame.loc[order_by_time[i], "t"]))
    return np.array(op_rows), pd.DataFrame(key_rows, columns=["unit", "t"])


@pytest.mark.parametrize("lag,order", [(1, 1), (4, 1), (1, 2)])
def test_difference_operator_matches_hand_built_matrix(lag, order):
    frame = _panel_frame()
    operator, keys = difference_operator(frame, "t", "unit", lag=lag, order=order)
    expected_operator, expected_keys = _expected_difference(frame, lag, order)

    np.testing.assert_array_equal(operator.toarray(), expected_operator)
    pd.testing.assert_frame_equal(keys.reset_index(drop=True), expected_keys)


def test_difference_operator_applies_to_known_vector():
    frame = _panel_frame()
    operator, keys = difference_operator(frame, "t", "unit", lag=1, order=1)
    x = frame["x"].to_numpy()
    result = operator @ x

    # x is base + t, so lag-1 differences are all 1.0.
    np.testing.assert_allclose(result, np.ones(len(keys)))


def test_difference_operator_duplicate_time_raises():
    frame = _panel_frame()
    frame.loc[frame.index[0], "t"] = frame.loc[frame.index[1], "t"]
    with pytest.raises(ValueError, match="duplicate"):
        difference_operator(frame, "t", "unit")


def test_difference_operator_bad_lag_or_order_raises():
    frame = _panel_frame()
    with pytest.raises(ValueError):
        difference_operator(frame, "t", "unit", lag=0)
    with pytest.raises(ValueError):
        difference_operator(frame, "t", "unit", order=0)


# --------------------------------------------------------------------------
# cumulation_operator
# --------------------------------------------------------------------------


def test_cumulation_operator_matches_per_unit_cumsum():
    frame = _panel_frame()
    operator = cumulation_operator(frame, "t", "unit")
    x = frame["x"].to_numpy()
    result = operator @ x

    expected = np.empty(len(frame))
    for unit in frame["unit"].unique():
        positions = frame.index[frame["unit"] == unit].to_numpy()
        order = positions[np.argsort(frame.loc[positions, "t"].to_numpy(), kind="stable")]
        expected[order] = np.cumsum(x[order])

    np.testing.assert_allclose(result, expected)


def test_cumulation_operator_strided_lag_matches_strided_cumsum():
    frame = _panel_frame()
    operator = cumulation_operator(frame, "t", "unit", lag=2)
    x = frame["x"].to_numpy()
    result = operator @ x

    expected = np.empty(len(frame))
    for unit in frame["unit"].unique():
        positions = frame.index[frame["unit"] == unit].to_numpy()
        order = positions[np.argsort(frame.loc[positions, "t"].to_numpy(), kind="stable")]
        values = x[order]
        strided = np.zeros_like(values)
        for lag_class in range(2):
            members = np.arange(lag_class, len(values), 2)
            strided[members] = np.cumsum(values[members])
        expected[order] = strided

    np.testing.assert_allclose(result, expected)


def test_difference_and_cumulation_are_right_inverses_past_the_first_position():
    frame = _panel_frame()
    difference, keys = difference_operator(frame, "t", "unit", lag=1, order=1)
    cumulation = cumulation_operator(frame, "t", "unit", lag=1)
    product = (difference @ cumulation).toarray()

    # Each difference row targets position i>=1 of its unit; D @ C selects
    # that row of the identity.
    expected = np.zeros((len(keys), len(frame)))
    for r, (unit, t) in keys.iterrows():
        target = frame.index[(frame["unit"] == unit) & (frame["t"] == t)][0]
        expected[r, target] = 1.0
    np.testing.assert_allclose(product, expected)


# --------------------------------------------------------------------------
# compose
# --------------------------------------------------------------------------


def test_compose_shape_mismatch_names_both_shapes():
    with pytest.raises(ValueError, match=r"\(2, 2\).*\(3, 3\)"):
        compose(np.eye(2), np.eye(3))


def test_compose_result_equals_dense_product():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.eye(2) * 2.0

    result = compose(a, b, c)
    np.testing.assert_allclose(result.toarray(), a @ b @ c)


# --------------------------------------------------------------------------
# End-to-end: differences grid, level totals per unit as a linear observation
# --------------------------------------------------------------------------


def test_composed_operator_recovers_level_totals_in_a_fit():
    rows = [
        {"row": f"{unit}_{t}", "unit": unit, "t": t, "d": 0.0}
        for unit in ("u1", "u2")
        for t in range(6)
    ]
    frame = pd.DataFrame(rows)
    shuffle = [7, 0, 4, 2, 9, 1, 8, 3, 6, 5, 10, 11]
    frame = frame.iloc[shuffle].reset_index(drop=True)

    model = LGM(
        response="y",
        likelihood=Gaussian(1.0),
        predictor=IID("d", index="row", precision=1.0),
        panel=("row",),
    )
    operator = compose(
        aggregation_operator(frame, "unit")[0],
        cumulation_operator(frame, "t", "unit"),
    )
    observation = LinearObservation([30.0, -12.0], operator, sigma=1e-4)
    result = model.fit(frame, engine="exact_gaussian", observations=[observation])

    np.testing.assert_allclose(operator @ result.predictive_mean, [30.0, -12.0], atol=1e-3)

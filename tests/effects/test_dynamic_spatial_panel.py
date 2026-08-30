import numpy as np
import pandas as pd
import pytest

from pylgm.effects.sar import build_dynamic_spatial_panel, build_sar


def _graph():
    return {"a": ["b"], "b": ["c"], "c": ["a"]}


def _panel_frame():
    rows = [(u, t) for t in ("1", "2") for u in ("a", "b", "c")]
    return pd.DataFrame(
        {"unit": [u for u, _ in rows], "period": [t for _, t in rows], "y": np.arange(6.0)}
    )


def test_t1_reduces_to_static_sar():
    frame = pd.DataFrame({"unit": ["a", "b", "c"], "period": ["1", "1", "1"], "y": [1.0, 2.0, 3.0]})
    panel = build_dynamic_spatial_panel(
        frame, "d", "unit", "period", {"1": _graph()}, rho=0.4, gamma=0.0, eta=0.0, precision=2.0
    )
    static = build_sar(pd.DataFrame({"region": ["a", "b", "c"]}), "s", "region", _graph(), 0.4, 2.0)
    assert np.allclose(panel.precision.toarray(), static.precision.toarray())


def test_precision_is_symmetric_pd():
    panel = build_dynamic_spatial_panel(
        _panel_frame(), "d", "unit", "period",
        {"1": _graph(), "2": _graph()}, rho=0.5, gamma=0.3, eta=0.2, precision=1.0,
    )
    q = panel.precision.toarray()
    assert q.shape == (6, 6)
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) > 0)


def test_labels_are_time_major():
    panel = build_dynamic_spatial_panel(
        _panel_frame(), "d", "unit", "period",
        {"1": _graph(), "2": _graph()}, rho=0.3, gamma=0.0, eta=0.0, precision=1.0,
    )
    assert panel.labels == ("a@1", "b@1", "c@1", "a@2", "b@2", "c@2")


def test_design_maps_unit_time_to_time_major_cell():
    frame = _panel_frame()
    panel = build_dynamic_spatial_panel(
        frame, "d", "unit", "period",
        {"1": _graph(), "2": _graph()}, rho=0.3, gamma=0.0, eta=0.0, precision=1.0,
    )
    design = panel.design.toarray()
    # row 3 is (a, period 2) -> column time_pos(1)*3 + unit_pos(0) = 3
    assert design[3, 3] == 1.0
    assert design.sum() == 6.0


def test_eta_zero_is_ar_link_block_bidiagonal():
    # gamma!=0, eta=0: sub-diagonal blocks are -gamma*I (pure temporal AR link)
    from pylgm.effects.sar import _panel_networks, _sdpd_operator

    _, _, ws = _panel_networks({"1": _graph(), "2": _graph()})
    m = _sdpd_operator(ws, rho=0.0, gamma=0.5, eta=0.0).toarray()
    n = 3
    assert np.allclose(m[n:, :n], -0.5 * np.eye(n))  # sub-diagonal = -gamma*I


def test_unobserved_time_without_graph_rejected():
    frame = pd.DataFrame({"unit": ["a"], "period": ["9"], "y": [1.0]})
    with pytest.raises(ValueError, match="time"):
        build_dynamic_spatial_panel(
            frame, "d", "unit", "period", {"1": _graph()}, rho=0.3, gamma=0.0, eta=0.0, precision=1.0
        )


def test_numeric_period_labels_sort_numerically_not_lexically():
    # Panel keyed "1".."11": lexicographic order would place "10","11" before "2",
    # silently mis-specifying SDPD temporal adjacency. Must be true numeric order.
    from pylgm.effects.sar import _panel_networks

    graphs = {str(t): {"a": ["b"], "b": ["a"]} for t in range(1, 12)}
    _units, times, _ws = _panel_networks(graphs)
    assert times == tuple(str(t) for t in range(1, 12))

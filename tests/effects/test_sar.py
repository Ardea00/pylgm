import numpy as np
import pandas as pd
import pytest

from pylgm.effects.directed_graph import normalize_directed_graph, row_standardize
from pylgm.effects.sar import build_sar


def _graph():
    # directed ring a->b->c->a, plus a->c
    return {"a": ["b", "c"], "b": ["c"], "c": ["a"]}


def _frame():
    return pd.DataFrame({"region": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})


def test_precision_is_gram_of_i_minus_rho_w():
    rho, precision = 0.4, 2.0
    block = build_sar(_frame(), "s", "region", _graph(), rho, precision)
    nodes, w = normalize_directed_graph(_graph())
    w = row_standardize(w)
    m = np.eye(len(nodes)) - rho * w.toarray()
    expected = precision * (m.T @ m)
    assert np.allclose(block.precision.toarray(), expected)


def test_precision_is_symmetric_and_positive_definite():
    block = build_sar(_frame(), "s", "region", _graph(), 0.6, 1.0)
    q = block.precision.toarray()
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) > 0)


def test_constraints_are_empty_proper_field():
    block = build_sar(_frame(), "s", "region", _graph(), 0.0, 1.0)
    assert block.constraints.shape == (0, 3)


def test_design_maps_rows_to_nodes():
    block = build_sar(_frame(), "s", "region", _graph(), 0.3, 1.0)
    assert block.labels == ("a", "b", "c")
    assert block.design.shape == (3, 3)
    assert np.allclose(block.design.toarray(), np.eye(3))


def test_rho_out_of_bounds_rejected():
    with pytest.raises(ValueError, match="rho"):
        build_sar(_frame(), "s", "region", _graph(), 1.0, 1.0)


def test_rho_zero_is_identity_scaled():
    block = build_sar(_frame(), "s", "region", _graph(), 0.0, 3.0)
    assert np.allclose(block.precision.toarray(), 3.0 * np.eye(3))

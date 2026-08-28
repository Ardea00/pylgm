# tests/test_spacetime_builder.py
import numpy as np
import pandas as pd
import pytest

from pylgm.effects.spacetime import build_spacetime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}  # path a-b-c, connected


def _frame(areas, times):
    rows = [(s, t) for s in areas for t in times]
    return pd.DataFrame({"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": 0.0})


def _dense(block):
    return block.precision.toarray()


def _rw_structure(T, order):
    if order == 1:
        d = np.zeros((T - 1, T))
        for i in range(T - 1):
            d[i, i], d[i, i + 1] = -1.0, 1.0
    else:
        d = np.zeros((T - 2, T))
        for i in range(T - 2):
            d[i, i], d[i, i + 1], d[i, i + 2] = 1.0, -2.0, 1.0
    return d.T @ d


def test_type_i_precision_is_identity_kron():
    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "I", 1, 2.0, True)
    S, T = 3, 4
    assert np.allclose(_dense(block), 2.0 * np.eye(S * T))
    assert block.constraints.shape == (0, S * T)


def test_type_iii_precision_is_scaled_besag_kron_identity():
    from pylgm.effects.besag import _scaled_structure
    from pylgm.effects.graph import normalize_graph

    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "III", 1, 1.0, True)
    nodes, w = normalize_graph(GRAPH)
    R_s = _scaled_structure(w, nodes, True)
    expected = np.kron(R_s, np.eye(4))
    assert np.allclose(_dense(block), expected)


def test_type_ii_marginal_variance_normalised():
    # Type II = I_s (x) R_t; each area gets the scaled temporal structure.
    frame = _frame(["a", "b", "c"], list(range(6)))
    block = build_spacetime(frame, "st", "area", "t", None, "II", 1, 1.0, True)
    S, T = 3, 6
    R_t_block = _rw_structure(T, 1)
    # scaled so geometric-mean generalized variance == 1
    eig, vec = np.linalg.eigh(R_t_block)
    inv = np.zeros_like(eig)
    inv[1:] = 1.0 / eig[1:]
    var = np.einsum("ij,j,ij->i", vec, inv, vec)
    factor = float(np.exp(np.mean(np.log(var))))
    expected = np.kron(np.eye(S), factor * R_t_block)
    assert np.allclose(_dense(block), expected)


@pytest.mark.parametrize(
    "interaction,order,expected_rows",
    [("I", 1, 0), ("II", 1, 3), ("II", 2, 6), ("III", 1, 4), ("IV", 1, 3 + 4 - 1), ("IV", 2, 6 + 4 - 2)],
)
def test_constraint_row_counts(interaction, order, expected_rows):
    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    graph = None if interaction in ("I", "II") else GRAPH
    block = build_spacetime(frame, "st", "area", "t", graph, interaction, order, 1.0, True)
    assert block.constraints.shape[0] == expected_rows


def test_constraints_span_null_space_type_iv():
    frame = _frame(["a", "b", "c"], [0, 1, 2, 3])
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "IV", 1, 1.0, True)
    P = block.precision.toarray()
    # The constraint rows must EQUAL an orthonormal basis of null(P), which
    # pins two directions:
    #   (a) span(C.T) subseteq null(P) -- every row is a genuine null vector,
    #   (b) null(P) subseteq span(C.T) -- the rows cover the whole null space.
    # (a): P annihilates every constraint row.
    assert np.allclose(P @ block.constraints.T, 0.0, atol=1e-8)
    # (b): projecting each null vector onto span(C.T) leaves zero residual (the
    # rows *span* null(P) -- see besag._component_constraints and
    # laplace._constraint_null_space -- not orthogonal to it, which would be
    # impossible for any full-rank set of rows drawn from that subspace).
    eig, vec = np.linalg.eigh(P)
    null_vectors = vec[:, eig < 1e-9 * eig.max()]
    basis, _ = np.linalg.qr(block.constraints.T)
    residual = null_vectors - basis @ (basis.T @ null_vectors)
    assert np.allclose(residual, 0.0, atol=1e-8)
    # and the constraint rows are full row-rank (so (a)+(b) give exact equality)
    assert np.linalg.matrix_rank(block.constraints) == block.constraints.shape[0]


def test_design_is_area_major_one_hot():
    frame = pd.DataFrame({"area": ["a", "c"], "t": [1, 3], "y": 0.0})
    block = build_spacetime(frame, "st", "area", "t", GRAPH, "I", 1, 1.0, True)
    # areas sorted a,b,c -> positions 0,1,2 ; times 1,3 (observed) -> but time
    # universe is observed levels {1,3}; T=2. cell(a,1)=0*2+0=0; cell(c,3)=2*2+1=5.
    design = block.design.toarray()
    assert design.shape == (2, 3 * 2)
    assert design[0, 0] == 1.0 and design[0].sum() == 1.0
    assert design[1, 5] == 1.0 and design[1].sum() == 1.0


def test_rejects_too_few_time_levels_for_rw():
    frame = _frame(["a", "b", "c"], [0, 1])  # T=2, order=2 needs T>2
    with pytest.raises(ValueError):
        build_spacetime(frame, "st", "area", "t", None, "II", 2, 1.0, True)


def test_space_structured_rejects_isolated_area():
    # Besag/BYM2 treat an isolated node as an IID singleton, but a space-
    # structured interaction's null-basis counts one indicator per component,
    # so an isolated (nulless) area would be over-constrained. It stays rejected.
    graph = {"a": ["b"], "b": ["a"], "c": []}  # c isolated
    frame = _frame(["a", "b", "c"], [0, 1, 2])
    with pytest.raises(ValueError, match="isolated area"):
        build_spacetime(frame, "st", "area", "t", graph, "III", 1, 1.0, True)

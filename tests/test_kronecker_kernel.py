import numpy as np
from scipy.sparse import csr_matrix, identity

from pylgm.effects.kronecker import kron_block, kron_null_constraints


def test_precision_is_the_kronecker_product_in_outer_major_order():
    outer = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    inner = csr_matrix(np.diag([1.0, 3.0, 5.0]))
    block = kron_block(
        "u", ("g1", "g2"), outer, np.zeros((2, 0)),
        ("a", "b", "c"), inner, np.zeros((3, 0)),
        np.array([0, 1]), np.array([2, 0]), "@", False,
    )
    assert np.allclose(block.precision.toarray(), np.kron(outer.toarray(), inner.toarray()))


def test_labels_pair_outer_major_with_the_given_separator():
    block = kron_block(
        "u", ("g1", "g2"), identity(2, format="csr"), np.zeros((2, 0)),
        ("a", "b"), identity(2, format="csr"), np.zeros((2, 0)),
        np.array([0]), np.array([0]), "|", False,
    )
    assert block.labels == ("g1|a", "g1|b", "g2|a", "g2|b")


def test_design_places_each_row_at_outer_times_inner_plus_inner():
    block = kron_block(
        "u", ("g1", "g2"), identity(2, format="csr"), np.zeros((2, 0)),
        ("a", "b", "c"), identity(3, format="csr"), np.zeros((3, 0)),
        np.array([0, 1, 1]), np.array([2, 0, 2]), "@", False,
    )
    dense = block.design.toarray()
    assert dense.shape == (3, 6)
    assert [row.argmax() for row in dense] == [0 * 3 + 2, 1 * 3 + 0, 1 * 3 + 2]
    assert np.allclose(dense.sum(axis=1), 1.0)


def test_no_null_on_either_factor_gives_no_constraints():
    got = kron_null_constraints(np.zeros((2, 0)), np.zeros((3, 0)), 2, 3, True)
    assert got.shape == (0, 6)


def test_a_single_part_is_left_alone_when_not_orthonormalising():
    """Slice 3's bit-for-bit equality with AR1(group=) depends on this.

    The SVD spans the same row space but returns different rows -- [1, 1, 1]
    comes back as [-0.577, -0.577, -0.577] -- so Replicated must get the
    literal kron(I_R, C) it has always produced.
    """
    inner_null = np.ones((3, 1))
    got = kron_null_constraints(np.zeros((2, 0)), inner_null, 2, 3, False)
    assert np.array_equal(got, np.kron(np.eye(2), inner_null.T))


def test_a_single_part_is_orthonormalised_when_asked():
    """SpaceTime types II and III have one part and orthonormalise today."""
    got = kron_null_constraints(np.zeros((2, 0)), np.ones((3, 1)), 2, 3, True)
    assert got.shape == (2, 6)
    assert np.allclose(got @ got.T, np.eye(2))


def test_two_parts_always_orthonormalise_even_when_not_asked():
    """The two spans share 1_out (x) 1_in; keeping it twice is rank-deficient.

    Stacking them raw gives 2 + 3 = 5 rows for a 4-dimensional space, and a
    rank-deficient constraint matrix breaks the constrained solve.
    """
    got = kron_null_constraints(np.ones((2, 1)), np.ones((3, 1)), 2, 3, False)
    assert got.shape == (4, 6)
    assert np.linalg.matrix_rank(got) == 4
    assert np.allclose(got @ got.T, np.eye(4))


def test_the_constraint_span_is_the_precision_null_space():
    """The invariant the whole slice rests on, checked directly."""
    outer = csr_matrix(np.array([[1.0, -1.0], [-1.0, 1.0]]))     # null = span{1}
    inner = csr_matrix(np.diag([1.0, 2.0]))                       # proper
    got = kron_null_constraints(np.ones((2, 1)), np.zeros((2, 0)), 2, 2, False)
    q = np.kron(outer.toarray(), inner.toarray())
    assert np.allclose(q @ got.T, 0.0)
    assert got.shape[0] == q.shape[0] - np.linalg.matrix_rank(q)

import numpy as np
import pytest

from pylgm.effects.structures import (
    AR1Structure, BesagStructure, IIDStructure, RW1Structure, RW2Structure,
)

LEVELS = ("g1", "g2", "g3", "g4")
GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}


def test_iid_is_the_identity_with_no_null():
    s = IIDStructure()
    assert np.allclose(s.precision(LEVELS).toarray(), np.eye(4))
    assert s.null_basis(LEVELS).shape == (4, 0)
    assert s.levels(LEVELS) == LEVELS


def test_ar1_is_proper_so_it_has_no_null():
    s = AR1Structure(rho=0.6)
    q = s.precision(LEVELS).toarray()
    assert q.shape == (4, 4)
    assert np.linalg.matrix_rank(q) == 4
    assert s.null_basis(LEVELS).shape == (4, 0)


def test_ar1_structure_matches_the_ar1_effect_builder():
    from pylgm.effects.ar1 import ar1_structure
    assert np.allclose(
        AR1Structure(rho=0.6).precision(LEVELS).toarray(), ar1_structure(4, 0.6).toarray()
    )


def test_ar1_rejects_a_rho_outside_the_stationary_range():
    with pytest.raises(ValueError, match="rho"):
        AR1Structure(rho=1.0)


@pytest.mark.parametrize("structure,null_dim", [(RW1Structure(), 1), (RW2Structure(), 2)])
def test_random_walk_null_dimension_matches_its_order(structure, null_dim):
    q = structure.precision(LEVELS).toarray()
    basis = structure.null_basis(LEVELS)
    assert basis.shape == (4, null_dim)
    assert np.allclose(q @ basis, 0.0)


def test_rw2_null_is_the_constant_and_the_centred_ramp():
    basis = RW2Structure().null_basis(LEVELS)
    assert np.allclose(basis[:, 0], np.ones(4))
    assert np.allclose(basis[:, 1], np.arange(4) - 1.5)


def test_besag_takes_its_universe_from_the_graph_not_the_observed_levels():
    """A node with no observations keeps its cell, so smoothing lends it strength."""
    s = BesagStructure(GRAPH)
    assert s.levels(("a", "b")) == ("a", "b", "c", "d")


def test_besag_precision_and_null_come_from_the_graph():
    s = BesagStructure(GRAPH)
    nodes = s.levels(("a",))
    q = s.precision(nodes).toarray()
    basis = s.null_basis(nodes)
    assert q.shape == (4, 4)
    assert basis.shape == (4, 1)          # one connected component
    assert np.allclose(q @ basis, 0.0)


def test_besag_null_has_one_column_per_connected_component():
    s = BesagStructure({"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]})
    assert s.null_basis(s.levels(())).shape == (4, 2)


def test_besag_rejects_levels_that_are_not_the_universe_it_returned():
    """Silently ignoring the argument is the permuted-neighbourhood bug."""
    s = BesagStructure(GRAPH)
    with pytest.raises(ValueError, match="graph orders nodes"):
        s.precision(("b", "a", "c", "d"))


def test_a_level_outside_the_graph_is_rejected_by_name():
    """Aligning by position instead would permute the neighbourhood silently."""
    with pytest.raises(ValueError, match="zz"):
        BesagStructure(GRAPH).levels(("a", "zz"))


def test_an_anonymous_structure_keeps_the_observed_levels():
    for s in (IIDStructure(), AR1Structure(rho=0.3), RW1Structure(), RW2Structure()):
        assert s.levels(("x", "y")) == ("x", "y")

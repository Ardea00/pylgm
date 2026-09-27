"""Grouped reproduces SpaceTime's four Knorr-Held interaction types.

SpaceTime is a shipped, tested implementation of the same Kronecker mechanism,
which makes it an oracle predating this slice -- the same role AR1(group=)
played for Replicated.

Constraints are compared by SPAN, not row by row: SpaceTime orthonormalises and
Grouped does not, so the bases differ while the constrained subspace does not.

Types II and IV (the RW-based interactions) disagree with SpaceTime by exactly
one global scalar on ``precision``, never on ``design``: ``build_spacetime``
always builds its time factor as ``rw_structure(T, order, scale=True)``
(Sørbye-Rue variance-scaled), while ``Grouped``'s inner ``RW1``/``RW2`` compiles
through the ordinary, unscaled ``build_random_walk`` -- the same builder every
standalone ``RW1``/``RW2`` effect in this library uses. ``sorbye_rue_scale``
returns ``factor * structure`` for one scalar ``factor``, so the mismatch is a
single multiplicative constant threaded through the Kronecker product, not a
reordering, transposition, or indexing bug. This is a pre-existing scaling
inconsistency between ``RW1``/``RW2`` and ``build_spacetime``'s internal
convention -- recorded here, not fixed: neither may change in this slice.
Types I and III (whose inner factor is a plain ``IID``) are unaffected and are
checked for exact equality, including on the constraint span.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Grouped, IID, IIDStructure, RW1, RW2,
)
from pylgm.compiler import _effect_block
from pylgm.effects.random_walk import rw_structure
from pylgm.effects.spacetime import build_spacetime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}


def _frame():
    return pd.DataFrame(
        [{"s": a, "t": t, "y": 0.0} for a in ("a", "b", "c", "d") for t in range(5)]
    )


def _same_span(first: np.ndarray, second: np.ndarray) -> bool:
    """Two constraint matrices span the same row space.

    Symmetric on purpose. Testing only ``rowspace(second) <= rowspace(first)``
    is safe here solely because ``reference.constraints`` happens to be full
    row rank; were the arguments ever swapped, a degenerate or zero ``first``
    would pass silently. Requiring both ranks to equal the stacked rank costs
    one call and removes the dependence on argument order.
    """
    if first.shape != second.shape:
        return False
    if first.shape[0] == 0:
        return True
    rank = np.linalg.matrix_rank(np.vstack([first, second]))
    return rank == np.linalg.matrix_rank(first) == np.linalg.matrix_rank(second)


def _rw_scale_ratio(time_count: int, order: int) -> float:
    """The scalar by which ``build_spacetime``'s scaled RW factor differs from
    ``Grouped``'s (via ``RW1``/``RW2``) unscaled one, computed from
    ``rw_structure`` itself -- never hard-coded.
    """
    scaled = rw_structure(time_count, order, scale=True)
    raw = rw_structure(time_count, order, scale=False)
    nonzero = raw != 0
    ratios = scaled[nonzero] / raw[nonzero]
    assert np.allclose(ratios, ratios[0]), "sorbye_rue_scale is no longer a single scalar factor"
    return float(ratios[0])


def _inner(interaction, order):
    return IID("st", index="t") if interaction in ("I", "III") else (
        RW1("st", index="t") if order == 1 else RW2("st", index="t")
    )


def _structure(interaction):
    return IIDStructure() if interaction in ("I", "II") else BesagStructure(GRAPH)


@pytest.mark.parametrize("interaction,order", [
    ("I", 1), ("II", 1), ("II", 2), ("III", 1), ("IV", 1), ("IV", 2),
])
def test_grouped_reproduces_the_knorr_held_interaction(interaction, order):
    frame = _frame()
    reference = build_spacetime(
        frame, "st", "s", "t", GRAPH, interaction, order, precision=1.0
    )
    grouped = _effect_block(
        Grouped(_inner(interaction, order), over="s", structure=_structure(interaction)),
        frame,
    )
    assert np.allclose(grouped.design.toarray(), reference.design.toarray())

    if interaction in ("II", "IV"):
        # RW-based: precision matches up to the one global Sørbye-Rue scalar
        # documented at module level -- never a hard-coded literal.
        ratio = _rw_scale_ratio(time_count=frame["t"].nunique(), order=order)
        assert np.allclose(grouped.precision.toarray() * ratio, reference.precision.toarray())
    else:
        assert np.allclose(grouped.precision.toarray(), reference.precision.toarray())

    assert _same_span(grouped.constraints, reference.constraints)


def test_the_two_differ_only_in_their_label_separator():
    """SpaceTime's `|` is user-visible in result.labels and cannot change."""
    frame = _frame()
    reference = build_spacetime(frame, "st", "s", "t", GRAPH, "IV", 1, precision=1.0)
    grouped = _effect_block(
        Grouped(RW1("st", index="t"), over="s", structure=BesagStructure(GRAPH)), frame
    )
    assert [la.replace("@", "|") for la in grouped.labels] == list(reference.labels)


def test_same_span_discriminates_different_subspaces():
    """A helper that always returns True would make the whole oracle vacuous."""
    same_basis = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    reference_basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert _same_span(reference_basis, same_basis)

    different_basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not _same_span(reference_basis, different_basis)


def test_rw_scaling_is_explicit_and_consistent():
    """The RW scaling divergence, reconciled.

    ``RW1``/``RW2`` and ``RW1Structure``/``RW2Structure`` now share one default
    (unscaled) and one ``scale`` flag, so two things spelled RW1 in one call are
    the same matrix; and asking for scaling on the inner effect reproduces
    ``SpaceTime``, whose own ``scale`` flag defaults to True.
    """
    from pylgm.effects.structures import RW1Structure

    frame = _frame()
    grouped = _effect_block(
        Grouped(RW1("u", index="t"), over="s", structure=RW1Structure()), frame
    )
    groups = frame["s"].nunique()
    periods = frame["t"].nunique()
    expected = np.kron(rw_structure(groups, 1, scale=False), rw_structure(periods, 1, scale=False))
    assert np.allclose(grouped.precision.toarray(), expected)

    reference = build_spacetime(frame, "st", "s", "t", GRAPH, "II", 1, precision=1.0)
    unscaled = _effect_block(
        Grouped(RW1("st", index="t"), over="s", structure=IIDStructure()), frame
    )
    scaled = _effect_block(
        Grouped(RW1("st", index="t", scale=True), over="s", structure=IIDStructure()), frame
    )
    assert not np.allclose(unscaled.precision.toarray(), reference.precision.toarray())
    assert np.allclose(scaled.precision.toarray(), reference.precision.toarray())


def test_same_span_rejects_a_degenerate_first_argument():
    """The asymmetry that used to make a zero constraint matrix pass."""
    real = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    degenerate = np.zeros((2, 3))
    assert not _same_span(degenerate, real)
    assert not _same_span(real, degenerate)

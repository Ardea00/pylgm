"""build_spacetime's output, pinned bit-for-bit across the kernel refactor.

The snapshots were generated from the pre-refactor implementation (the
original interaction x order grid) plus a widened grid pinned post-refactor
-- see tests/data/generate_spacetime_snapshots.py for the fixture inputs and
provenance. A difference here is a regression rather than an updated
expectation: this is the only guard on a released, tested path being
rewritten underneath.

Compared to tolerance, not bit-for-bit, and constraints compared as a
subspace. ``build_spacetime`` reaches LAPACK (eigendecomposition, null
space), and LAPACK is free to answer differently on different builds:

* the precision entries drift in the last ulp across BLAS implementations;
* the constraint rows are an *arbitrary* orthonormal basis of a null space.
  Where an eigenvalue is degenerate any rotation within its eigenspace is an
  equally correct answer, and a global sign is always free. Both were
  observed against this fixture -- a whole-row sign flip on II_1, a genuine
  change of basis on IV_2 -- on Linux and Windows against a snapshot
  generated on macOS.

So the constraint check pins what a constraint set actually means: its row
space, via the projector ``C^T C``, plus its shape and the orthonormality of
its rows. ``C x = 0`` depends on nothing else. That is strictly more than the
old ``array_equal`` verified about the mathematics -- orthonormality was
never asserted before -- and strictly less about LAPACK's basis bookkeeping,
which is not ours to pin. The tolerances are far tighter than any real
regression: a changed interaction or order moves these entries by O(1).
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from pylgm.effects.spacetime import build_spacetime

_DATA_DIR = pathlib.Path(__file__).parent / "data"

# Load the generator as a plain file (not a package import): tests/ and
# tests/data/ have no __init__.py, so this is the only cwd-independent way to
# share GRID/frame_for with the script that produced the snapshot.
_spec = importlib.util.spec_from_file_location(
    "generate_spacetime_snapshots", _DATA_DIR / "generate_spacetime_snapshots.py"
)
_generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator)
GRID, frame_for = _generator.GRID, _generator.frame_for

SNAPSHOTS = np.load(_DATA_DIR / "spacetime_snapshots.npz")


@pytest.mark.parametrize("case", GRID, ids=[case.key for case in GRID])
def test_spacetime_output_is_unchanged_by_the_kernel_refactor(case):
    block = build_spacetime(
        frame_for(), "st", "s", "t", case.graph, case.interaction, case.order,
        precision=1.5, scale=case.scale,
    )
    np.testing.assert_allclose(
        block.precision.toarray(), SNAPSHOTS[f"{case.key}_q"], rtol=1e-12, atol=1e-12
    )
    _assert_same_row_space(block.constraints, SNAPSHOTS[f"{case.key}_c"])
    assert np.array_equal(block.design.toarray(), SNAPSHOTS[f"{case.key}_d"])
    assert block.labels == tuple(SNAPSHOTS[f"{case.key}_labels"])


def _assert_same_row_space(actual: np.ndarray, expected: np.ndarray) -> None:
    """Same constraint set, up to the basis LAPACK happened to choose.

    Rows orthonormal, so the projector onto the row space is ``C^T C``; it is
    invariant under a sign flip or any rotation within a degenerate eigenspace,
    which is exactly the freedom LAPACK exercises between platforms.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    if actual.size == 0:
        return
    np.testing.assert_allclose(
        actual @ actual.T, np.eye(actual.shape[0]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(actual.T @ actual, expected.T @ expected, rtol=0, atol=1e-12)


def test_row_space_check_absorbs_lapack_basis_freedom_but_not_a_real_change():
    """The loosened constraint check must still fail on a genuine regression.

    Sign flip and in-subspace rotation are what LAPACK varies between
    platforms; a different constraint set is what this fixture exists to catch.
    """
    constraints = SNAPSHOTS["IV_2_c"]
    rotation, _ = np.linalg.qr(
        np.random.default_rng(0).normal(size=(constraints.shape[0],) * 2)
    )

    _assert_same_row_space(-constraints, constraints)
    _assert_same_row_space(rotation @ constraints, constraints)

    different = SNAPSHOTS["IV_2_disconnected_c"][: constraints.shape[0]]
    with pytest.raises(AssertionError):
        _assert_same_row_space(different, constraints)


def test_precision_tolerance_absorbs_ulp_drift_but_not_a_real_change():
    precision = SNAPSHOTS["IV_1_q"]

    np.testing.assert_allclose(
        precision * (1.0 + 3e-16), precision, rtol=1e-12, atol=1e-12
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            SNAPSHOTS["IV_2_q"], precision, rtol=1e-12, atol=1e-12
        )

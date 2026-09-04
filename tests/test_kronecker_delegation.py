"""build_spacetime's output, pinned bit-for-bit across the kernel refactor.

The snapshots were generated from the pre-refactor implementation (the
original interaction x order grid) plus a widened grid pinned post-refactor
-- see tests/data/generate_spacetime_snapshots.py for the fixture inputs and
provenance. A difference here is a regression rather than an updated
expectation: this is the only guard on a released, tested path being
rewritten underneath.
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
    assert np.array_equal(block.precision.toarray(), SNAPSHOTS[f"{case.key}_q"])
    assert np.array_equal(block.constraints, SNAPSHOTS[f"{case.key}_c"])
    assert np.array_equal(block.design.toarray(), SNAPSHOTS[f"{case.key}_d"])
    assert block.labels == tuple(SNAPSHOTS[f"{case.key}_labels"])

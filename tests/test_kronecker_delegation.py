"""build_spacetime's output, pinned bit-for-bit across the kernel refactor.

The snapshots were generated from the pre-refactor implementation, so a
difference here is a regression rather than an updated expectation. This is the
only guard on a released, tested path being rewritten underneath.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm.effects.spacetime import build_spacetime

# Path graph a-b-c-d. (The brief's original GRAPH had "d": ["c"] without the
# matching "c": ["b", "d"] reverse edge; normalize_graph now rejects
# asymmetric graphs, so the edge set is completed here to stay symmetric.)
GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
SNAPSHOTS = np.load("tests/data/spacetime_snapshots.npz")


def _frame():
    return pd.DataFrame(
        [{"s": a, "t": t, "y": 0.0} for a in ("a", "b", "c", "d") for t in range(5)]
    )


@pytest.mark.parametrize("interaction", ["I", "II", "III", "IV"])
@pytest.mark.parametrize("order", [1, 2])
def test_spacetime_output_is_unchanged_by_the_kernel_refactor(interaction, order):
    block = build_spacetime(
        _frame(), "st", "s", "t", GRAPH, interaction, order, precision=1.5
    )
    key = f"{interaction}_{order}"
    assert np.array_equal(block.precision.toarray(), SNAPSHOTS[f"{key}_q"])
    assert np.array_equal(block.constraints, SNAPSHOTS[f"{key}_c"])
    assert np.array_equal(block.design.toarray(), SNAPSHOTS[f"{key}_d"])
    assert block.labels[:3] == ("a|0", "a|1", "a|2")

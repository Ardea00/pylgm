# tests/data/generate_spacetime_snapshots.py
"""Regenerates spacetime_snapshots.npz, the regression fixture pinning
``build_spacetime``'s output across the Kronecker-kernel refactor.

Run from anywhere (paths resolve relative to this file):

    PYTHONPATH=src python tests/data/generate_spacetime_snapshots.py

``tests/test_kronecker_delegation.py`` imports ``GRAPH``, ``frame_for`` and
``GRID`` from this module, so the fixture inputs are defined exactly once
and the test cannot drift from the data that produced the snapshot.

Provenance
----------
The original ``interaction`` x ``order`` grid (8 rows: I/II/III/IV x order
1/2, ``GRAPH``, ``scale=True``) was generated **pre-refactor**, against the
``build_spacetime`` that existed before commit ab76780 moved it onto the
shared Kronecker kernel. Both the reviewer's full differential and a
regeneration from the current source confirm those 8 rows are still
bit-for-bit unchanged -- that equality was already established and is not
what this script re-proves.

The widened rows below (``scale=False``, ``DISCONNECTED_GRAPH``) were never
exercised by the original grid. They are pinned here for the first time, as
of this commit, generated from the current (post-refactor) source -- there
is no pre-refactor baseline for them to match.
"""

import pathlib
from typing import NamedTuple

import numpy as np
import pandas as pd

from pylgm.effects.spacetime import build_spacetime

AREAS = ("a", "b", "c", "d")

# Path graph a-b-c-d, one connected component.
GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}

# Two components, {a, b} and {c, d}: exercises the multi-component space null
# basis (_space_null_basis emits one indicator column per component), which
# GRAPH -- being connected -- never builds.
DISCONNECTED_GRAPH = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}


def frame_for(areas=AREAS):
    return pd.DataFrame([{"s": a, "t": t, "y": 0.0} for a in areas for t in range(5)])


class Case(NamedTuple):
    key: str
    interaction: str
    order: int
    graph: dict
    scale: bool


GRID = [
    *(
        Case(f"{interaction}_{order}", interaction, order, GRAPH, True)
        for interaction in ("I", "II", "III", "IV")
        for order in (1, 2)
    ),
    # scale=False: the Sørbye-Rue branch, never exercised above.
    Case("IV_1_noscale", "IV", 1, GRAPH, False),
    Case("IV_2_noscale", "IV", 2, GRAPH, False),
    # Disconnected graph: multi-component space null basis (2 columns).
    Case("III_1_disconnected", "III", 1, DISCONNECTED_GRAPH, True),
    Case("IV_2_disconnected", "IV", 2, DISCONNECTED_GRAPH, True),
]


def snapshot():
    data = {}
    for case in GRID:
        block = build_spacetime(
            frame_for(),
            "st", "s", "t", case.graph, case.interaction, case.order,
            precision=1.5, scale=case.scale,
        )
        data[f"{case.key}_q"] = block.precision.toarray()
        data[f"{case.key}_c"] = block.constraints
        data[f"{case.key}_d"] = block.design.toarray()
        data[f"{case.key}_labels"] = np.array(block.labels)
    return data


if __name__ == "__main__":
    data = snapshot()
    out = pathlib.Path(__file__).parent / "spacetime_snapshots.npz"
    np.savez(out, **data)
    print(f"wrote {out} {len(data)} arrays")

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import block_diag, csr_matrix

from pylgm.effects.besag import build_besag
from pylgm.effects.fixed import build_fixed
from pylgm.effects.proper_car import build_proper_car
from pylgm.ir.model import CompiledLGM
from pylgm.likelihoods import CompiledGaussian

# Small connected chain graph over regions "0".."3": i <-> i-1, i <-> i+1.
_GRAPH = {
    str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 3]
    for i in range(4)
}


@pytest.fixture
def besag_with_intercept_model() -> CompiledLGM:
    """A compiled ``Fixed("1") + Besag(...)`` model on a tiny connected graph.

    Built from the real ``build_fixed`` / ``build_besag`` compiler helpers and
    the ``LatentBlock`` / ``CompiledLGM`` constructors (the same direct-
    construction pattern ``tests/inference/test_gaussian.py`` uses), so the
    fixed block carries its real ``1e-6`` ridge precision and an all-ones
    intercept design. ``_partition_blocks`` routes it to the dense block by
    design-column density (the intercept touches every observation), which is
    what Approach A must quarantine.
    """
    regions = [str(i) for i in range(4)]
    frame = pd.DataFrame({"region": regions, "y": [1.0, 2.0, 1.5, 2.5]})

    fixed_block = build_fixed(frame, "1", 1e-6)
    region_block = build_besag(frame, "region", "region", _GRAPH, precision=1.0)

    blocks = (fixed_block, region_block)
    design = csr_matrix(np.hstack([fixed_block.design.toarray(), region_block.design.toarray()]))
    precision = block_diag([fixed_block.precision, region_block.precision], format="csr")
    constraints = np.vstack(
        [
            np.hstack([fixed_block.constraints, np.zeros((fixed_block.constraints.shape[0], 4))]),
            np.hstack([np.zeros((region_block.constraints.shape[0], 1)), region_block.constraints]),
        ]
    )
    labels = tuple(f"{block.name}:{label}" for block in blocks for label in block.labels)

    return CompiledLGM(
        y=frame["y"].to_numpy(dtype=float),
        observed=np.ones(4, dtype=bool),
        offset=np.zeros(4),
        design=design,
        precision=precision,
        constraints=constraints,
        labels=labels,
        likelihood=CompiledGaussian(0.1),
        blocks=blocks,
    )


@pytest.fixture
def proper_car_with_intercept_model() -> CompiledLGM:
    """A compiled ``Fixed("1") + proper_car(...)`` model on a tiny connected graph.

    The proper-CAR precision ``rho`` (0.5) sits inside the graph's validity
    interval, so the field prior is full rank and the model carries **no**
    constraints -- the no-constraint deliverable Task 4a must match dense on.
    Two of the six rows are unobserved, exercising the ``observed`` mask path.
    """
    frame = pd.DataFrame(
        {
            "region": ["0", "1", "2", "3", "1", "2"],
            "y": [1.0, 2.0, 1.5, 2.5, 0.0, 0.0],
        }
    )
    observed = np.array([True, True, True, True, False, False])

    fixed_block = build_fixed(frame, "1", 1e-6)
    car_block = build_proper_car(frame, "car", "region", _GRAPH, rho=0.5, precision=2.0)

    blocks = (fixed_block, car_block)
    design = csr_matrix(np.hstack([fixed_block.design.toarray(), car_block.design.toarray()]))
    precision = block_diag([fixed_block.precision, car_block.precision], format="csr")
    constraints = np.vstack(
        [
            np.hstack([fixed_block.constraints, np.zeros((fixed_block.constraints.shape[0], 4))]),
            np.hstack([np.zeros((car_block.constraints.shape[0], 1)), car_block.constraints]),
        ]
    )
    labels = tuple(f"{block.name}:{label}" for block in blocks for label in block.labels)

    return CompiledLGM(
        y=frame["y"].to_numpy(dtype=float),
        observed=observed,
        offset=np.zeros(len(frame)),
        design=design,
        precision=precision,
        constraints=constraints,
        labels=labels,
        likelihood=CompiledGaussian(0.1),
        blocks=blocks,
    )

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import block_diag, csr_matrix

from pylgm.effects.besag import build_besag
from pylgm.ir.model import CompiledLGM, LatentBlock
from pylgm.likelihoods import CompiledGaussian

# Small connected chain graph over regions "0".."3": i <-> i-1, i <-> i+1.
_GRAPH = {
    str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= 3]
    for i in range(4)
}


@pytest.fixture
def besag_with_intercept_model() -> CompiledLGM:
    """A compiled ``Fixed("1") + Besag(...)`` model on a tiny connected graph.

    Built from the same ``LatentBlock``/``CompiledLGM`` constructors and
    ``build_besag`` helper the rest of the compiler uses (see
    ``tests/inference/test_gaussian.py`` for the same direct-construction
    pattern), rather than through ``LGM(...).fit(...)``: the public ``Fixed``
    effect (``src/pylgm/effects/spec.py``) enforces ``prior_precision > 0``
    (default ``1e-6`` ridge), so it can never compile to an exactly
    zero-nnz precision block. ``_partition_blocks``'s dense/sparse criterion
    is ``precision.nnz == 0``, so the fixed block here is built with a real
    all-zero precision to exercise that branch; a formula-compiled Fixed
    intercept currently always keeps its tiny ridge and would (correctly, by
    the stated criterion) land in the sparse block instead.
    """
    regions = [str(i) for i in range(4)]
    frame = pd.DataFrame({"region": regions, "y": [1.0, 2.0, 1.5, 2.5]})

    fixed_block = LatentBlock(
        name="fixed",
        labels=("Intercept",),
        design=csr_matrix(np.ones((4, 1))),
        precision=csr_matrix((1, 1)),
        constraints=np.empty((0, 1)),
    )
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

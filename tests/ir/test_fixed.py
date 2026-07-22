import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pylgm.effects.fixed import build_fixed
from pylgm.ir import CompiledLGM, LatentBlock


def test_fixed_block_has_stable_columns_and_precision() -> None:
    frame = pd.DataFrame({"x": [2.0, 3.0]})
    block = build_fixed(frame, "1 + x", prior_precision=0.25)
    assert block.name == "fixed"
    assert block.labels == ("Intercept", "x")
    np.testing.assert_allclose(block.design.toarray(), [[1.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(block.precision.toarray(), np.eye(2) * 0.25)
    assert block.constraints.shape == (0, 2)


def test_latent_block_rejects_dense_and_sparse_payload_mutation() -> None:
    block = LatentBlock(
        name="block",
        labels=("coefficient",),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[2.0]]),
        constraints=np.array([[3.0]]),
    )

    with pytest.raises(ValueError):
        block.constraints[0, 0] = 4.0
    with pytest.raises(ValueError):
        block.design.data[0] = 4.0


def test_compiled_lgm_rejects_dense_payload_mutation() -> None:
    model = CompiledLGM(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.array([0.0]),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[2.0]]),
        constraints=np.empty((0, 1)),
        labels=("coefficient",),
        sigma=1.0,
        blocks=(),
    )

    with pytest.raises(ValueError):
        model.y[0] = 2.0

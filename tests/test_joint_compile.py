import numpy as np
from scipy.sparse import csr_matrix

from pylgm.ir.model import LatentBlock
from pylgm.joint import _pad_block_rows


def _block():
    design = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    precision = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    constraints = np.array([[1.0, 1.0]])
    return LatentBlock("u", ("a", "b"), design, precision, constraints)


def test_pad_block_rows_zero_pads_design_and_preserves_everything_else():
    block = _block()
    padded = _pad_block_rows(block, before=2, after=1)

    assert padded.design.shape == (6, 2)
    assert np.allclose(padded.design.toarray()[:2], 0.0)
    assert np.allclose(padded.design.toarray()[2:5], block.design.toarray())
    assert np.allclose(padded.design.toarray()[5:], 0.0)

    assert padded.name == block.name
    assert padded.labels == block.labels
    assert np.allclose(padded.precision.toarray(), block.precision.toarray())
    assert np.allclose(padded.constraints, block.constraints)


def test_pad_block_rows_with_no_padding_is_an_identity_on_the_design():
    block = _block()
    padded = _pad_block_rows(block, before=0, after=0)
    assert np.allclose(padded.design.toarray(), block.design.toarray())

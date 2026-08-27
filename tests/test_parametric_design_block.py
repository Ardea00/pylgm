import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.exceptions import NumericalError
from pylgm.ir.family import ParametricDesignBlock
from pylgm.ir.model import LatentBlock


def _template(width_rows=4):
    design = csr_matrix(np.ones((width_rows, 1)))
    precision = csr_matrix([[1e-6]])
    return LatentBlock("m", ("m",), design, precision, np.empty((0, 1)))


def test_materialize_rebuilds_design_keeps_precision():
    template = _template()

    def build(resolved):
        return csr_matrix((resolved["theta"] * np.arange(4.0)).reshape(-1, 1))

    block = ParametricDesignBlock(template, ("theta",), build)
    out = block.materialize({"theta": 2.0})
    assert out.name == "m" and out.labels == ("m",)
    np.testing.assert_allclose(out.design.toarray().ravel(), [0.0, 2.0, 4.0, 6.0])
    # precision + constraints carried from the template, unchanged
    np.testing.assert_allclose(out.precision.toarray(), [[1e-6]])
    assert out.constraints.shape == (0, 1)


def test_materialize_rejects_nonfinite_design():
    block = ParametricDesignBlock(
        _template(), ("theta",), lambda r: csr_matrix(np.array([[np.inf]] * 4))
    )
    with pytest.raises(NumericalError):
        block.materialize({"theta": 1.0})

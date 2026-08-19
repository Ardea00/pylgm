import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from pylgm.ir import CompiledFamily
from pylgm.ir.family import ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian, CompiledPoisson


def _block(name, precision_value):
    return LatentBlock(
        name=name,
        labels=("a", "b"),
        design=csr_matrix(np.eye(2)),
        precision=eye(2, format="csr") * precision_value,
        constraints=np.empty((0, 2), dtype=float),
    )


def _family(factory, parameter_names, blocks):
    return CompiledFamily(
        y=np.array([1.0, 2.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        blocks=blocks,
        parameter_names=parameter_names,
        likelihood_factory=factory,
    )


def test_compiled_family_scales_bound_block_and_builds_likelihood():
    blocks = (ScalableBlock(_block("g", 1.0), "g_prec", 1.0),)
    family = _family(lambda r: CompiledGaussian(0.5), ("g_prec",), blocks)
    compiled = family.materialize({"g_prec": 4.0})
    # unit-base precision (eye) scaled by 4.0
    np.testing.assert_allclose(compiled.precision.toarray(), np.eye(2) * 4.0)
    assert isinstance(compiled.likelihood, CompiledGaussian)
    assert compiled.likelihood.sigma == 0.5


def test_compiled_family_non_gaussian_factory():
    blocks = (ScalableBlock(_block("g", 2.0), None, 1.0),)
    family = _family(lambda r: CompiledPoisson(), (), blocks)
    compiled = family.materialize({})
    assert isinstance(compiled.likelihood, CompiledPoisson)
    np.testing.assert_allclose(compiled.precision.toarray(), np.eye(2) * 2.0)


def test_compiled_family_rejects_wrong_parameters():
    blocks = (ScalableBlock(_block("g", 1.0), "g_prec", 1.0),)
    family = _family(lambda r: CompiledPoisson(), ("g_prec",), blocks)
    with pytest.raises(Exception):
        family.materialize({"wrong": 1.0})

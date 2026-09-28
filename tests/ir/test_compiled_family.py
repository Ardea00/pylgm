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


def test_materialize_fast_path_matches_fully_validated_assembly():
    from pylgm.ir.family import ParametricBlock, _assemble_compiled_model, _materialize_blocks

    rw = LatentBlock("rw", ("a", "b"), csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
                     csr_matrix([[1.0, -1.0], [-1.0, 1.0]]), np.ones((1, 2)))
    par = ParametricBlock(_block("p", 1.0), ("rho",),
                          lambda r: eye(2, format="csr") * (1.0 + r["rho"]))
    family = CompiledFamily(
        y=np.array([1.0, 2.0]), observed=np.array([True, True]), offset=np.zeros(2),
        blocks=(ScalableBlock(rw, "tau", 1.0), ScalableBlock(_block("g", 2.0), None, 3.0), par),
        parameter_names=("sigma", "tau", "rho"),
        likelihood_factory=lambda r: CompiledGaussian(r["sigma"]),
        extra_constraints=np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    for theta in ({"sigma": 0.5, "tau": 4.0, "rho": 0.2}, {"sigma": 2.0, "tau": 0.1, "rho": 3.0}):
        fast = family.materialize(theta)
        slow = _assemble_compiled_model(
            family._y, family._observed, family._offset,
            _materialize_blocks(family.blocks, theta), CompiledGaussian(theta["sigma"]),
            extra_constraints=family._extra_constraints,
            extra_constraint_rhs=family._extra_constraint_rhs,
        )
        for name in ("y", "observed", "offset", "constraints", "constraint_rhs",
                     "extra_constraints", "prediction_offset"):
            np.testing.assert_array_equal(getattr(fast, name), getattr(slow, name))
        for name in ("design", "precision", "prediction_design"):
            got, want = getattr(fast, name), getattr(slow, name)
            assert (got != want).nnz == 0 and not got.data.flags.writeable
        for got, want in zip(fast.blocks, slow.blocks, strict=True):
            assert got.name == want.name and (got.precision != want.precision).nnz == 0
        assert fast.labels == slow.labels and fast.likelihood.sigma == theta["sigma"]

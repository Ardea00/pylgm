import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags, eye

from pylgm.exceptions import ModelValidationError
from pylgm.ir.family import CompiledGaussianFamily, Hyperparameters, ParametricBlock, ScalableBlock
from pylgm.ir.model import LatentBlock
from pylgm.likelihoods import CompiledGaussian


def _car_block(name, d, w):
    template = LatentBlock(
        name, tuple(str(i) for i in range(len(d))),
        csr_matrix(np.eye(len(d))),
        csr_matrix(diags(d) - 0.5 * w),
        np.empty((0, len(d))),
    )

    def build(values):
        return csr_matrix(values[f"{name}.precision"] * (diags(d) - values[f"{name}.rho"] * w))

    return ParametricBlock(template, (f"{name}.precision", f"{name}.rho"), build)


def test_parametric_block_materializes_tau_d_minus_rho_w():
    d = np.array([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    block = _car_block("region", d, w)
    materialized = block.materialize({"region.precision": 3.0, "region.rho": 0.5})
    assert np.allclose(materialized.precision.toarray(), 3.0 * (np.diag(d) - 0.5 * w))
    assert materialized.labels == block.block.labels
    assert materialized.constraints.shape == (0, 3)


def test_parametric_block_rejects_non_latent_block():
    with pytest.raises(ModelValidationError):
        ParametricBlock(object(), ("region.precision",), lambda values: csr_matrix(np.eye(2)))


def test_parametric_block_rejects_empty_parameters():
    d = np.array([1.0, 2.0, 1.0])
    w = np.zeros((3, 3))
    block = _car_block("region", d, w).block
    with pytest.raises(ModelValidationError):
        ParametricBlock(block, (), lambda values: csr_matrix(np.eye(3)))


def test_parametric_block_rejects_non_callable_build():
    d = np.array([1.0, 2.0, 1.0])
    w = np.zeros((3, 3))
    block = _car_block("region", d, w).block
    with pytest.raises(ModelValidationError):
        ParametricBlock(block, ("region.precision",), "not callable")


def _fixed_block(name, n_obs, width):
    design = np.zeros((n_obs, width))
    for row in range(n_obs):
        design[row, row % width] = 1.0
    return LatentBlock(
        name,
        tuple(str(i) for i in range(width)),
        csr_matrix(design),
        eye(width, format="csr"),
        np.empty((0, width)),
    )


def test_compiled_gaussian_family_mixes_scalable_and_parametric_blocks():
    d = np.array([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    n_obs = len(d)
    parametric = _car_block("region", d, w)
    scalable = ScalableBlock(_fixed_block("fixed", n_obs, 2), None, 1.0)

    family = CompiledGaussianFamily(
        y=np.zeros(n_obs),
        observed=np.ones(n_obs, dtype=bool),
        offset=np.zeros(n_obs),
        blocks=(scalable, parametric),
        parameter_names=("sigma", "region.precision", "region.rho"),
        initial=Hyperparameters(sigma=1.0, precisions={}),
    )

    compiled = family.materialize({"sigma": 0.5, "region.precision": 3.0, "region.rho": 0.5})

    assert isinstance(compiled.likelihood, CompiledGaussian)
    assert compiled.likelihood.sigma == 0.5

    fixed_slice = family.block_slice("fixed")
    region_slice = family.block_slice("region")
    np.testing.assert_allclose(
        compiled.precision[fixed_slice, fixed_slice].toarray(), np.eye(2)
    )
    np.testing.assert_allclose(
        compiled.precision[region_slice, region_slice].toarray(),
        3.0 * (np.diag(d) - 0.5 * w),
    )
    # off-diagonal blocks stay zero (block-diagonal assembly)
    np.testing.assert_allclose(
        compiled.precision[fixed_slice, region_slice].toarray(), np.zeros((2, 3))
    )


def test_compiled_gaussian_family_rejects_parametric_parameters_missing_from_names():
    d = np.array([1.0, 2.0, 1.0])
    w = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    parametric = _car_block("region", d, w)

    n_obs = len(d)
    with pytest.raises(ModelValidationError):
        CompiledGaussianFamily(
            y=np.zeros(n_obs),
            observed=np.ones(n_obs, dtype=bool),
            offset=np.zeros(n_obs),
            blocks=(parametric,),
            parameter_names=("region.precision",),  # missing "region.rho"
            initial=Hyperparameters(sigma=1.0, precisions={}),
        )


def test_family_carries_parameter_priors():
    block = LatentBlock(
        "latent", ("x",), csr_matrix([[1.0]]), csr_matrix([[1.0]]), np.empty((0, 1))
    )

    class _Prior:
        def logpdf(self, value):
            return -value

    family = CompiledGaussianFamily(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(ScalableBlock(block, "latent.precision", 1.0),),
        parameter_names=("latent.precision",),
        initial=Hyperparameters(sigma=1.0, precisions={"latent": 1.0}),
        parameter_priors={"latent.precision": _Prior()},
    )
    assert family.parameter_priors["latent.precision"].logpdf(2.0) == -2.0

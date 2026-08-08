from types import MappingProxyType

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.inference import fit_gaussian
from pylgm.ir import CompiledLGM
from pylgm.likelihoods import CompiledGaussian


def _fitted_result():
    model = CompiledLGM(
        y=np.array([2.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix(np.eye(2)),
        constraints=np.empty((0, 2)),
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )
    return fit_gaussian(model)


def test_gaussian_result_exposes_common_marginals():
    result = _fitted_result()

    marginals = result.latent_marginals()

    assert np.array_equal(marginals.mean, result.mean)
    assert np.array_equal(marginals.variance, np.diag(result.covariance))
    assert result.engine == "exact_gaussian"
    assert result.converged is True
    assert result.hyperparameter_marginals() == {}


def test_linear_combinations_propagate_covariance():
    result = _fitted_result()
    weights = csr_matrix([[1.0, -1.0]])

    combined = result.linear_combinations(weights)

    expected = weights.toarray() @ result.covariance @ weights.toarray().T
    assert combined.mean[0] == pytest.approx(weights.toarray()[0] @ result.mean)
    assert combined.variance[0] == pytest.approx(expected[0, 0])


def test_marginals_are_defensive_and_support_normal_quantiles():
    marginals = _fitted_result().latent_marginals()

    assert marginals.std[0] == pytest.approx(np.sqrt(0.5))
    assert marginals.quantile(0.5)[0] == pytest.approx(1.0)
    assert marginals.quantile(0.975)[0] == pytest.approx(1.0 + 1.959963984540054 * np.sqrt(0.5))
    with pytest.raises(ValueError, match="probability"):
        marginals.quantile(1.0)
    with pytest.raises(ValueError):
        marginals.mean[0] = 100.0


@pytest.mark.parametrize(
    "weights",
    [np.ones((1, 3)), np.array([[np.nan, 1.0]]), np.array([[1.0 + 1.0j, 0.0]])],
)
def test_linear_combinations_reject_invalid_weight_matrices(weights):
    result = _fitted_result()

    with pytest.raises((TypeError, ValueError)):
        result.linear_combinations(weights)


def test_result_metadata_is_immutable_mapping():
    result = _fitted_result()

    assert isinstance(result.block_slices, MappingProxyType)
    assert isinstance(result.diagnostics, MappingProxyType)
    assert result.diagnostics == {
        "latent_dimension": 2,
        "observed_count": 2,
        "constraint_count": 0,
    }
    with pytest.raises(TypeError):
        result.diagnostics["latent_dimension"] = 3  # type: ignore[index]

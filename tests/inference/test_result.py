from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pylgm.inference import GaussianResult, fit_gaussian
from pylgm.ir import CompiledLGM
from pylgm.likelihoods import CompiledGaussian


def test_result_preserves_prediction_keys_by_defensive_copy():
    keys = pd.DataFrame({"region": ["A", "B"], "time": [1, 2]})
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.eye(1),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([1.0, 2.0]),
        predictive_variance=np.array([1.0, 1.0]),
        prediction_keys=keys,
    )

    keys.loc[0, "region"] = "mutated"
    exposed = result.prediction_keys

    assert exposed is not None
    assert exposed["region"].tolist() == ["A", "B"]
    exposed.loc[1, "time"] = 99
    assert result.prediction_keys["time"].tolist() == [1, 2]


def test_result_rejects_prediction_keys_with_wrong_row_count():
    with pytest.raises(ValueError, match="prediction keys.*row count"):
        GaussianResult(
            labels=("x",),
            mean=np.array([0.0]),
            covariance=np.eye(1),
            log_marginal_likelihood=0.0,
            predictive_mean=np.array([1.0, 2.0]),
            predictive_variance=np.array([1.0, 1.0]),
            prediction_keys=pd.DataFrame({"time": [1]}),
        )


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


class _NoQuadraticTranspose(np.ndarray):
    @property
    def T(self):
        raise AssertionError(
            "linear combinations must not form a combination-by-combination matrix"
        )


class _NoQuadraticTransposeCSR(csr_matrix):
    def transpose(self, axes=None, copy=False):
        raise AssertionError(
            "linear combinations must not form a combination-by-combination matrix"
        )


def test_many_linear_combinations_do_not_require_a_quadratic_matrix():
    combination_count = 20_000
    weights = np.tile([1.0, -1.0], (combination_count, 1)).view(
        _NoQuadraticTranspose
    )
    result = GaussianResult(
        labels=("a", "b"),
        mean=np.array([1.0, 2.0]),
        covariance=np.array([[2.0, 0.5], [0.5, 1.0]]),
        log_marginal_likelihood=0.0,
        predictive_mean=np.empty(0),
        predictive_variance=np.empty(0),
    )

    combined = result.linear_combinations(weights)

    np.testing.assert_allclose(combined.mean, np.full(combination_count, -1.0))
    np.testing.assert_allclose(combined.variance, np.full(combination_count, 2.0))


def test_many_sparse_linear_combinations_do_not_require_a_quadratic_matrix():
    combination_count = 20_000
    weights = _NoQuadraticTransposeCSR(
        np.tile([1.0, -1.0], (combination_count, 1))
    )
    result = GaussianResult(
        labels=("a", "b"),
        mean=np.array([1.0, 2.0]),
        covariance=np.array([[2.0, 0.5], [0.5, 1.0]]),
        log_marginal_likelihood=0.0,
        predictive_mean=np.empty(0),
        predictive_variance=np.empty(0),
    )

    combined = result.linear_combinations(weights)

    np.testing.assert_allclose(combined.mean, np.full(combination_count, -1.0))
    np.testing.assert_allclose(combined.variance, np.full(combination_count, 2.0))


@pytest.mark.parametrize(
    "weights",
    [
        np.array([[1.0, -1.0], [2.0, 0.5], [0.0, 3.0]]),
        csr_matrix([[1.0, -1.0], [2.0, 0.5], [0.0, 3.0]]),
    ],
)
def test_linear_combination_variances_match_for_dense_and_sparse_weights(weights):
    dense_weights = weights.toarray() if isinstance(weights, csr_matrix) else weights
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    result = GaussianResult(
        labels=("a", "b"),
        mean=np.array([1.0, 2.0]),
        covariance=covariance,
        log_marginal_likelihood=0.0,
        predictive_mean=np.empty(0),
        predictive_variance=np.empty(0),
    )

    combined = result.linear_combinations(weights)

    expected = np.einsum("ij,jk,ik->i", dense_weights, covariance, dense_weights)
    np.testing.assert_allclose(combined.variance, expected)


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


def test_linear_combinations_tolerate_zero_variance_constraint_roundoff():
    latent_dimension = 30
    model = CompiledLGM(
        y=np.zeros(latent_dimension),
        observed=np.zeros(latent_dimension, dtype=bool),
        offset=np.zeros(latent_dimension),
        design=csr_matrix(np.eye(latent_dimension)),
        precision=csr_matrix(np.eye(latent_dimension)),
        constraints=np.ones((1, latent_dimension)),
        labels=tuple(f"x{index}" for index in range(latent_dimension)),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    combined = fit_gaussian(model).linear_combinations(np.ones((1, latent_dimension)))

    np.testing.assert_allclose(combined.mean, [0.0])
    np.testing.assert_allclose(combined.variance, [0.0], atol=1e-12)


def test_linear_combinations_reject_materially_negative_variance():
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.array([[-1e-6]]),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([0.0]),
        predictive_variance=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="variance must be non-negative"):
        result.linear_combinations(np.array([[1.0]]))


def test_linear_combinations_reject_nonfinite_variance_from_finite_weights():
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.array([[-1.0]]),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([0.0]),
        predictive_variance=np.array([1.0]),
    )

    with pytest.raises(ValueError, match="propagated covariance"):
        result.linear_combinations(np.array([[np.finfo(float).max]]))


def test_result_diagnostics_reject_mutable_values_and_isolate_input_mapping():
    source = {"latent_dimension": 1}
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.array([[1.0]]),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([0.0]),
        predictive_variance=np.array([1.0]),
        diagnostics=source,
    )
    source["latent_dimension"] = 99

    assert result.diagnostics == {"latent_dimension": 1}
    with pytest.raises(TypeError, match="immutable scalar"):
        GaussianResult(
            labels=("x",),
            mean=np.array([0.0]),
            covariance=np.array([[1.0]]),
            log_marginal_likelihood=0.0,
            predictive_mean=np.array([0.0]),
            predictive_variance=np.array([1.0]),
            diagnostics={"latent_dimension": np.array([1])},
        )


def test_result_rejects_nonfinite_covariance_at_construction():
    with pytest.raises(ValueError, match="covariance must be finite"):
        GaussianResult(
            labels=("x",),
            mean=np.array([0.0]),
            covariance=np.array([[np.inf]]),
            log_marginal_likelihood=0.0,
            predictive_mean=np.array([0.0]),
            predictive_variance=np.array([1.0]),
        )


def test_result_diagnostics_reject_structured_numpy_scalars_with_object_fields():
    mutable_payload: list[object] = []
    diagnostic = np.array(
        [(mutable_payload,)], dtype=[("payload", object)]
    )[0]

    with pytest.raises(TypeError, match="immutable scalar"):
        GaussianResult(
            labels=("x",),
            mean=np.array([0.0]),
            covariance=np.array([[1.0]]),
            log_marginal_likelihood=0.0,
            predictive_mean=np.array([0.0]),
            predictive_variance=np.array([1.0]),
            diagnostics={"payload": diagnostic},
        )


def test_result_diagnostics_normalize_safe_numpy_scalars_to_python_values():
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.array([[1.0]]),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([0.0]),
        predictive_variance=np.array([1.0]),
        diagnostics={"latent_dimension": np.int64(1)},
    )

    assert result.diagnostics == {"latent_dimension": 1}
    assert type(result.diagnostics["latent_dimension"]) is int


def test_linear_combinations_support_empty_latent_dimensions():
    result = GaussianResult(
        labels=(),
        mean=np.empty(0),
        covariance=np.empty((0, 0)),
        log_marginal_likelihood=0.0,
        predictive_mean=np.empty(0),
        predictive_variance=np.empty(0),
    )

    combined = result.linear_combinations(np.empty((1, 0)))

    np.testing.assert_allclose(combined.mean, [0.0])
    np.testing.assert_allclose(combined.variance, [0.0])

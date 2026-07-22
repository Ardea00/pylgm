import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.inference import fit_gaussian
from pylgm.ir.model import CompiledLGM


def test_scalar_gaussian_matches_conjugate_solution() -> None:
    model = CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        sigma=1.0,
        blocks=(),
    )

    result = fit_gaussian(model)

    np.testing.assert_allclose(result.mean, [1.0])
    np.testing.assert_allclose(result.covariance, [[0.5]])
    np.testing.assert_allclose(result.predictive_mean, [1.0])
    np.testing.assert_allclose(result.predictive_variance, [1.5])
    np.testing.assert_allclose(
        result.log_marginal_likelihood, -0.5 * (np.log(4.0 * np.pi) + 2.0)
    )


def test_sum_to_zero_constraint_is_satisfied() -> None:
    model = CompiledLGM(
        y=np.array([1.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        constraints=np.array([[1.0, 1.0]]),
        labels=("a", "b"),
        sigma=1.0,
        blocks=(),
    )

    result = fit_gaussian(model)

    np.testing.assert_allclose(model.constraints @ result.mean, 0.0, atol=1e-12)


def test_missing_observation_is_excluded_from_posterior_and_likelihood() -> None:
    model = CompiledLGM(
        y=np.array([2.0, np.nan]),
        observed=np.array([True, False]),
        offset=np.zeros(2),
        design=csr_matrix([[1.0], [3.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        sigma=1.0,
        blocks=(),
    )

    result = fit_gaussian(model)

    np.testing.assert_allclose(result.mean, [1.0])
    np.testing.assert_allclose(result.predictive_mean, [1.0, 3.0])
    np.testing.assert_allclose(result.predictive_variance, [1.5, 5.5])
    np.testing.assert_allclose(
        result.log_marginal_likelihood, -0.5 * (np.log(4.0 * np.pi) + 2.0)
    )


def test_fully_constrained_latent_space_uses_noise_only_likelihood() -> None:
    model = CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.array([0.5]),
        design=csr_matrix([[4.0]]),
        precision=csr_matrix([[2.0]]),
        constraints=np.array([[1.0]]),
        labels=("x",),
        sigma=2.0,
        blocks=(),
    )

    result = fit_gaussian(model)

    np.testing.assert_allclose(result.mean, [0.0])
    np.testing.assert_allclose(result.covariance, [[0.0]])
    np.testing.assert_allclose(result.predictive_mean, [0.5])
    np.testing.assert_allclose(result.predictive_variance, [4.0])
    np.testing.assert_allclose(
        result.log_marginal_likelihood,
        -0.5 * (np.log(8.0 * np.pi) + 1.5**2 / 4.0),
    )


def test_result_arrays_are_value_isolated() -> None:
    model = CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        sigma=1.0,
        blocks=(),
    )

    result = fit_gaussian(model)

    with pytest.raises(ValueError):
        result.mean[0] = 100.0
    np.testing.assert_allclose(result.mean, [1.0])


def test_non_symmetric_precision_fails_explicitly() -> None:
    model = CompiledLGM(
        y=np.empty(0),
        observed=np.empty(0, dtype=bool),
        offset=np.empty(0),
        design=csr_matrix((0, 2)),
        precision=csr_matrix([[1.0, 5.0], [0.0, 1.0]]),
        constraints=np.empty((0, 2)),
        labels=("a", "b"),
        sigma=1.0,
        blocks=(),
    )

    with pytest.raises(ValueError, match="precision must be symmetric"):
        fit_gaussian(model)


def test_singular_reduced_prior_fails_even_when_likelihood_is_informative() -> None:
    model = CompiledLGM(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[0.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        sigma=1.0,
        blocks=(),
    )

    with pytest.raises(
        np.linalg.LinAlgError, match="reduced prior precision must be positive definite"
    ):
        fit_gaussian(model)

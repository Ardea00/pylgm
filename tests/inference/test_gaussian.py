import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.exceptions import (
    DenseReferenceLimitError,
    ModelValidationError,
    NumericalError,
    UnsupportedEngineError,
)
from pylgm.inference import gaussian
from pylgm.inference import fit_gaussian
from pylgm.ir.model import CompiledLGM, LatentBlock
from pylgm.likelihoods import CompiledGaussian


def _compiled(likelihood: object) -> CompiledLGM:
    return CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=likelihood,
        blocks=(),
    )


def test_exact_gaussian_rejects_an_incompatible_likelihood() -> None:
    with pytest.raises(UnsupportedEngineError, match="Gaussian"):
        fit_gaussian(_compiled(object()))


def test_scalar_gaussian_matches_conjugate_solution() -> None:
    model = CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=CompiledGaussian(1.0),
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


def test_result_exposes_named_block_marginals_and_block_slices() -> None:
    first = LatentBlock(
        name="first",
        labels=("x",),
        design=csr_matrix([[1.0], [0.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
    )
    second = LatentBlock(
        name="second",
        labels=("y",),
        design=csr_matrix([[0.0], [1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
    )
    model = CompiledLGM(
        y=np.array([2.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix(np.eye(2)),
        constraints=np.empty((0, 2)),
        labels=("first:x", "second:y"),
        likelihood=CompiledGaussian(1.0),
        blocks=(first, second),
    )

    result = fit_gaussian(model)

    assert result.block_slices == {"first": slice(0, 1), "second": slice(1, 2)}
    np.testing.assert_allclose(result.latent_marginals("second").mean, [-0.5])


def test_sum_to_zero_constraint_is_satisfied() -> None:
    model = CompiledLGM(
        y=np.array([1.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        constraints=np.array([[1.0, 1.0]]),
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
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
        likelihood=CompiledGaussian(1.0),
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
        likelihood=CompiledGaussian(2.0),
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
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    result = fit_gaussian(model)

    with pytest.raises(ValueError):
        result.mean[0] = 100.0
    np.testing.assert_allclose(result.mean, [1.0])


def test_non_symmetric_precision_fails_explicitly() -> None:
    with pytest.raises(ModelValidationError, match="precision must be symmetric"):
        CompiledLGM(
            y=np.empty(0),
            observed=np.empty(0, dtype=bool),
            offset=np.empty(0),
            design=csr_matrix((0, 2)),
            precision=csr_matrix([[1.0, 5.0], [0.0, 1.0]]),
            constraints=np.empty((0, 2)),
            labels=("a", "b"),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


def test_singular_reduced_prior_fails_even_when_likelihood_is_informative() -> None:
    model = CompiledLGM(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[0.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    with pytest.raises(
        np.linalg.LinAlgError, match="reduced prior precision must be positive definite"
    ):
        fit_gaussian(model)


def test_tiny_nonzero_constraint_row_is_not_discarded() -> None:
    model = CompiledLGM(
        y=np.empty(0),
        observed=np.empty(0, dtype=bool),
        offset=np.empty(0),
        design=csr_matrix((0, 2)),
        precision=csr_matrix(np.eye(2)),
        constraints=np.diag([1.0, 1e-20]),
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    result = fit_gaussian(model)

    np.testing.assert_allclose(result.mean, [0.0, 0.0])
    np.testing.assert_allclose(result.covariance, np.zeros((2, 2)))
    np.testing.assert_allclose(model.constraints @ result.mean, [0.0, 0.0])


def test_redundant_scaled_constraints_preserve_the_same_constrained_posterior() -> None:
    constraints = np.array([[1.0, 1.0], [1e-20, 1e-20]])
    model = CompiledLGM(
        y=np.array([2.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        constraints=constraints,
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    result = fit_gaussian(model)

    np.testing.assert_allclose(constraints @ result.mean, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.mean, [0.5, -0.5])


def test_near_asymmetric_precision_is_rejected() -> None:
    with pytest.raises(ModelValidationError, match="precision must be symmetric"):
        CompiledLGM(
            y=np.empty(0),
            observed=np.empty(0, dtype=bool),
            offset=np.empty(0),
            design=csr_matrix((0, 2)),
            precision=csr_matrix([[2.0, 1e-9], [0.0, 2.0]]),
            constraints=np.empty((0, 2)),
            labels=("a", "b"),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


def test_integer_observation_mask_is_rejected() -> None:
    with pytest.raises(ModelValidationError, match="one-dimensional boolean array"):
        CompiledLGM(
            y=np.array([1.0, 2.0]),
            observed=np.array([1, 0]),
            offset=np.zeros(2),
            design=csr_matrix([[1.0], [1.0]]),
            precision=csr_matrix([[1.0]]),
            constraints=np.empty((0, 1)),
            labels=("x",),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


def test_observation_lengths_must_match_design_rows() -> None:
    with pytest.raises(ModelValidationError, match="row counts must match"):
        CompiledLGM(
            y=np.array([1.0, 2.0]),
            observed=np.array([True, False]),
            offset=np.zeros(2),
            design=csr_matrix([[1.0]]),
            precision=csr_matrix([[1.0]]),
            constraints=np.empty((0, 1)),
            labels=("x",),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


@pytest.mark.parametrize("sigma", [1e-200, np.finfo(float).max])
def test_sigma_must_have_a_finite_positive_variance(sigma: float) -> None:
    model = CompiledLGM(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=CompiledGaussian(float(sigma)),
        blocks=(),
    )

    with pytest.raises(ValueError, match="sigma squared must be finite and positive"):
        fit_gaussian(model)


def test_observed_nan_response_fails_before_zero_dimensional_inference() -> None:
    with pytest.raises(ModelValidationError, match="observed y values must be finite"):
        CompiledLGM(
            y=np.array([np.nan]),
            observed=np.array([True]),
            offset=np.zeros(1),
            design=csr_matrix([[1.0]]),
            precision=csr_matrix([[1.0]]),
            constraints=np.array([[1.0]]),
            labels=("x",),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


@pytest.mark.parametrize("bad_offset", [np.nan, np.inf])
def test_nonfinite_offset_is_rejected_for_predictions(bad_offset: float) -> None:
    with pytest.raises(ModelValidationError, match="offset must be finite"):
        CompiledLGM(
            y=np.array([1.0, np.nan]),
            observed=np.array([True, False]),
            offset=np.array([0.0, bad_offset]),
            design=csr_matrix([[1.0], [1.0]]),
            precision=csr_matrix([[1.0]]),
            constraints=np.empty((0, 1)),
            labels=("x",),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


@pytest.mark.parametrize("bad_design_value", [np.nan, np.inf])
def test_nonfinite_sparse_design_data_is_rejected(bad_design_value: float) -> None:
    with pytest.raises(ModelValidationError, match="design data must be finite"):
        CompiledLGM(
            y=np.array([1.0]),
            observed=np.array([True]),
            offset=np.zeros(1),
            design=csr_matrix([[bad_design_value]]),
            precision=csr_matrix([[1.0]]),
            constraints=np.empty((0, 1)),
            labels=("x",),
            likelihood=CompiledGaussian(1.0),
            blocks=(),
        )


def test_extreme_finite_response_raises_typed_numerical_error_without_warnings() -> None:
    model = CompiledLGM(
        y=np.array([1e308]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(NumericalError, match="non-finite|numerical"):
            fit_gaussian(model)

    assert not recorded


def test_dense_reference_dimension_limit_runs_before_dense_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompiledLGM(
        y=np.array([1.0, 2.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix(np.eye(2)),
        constraints=np.empty((0, 2)),
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )
    monkeypatch.setattr(gaussian, "_MAX_DENSE_LATENT_DIMENSION", 1)
    monkeypatch.setattr(
        gaussian,
        "null_space",
        lambda constraints: pytest.fail("dense algebra ran before preflight"),
    )

    with pytest.raises(DenseReferenceLimitError, match="allow_large_dense"):
        fit_gaussian(model)


def test_dense_reference_estimated_byte_limit_can_be_explicitly_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompiledLGM(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        design=csr_matrix([[1.0]]),
        precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)),
        labels=("x",),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )
    monkeypatch.setattr(gaussian, "_MAX_DENSE_BYTES", 1)

    with pytest.raises(DenseReferenceLimitError, match="estimated dense workspace"):
        fit_gaussian(model)

    result = fit_gaussian(model, allow_large_dense=True)
    np.testing.assert_allclose(result.mean, [1.0])


def test_constrained_gaussian_matches_independent_one_coordinate_solution() -> None:
    model = CompiledLGM(
        y=np.array([2.0, -1.0]),
        observed=np.array([True, True]),
        offset=np.zeros(2),
        design=csr_matrix(np.eye(2)),
        precision=csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        constraints=np.array([[1.0, 1.0]]),
        labels=("a", "b"),
        likelihood=CompiledGaussian(1.0),
        blocks=(),
    )

    result = fit_gaussian(model)

    basis = np.array([1.0, -1.0]) / np.sqrt(2.0)
    prior_precision = float(basis @ model.precision.toarray() @ basis)
    posterior_precision = prior_precision + float(basis @ basis)
    score = float(basis @ model.y)
    coordinate_mean = score / posterior_precision
    expected_mean = basis * coordinate_mean
    expected_covariance = np.outer(basis, basis) / posterior_precision
    expected_quadratic = float(
        np.sum((model.y - basis * coordinate_mean) ** 2)
        + prior_precision * coordinate_mean**2
    )
    expected_log_marginal = -0.5 * (
        2 * np.log(2 * np.pi)
        - np.log(prior_precision)
        + np.log(posterior_precision)
        + expected_quadratic
    )
    np.testing.assert_allclose(result.mean, expected_mean)
    np.testing.assert_allclose(result.covariance, expected_covariance)
    np.testing.assert_allclose(result.log_marginal_likelihood, expected_log_marginal)

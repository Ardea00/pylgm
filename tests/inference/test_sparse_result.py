# tests/inference/test_sparse_result.py
import numpy as np
import pytest

from pylgm.inference.result import GaussianResult


def _covless_result():
    return GaussianResult(
        labels=("a:1", "a:2"),
        mean=np.array([1.0, 2.0]),
        covariance=None,
        log_marginal_likelihood=-3.0,
        predictive_mean=np.array([0.5, 1.5]),
        predictive_variance=None,
        observation_variance=1.0,
        block_slices={"a": slice(0, 2)},
    )


def test_covless_result_exposes_mean_and_lml():
    result = _covless_result()
    assert np.allclose(result.mean, [1.0, 2.0])
    assert result.log_marginal_likelihood == -3.0
    assert np.allclose(result.predictive_mean, [0.5, 1.5])


def test_covless_result_raises_on_uncertainty():
    result = _covless_result()
    for accessor in (
        lambda: result.covariance,
        lambda: result.predictive_variance,
        lambda: result.latent_marginals(),
        lambda: result.linear_combinations(np.eye(2)),
    ):
        with pytest.raises(NotImplementedError, match="E-sparse-C"):
            accessor()


def test_rebuild_preserves_absent_covariance():
    from pylgm.model import _rebuild_result

    result = _covless_result()
    rebuilt = _rebuild_result(result, hyperparameters={"tau": 1.0})
    assert rebuilt._covariance is None
    assert rebuilt._predictive_variance is None
    assert np.allclose(rebuilt.mean, [1.0, 2.0])
    assert rebuilt.hyperparameters["tau"] == 1.0


def test_align_reorders_without_covariance():
    from pylgm.model import _align_predictions_with_source_rows

    result = _covless_result()
    aligned = _align_predictions_with_source_rows(result, np.array([1, 0]))
    assert aligned._covariance is None
    assert aligned._predictive_variance is None

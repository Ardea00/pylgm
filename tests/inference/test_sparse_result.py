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

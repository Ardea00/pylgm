# tests/inference/test_sparse_result.py
import numpy as np
import pytest

from pylgm.inference.result import GaussianResult


@pytest.fixture
def small_model_forced_sparse(proper_car_with_intercept_model):
    """A small, well-conditioned model passed straight to ``_fit_sparse``.

    ``_fit_sparse`` does not consult the dense-size guard, so any compiled
    model works; reuses the well-conditioned no-constraint proper-CAR fixture
    from ``tests/inference/conftest.py`` (rather than the Besag+intercept
    fixture) because the intrinsic Besag field is ill-conditioned
    (cond ~ 2.5e9, per Tasks 3-4), which makes the tight ``atol=1e-7``
    marginal-variance comparison against the dense reference too fragile for
    reasons unrelated to this task's routing logic.
    """
    return proper_car_with_intercept_model


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


def test_sparse_result_marginals_match_dense_but_covariance_guarded(small_model_forced_sparse):
    """Past the guard: scoped accessors work via the posterior; covariance raises."""
    from pylgm.exceptions import DenseReferenceLimitError
    from pylgm.inference.gaussian import _fit_dense, _fit_sparse

    model = small_model_forced_sparse
    dense = _fit_dense(model)
    sparse = _fit_sparse(model)

    # marginal variances match
    got = sparse.latent_marginals().variance
    assert np.allclose(got, dense.latent_marginals().variance, atol=1e-7)
    # linear combinations match too
    weights = np.eye(len(model.labels))
    got_combo = sparse.linear_combinations(weights).variance
    want_combo = dense.linear_combinations(weights).variance
    assert np.allclose(got_combo, want_combo, atol=1e-7)
    # predictive variance materialised on the result
    assert np.allclose(sparse.predictive_variance, dense.predictive_variance, atol=1e-7)
    # covariance is guarded (scale error, NOT pending-C)
    with pytest.raises(DenseReferenceLimitError) as exc:
        _ = sparse.covariance
    assert "E-sparse-C" not in str(exc.value)


def test_rebuild_preserves_sparse_posterior(small_model_forced_sparse):
    from pylgm.inference.gaussian import _fit_sparse
    from pylgm.model import _rebuild_result

    sparse = _fit_sparse(small_model_forced_sparse)
    order = np.arange(sparse.predictive_mean.size)[::-1]
    rebuilt = _rebuild_result(sparse, caller_order=order)
    assert rebuilt._sparse_posterior is sparse._sparse_posterior
    # latent marginals still computable after rebuild
    assert np.all(np.isfinite(rebuilt.latent_marginals().variance))


def test_sparse_predict_matches_dense(proper_car_with_intercept_model):
    import pandas as pd

    from pylgm.inference.gaussian import _fit_sparse
    from pylgm.inference.prediction import PredictionContext, predict_from
    from pylgm.model import _rebuild_result

    model = proper_car_with_intercept_model
    sparse = _fit_sparse(model)
    posterior = sparse._sparse_posterior
    dense_cov = posterior.covariance_dense()

    labels = tuple(str(label) for label in model.labels)
    # A one-hot structured block spanning the full latent vector; new_data
    # picks the first few latent rows. width == latent_size so design @ mean
    # conforms -- this is a routing test, not a domain-realistic design.
    context = PredictionContext(
        entries=(("structured", ("node", "node", labels)),),
        likelihood=model.likelihood,
        offset=None,
        width=len(labels),
    )
    new_data = pd.DataFrame({"node": list(labels[:5])})

    want = predict_from(context, sparse.mean, dense_cov, new_data)
    rebuilt = _rebuild_result(sparse, prediction_context=context)
    got = rebuilt.predict(new_data)

    assert np.allclose(got.predictive_mean, want.predictive_mean, atol=1e-7)
    assert np.allclose(got.predictive_variance, want.predictive_variance, atol=1e-7)
    assert np.allclose(got.fitted_mean, want.fitted_mean, atol=1e-7)

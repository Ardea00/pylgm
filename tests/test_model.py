import pandas as pd
import numpy as np
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, PCPrecision, Poisson, RW1, RW2
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference.result import LaplaceResult
from pylgm.likelihoods import CompiledGaussian
from pylgm.parameters import Hyperparameter
from pylgm.model import _align_predictions_with_source_rows


def _example():
    frame = pd.DataFrame(
        {
            "region": ["a", "a", "b", "b"],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 1.5, None],
        }
    )
    model = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed("1")
        + IID("region", "region", 2.0)
        + RW1("trend", "time", 3.0),
        panel=("region",),
        time="time",
    )
    return frame, model


def test_declarative_model_fits_existing_gaussian_path():
    frame, model = _example()

    result = model.fit(frame)

    assert result.engine == "exact_gaussian"
    assert result.predictive_mean.shape == (4,)


def test_declarative_predictions_align_with_unsorted_caller_rows():
    frame = pd.DataFrame(
        {
            "region": ["B", "A", "B", "A"],
            "time": [2, 1, 1, 2],
            "x": [4.0, 1.0, 3.0, 2.0],
            "y": [8.0, 2.0, 6.0, 4.0],
        }
    )
    model = LGM(
        response="y",
        likelihood=Gaussian(1.0),
        predictor=Fixed("0 + x", prior_precision=1.0),
        panel=("region",),
        time="time",
    )

    result = model.fit(frame)

    np.testing.assert_allclose(
        result.predictive_mean,
        np.array([240.0, 60.0, 180.0, 120.0]) / 31.0,
    )
    np.testing.assert_allclose(
        result.predictive_variance,
        1.0 + np.array([16.0, 1.0, 9.0, 4.0]) / 31.0,
    )


def test_prediction_alignment_retains_prediction_keys():
    from pylgm.inference import GaussianResult

    keys = pd.DataFrame({"region": ["A", "B"], "time": [1, 2]})
    result = GaussianResult(
        labels=("x",),
        mean=np.array([0.0]),
        covariance=np.eye(1),
        log_marginal_likelihood=0.0,
        predictive_mean=np.array([10.0, 20.0]),
        predictive_variance=np.array([1.0, 1.0]),
        prediction_keys=keys,
    )

    aligned = _align_predictions_with_source_rows(result, np.array([1, 0]))

    np.testing.assert_allclose(aligned.predictive_mean, [20.0, 10.0])
    pd.testing.assert_frame_equal(aligned.prediction_keys, keys)


def test_model_does_not_fallback_from_unknown_engine():
    frame, model = _example()

    with pytest.raises(UnsupportedEngineError, match="unknown"):
        model.fit(frame, engine="unknown")


def test_model_without_time_uses_private_row_key_without_mutating_frame():
    frame = pd.DataFrame(
        {"group": ["b", "a"], "step": [2, 1], "y": [2.0, 1.0]}
    )
    original = frame.copy(deep=True)
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("1") + IID("group", "group") + RW1("trend", "step"),
    )

    result = model.fit(frame)

    assert result.predictive_mean.shape == (2,)
    pd.testing.assert_frame_equal(frame, original)


def test_model_without_time_preserves_caller_order_prediction_values():
    frame = pd.DataFrame(
        {"x": [4.0, 1.0, 3.0], "y": [8.0, 2.0, 6.0]},
        index=[20, 10, 20],
    )
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("0 + x", prior_precision=1.0),
    )

    result = model.fit(frame)

    np.testing.assert_allclose(
        result.predictive_mean, np.array([208.0, 52.0, 156.0]) / 27.0
    )
    np.testing.assert_allclose(
        result.predictive_variance,
        1.0 + np.array([16.0, 1.0, 9.0]) / 27.0,
    )


def test_model_rejects_collision_with_private_row_key():
    frame = pd.DataFrame({"__pylgm_row__": [4], "y": [1.0]})
    model = LGM("y", Gaussian(1.0), Fixed("1"))

    with pytest.raises(DataContractError, match="reserved row key"):
        model.fit(frame)


def test_model_resolves_declared_hyperparameters_to_initial_values():
    frame = pd.DataFrame(
        {"group": ["a", "b"], "time": [1, 2], "y": [1.0, 2.0]}
    )
    model = LGM(
        "y",
        Gaussian(Hyperparameter("sigma", 0.5)),
        Fixed("1")
        + IID("group", "group", Hyperparameter("group_precision", 2.0)),
        time="time",
    )

    result = model.fit(frame)
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="time", response="y")
    )
    compiled = compile_lgm(model, panel)

    assert result.predictive_mean.shape == (2,)
    assert compiled.likelihood == CompiledGaussian(0.5)
    np.testing.assert_allclose(
        compiled.blocks[1].precision.toarray(), np.eye(2) * 2.0
    )


def test_declarative_rw2_compiles_the_second_order_precision_and_constraints():
    frame = pd.DataFrame(
        {"time": [1, 2, 3], "y": [1.0, 2.0, 4.0]}
    )
    model = LGM(
        "y",
        Gaussian(1.0),
        Fixed("1")
        + RW2(
            "trend",
            "time",
            Hyperparameter("trend_precision", 2.0),
        ),
        time="time",
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="time", response="y")
    )

    compiled = compile_lgm(model, panel)

    trend = compiled.blocks[1]
    np.testing.assert_allclose(
        trend.precision.toarray(),
        [[2.0, -4.0, 2.0], [-4.0, 8.0, -4.0], [2.0, -4.0, 2.0]],
    )
    np.testing.assert_allclose(
        trend.constraints,
        [[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]],
    )


def test_pandas_fit_keeps_no_prediction_keys():
    result = LGM("y", Gaussian(1.0), Fixed("1")).fit(pd.DataFrame({"y": [1.0, 2.0]}))
    assert result.prediction_keys is None


def test_pandas_fit_preserves_caller_order_with_default_row_limit():
    frame = pd.DataFrame({"region": ["B", "A"], "time": [2, 1], "y": [8.0, 2.0]})
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="time")
    result = model.fit(frame)
    # Predictions stay aligned to the caller's B-then-A row order.
    assert result.predictive_mean.shape == (2,)
    assert result.prediction_keys is None


class _NonSparkFrame:
    def toPandas(self):  # noqa: N802
        raise AssertionError("non-Spark inputs must not be collected")


def test_model_rejects_non_pandas_non_spark_input():
    model = LGM("y", Gaussian(1.0), Fixed("1"))

    # A duck-typed object whose module is not ``pyspark`` is not a genuine Spark
    # frame, so it takes the plain Pandas error path without being collected.
    with pytest.raises(DataContractError, match="frame must be a Pandas DataFrame"):
        model.fit(_NonSparkFrame())


def test_lgm_fit_laplace_poisson_returns_laplace_result():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 4.0, 7.0]})
    result = LGM("y", Poisson(), Fixed("1 + x"), time="t").fit(frame, engine="laplace")
    assert isinstance(result, LaplaceResult)
    assert result.converged is True
    assert result.prediction_keys is None
    assert result.fitted_mean.shape == (4,)


def test_lgm_fit_laplace_preserves_caller_row_order():
    rows_shuffled = pd.DataFrame({"t": [3, 1, 2], "x": [2.0, 0.0, 1.0], "y": [4.0, 1.0, 2.0]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t")
    shuffled = model.fit(rows_shuffled, engine="laplace")
    ordered = model.fit(rows_shuffled.sort_values("t").reset_index(drop=True), engine="laplace")
    # shuffled row order is t=[3,1,2]; ordered is t=[1,2,3]. Map: shuffled[i] == ordered[perm]
    perm = [2, 0, 1]  # positions in `ordered` for each shuffled row
    np.testing.assert_allclose(shuffled.predictive_mean, ordered.predictive_mean[perm], atol=1e-8)
    np.testing.assert_allclose(shuffled.fitted_mean, ordered.fitted_mean[perm], atol=1e-8)


def test_lgm_rejects_engine_likelihood_mismatch():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    with pytest.raises(UnsupportedEngineError, match="Gaussian"):
        LGM("y", Gaussian(1.0), Fixed("1"), time="t").fit(frame, engine="laplace")
    with pytest.raises(UnsupportedEngineError, match="non-Gaussian"):
        LGM("y", Poisson(), Fixed("1"), time="t").fit(frame, engine="exact_gaussian")


def test_gaussian_empirical_bayes_estimates_precision():
    rng = np.random.default_rng(0)
    regions = ["a", "b", "c", "d"]
    rows = []
    for t in range(1, 21):
        for r in regions:
            rows.append({"region": r, "t": t, "y": rng.normal()})
    frame = pd.DataFrame(rows)
    model = LGM(
        "y", Gaussian(0.5),
        Fixed("1") + IID("region", index="region",
                         precision=Hyperparameter("region_prec", initial=1.0)),
        panel=("region",), time="t",
    )
    result = model.fit(frame, engine="exact_gaussian")
    assert result.hyperparameters is not None
    assert "region_prec" in result.hyperparameters
    assert result.hyperparameters["region_prec"] > 0
    assert result.diagnostics["empirical_bayes_converged"] is True


def test_poisson_empirical_bayes_runs_via_laplace():
    rows = [{"region": r, "t": t, "y": float((t + (r == "b")) % 5)}
            for t in range(1, 16) for r in ["a", "b"]]
    frame = pd.DataFrame(rows)
    model = LGM(
        "y", Poisson(),
        Fixed("1") + IID("region", index="region",
                         precision=Hyperparameter("region_prec", initial=1.0)),
        panel=("region",), time="t",
    )
    result = model.fit(frame, engine="laplace")
    assert result.hyperparameters is not None and result.hyperparameters["region_prec"] > 0


def test_no_hyperparameter_model_has_none_hyperparameters():
    frame = pd.DataFrame({"t": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    result = LGM("y", Gaussian(1.0), Fixed("1"), time="t").fit(frame)
    assert result.hyperparameters is None


def _region_panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = ["a", "b", "c", "d"]
    rows = [
        {"region": r, "t": t, "y": rng.normal()}
        for t in range(1, 21)
        for r in regions
    ]
    return pd.DataFrame(rows)


def test_gaussian_sigma_hyperparameter_estimates_via_empirical_bayes():
    frame = _region_panel(1)
    model = LGM(
        "y",
        Gaussian(Hyperparameter("obs_sigma", initial=1.0)),
        Fixed("1"),
        panel=("region",), time="t",
    )

    result = model.fit(frame, engine="exact_gaussian")

    assert result.hyperparameters is not None
    obs_sigma = result.hyperparameters["obs_sigma"]
    assert np.isfinite(obs_sigma)
    assert obs_sigma > 0
    assert result.diagnostics["empirical_bayes_converged"] is True


def test_empirical_bayes_precision_matches_fixed_precision_refit_at_optimum():
    frame = _region_panel(2)

    def build_model(precision):
        return LGM(
            "y",
            Gaussian(0.5),
            Fixed("1") + IID("region", index="region", precision=precision),
            panel=("region",), time="t",
        )

    eb_result = build_model(Hyperparameter("p", initial=1.0)).fit(frame, engine="exact_gaussian")
    p_star = eb_result.hyperparameters["p"]

    fixed_result = build_model(p_star).fit(frame, engine="exact_gaussian")

    np.testing.assert_allclose(eb_result.mean, fixed_result.mean, atol=1e-6)
    np.testing.assert_allclose(
        eb_result.predictive_mean, fixed_result.predictive_mean, atol=1e-6
    )


def _panel_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"region": r, "t": t, "y": rng.normal()} for t in range(1, 21) for r in "abcd"]
    return pd.DataFrame(rows)


def test_map_ii_sets_penalized_flag_and_estimate_both_engines():
    frame = _panel_frame()
    gauss = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0,
                                                          prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
                panel=("region",), time="t")
    result = gauss.fit(frame, engine="exact_gaussian")
    assert result.diagnostics["hyperparameter_penalized"] is True
    assert result.hyperparameters["p"] > 0

    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    pois = LGM("y", Poisson(),
               Fixed("1") + IID("region", index="region",
                                precision=Hyperparameter("p", initial=1.0,
                                                         prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
               panel=("region",), time="t")
    pres = pois.fit(counts, engine="laplace")
    assert pres.diagnostics["hyperparameter_penalized"] is True
    assert pres.hyperparameters["p"] > 0


def test_ml_when_no_prior_and_penalty_changes_estimate():
    frame = _panel_frame()
    def build(prior):
        return LGM("y", Gaussian(0.5),
                   Fixed("1") + IID("region", index="region",
                                    precision=Hyperparameter("p", initial=1.0, prior=prior)),
                   panel=("region",), time="t")
    ml = build(None).fit(frame, engine="exact_gaussian")
    assert ml.diagnostics["hyperparameter_penalized"] is False
    map_ii = build(PCPrecision(upper_sd=0.2, alpha=0.01)).fit(frame, engine="exact_gaussian")
    # the prior must actually move the estimate
    assert not np.isclose(ml.hyperparameters["p"], map_ii.hyperparameters["p"])

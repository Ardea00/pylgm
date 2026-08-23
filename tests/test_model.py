import pandas as pd
import numpy as np
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, PCPrecision, Poisson, RW1, RW2
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference.result import INLAResult, LaplaceResult, ModelCriteria
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


def test_mixed_hyperparameters_penalize_only_the_prior_bearing_one():
    frame = _panel_frame()
    model = LGM(
        "y", Gaussian(0.5),
        Fixed("1")
        + IID("region", index="region",
              precision=Hyperparameter("region_precision", initial=1.0,
                                       prior=PCPrecision(upper_sd=1.0, alpha=0.01)))
        + IID("time_effect", index="t",
              precision=Hyperparameter("time_precision", initial=1.0)),
        panel=("region",), time="t",
    )
    result = model.fit(frame, engine="exact_gaussian")
    assert result.diagnostics["hyperparameter_penalized"] is True
    assert "region_precision" in result.hyperparameters
    assert "time_precision" in result.hyperparameters
    assert result.hyperparameters["region_precision"] > 0
    assert result.hyperparameters["time_precision"] > 0
    assert result.diagnostics["empirical_bayes_converged"] is True


def _region_panel(seed=2):
    # Real per-region effects (not pure noise) so the region precision
    # hyperparameter has a non-degenerate interior posterior mode: an
    # empirical Bayes fit on iid-across-region data drives the precision to
    # its upper bound (zero region variance), which collapses the INLA grid
    # to a single point. See tests/optimization/test_inla.py's
    # `_one_hyperparameter_family` for the same "informative" requirement.
    # seed=2 also keeps every grid-point Laplace conditional fit (Poisson
    # branch) inside its Newton-convergence tolerance.
    rng = np.random.default_rng(seed)
    regions = list("abcd")
    effects = {r: rng.normal(0.0, 1.0) for r in regions}
    return pd.DataFrame([{"region": r, "t": t, "y": effects[r] + rng.normal(0.0, 0.5)}
                         for t in range(1, 21) for r in regions])


def test_fit_integrate_returns_inla_result_both_engines():
    frame = _region_panel()
    gauss = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    result = gauss.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    assert isinstance(result, INLAResult)
    assert result.engine == "inla"
    assert "p" in result.hyperparameter_marginals()
    assert result.diagnostics["inla_grid_points"] >= 3

    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    pois = LGM("y", Poisson(),
               Fixed("1") + IID("region", index="region",
                                precision=Hyperparameter("p", initial=1.0)),
               panel=("region",), time="t")
    pres = pois.fit(counts, engine="laplace", hyperparameters="integrate")
    assert isinstance(pres, INLAResult)
    assert pres.fitted_mean is not None


def test_integrate_requires_a_hyperparameter():
    frame = _region_panel()
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="t")
    with pytest.raises(ValueError, match="integrate"):
        model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")


def test_inla_result_exposes_model_criteria_both_engines():
    rng = np.random.default_rng(0)
    regions = [f"r{i}" for i in range(30)]
    effects = {r: rng.normal() for r in regions}
    rows = [{"region": r, "t": t, "y": effects[r] + rng.normal()}
            for t in range(6) for r in regions]
    frame = pd.DataFrame(rows)
    gauss = LGM("y", Gaussian(1.0),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    result = gauss.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    crit = result.criteria
    assert isinstance(crit, ModelCriteria)
    assert np.isfinite(crit.dic) and np.isfinite(crit.waic)
    n_obs = len(frame)
    assert crit.cpo.shape == (n_obs,) and crit.pit.shape == (n_obs,)
    assert np.all((crit.pit >= 0) & (crit.pit <= 1))

    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    pois = LGM("y", Poisson(),
               Fixed("1") + IID("region", index="region",
                                precision=Hyperparameter("p", initial=1.0)),
               panel=("region",), time="t")
    pres = pois.fit(counts, engine="laplace", hyperparameters="integrate")
    assert np.isfinite(pres.criteria.waic)


def test_fit_optimize_default_unchanged():
    frame = _region_panel()
    model = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    default = model.fit(frame, engine="exact_gaussian")
    explicit = model.fit(frame, engine="exact_gaussian", hyperparameters="optimize")
    np.testing.assert_allclose(default.hyperparameters["p"], explicit.hyperparameters["p"])


def test_invalid_hyperparameters_mode_rejected():
    frame = _region_panel()
    model = LGM("y", Gaussian(1.0), Fixed("1"), panel=("region",), time="t")
    with pytest.raises(ValueError, match="hyperparameters"):
        model.fit(frame, engine="exact_gaussian", hyperparameters="nonsense")


def test_criteria_cpo_pit_are_row_aligned_with_caller_order():
    # cpo/pit are computed internally in CanonicalPanel (sorted) order but must be
    # realigned to caller row order just like predictive_mean/predictive_variance,
    # since result.criteria.cpo[i] is documented to describe the same observation
    # as result.predictive_mean[i].
    base = _region_panel()
    sort_order = base.sort_values(["region", "t"]).index.to_numpy()
    sorted_frame = base.iloc[sort_order].reset_index(drop=True)
    rng = np.random.default_rng(7)
    shuffled_index = rng.permutation(len(base))
    shuffled_frame = base.iloc[shuffled_index].reset_index(drop=True)

    model = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")

    sorted_result = model.fit(sorted_frame, engine="exact_gaussian", hyperparameters="integrate")
    shuffled_result = model.fit(shuffled_frame, engine="exact_gaussian", hyperparameters="integrate")

    assert shuffled_result.criteria.cpo.shape[0] == len(shuffled_frame)
    assert shuffled_result.criteria.pit.shape[0] == len(shuffled_frame)
    assert shuffled_result.criteria.cpo.shape[0] == shuffled_result.predictive_mean.shape[0]

    # Scatter the shuffled fit's cpo/pit back into base's original row positions
    # (recovered_cpo[shuffled_index[k]] = shuffled_result.cpo[k]), then apply the
    # same region/t sort used to build sorted_frame: the two fits operate on the
    # same 80 observations, so once both are viewed in that common canonical row
    # order they must match exactly (INLA fitting itself is caller-order-independent;
    # only the final realignment differs).
    recovered_cpo = np.empty_like(shuffled_result.criteria.cpo)
    recovered_cpo[shuffled_index] = shuffled_result.criteria.cpo
    recovered_pit = np.empty_like(shuffled_result.criteria.pit)
    recovered_pit[shuffled_index] = shuffled_result.criteria.pit

    np.testing.assert_allclose(recovered_cpo[sort_order], sorted_result.criteria.cpo)
    np.testing.assert_allclose(recovered_pit[sort_order], sorted_result.criteria.pit)


def test_criteria_cpo_pit_have_nan_at_unobserved_rows():
    frame = _region_panel().reset_index(drop=True)
    unobserved_positions = [0, 5, 17]
    frame.loc[unobserved_positions, "y"] = np.nan

    model = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    result = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")

    cpo = result.criteria.cpo
    pit = result.criteria.pit
    assert cpo.shape[0] == len(frame) == pit.shape[0]

    observed_mask = frame["y"].notna().to_numpy()
    assert np.all(np.isnan(cpo[~observed_mask]))
    assert np.all(np.isnan(pit[~observed_mask]))
    assert np.all(np.isfinite(cpo[observed_mask]))
    assert np.all(np.isfinite(pit[observed_mask]))


def test_sla_latent_strategy_returns_skew_marginals_both_engines():
    from pylgm import Fixed, IID, LGM, Poisson, Hyperparameter
    from pylgm.inference.result import SkewNormalMarginals, GaussianMarginals

    rng = np.random.default_rng(0)
    regions = [f"r{i}" for i in range(25)]
    eff = {r: rng.normal() for r in regions}
    rows = [{"region": r, "t": t, "y": eff[r] + rng.normal()} for t in range(6) for r in regions]
    frame = pd.DataFrame(rows)
    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    model = LGM("y", Poisson(),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    sla = model.fit(counts, engine="laplace", hyperparameters="integrate",
                    latent_strategy="simplified_laplace")
    assert isinstance(sla.latent_marginals("region"), SkewNormalMarginals)
    # default stays Gaussian
    gauss = model.fit(counts, engine="laplace", hyperparameters="integrate")
    assert isinstance(gauss.latent_marginals("region"), GaussianMarginals)


def test_sla_gaussian_exact_gaussian_integrated_mean_variance_matches_gaussian_strategy():
    # Per grid point, a Gaussian likelihood's SLA correction is exactly zero (d3=0),
    # so each conditional marginal is exactly Gaussian and the SLA and Gaussian
    # strategies share the same underlying grid mixture -> integrated mean/variance
    # must match to tight tolerance. The *integrated* marginal is a grid-weighted
    # mixture of Gaussians with theta-varying means, which is itself skewed in
    # general -- this test intentionally does NOT assert skewness == 0.
    frame = _region_panel()
    model = LGM("y", Gaussian(0.5),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    sla = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate",
                    latent_strategy="simplified_laplace")
    gauss = model.fit(frame, engine="exact_gaussian", hyperparameters="integrate")
    assert sla.diagnostics["inla_grid_points"] >= 3  # genuine multi-grid mixture
    assert "skew_clamped" in sla.diagnostics

    sla_marg = sla.latent_marginals("region")
    gauss_marg = gauss.latent_marginals("region")
    np.testing.assert_allclose(sla_marg.mean, gauss_marg.mean, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(sla_marg.variance, gauss_marg.variance, atol=1e-8, rtol=1e-6)
    assert np.isfinite(sla_marg.skewness).all()


def test_sla_requires_integration_and_valid_strategy():
    from pylgm import Fixed, Gaussian, IID, LGM, Hyperparameter
    frame = pd.DataFrame({"region": ["a", "b"] * 10, "t": list(range(20)),
                          "y": np.random.default_rng(0).normal(size=20)})
    model = LGM("y", Gaussian(1.0),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    with pytest.raises(ValueError, match="integrate"):
        model.fit(frame, hyperparameters="optimize", latent_strategy="simplified_laplace")
    with pytest.raises(ValueError, match="latent_strategy"):
        model.fit(frame, hyperparameters="integrate", latent_strategy="nonsense")


def test_full_laplace_latent_strategy_returns_tabulated_both_engines():
    from pylgm import Fixed, Gaussian, IID, LGM, Poisson, Hyperparameter
    from pylgm.inference.result import TabulatedMarginals
    rng = np.random.default_rng(0)
    regions = [f"r{i}" for i in range(25)]
    eff = {r: rng.normal() for r in regions}
    rows = [{"region": r, "t": t, "x": rng.normal(), "y": eff[r] + rng.normal()}
            for t in range(6) for r in regions]
    frame = pd.DataFrame(rows)
    counts = frame.assign(y=(frame["y"].abs() * 3).round())
    # UNCONSTRAINED: Fixed + IID (no RW)
    model = LGM("y", Poisson(),
                Fixed("1 + x") + IID("region", index="region",
                                     precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    res = model.fit(counts, engine="laplace", hyperparameters="integrate",
                    latent_strategy="laplace")
    assert isinstance(res.latent_marginals("region"), TabulatedMarginals)

    gauss_model = LGM("y", Gaussian(1.0),
                      Fixed("1") + IID("region", index="region",
                                       precision=Hyperparameter("p", initial=1.0)),
                      panel=("region",), time="t")
    gauss_frame = frame.drop(columns=["x"])
    gauss_res = gauss_model.fit(gauss_frame, engine="exact_gaussian", hyperparameters="integrate",
                                latent_strategy="laplace")
    assert isinstance(gauss_res.latent_marginals("region"), TabulatedMarginals)


def test_full_laplace_rejects_constrained_effects():
    from pylgm import Fixed, Gaussian, LGM, RW1, Hyperparameter
    from pylgm.exceptions import UnsupportedEngineError
    rng = np.random.default_rng(0)
    rows = [{"t": t, "y": rng.normal()} for t in range(1, 40)]
    frame = pd.DataFrame(rows)
    model = LGM("y", Gaussian(1.0),
                Fixed("1") + RW1("trend", index="t",
                                 precision=Hyperparameter("p", initial=1.0)),
                time="t")
    with pytest.raises(UnsupportedEngineError, match="constrained|laplace"):
        model.fit(frame, engine="exact_gaussian", hyperparameters="integrate",
                  latent_strategy="laplace")


def test_full_laplace_requires_integration():
    from pylgm import Fixed, Gaussian, IID, LGM, Hyperparameter
    frame = pd.DataFrame({"region": ["a", "b"] * 10, "t": list(range(20)),
                          "y": np.random.default_rng(0).normal(size=20)})
    model = LGM("y", Gaussian(1.0),
                Fixed("1") + IID("region", index="region",
                                 precision=Hyperparameter("p", initial=1.0)),
                panel=("region",), time="t")
    with pytest.raises(ValueError, match="integrate"):
        model.fit(frame, hyperparameters="optimize", latent_strategy="laplace")


def test_penalty_prefers_family_parameter_priors():
    # A prior carried on the family (e.g. a graph-bound PC prior) must drive the
    # MAP-II penalty, not just the prior declared on the Hyperparameter.
    from pylgm.compiler import compile_family

    frame = _panel_frame()
    model = LGM(
        "y", Gaussian(0.5),
        Fixed("1") + IID("region", index="region",
                         precision=Hyperparameter("p", initial=1.0)),
        panel=("region",), time="t",
    )
    data = DataConfig(time="t", response="y", panel=("region",))
    panel = CanonicalPanel.from_frame(frame, data)
    family = compile_family(model, panel)
    assert family is not None
    assert family.parameter_priors == {}

    calls = []

    class _SentinelPrior:
        def logpdf(self, value):
            calls.append(value)
            return -7.0

    object.__setattr__(family, "parameter_priors", {"p": _SentinelPrior()})

    bounds, initial, penalty = model._family_optimization_inputs(family)
    assert penalty is not None
    assert penalty({"p": 3.0}) == -7.0
    assert calls == [3.0]


def test_penalty_avoids_double_counting_family_and_hyperparameter_priors():
    # When a parameter has both a family-bound prior and its own Hyperparameter
    # prior (same name), only the family's bound prior must contribute -- the
    # Hyperparameter's own prior (a real PCPrecision, which would evaluate to a
    # very different, finite number here) must not also be added in.
    from pylgm.compiler import compile_family

    frame = _panel_frame()
    model = LGM(
        "y", Gaussian(0.5),
        Fixed("1") + IID("region", index="region",
                         precision=Hyperparameter("p", initial=1.0,
                                                  prior=PCPrecision(upper_sd=1.0, alpha=0.01))),
        panel=("region",), time="t",
    )
    data = DataConfig(time="t", response="y", panel=("region",))
    panel = CanonicalPanel.from_frame(frame, data)
    family = compile_family(model, panel)
    assert family is not None

    family_calls = []

    class _SentinelPrior:
        def logpdf(self, value):
            family_calls.append(value)
            return -1000.0

    object.__setattr__(family, "parameter_priors", {"p": _SentinelPrior()})

    _, _, penalty = model._family_optimization_inputs(family)
    assert penalty is not None
    assert penalty({"p": 2.0}) == -1000.0
    assert family_calls == [2.0]

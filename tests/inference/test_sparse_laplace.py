"""Sparse Laplace engine: the dense Laplace engine is the oracle below the guard.

Both run the same Newton iteration to the same mode, so mode, log marginal
likelihood, marginals and predictions must agree to solver tolerance.
"""

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import (
    AR1, IID, LGM, RW1, RW2, Besag, Binomial, Fixed, Gaussian, NegativeBinomial, Poisson, SpaceTime,
    WeibullSurv,
)
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.inference.gaussian import fit_gaussian
from pylgm.inference.laplace import fit_laplace

REGIONS, PERIODS = 6, 12
RING = {f"r{i}": [f"r{(i - 1) % REGIONS}", f"r{(i + 1) % REGIONS}"] for i in range(REGIONS)}


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        [(f"r{r}", t) for r in range(REGIONS) for t in range(PERIODS)], columns=["region", "t"]
    )
    frame["x"] = rng.normal(size=len(frame))
    spatial = 0.5 * np.sin(np.arange(REGIONS))[frame["region"].str[1:].astype(int)]
    trend = 0.3 * np.sin(frame["t"] / 3.0)
    eta = 0.4 + 0.3 * frame["x"] + spatial + trend
    frame["E"] = rng.uniform(2.0, 6.0, size=len(frame))
    frame["logE"] = np.log(frame["E"])
    frame["count"] = rng.poisson(frame["E"] * np.exp(eta)).astype(float)
    frame["n"] = rng.integers(3, 12, size=len(frame)).astype(float)
    frame["k"] = rng.binomial(frame["n"].astype(int), 1 / (1 + np.exp(-eta))).astype(float)
    frame["nb"] = rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(eta))).astype(float)
    frame["time"] = rng.exponential(np.exp(-eta)) + 0.05
    frame["event"] = (rng.uniform(size=len(frame)) < 0.8).astype(float)
    frame.loc[[3, 40], ["count", "k", "nb"]] = np.nan  # prediction-only rows
    return frame


MODELS = {
    "poisson_besag_iid": LGM(
        response="count", likelihood=Poisson(), offset="logE", panel=("region",), time="t",
        predictor=Fixed("1 + x") + Besag("s", index="region", graph=RING, precision=3.0)
        + IID("v", index="t", precision=10.0),
    ),
    "binomial_rw1_iid": LGM(
        response="k", likelihood=Binomial(trials="n"), panel=("region",), time="t",
        predictor=Fixed("1") + RW1("trend", index="t", precision=5.0)
        + IID("u", index="region", precision=2.0),
    ),
    "negbin_iid_ar1": LGM(
        response="nb", likelihood=NegativeBinomial(phi=2.0), panel=("region",), time="t",
        predictor=Fixed("1 + x") + IID("u", index="region", precision=2.0)
        + AR1("a", index="t", precision=4.0, rho=0.6),
    ),
    "weibull_iid": LGM(
        response="time", likelihood=WeibullSurv(event="event", shape=1.3), panel=("region",),
        time="t", predictor=Fixed("1 + x") + IID("u", index="region", precision=2.0),
    ),
}


def _fit(model, frame, monkeypatch, sparse):
    if sparse:
        monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    try:
        return model.fit(frame, engine="laplace")
    finally:
        monkeypatch.undo()


def _assert_same(sparse, dense):
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-7)
    np.testing.assert_allclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-7)
    for block in dense.block_slices:
        np.testing.assert_allclose(
            sparse.latent_marginals(block).std, dense.latent_marginals(block).std, rtol=1e-6
        )
    np.testing.assert_allclose(sparse.predictive_mean, dense.predictive_mean, atol=1e-7)
    np.testing.assert_allclose(sparse.predictive_variance, dense.predictive_variance, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(sparse.fitted_mean, dense.fitted_mean, rtol=1e-6)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_sparse_laplace_matches_dense(name, monkeypatch):
    frame = _frame()
    dense = _fit(MODELS[name], frame, monkeypatch, sparse=False)
    sparse = _fit(MODELS[name], frame, monkeypatch, sparse=True)
    assert dense.covariance is not None and sparse._covariance is None  # really two paths
    _assert_same(sparse, dense)
    draws = sparse.sample(20_000, rng=0)
    np.testing.assert_allclose(draws.mean(axis=0), dense.predictive_mean, atol=0.05)


def test_sparse_laplace_on_a_gaussian_likelihood_is_exact(monkeypatch):
    """The correctness anchor fit_laplace promises: Laplace is exact for a Gaussian."""
    frame = _frame().assign(y=lambda f: f["x"] + f["t"] / 10.0)
    model = LGM("y", Gaussian(0.7), Fixed("1 + x") + RW1("trend", index="t", precision=4.0)
                + IID("u", index="region", precision=2.0), panel=("region",), time="t")
    compiled = compile_lgm(model, CanonicalPanel.from_frame(
        frame, DataConfig(time="t", response="y", panel=("region",))))
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    laplace, exact = fit_laplace(compiled), fit_gaussian(compiled)
    np.testing.assert_allclose(laplace.mean, exact.mean, atol=1e-8)
    np.testing.assert_allclose(laplace.log_marginal_likelihood, exact.log_marginal_likelihood, atol=1e-8)
    np.testing.assert_allclose(laplace.predictive_variance, exact.predictive_variance, atol=1e-10)


CONFOUNDED = {
    # Besag + RW1: A_ss is singular along (1_s, -1_t) before the constraints.
    "gaussian_besag_rw1": (Gaussian(0.7), "x", Fixed("1") + Besag("s", index="region", graph=RING)
                           + RW1("trend", index="t")),
    "poisson_besag_rw1": (Poisson(), "count", Fixed("1 + x") + Besag("s", index="region", graph=RING)
                          + RW1("trend", index="t", precision=4.0)),
    # RW2 + RW1 on one index: the shared constant.
    "binomial_rw2_rw1": (Binomial(trials="n"), "k", Fixed("1") + RW2("r2", index="t", precision=20.0)
                         + RW1("r1", index="t", precision=5.0)),
    # Knorr-Held type IV: main effects plus an interaction whose null space overlaps both.
    "poisson_knorr_held": (Poisson(), "count", Fixed("1") + Besag("s", index="region", graph=RING)
                           + RW1("trend", index="t", precision=4.0)
                           + SpaceTime("st", space="region", time="t", graph=RING, interaction="IV")),
}


@pytest.mark.parametrize("name", sorted(CONFOUNDED))
def test_confounded_intrinsic_effects_match_dense(name, monkeypatch):
    """E-sparse-D2: the sparse path regularises with the intrinsic blocks' own
    constraint rows (exact on the constraint set) and SMW-updates a grounded
    factor. It used to refuse these models -- and before that, silently return
    a mean violating the sum-to-zero rows."""
    likelihood, response, predictor = CONFOUNDED[name]
    model = LGM(response, likelihood, predictor, offset="logE" if response == "count" else None,
                panel=("region",), time="t")
    engine = "exact_gaussian" if response == "x" else "laplace"
    frame = _frame()
    dense = model.fit(frame, engine=engine)
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = model.fit(frame, engine=engine)
    assert sparse._covariance is None
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-7)
    np.testing.assert_allclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-7)
    np.testing.assert_allclose(sparse.predictive_variance, dense.predictive_variance, rtol=1e-6, atol=1e-10)
    for block in dense.block_slices:
        np.testing.assert_allclose(sparse.latent_marginals(block).std, dense.latent_marginals(block).std,
                                   rtol=1e-6)
    draws = sparse.sample(20_000, rng=1)
    np.testing.assert_allclose(draws.std(axis=0), np.sqrt(dense.predictive_variance), rtol=0.05)


@pytest.mark.parametrize("hyperparameters", ["optimize", "integrate"])
def test_sparse_laplace_estimates_hyperparameters_like_dense(hyperparameters, monkeypatch):
    """End to end: a declared precision must steer the sparse fit exactly as the dense one."""
    from pylgm import Hyperparameter, PCPrecision

    model = LGM(
        response="count", likelihood=Poisson(), offset="logE", panel=("region",), time="t",
        predictor=Fixed("1 + x") + Besag(
            "s", index="region", graph=RING,
            precision=Hyperparameter("s_prec", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01)),
        ),
    )
    frame = _frame()
    dense = model.fit(frame, engine="laplace", hyperparameters=hyperparameters)
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = model.fit(frame, engine="laplace", hyperparameters=hyperparameters)
    def estimate(result):
        if hyperparameters == "optimize":
            return result.hyperparameters["s_prec"]
        return float(result.hyperparameter_marginals()["s_prec"].mean[0])

    # The EB search stops on a plateau (objective_tolerance 1e-5) of a flat optimum,
    # so the two argmaxes agree to ~1e-4, not to solver tolerance.
    assert estimate(sparse) == pytest.approx(estimate(dense), rel=1e-3)
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-4)
    # The reported lml is not stationary at the penalised optimum (the PC prior's
    # slope balances it there), so the argmax gap enters it at first order.
    np.testing.assert_allclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, rtol=1e-5)


def _dense_and_sparse(fit, monkeypatch):
    dense = fit()
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = fit()
    monkeypatch.undo()
    assert sparse._covariance is None and dense._covariance is not None
    return sparse, dense


BASE = dict(response="count", offset="logE", panel=("region",), time="t")
FIELD = Fixed("1 + x") + Besag("s", index="region", graph=RING, precision=3.0) \
    + IID("v", index="t", precision=10.0)


def test_nonzero_rhs_label_constraint(monkeypatch):
    model = LGM(likelihood=Poisson(), predictor=FIELD,
                constraints=[({"v:0": 1.0, "v:1": 1.0}, 0.5)], **BASE)
    sparse, dense = _dense_and_sparse(lambda: model.fit(_frame(), engine="laplace"), monkeypatch)
    _assert_same(sparse, dense)
    labels = list(sparse.labels)
    assert sparse.mean[labels.index("v:0")] + sparse.mean[labels.index("v:1")] == pytest.approx(0.5)


def test_mean_correction(monkeypatch):
    model = LGM(likelihood=Poisson(), predictor=FIELD, **BASE)
    sparse, dense = _dense_and_sparse(
        lambda: model.fit(_frame(), engine="laplace", mean_correction=True), monkeypatch
    )
    _assert_same(sparse, dense)


def test_zero_inflated_poisson(monkeypatch):
    from pylgm import ZeroInflated

    frame = _frame()
    frame.loc[::5, "count"] = 0.0
    model = LGM(likelihood=ZeroInflated(Poisson(), pi=0.2), predictor=FIELD, **BASE)
    sparse, dense = _dense_and_sparse(lambda: model.fit(frame, engine="laplace"), monkeypatch)
    _assert_same(sparse, dense)


def test_joint_with_a_shared_field_and_a_data_constraint(monkeypatch):
    """Two outcomes on one latent field, one of them pinned by an exact aggregate."""
    from pylgm.joint import Joint, Shared
    from pylgm.observations import LinearConstraint

    frame = _frame()
    frame["level"] = np.log1p(frame["count"].fillna(2.0)) + 0.1 * frame["x"]
    joint = Joint(
        [LGM(response="level", likelihood=Gaussian(0.5), predictor=Fixed("1 + x"),
             panel=("region",), time="t"),
         LGM(response="count", likelihood=Poisson(), offset="logE", predictor=Fixed("1 + x"),
             panel=("region",), time="t")],
        shared=[Shared(Besag("s", index="region", graph=RING, precision=3.0), scale=(1.0, 0.8))],
    )
    operator = np.zeros((1, len(frame)))
    operator[0, :REGIONS] = 1.0 / REGIONS
    constraint = LinearConstraint(operator, [1.0])
    sparse, dense = _dense_and_sparse(
        lambda: joint.fit(frame, constraints={"level": [constraint]}), monkeypatch
    )
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-7)
    np.testing.assert_allclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-7)
    np.testing.assert_allclose(sparse.predictive_variance, dense.predictive_variance, rtol=1e-6, atol=1e-10)


def test_mean_correction_with_data_constraints_matches_dense(monkeypatch):
    """The shift uses the eta variances *before* conditioning on the data rows
    (the dense engine shifts, then conditions); used to be refused above the guard."""
    from pylgm.joint import Joint
    from pylgm.observations import LinearConstraint

    frame = _frame()
    frame["level"] = np.log1p(frame["count"].fillna(2.0))
    joint = Joint([
        LGM(response="level", likelihood=Gaussian(0.5), predictor=FIELD, panel=("region",), time="t"),
        LGM(response="count", likelihood=Poisson(), offset="logE", predictor=Fixed("1 + x"),
            panel=("region",), time="t"),
    ])
    operator = np.zeros((1, len(frame)))
    operator[0, :REGIONS] = 1.0 / REGIONS
    constraint = LinearConstraint(operator, [1.0])
    sparse, dense = _dense_and_sparse(
        lambda: joint.fit(frame, constraints={"level": [constraint]}, mean_correction=True),
        monkeypatch,
    )
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-7)
    np.testing.assert_allclose(sparse.predictive_mean, dense.predictive_mean, atol=1e-7)


@pytest.mark.parametrize("sparse", [False, True])
def test_warm_start_changes_iterations_not_the_answer(sparse, monkeypatch):
    """Same mode and lml from a warm start, in fewer iterations -- and to the
    precision a finite-difference Hessian over theta needs: without the polish
    step a warm start stopped just under the gradient tolerance and moved the
    INLA Hessian by 0.2%."""
    from pylgm import Hyperparameter
    from pylgm.compiler import compile_family

    if sparse:
        monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    panel = CanonicalPanel.from_frame(_frame(), DataConfig(time="t", response="count", panel=("region",)))
    model = LGM(likelihood=Poisson(), predictor=Fixed("1 + x") + Besag(
        "s", index="region", graph=RING, precision=Hyperparameter("tau", initial=1.0)), **BASE)
    family = compile_family(model, panel)
    start = fit_laplace(family.materialize({"tau": 3.0})).mean

    def lml(tau, **kw):
        return fit_laplace(family.materialize({"tau": tau}), **kw)

    cold, warm = lml(3.3), lml(3.3, initial_mode=start)
    np.testing.assert_allclose(warm.mean, cold.mean, atol=1e-10)
    assert warm.log_marginal_likelihood == pytest.approx(cold.log_marginal_likelihood, abs=1e-11)
    assert warm.diagnostics["newton_iterations"] < cold.diagnostics["newton_iterations"]
    h = 1e-3
    u = np.log(3.0)

    def hessian(**kw):
        f = [lml(float(np.exp(u + k * h)), **kw).log_marginal_likelihood for k in (-1, 0, 1)]
        return (f[0] - 2 * f[1] + f[2]) / h**2

    assert hessian(initial_mode=start) == pytest.approx(hessian(), rel=1e-5)


def test_augmented_bym2_confounded_with_rw1_matches_dense(monkeypatch):
    """BYM2's augmented block pins its null vector g = (sqrt(phi) 1, 1) through a
    row on the u* half only, so its constraint row is not a null vector; the
    regularisation must still ground g, the block's true null vector."""
    from pylgm import BYM2

    import pylgm.effects.bym2 as bym2

    monkeypatch.setattr(bym2, "_BYM2_AUGMENT_NODES", 1)  # augmented on a small graph
    model = LGM(likelihood=Poisson(), predictor=Fixed("1 + x")
                + BYM2("b", index="region", graph=RING, precision=2.0, phi=0.5)
                + RW1("trend", index="t", precision=4.0), **BASE)
    frame = _frame()
    dense = model.fit(frame, engine="laplace")
    assert any(label.endswith("__u") for label in dense.labels)  # really augmented
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = model.fit(frame, engine="laplace")
    assert sparse._covariance is None
    _assert_same(sparse, dense)


def test_simplified_laplace_marginals_above_the_guard_match_dense(monkeypatch):
    """SLA needs cov(x_i, eta_j) = (Sigma A^T)_ij; above the guard it is
    accumulated in column batches from the sparse posterior instead of the
    dense covariance. Used to raise UnsupportedEngineError."""
    from pylgm import Hyperparameter

    model = LGM(likelihood=Poisson(), predictor=Fixed("1 + x") + Besag(
        "s", index="region", graph=RING, precision=Hyperparameter("tau", initial=1.0)), **BASE)
    frame = _frame()

    def fit():
        return model.fit(frame, engine="laplace", hyperparameters="integrate",
                         latent_strategy="simplified_laplace")

    dense = fit()
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = fit()
    assert sparse._covariance is None
    for block in ("fixed", "s"):
        d, s = dense.latent_marginals(block), sparse.latent_marginals(block)
        np.testing.assert_allclose(s.mean, d.mean, atol=1e-6)
        np.testing.assert_allclose(s.std, d.std, rtol=1e-5)
        np.testing.assert_allclose(s.quantile(0.975), d.quantile(0.975), atol=1e-6)


@pytest.mark.parametrize("likelihood,response", [(Gaussian(0.7), "x"), (Poisson(), "count")])
def test_cross_block_constraint_matches_dense(likelihood, response, monkeypatch):
    """A label constraint coupling two blocks: the prior logdet is no longer
    block-separable. Used to raise NotImplementedError on the sparse path."""
    model = LGM(response, likelihood, FIELD, offset="logE" if response == "count" else None,
                panel=("region",), time="t",
                constraints=[({"s:r0": 1.0, "v:3": -1.0}, 0.25)])
    engine = "laplace" if response == "count" else "exact_gaussian"
    frame = _frame()
    dense = model.fit(frame, engine=engine)
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = model.fit(frame, engine=engine)
    assert sparse._covariance is None
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-7)
    np.testing.assert_allclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-7)
    np.testing.assert_allclose(sparse.predictive_variance, dense.predictive_variance, rtol=1e-6, atol=1e-10)
    labels = list(sparse.labels)
    assert sparse.mean[labels.index("s:r0")] - sparse.mean[labels.index("v:3")] == pytest.approx(0.25)


def test_coupled_prior_logdet_equals_the_block_separable_one():
    """Where both apply (every row inside one block), the general grounded
    formula must reproduce the block-separable matrix-tree/cofactor one."""
    from pylgm.inference.sparse import (
        _block_column_confinement, _coupled_prior_logdet, _prior_logdet,
    )

    model = LGM("x", Gaussian(0.7), Fixed("1") + Besag("s", index="region", graph=RING)
                + RW2("r2", index="t", precision=3.0) + IID("u", index="region"),
                panel=("region",), time="t")
    compiled = compile_lgm(model, CanonicalPanel.from_frame(
        _frame(), DataConfig(time="t", response="x", panel=("region",))))
    rows = compiled.constraints
    separable = _prior_logdet(compiled, rows, _block_column_confinement(compiled, rows))
    assert _coupled_prior_logdet(compiled, rows) == pytest.approx(separable, abs=1e-8)


@pytest.mark.parametrize("latent_strategy", ["gaussian", "simplified_laplace"])
def test_sparse_path_is_identical_under_parallel_workers(latent_strategy, monkeypatch):
    """num_workers fans conditional fits out over threads (SuperLU and BLAS
    release the GIL). On the sparse path -- Laplace engine, warm starts,
    confounded-intrinsic grounding, sparse SLA -- the result must not depend
    on the worker count."""
    from pylgm import Hyperparameter

    model = LGM(likelihood=Poisson(), predictor=Fixed("1 + x")
                + Besag("s", index="region", graph=RING, precision=Hyperparameter("tau", initial=1.0))
                + RW1("trend", index="t", precision=Hyperparameter("rho", initial=4.0)), **BASE)
    frame = _frame()
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)

    def fit(workers):
        return model.fit(frame, engine="laplace", hyperparameters="integrate",
                         latent_strategy=latent_strategy, num_workers=workers)

    serial, threaded = fit(1), fit(4)
    assert serial._covariance is None
    np.testing.assert_array_equal(threaded.mean, serial.mean)
    assert threaded.log_marginal_likelihood == serial.log_marginal_likelihood
    np.testing.assert_array_equal(threaded.predictive_variance, serial.predictive_variance)
    for block in ("fixed", "s", "trend"):
        np.testing.assert_array_equal(threaded.latent_marginals(block).std,
                                      serial.latent_marginals(block).std)

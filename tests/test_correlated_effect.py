"""Correlated k-variate IID effect, against closed forms."""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from pylgm import LGM, Correlated, Fixed, Gaussian, Hyperparameter, SymmetricBeta
from pylgm.effects.correlated import component_precision, correlation_cholesky

NODES, SIGMA = 12, 0.5


def _network(nodes=NODES, edges_per_node=6, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    sender = np.repeat(np.arange(nodes), edges_per_node)
    receiver = (sender + rng.integers(1, nodes, sender.size)) % nodes
    cov = np.array([[1.0, rho], [rho, 1.0]])
    effects = rng.multivariate_normal(np.zeros(2), cov, size=nodes)
    y = effects[sender, 0] + effects[receiver, 1] + SIGMA * rng.normal(size=sender.size)
    return pd.DataFrame({"sender": sender, "receiver": receiver, "y": y})


def test_gaussian_posterior_and_evidence_match_the_closed_form():
    frame = _network()
    taus, rho = (2.0, 0.5), 0.4
    model = LGM(response="y", likelihood=Gaussian(SIGMA),
                predictor=Correlated("node", index=("sender", "receiver"),
                                     precision=taus, correlation=rho))
    result = model.fit(frame)

    # Independent build: x = (s_0..s_n-1, r_0..r_n-1), Cov = Sigma ⊗ I_n.
    n, m = NODES, len(frame)
    sd = 1 / np.sqrt(taus)
    sigma = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
    prior = np.kron(sigma, np.eye(n))
    design = np.zeros((m, 2 * n))
    design[np.arange(m), frame["sender"]] = 1.0
    design[np.arange(m), n + frame["receiver"]] = 1.0
    marginal = design @ prior @ design.T + SIGMA**2 * np.eye(m)
    y = frame["y"].to_numpy()
    mean = prior @ design.T @ np.linalg.solve(marginal, y)
    evidence = -0.5 * (y @ np.linalg.solve(marginal, y) + np.linalg.slogdet(marginal)[1]
                       + m * np.log(2 * np.pi))
    np.testing.assert_allclose(result.mean, mean, atol=1e-9)
    assert result.log_marginal_likelihood == pytest.approx(evidence, abs=1e-8)
    np.testing.assert_allclose(result.predict(frame).predictive_mean, result.predictive_mean,
                               atol=1e-12)


def test_empirical_bayes_recovers_the_correlation():
    frame = _network(nodes=200, edges_per_node=10, rho=0.6, seed=3)
    model = LGM(response="y", likelihood=Gaussian(SIGMA), predictor=Fixed("1") + Correlated(
        "node", index=("sender", "receiver"),
        precision=(Hyperparameter("tau_s", initial=1.0), Hyperparameter("tau_r", initial=1.0)),
        correlation=Hyperparameter("rho", initial=0.0, transform="logit", lower=-0.99, upper=0.99),
    ))
    result = model.fit(frame)
    assert result.hyperparameters["rho"] == pytest.approx(0.6, abs=0.15)
    assert result.hyperparameters["tau_s"] == pytest.approx(1.0, rel=0.4)


def test_absent_components_and_prediction():
    frame = _network()
    frame.loc[::5, "receiver"] = np.nan  # a row without a receiver component
    model = LGM(response="y", likelihood=Gaussian(SIGMA),
                predictor=Correlated("node", index=("sender", "receiver"), correlation=0.3))
    result = model.fit(frame)
    np.testing.assert_allclose(result.predict(frame).predictive_mean, result.predictive_mean,
                               atol=1e-12)


def test_canonical_partial_correlations_give_lkj():
    rng = np.random.default_rng(0)
    for k in (2, 3, 4):
        draws = []
        for _ in range(4000):
            cpcs = [2 * rng.beta(b, b) - 1
                    for j in range(k - 1) for b in [1 + (k - 2 - j) / 2] * (k - 1 - j)]
            factor = correlation_cholesky(np.array(cpcs), k)
            correlation = factor @ factor.T
            np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-12)
            draws.append(correlation[np.triu_indices(k, 1)])
        # LKJ(1) is uniform on correlation matrices: each r_ij ~ Beta(k/2, k/2) on
        # (-1, 1), mean 0 and variance 1 / (k + 1).
        draws = np.array(draws)
        np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=0.04)
        np.testing.assert_allclose(draws.var(axis=0), 1 / (k + 1), rtol=0.08)


def test_component_precision_and_symmetric_beta():
    q = component_precision([4.0, 1.0], [0.5])
    covariance = np.linalg.inv(q)
    np.testing.assert_allclose(covariance, [[0.25, 0.25], [0.25, 1.0]], atol=1e-12)
    for shape in (0.7, 1.0, 2.5):
        total, _ = quad(lambda x: np.exp(SymmetricBeta(shape).logpdf(x)), -1, 1)
        assert total == pytest.approx(1.0, abs=1e-6)


def test_validation():
    with pytest.raises(ValueError, match="at least two"):
        Correlated("x", index=("a",))
    with pytest.raises(ValueError, match="canonical partial"):
        Correlated("x", index=("a", "b", "c"), correlation=(0.1,))
    with pytest.raises(ValueError, match=r"\(-1, 1\)"):
        Correlated("x", index=("a", "b"), correlation=1.0)
    lkj = Correlated("x", index=("a", "b", "c"), correlation=tuple(
        Hyperparameter(f"z{i}", initial=0.0, transform="logit", lower=-0.9, upper=0.9)
        for i in range(3)
    ))
    assert [c.prior.shape for c in lkj.correlation] == [1.5, 1.5, 1.0]


def _two_outcomes(firms=10, rows_per_firm=4, seed=1):
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(firms), rows_per_firm)
    effects = rng.multivariate_normal([0, 0], [[1.0, 0.7], [0.7, 1.0]], size=firms)
    return pd.DataFrame({
        "firm": firm,
        "a": effects[firm, 0] + 0.4 * rng.normal(size=firm.size),
        "b": effects[firm, 1] + 0.6 * rng.normal(size=firm.size),
    })


def test_shared_correlated_across_outcomes_matches_the_closed_form():
    from pylgm import Joint, Shared

    frame = _two_outcomes()
    taus, rho = (1.5, 0.8), 0.5
    frame["row"] = np.arange(len(frame))  # panel keys must be unique per row
    joint = Joint(
        [LGM(response="a", likelihood=Gaussian(0.4), predictor=Fixed("1", prior_precision=1.0),
             panel=("row",)),
         LGM(response="b", likelihood=Gaussian(0.6), predictor=Fixed("1", prior_precision=1.0),
             panel=("row",))],
        shared=[Shared(Correlated("firm_eff", index=("firm", "firm"), precision=taus,
                                  correlation=rho))],
    )
    result = joint.fit(frame)

    # Closed form: y = (y_a, y_b) = X beta + Z u + e, all Gaussian.
    m, n = len(frame), frame["firm"].nunique()
    sd = 1 / np.sqrt(taus)
    sigma_u = np.kron(np.array([[sd[0] ** 2, rho * sd[0] * sd[1]],
                                [rho * sd[0] * sd[1], sd[1] ** 2]]), np.eye(n))
    z = np.zeros((2 * m, 2 * n))
    z[np.arange(m), frame["firm"]] = 1.0
    z[m + np.arange(m), n + frame["firm"]] = 1.0
    x = np.zeros((2 * m, 2))
    x[:m, 0] = x[m:, 1] = 1.0
    noise = np.diag(np.r_[np.full(m, 0.4**2), np.full(m, 0.6**2)])
    marginal = x @ x.T + z @ sigma_u @ z.T + noise  # beta ~ N(0, I)
    y = np.r_[frame["a"], frame["b"]]
    evidence = -0.5 * (y @ np.linalg.solve(marginal, y) + np.linalg.slogdet(marginal)[1]
                       + 2 * m * np.log(2 * np.pi))
    assert result.log_marginal_likelihood == pytest.approx(evidence, abs=1e-6)
    np.testing.assert_allclose(
        result.predict(frame, outcome="b").predictive_mean,
        result.predictive_mean[m:], atol=1e-10,
    )


def test_shared_correlated_estimates_with_a_censored_hurdle_pair():
    from pylgm import Bernoulli, Joint, Shared

    frame = _two_outcomes(firms=60, rows_per_firm=8, seed=5)
    frame["row"] = np.arange(len(frame))
    frame["a"] = (frame["a"] > 0).astype(float)
    joint = Joint(
        [LGM(response="a", likelihood=Bernoulli(), predictor=Fixed("1"), panel=("row",)),
         LGM(response="b", likelihood=Gaussian(0.6), predictor=Fixed("1"), panel=("row",))],
        shared=[Shared(Correlated(
            "firm_eff", index=("firm", "firm"),
            precision=(Hyperparameter("tau_a", initial=1.0), Hyperparameter("tau_b", initial=1.0)),
            correlation=Hyperparameter("rho", initial=0.0, transform="logit", lower=-0.95, upper=0.95),
        ))],
    )
    result = joint.fit(frame)
    assert result.hyperparameters["rho"] > 0.3  # simulated at 0.7
    with pytest.raises(ValueError, match="one component"):
        Joint(joint.submodels, shared=[Shared(Correlated("f", index=("firm", "firm", "firm")))])


def test_reciprocity_is_recovered_from_dyads():
    from pylgm import dyad_columns

    rng = np.random.default_rng(7)
    nodes = 80
    pairs = np.array([(a, b) for a in range(nodes) for b in range(a + 1, nodes)])
    pairs = pairs[rng.random(len(pairs)) < 0.25]
    u = rng.multivariate_normal([0, 0], [[1.0, 0.8], [0.8, 1.0]], size=len(pairs))
    frame = pd.DataFrame({
        "sender": np.r_[pairs[:, 0], pairs[:, 1]],
        "receiver": np.r_[pairs[:, 1], pairs[:, 0]],
        "y": np.r_[u[:, 0], u[:, 1]] + 0.3 * rng.normal(size=2 * len(pairs)),
    })
    frame = frame.join(dyad_columns(frame, "sender", "receiver"))
    tau = Hyperparameter("tau", initial=1.0)
    model = LGM(response="y", likelihood=Gaussian(0.3), predictor=Correlated(
        "dyad", index=("forward", "backward"), precision=(tau, tau),
        correlation=Hyperparameter("reciprocity", initial=0.0, transform="logit",
                                   lower=-0.99, upper=0.99),
    ))
    result = model.fit(frame)
    assert result.hyperparameters["reciprocity"] == pytest.approx(0.8, abs=0.1)
    assert set(result.hyperparameters) == {"tau", "reciprocity"}

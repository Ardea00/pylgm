"""Censored hurdle on a bank-firm register, against an exact numpy oracle.

A register reports an edge's amount only when it is at least ``c``. An absent
candidate edge contributes ``log[(1 - p) + p Phi((log c - b) / sigma)]``,
coupling the link predictor ``a`` and the log-amount predictor ``b``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import log_expit, log_ndtr

from pylgm import IID, LGM, Bernoulli, CensoredHurdle, Fixed, Gaussian, Hyperparameter, Joint
from pylgm.exceptions import DataContractError, UnsupportedEngineError
from pylgm.inference import gaussian as gaussian_engine

FIRMS, BANKS = 7, 3
TAU, SIGMA, LOG_C, DIFFUSE = 2.0, 0.8, 1.0, 1e-6


def _frame():
    rng = np.random.default_rng(4)
    firm = np.repeat(np.arange(FIRMS), BANKS)
    bank = np.tile(np.arange(BANKS), FIRMS)
    linked = rng.random(firm.size) < 0.6
    amount = 1.2 + 0.4 * rng.normal(size=FIRMS)[firm] + SIGMA * rng.normal(size=firm.size)
    reported = linked & (amount >= LOG_C)
    # Some absences are known non-links (a 0 response, not censored); without
    # them only the amount tail separates "no link" from "a link below c".
    known_absent = ~linked & (rng.random(firm.size) < 0.5)
    return pd.DataFrame({
        "firm": [f"f{i}" for i in firm], "bank": [f"b{j}" for j in bank],
        "linked": np.where(reported, 1.0, np.where(known_absent, 0.0, np.nan)),
        "log_amount": np.where(reported, amount, np.nan),
        "unreported": ~reported & ~known_absent,
    })


def _joint(sigma=SIGMA):
    link = LGM(
        response="linked", likelihood=Bernoulli(), panel=("firm", "bank"),
        predictor=Fixed("1") + IID("fl", index="firm", precision=TAU)
        + IID("bl", index="bank", precision=TAU),
    )
    amount = LGM(
        response="log_amount", likelihood=Gaussian(sigma), panel=("firm", "bank"),
        predictor=Fixed("1") + IID("fa", index="firm", precision=TAU),
    )
    return Joint([link, amount], censoring=CensoredHurdle(
        link="linked", amount="log_amount", censored="unreported", threshold=LOG_C,
    ))


def _oracle_objective(frame):
    """Negative log posterior over z = (alpha, u_firm, u_bank, beta, v_firm)."""
    firm = frame["firm"].str[1:].astype(int).to_numpy()
    bank = frame["bank"].str[1:].astype(int).to_numpy()
    reported = frame["linked"].to_numpy() == 1.0
    absent = frame["linked"].to_numpy() == 0.0
    censored = frame["unreported"].to_numpy()
    y = frame["log_amount"].to_numpy()

    def split(z):
        alpha, u, w = z[0], z[1:1 + FIRMS], z[1 + FIRMS:1 + FIRMS + BANKS]
        beta, v = z[1 + FIRMS + BANKS], z[2 + FIRMS + BANKS:]
        return alpha + u[firm] + w[bank], beta + v[firm], np.r_[u, w, v], np.r_[alpha, beta]

    def objective(z):
        a, b, random, fixed = split(z)
        value = 0.5 * TAU * random @ random + 0.5 * DIFFUSE * fixed @ fixed
        value -= log_expit(a[reported]).sum() + log_expit(-a[absent]).sum()
        r = (y[reported] - b[reported]) / SIGMA
        value += np.sum(0.5 * r * r + np.log(SIGMA) + 0.5 * np.log(2 * np.pi))
        a_c, b_c = a[censored], b[censored]
        value -= np.logaddexp(log_expit(-a_c), log_expit(a_c) + log_ndtr((LOG_C - b_c) / SIGMA)).sum()
        return value

    dimension = 2 + 2 * FIRMS + BANKS
    precision_logdet = (2 * FIRMS + BANKS) * np.log(TAU) + 2 * np.log(DIFFUSE)
    return objective, split, dimension, precision_logdet


def _hessian(f, z, h=1e-4):
    n = z.size
    eye = np.eye(n) * h
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            out[i, j] = out[j, i] = (
                f(z + eye[i] + eye[j]) - f(z + eye[i] - eye[j])
                - f(z - eye[i] + eye[j]) + f(z - eye[i] - eye[j])
            ) / (4 * h * h)
    return out


def _gradient(f, z, h=1e-6):
    eye = np.eye(z.size) * h
    return np.array([(f(z + e) - f(z - e)) / (2 * h) for e in eye])


def _sorted_eta(result, frame, outcome):
    """Fitted link-scale predictor of ``outcome`` in the caller's frame order."""
    return result.predict(frame, outcome=outcome).predictive_mean


def test_mode_and_evidence_match_the_exact_oracle():
    frame = _frame()
    result = _joint().fit(frame)
    objective, split, dimension, precision_logdet = _oracle_objective(frame)
    oracle = minimize(objective, np.zeros(dimension), jac=lambda z: _gradient(objective, z),
                      method="BFGS", options={"gtol": 1e-8})
    a, b, _, _ = split(oracle.x)
    np.testing.assert_allclose(_sorted_eta(result, frame, "linked"), a, atol=1e-5)
    np.testing.assert_allclose(_sorted_eta(result, frame, "log_amount"), b, atol=1e-5)

    hessian = _hessian(objective, oracle.x)
    laplace = -objective(oracle.x) + 0.5 * precision_logdet - 0.5 * np.linalg.slogdet(hessian)[1]
    assert result.log_marginal_likelihood == pytest.approx(laplace, abs=1e-4)


def test_dense_and_sparse_engines_agree(monkeypatch):
    frame = _frame()
    dense = _joint().fit(frame)
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = _joint().fit(frame)
    assert sparse._covariance is None and dense._covariance is not None
    np.testing.assert_allclose(sparse.mean, dense.mean, atol=1e-8)
    np.testing.assert_allclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-8)
    np.testing.assert_allclose(sparse.predictive_variance, dense.predictive_variance, rtol=1e-6)


def test_estimated_sigma_moves_the_coupled_term_too():
    sigma = Hyperparameter("sigma", initial=0.5, lower=0.05, upper=5.0)
    result = _joint(sigma).fit(_frame())
    assert np.isfinite(result.log_marginal_likelihood)
    assert 0.05 < result.hyperparameters["sigma"] < 5.0
    integrated = _joint(sigma).fit(_frame(), hyperparameters="integrate")
    assert np.isfinite(integrated.log_marginal_likelihood)
    with pytest.raises(ValueError, match="censored hurdle"):
        integrated.criteria  # per-row criteria do not apply to a two-row observation


def test_unsupported_paths_are_refused():
    sigma = Hyperparameter("sigma", initial=0.8, lower=0.05, upper=5.0)
    with pytest.raises(UnsupportedEngineError, match="censored hurdle"):
        _joint().fit(_frame(), mean_correction=True)
    with pytest.raises(UnsupportedEngineError, match="censored hurdle"):
        _joint(sigma).fit(_frame(), hyperparameters="integrate", latent_strategy="simplified_laplace")


def test_censored_rows_must_carry_no_response():
    frame = _frame()
    frame.loc[frame.index[frame["unreported"]][0], "linked"] = 0.0
    with pytest.raises(DataContractError, match="must be NaN on censored rows"):
        _joint().fit(frame)

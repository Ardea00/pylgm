"""The sparse kriging capacitance K = A Q^-1 A^T must not be formed explicitly.

Near-unit-root AR1s and nearly redundant annual/national aggregates push
cond(K) past 1/eps, so Cholesky on K fails although K is positive definite, and
empirical Bayes loses every such evaluation. Factoring G = L^-1 A^T by QR, so
that K = R^T R, works with sqrt(cond K) instead.
"""

import numpy as np
import pandas as pd
import pytest

import pylgm.inference.gaussian as gaussian_engine
from pylgm import AR1, IID, Fixed, Gaussian, LGM, LinearConstraint

R, Q = 6, 40


def _panel():
    rng = np.random.default_rng(0)
    grid = pd.DataFrame([(f"r{r}", q) for r in range(R) for q in range(Q)], columns=["region", "q"])
    g = (100 + rng.normal(size=(R, 1)) * 10 + np.cumsum(rng.normal(size=(R, Q)), axis=1)).ravel()
    annual, national = np.zeros((R * Q // 4, R * Q)), np.zeros((Q, R * Q))
    for r in range(R):
        for q in range(Q):
            annual[r * (Q // 4) + q // 4, r * Q + q] = 1.0
            national[q, r * Q + q] = 1.0
    return grid, [LinearConstraint(annual, annual @ g), LinearConstraint(national, national @ g)]


def _model():
    return LGM(
        response="y", likelihood=Gaussian(1.0),
        predictor=Fixed("1") + IID("reg", "region", precision=10.0)
        + AR1("common", "q", precision=1e-5, rho=0.9999)
        + AR1("regional", "q", replicate="region", precision=1e4, rho=0.9999),
        panel=("region",), time="q",
    )


def test_an_ill_conditioned_capacitance_still_fits_and_matches_dense(monkeypatch):
    grid, constraints = _panel()
    dense = _model().fit(grid, constraints=constraints)
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    sparse = _model().fit(grid, constraints=constraints)

    assert sparse.log_marginal_likelihood == pytest.approx(dense.log_marginal_likelihood, rel=1e-6)
    np.testing.assert_allclose(sparse.predictive_mean, dense.predictive_mean, rtol=1e-6)
    for constraint in constraints:
        residual = constraint.operator @ sparse.predictive_mean - constraint.rhs
        assert np.abs(residual).max() <= 1e-9 * np.abs(constraint.rhs).max()
    draws = sparse.sample(50, 0)
    for constraint in constraints:
        residual = draws @ constraint.operator.T.toarray() - constraint.rhs
        assert np.abs(residual).max() <= 1e-8 * np.abs(constraint.rhs).max()


def test_the_factor_half_solves_are_a_square_root_of_the_inverse():
    from scipy.sparse import csr_matrix, random as sparse_random

    from pylgm.inference.sparse import SparseSpdFactor

    m = sparse_random(40, 40, density=0.1, random_state=3)
    matrix = (m @ m.T + csr_matrix(np.diag(np.linspace(0.1, 5.0, 40)))).toarray()
    factor = SparseSpdFactor(csr_matrix(matrix), "test")
    rhs = np.random.default_rng(0).normal(size=(40, 5))
    g = factor.half_solve(rhs)
    np.testing.assert_allclose(g.T @ g, rhs.T @ np.linalg.solve(matrix, rhs), rtol=1e-10)
    # half_solve_transpose maps N(0, I) to N(0, A^-1): its matrix M has M M^T = A^-1.
    transform = factor.half_solve_transpose(np.eye(40))
    np.testing.assert_allclose(transform @ transform.T, np.linalg.inv(matrix), atol=1e-10)

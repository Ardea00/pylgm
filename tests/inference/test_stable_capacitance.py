"""The sparse kriging capacitance K = A Q^-1 A^T must not be formed explicitly.

Near-unit-root AR1s under aggregate constraints push cond(K) toward 1/eps:
Cholesky on the formed K loses the digits the log marginal likelihood needs,
or fails outright although K is positive definite. Factoring G = L^-1 A^T by
QR, so that K = R^T R, works with sqrt(cond K) instead.

The reference log marginal likelihood was computed in 60-digit arithmetic
(mpmath) from the covariance form: condition N(0, Q^-1) on A x = e, then
evaluate y ~ N(Z m, Z S Z^T + sigma^2 I). It is the same for both
hyperparameter points below.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, random as sparse_random

import pylgm.inference.gaussian as gaussian_engine
from pylgm import AR1, Fixed, Gaussian, LGM
from pylgm.inference.sparse import SparseSpdFactor

R, Q = 6, 40
REFERENCE_LML = -68.27360345


def _model(precision, rho):
    rng = np.random.default_rng(0)
    grid = pd.DataFrame([(f"r{r}", q) for r in range(R) for q in range(Q)], columns=["region", "q"])
    g = (100 + rng.normal(size=(R, 1)) * 10 + np.cumsum(rng.normal(size=(R, Q)), axis=1)).ravel()
    grid["y"] = g + rng.normal(scale=0.5, size=g.size)
    grid.loc[grid["q"] >= 8, "y"] = np.nan

    def label(r, q):
        return f"regional:r{r}@{q}"

    # Annual sums per region and national quarterly sums, one national row per
    # year dropped so the rows stay independent.
    constraints = [
        ({label(r, q): 1.0 for q in range(4 * a, 4 * a + 4)},
         float(g[r * Q + 4 * a:r * Q + 4 * a + 4].sum()))
        for r in range(R) for a in range(Q // 4)
    ] + [
        ({label(r, q): 1.0 for r in range(R)}, float(g[q::Q].sum()))
        for q in range(Q) if q % 4 != 3
    ]
    model = LGM(
        response="y", likelihood=Gaussian(1.0), constraints=constraints,
        predictor=Fixed("1") + AR1("regional", "q", replicate="region", precision=precision, rho=rho),
        panel=("region",), time="q",
    )
    return model, grid


@pytest.mark.parametrize("precision, tolerance", [
    # Forming K: 3.1e-5 off; QR: 1.0e-5.
    (1e4, 2e-5),
    # Forming K: 0.69 off; QR: 0.16 -- still visibly short, but four times closer.
    (1e8, 0.3),
])
def test_an_ill_conditioned_capacitance_keeps_the_marginal_likelihood(
    precision, tolerance, monkeypatch
):
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    model, grid = _model(precision, 0.999999)
    assert model.fit(grid).log_marginal_likelihood == pytest.approx(REFERENCE_LML, abs=tolerance)


def test_the_factor_half_solve_is_a_square_root_of_the_inverse():
    m = sparse_random(40, 40, density=0.1, random_state=3)
    matrix = (m @ m.T + csr_matrix(np.diag(np.linspace(0.1, 5.0, 40)))).toarray()
    factor = SparseSpdFactor(csr_matrix(matrix), "test")
    rhs = np.random.default_rng(0).normal(size=(40, 5))
    g = factor.half_solve(rhs)
    np.testing.assert_allclose(g.T @ g, rhs.T @ np.linalg.solve(matrix, rhs), rtol=1e-10)


def test_spd_with_small_leading_pivot_is_accepted():
    # Partial pivoting swaps the rows here and yields diag(U) = [1, -9] on an
    # SPD matrix; the symmetric-mode factor must accept it (from PR #48).
    m = np.array([[1e-3, 1.0], [1.0, 1e4]])
    factor = SparseSpdFactor(csr_matrix(m), "test")
    sign, logdet = np.linalg.slogdet(m)
    assert sign > 0
    assert np.isclose(factor.logdet, logdet, atol=1e-10)
    b = np.array([1.0, 2.0])
    assert np.allclose(factor.solve(b), np.linalg.solve(m, b))

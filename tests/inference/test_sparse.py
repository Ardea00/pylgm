import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.sparse import csr_matrix

from pylgm.exceptions import NumericalError
from pylgm.inference.gaussian import _fit_dense
from pylgm.inference.sparse import (
    SparseSpdFactor,
    _partition_blocks,
    sparse_constrained_gaussian,
)


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_logdet_matches_slogdet():
    m = _spd(6)
    factor = SparseSpdFactor(csr_matrix(m), "test")
    sign, logdet = np.linalg.slogdet(m)
    assert sign > 0
    assert np.isclose(factor.logdet, logdet, atol=1e-8)


def test_solve_matches_dense():
    m = _spd(6)
    b = np.arange(6, dtype=float)
    factor = SparseSpdFactor(csr_matrix(m), "test")
    assert np.allclose(factor.solve(b), np.linalg.solve(m, b), atol=1e-8)
    # Multi-column solve (used for the Schur block B -> A_s^{-1} B).
    rhs = np.eye(6)[:, :2]
    assert np.allclose(factor.solve(rhs), np.linalg.solve(m, rhs), atol=1e-8)


def test_non_spd_raises():
    m = csr_matrix(np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(NumericalError):
        SparseSpdFactor(m, "indefinite")


def test_partition_splits_fixed_from_field(besag_with_intercept_model):
    # Fixture builds Fixed("1") + Besag(...) compiled model (see conftest).
    sparse_index, dense_index = _partition_blocks(besag_with_intercept_model)
    labels = besag_with_intercept_model.labels
    # The intercept (Fixed) has an all-ones design touching every
    # observation, so it lands in the dense block by design-column density.
    # Fixed("1") compiles the intercept column as "Intercept" (formula-library
    # convention), so the qualified label is "fixed:Intercept", not "fixed:1".
    dense_labels = {labels[i] for i in dense_index}
    assert any(label.endswith(":Intercept") for label in dense_labels)
    # Every Besag column has an incidence design (one cell each), so
    # it lands in the sparse block.
    assert len(sparse_index) + len(dense_index) == len(labels)
    assert set(sparse_index).isdisjoint(dense_index)


def _prior_logdet_blockwise(model):
    """logdet(basis^T Q_prior basis) as the per-block sum the engine will use.

    Each constraint row is assumed confined to a single block (holds for these
    fixtures); a block with confined rows contributes the reduced-block logdet,
    a block with none contributes its full sparse-block logdet.
    """
    a = model.constraints
    q = model.precision.toarray()
    total = 0.0
    start = 0
    for block in model.blocks:
        w = block.design.shape[1]
        sub = a[:, start : start + w]
        in_here = np.abs(sub).sum(axis=1) > 0
        elsewhere = np.abs(a).sum(axis=1) - np.abs(sub).sum(axis=1)
        confined = in_here & (elsewhere <= 1e-12)
        q_b = q[start : start + w, start : start + w]
        rows = a[confined][:, start : start + w]
        if rows.shape[0] == 0:
            total += np.linalg.slogdet(q_b)[1]
        else:
            basis_b = null_space(rows)
            total += np.linalg.slogdet(basis_b.T @ q_b @ basis_b)[1]
        start += w
    return total


def _check_kriging_identities(model):
    variance = float(model.likelihood.variance)
    q = model.precision.toarray()
    observed = model.observed
    z = model.design[observed].toarray()
    r0 = model.y[observed] - model.offset[observed]
    a = model.constraints
    e = model.constraint_rhs
    basis = null_space(a)

    q_post = q + z.T @ z / variance
    apply_inv = np.linalg.inv(q_post)  # spike: dense; engine uses the Schur solve
    g_full = z.T @ r0 / variance
    mu = apply_inv @ g_full

    # (1) kriged mean equals the dense constrained mean.
    w = apply_inv @ a.T
    m = a @ w
    mu_star = mu - w @ np.linalg.solve(m, a @ mu - e)
    dense = _fit_dense(model)
    assert np.allclose(mu_star, dense.mean, atol=1e-9), "kriged mean mismatch"

    # (2) reduced posterior logdet via the SPD identity.
    lhs_post = np.linalg.slogdet(basis.T @ q_post @ basis)[1]
    rhs_post = (
        np.linalg.slogdet(q_post)[1]
        + np.linalg.slogdet(m)[1]
        - np.linalg.slogdet(a @ a.T)[1]
    )
    assert np.isclose(lhs_post, rhs_post, atol=1e-9), "posterior logdet identity mismatch"

    # (3) reduced prior logdet: dense reference == per-block engine form.
    dense_prior = np.linalg.slogdet(basis.T @ q @ basis)[1]
    assert np.isclose(
        _prior_logdet_blockwise(model), dense_prior, atol=1e-9
    ), "prior logdet per-block mismatch"

    # (3b) tau-split: scaling a block's precision by tau shifts the reduced
    # logdet by rank_b * log(tau) (the Sorbye-Rue separability the plan uses).
    field = model.blocks[1]
    fstart = model.blocks[0].design.shape[1]
    fw = field.design.shape[1]
    fcols = slice(fstart, fstart + fw)
    frows = a[:, fcols]
    field_basis = null_space(frows[np.abs(frows).sum(axis=1) > 0])
    rank_b = field_basis.shape[1]
    tau = 3.7
    base = np.linalg.slogdet(field_basis.T @ q[fcols, fcols] @ field_basis)[1]
    scaled = np.linalg.slogdet(field_basis.T @ (tau * q[fcols, fcols]) @ field_basis)[1]
    assert np.isclose(scaled, base + rank_b * np.log(tau), atol=1e-9), "tau-split mismatch"

    # (4) quadratic against the dense oracle (backed out of the dense lml).
    #
    # The two-term form is used for BOTH cases (nu == 0 when e == 0). It is the
    # numerically robust one here: the intercept is confounded with the
    # intrinsic field's constant mode, so the unconstrained Q_post is severely
    # ill-conditioned (cond ~ 1/ridge). The kriged mean carries O(1e-7) error in
    # that confounded direction, but both Q (prior matvec) and Z (design)
    # annihilate it, so the two-term quadratic is accurate. The compact form
    # r0^T r0 / var - mu*^T g0 does NOT filter that direction and drifts ~1e-4.
    n_obs = int(np.count_nonzero(observed))
    dense_quad = (
        -2.0 * dense.log_marginal_likelihood
        - n_obs * np.log(2 * np.pi * variance)
        + dense_prior
        - lhs_post
    )
    if np.any(e):
        # Constrained prior mean nu = argmin_{A x = e} x^T Q x, via the augmented
        # SPD operator Q + A^T A (equal minimiser on the feasible set).
        q_aug = q + a.T @ a
        w_aug = np.linalg.inv(q_aug) @ a.T
        nu = w_aug @ np.linalg.solve(a @ w_aug, e)
    else:
        nu = np.zeros_like(mu_star)
    resid = r0 - z @ mu_star
    diff = mu_star - nu
    quad = float(resid @ resid / variance + diff @ (q @ diff))
    assert np.isclose(quad, dense_quad, rtol=1e-6, atol=1e-6), "quadratic mismatch"

    # (5) assembled lml matches dense end-to-end (Step-2's exact check).
    lml = -0.5 * (
        n_obs * np.log(2 * np.pi * variance) - dense_prior + lhs_post + quad
    )
    assert np.isclose(lml, dense.log_marginal_likelihood, atol=1e-6), "lml mismatch"


def test_kriging_identities_spike(besag_with_intercept_model):
    # Homogeneous: Besag sum-to-zero constraint only (e == 0).
    assert besag_with_intercept_model.constraints.shape[0] >= 1
    _check_kriging_identities(besag_with_intercept_model)


def test_kriging_identities_spike_extraconstr(besag_extraconstr_model):
    # Nonzero rhs: conditioning by kriging with e != 0.
    assert np.any(besag_extraconstr_model.constraint_rhs != 0.0)
    _check_kriging_identities(besag_extraconstr_model)


def test_sparse_matches_dense_no_constraints(proper_car_with_intercept_model):
    model = proper_car_with_intercept_model
    assert model.constraints.shape[0] == 0  # this fixture has no constraints
    dense = _fit_dense(model)
    sparse = sparse_constrained_gaussian(model)
    assert np.allclose(sparse.mean, dense.mean, atol=1e-7)
    assert np.isclose(
        sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-6
    )
    assert np.allclose(sparse.predictive_mean, dense.predictive_mean, atol=1e-7)


def test_sparse_matches_dense_besag(besag_with_intercept_model):
    model = besag_with_intercept_model  # intrinsic + sum-to-zero
    assert model.constraints.shape[0] >= 1
    dense = _fit_dense(model)
    sparse = sparse_constrained_gaussian(model)
    assert np.allclose(sparse.mean, dense.mean, atol=1e-7)
    assert np.isclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-6)
    assert np.allclose(sparse.predictive_mean, dense.predictive_mean, atol=1e-7)


def test_sparse_matches_dense_extraconstr(besag_extraconstr_model):
    # Model-level extraconstr with nonzero rhs (conditioning by kriging, e != 0).
    model = besag_extraconstr_model
    assert np.any(model.constraint_rhs != 0.0)
    dense = _fit_dense(model)
    sparse = sparse_constrained_gaussian(model)
    assert np.allclose(sparse.mean, dense.mean, atol=1e-7)
    assert np.isclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-6)


def test_selected_inverse_diagonal_raises():
    from scipy.sparse import csr_matrix
    from pylgm.inference.sparse import SparseSpdFactor, selected_inverse_diagonal

    factor = SparseSpdFactor(csr_matrix(np.eye(3)), "id")
    with pytest.raises(NotImplementedError, match="E-sparse-C"):
        selected_inverse_diagonal(factor, np.array([0, 1, 2]))

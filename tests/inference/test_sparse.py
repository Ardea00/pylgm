import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.sparse import coo_matrix, csr_matrix, tril
from scipy.sparse.linalg import splu

from pylgm.exceptions import DenseReferenceLimitError, NumericalError
from pylgm.inference.gaussian import _fit_dense, fit_gaussian
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


def test_spd_with_small_leading_pivot_is_accepted():
    # Partial pivoting (COLAMD, threshold 1) swaps the rows here and yields
    # diag(U) = [1, -9] on an SPD matrix; symmetric mode must accept it.
    m = np.array([[1e-3, 1.0], [1.0, 1e4]])
    factor = SparseSpdFactor(csr_matrix(m), "test")
    sign, logdet = np.linalg.slogdet(m)
    assert sign > 0
    assert np.isclose(factor.logdet, logdet, atol=1e-10)
    b = np.array([1.0, 2.0])
    assert np.allclose(factor.solve(b), np.linalg.solve(m, b))


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


def test_sparse_posterior_apply_inverse_reproduces_mean(proper_car_with_intercept_model):
    """The captured posterior's apply_inverse must reproduce the fit mean from the score.

    Uses the no-constraint proper-CAR fixture (full-rank field prior) rather than
    the Besag+intercept fixture: the latter's intrinsic field is exactly collinear
    with the intercept absent its sum-to-zero constraint, so its *unconstrained*
    posterior precision is near-singular (cond ~2.5e9) and a raw dense solve of it
    is not a meaningful reference at tight tolerance.
    """
    model = proper_car_with_intercept_model
    fit = sparse_constrained_gaussian(model)
    posterior = fit.posterior
    # rebuild the score g = Z^T r / sigma^2 and check apply_inverse(g) == unconstrained mean
    variance = float(model.likelihood.variance)
    observed = model.observed
    design = model.design[observed]
    residual = model.y[observed] - model.offset[observed]
    g = np.asarray(design.T @ residual).reshape(-1) / variance
    reconstructed = posterior.apply_inverse(g)
    # apply_inverse is the UNCONSTRAINED solve; compare to a direct dense solve of P.
    p = (model.precision + (design.T @ design) / variance).toarray()
    assert np.allclose(reconstructed, np.linalg.solve(p, g), atol=1e-7)


def test_sparse_matches_dense_extraconstr(besag_extraconstr_model):
    # Model-level extraconstr with nonzero rhs (conditioning by kriging, e != 0).
    model = besag_extraconstr_model
    assert np.any(model.constraint_rhs != 0.0)
    dense = _fit_dense(model)
    sparse = sparse_constrained_gaussian(model)
    assert np.allclose(sparse.mean, dense.mean, atol=1e-7)
    assert np.isclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-6)


def test_sparse_marginal_variances_match_dense_unconstrained(proper_car_with_intercept_model):
    """diag(Sigma) from the sparse posterior must equal _fit_dense's covariance
    diagonal on the unconstrained, well-conditioned proper-CAR fixture -- this
    exercises the Takahashi + Schur composition with w_constraint is None."""
    model = proper_car_with_intercept_model
    assert model.constraints.shape[0] == 0
    dense = _fit_dense(model)
    posterior = sparse_constrained_gaussian(model).posterior
    got = posterior.marginal_variances()
    assert np.allclose(got, np.diag(dense.covariance), atol=1e-7)


def test_sparse_marginal_variances_match_dense_besag(besag_with_intercept_model):
    """diag(Sigma_c) from the sparse posterior must equal _fit_dense's covariance
    diagonal on the Besag+intercept fixture -- this exercises the Rue-Held
    constrained correction.

    Tolerance: the unconstrained posterior precision here is confounded (the
    intrinsic field's null mode coincides with the intercept absent the
    sum-to-zero constraint, cond ~2.5e9), so _unconstrained_marginal computes
    diag(P^-1) for a near-singular P and the Rue-Held correction then subtracts
    the null direction back out -- a catastrophic-cancellation path. Measured
    relative error is ~1e-7, consistent with the ~1e9 condition number times
    machine epsilon; rtol=1e-6/atol=1e-8 covers that without masking a logic
    error (a wrong composition would miss by orders of magnitude, not ~1e-7).
    """
    model = besag_with_intercept_model
    assert model.constraints.shape[0] >= 1
    dense = _fit_dense(model)
    posterior = sparse_constrained_gaussian(model).posterior
    got = posterior.marginal_variances()
    want = np.diag(dense.covariance)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-8)


def test_sparse_predictive_and_covariance_match_dense_unconstrained(
    proper_car_with_intercept_model,
):
    """predictive_variances / linear_combination_variances / covariance_dense
    must equal the _fit_dense oracle on the well-conditioned, unconstrained
    proper-CAR fixture (w_constraint is None) -- this is the tight-tolerance
    correctness gate for the quadratic-form math itself."""
    model = proper_car_with_intercept_model
    assert model.constraints.shape[0] == 0
    dense = _fit_dense(model)
    posterior = sparse_constrained_gaussian(model).posterior

    design = model.design.toarray()
    want_pv = np.einsum("ij,jk,ik->i", design, dense.covariance, design)
    got_pv = posterior.predictive_variances(model.design)
    assert np.allclose(got_pv, want_pv, atol=1e-7)

    weights = design[:5]
    want_lc = np.einsum("ij,jk,ik->i", weights, dense.covariance, weights)
    assert np.allclose(posterior.linear_combination_variances(weights), want_lc, atol=1e-7)

    assert np.allclose(posterior.covariance_dense(), dense.covariance, atol=1e-7)


def test_sparse_predictive_and_covariance_match_dense_besag(besag_with_intercept_model):
    """Same three comparisons on the Besag+intercept fixture (w_constraint is
    not None) -- exercises the Rue-Held constrained correction in all three
    methods.

    Tolerance: as in test_sparse_marginal_variances_match_dense_besag, the
    unconstrained posterior precision here is confounded (intrinsic field's
    null mode == the intercept absent the sum-to-zero constraint, cond ~2.5e9),
    so this is a genuine catastrophic-cancellation path, not a bug: measured
    relative error ~cond * eps ~= 2.5e9 * 2.2e-16 ~= 5.5e-7. rtol=1e-6/atol=1e-8
    covers that without masking a wrong composition (which would miss by
    orders of magnitude, not ~1e-7).
    """
    model = besag_with_intercept_model
    assert model.constraints.shape[0] >= 1
    dense = _fit_dense(model)
    posterior = sparse_constrained_gaussian(model).posterior

    design = model.design.toarray()
    want_pv = np.einsum("ij,jk,ik->i", design, dense.covariance, design)
    got_pv = posterior.predictive_variances(model.design)
    assert np.allclose(got_pv, want_pv, rtol=1e-6, atol=1e-8)

    weights = design[:5]
    want_lc = np.einsum("ij,jk,ik->i", weights, dense.covariance, weights)
    got_lc = posterior.linear_combination_variances(weights)
    assert np.allclose(got_lc, want_lc, rtol=1e-6, atol=1e-8)

    got_cov = posterior.covariance_dense()
    assert np.allclose(got_cov, dense.covariance, rtol=1e-6, atol=1e-8)


def test_selected_inverse_matches_dense_inverse():
    """Takahashi selected inversion must equal diag(inv(Q)) on ring and grid GMRFs."""
    import numpy as np
    from scipy.sparse import lil_matrix
    from pylgm.inference.sparse import selected_inverse

    def ring(n, ridge=0.4):
        q = lil_matrix((n, n))
        for i in range(n):
            q[i, i] = 2.0 + ridge
            q[i, (i + 1) % n] = -1.0
            q[i, (i - 1) % n] = -1.0
        return q.tocsc()

    def grid(m, ridge=0.3):
        n = m * m
        q = lil_matrix((n, n))
        idx = lambda r, c: r * m + c  # noqa: E731
        for r in range(m):
            for c in range(m):
                i = idx(r, c)
                deg = 0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < m and 0 <= cc < m:
                        q[i, idx(rr, cc)] = -1.0
                        deg += 1
                q[i, i] = deg + ridge
        return q.tocsc()

    for mat in (ring(300), ring(1000), grid(20), grid(30)):
        got = selected_inverse(mat).diagonal()
        want = np.diag(np.linalg.inv(mat.toarray()))
        assert np.max(np.abs(got - want)) < 1e-9


def test_forced_sparse_under_guard_matches_dense(proper_car_with_intercept_model):
    # A model under the guard, forced sparse by monkeypatching the threshold low,
    # matches the dense fit on mean and lml.
    model = proper_car_with_intercept_model
    dense = fit_gaussian(model, allow_large_dense=True)  # dense
    import pylgm.inference.gaussian as g
    original = g._MAX_DENSE_LATENT_DIMENSION
    try:
        g._MAX_DENSE_LATENT_DIMENSION = 1     # force above-threshold routing
        sparse = fit_gaussian(model)          # sparse
    finally:
        g._MAX_DENSE_LATENT_DIMENSION = original
    assert np.allclose(sparse.mean, dense.mean, atol=1e-7)
    assert np.isclose(sparse.log_marginal_likelihood, dense.log_marginal_likelihood, atol=1e-6)
    # Task 5: the sparse posterior is now attached, so the full covariance is a
    # scale guard (DenseReferenceLimitError), not the posterior-less
    # NotImplementedError -- scoped accessors (latent_marginals/predictive_variance)
    # are exercised in tests/inference/test_sparse_result.py.
    with pytest.raises(DenseReferenceLimitError):
        _ = sparse.covariance


def test_bym2_augmented_prior_logdet_matches_dense():
    import numpy as np
    from scipy.linalg import null_space
    from scipy.sparse import bmat, csr_matrix, identity, lil_matrix

    from pylgm.inference.sparse import _bym2_augmented_logdet

    m = 5
    n = m * m
    r = lil_matrix((n, n))

    def idx(a, b):
        return a * m + b

    for a in range(m):
        for b in range(m):
            i = idx(a, b)
            deg = 0
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                aa, bb = a + da, b + db
                if 0 <= aa < m and 0 <= bb < m:
                    r[i, idx(aa, bb)] = -1.0
                    deg += 1
            r[i, i] = deg
    r = r.tocsr()
    w, v = np.linalg.eigh(r.toarray())
    inv = np.zeros_like(w)
    inv[1:] = 1 / w[1:]
    var = np.einsum("ij,j,ij->i", v, inv, v)
    s = np.exp(np.mean(np.log(var)))
    rstar = csr_matrix(r.toarray() * s)
    tau, phi = 1.3, 0.6
    a_ = 1 / (1 - phi)
    b_ = -np.sqrt(phi) / (1 - phi)
    d_ = phi / (1 - phi)
    ident = identity(n, format="csr")
    qj = tau * bmat([[a_ * ident, b_ * ident], [b_ * ident, rstar + d_ * ident]], format="csr")
    rows = np.zeros((1, 2 * n))
    rows[0, n:] = 1.0

    got = _bym2_augmented_logdet(rows, qj)
    basis = null_space(rows)
    _, want = np.linalg.slogdet(basis.T @ qj.toarray() @ basis)
    assert got is not None and abs(got - want) < 1e-7


def test_bym2_augmented_end_to_end_matches_dense(monkeypatch):
    """Public BYM2 fit via the augmented path: x-block marginals match the dense build."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    import pylgm.effects.bym2 as bym2_mod
    from pylgm.effects.bym2 import build_bym2
    from pylgm.inference.gaussian import _fit_dense
    from pylgm.inference.sparse import sparse_constrained_gaussian
    from pylgm.ir.model import CompiledLGM
    from pylgm.likelihoods import CompiledGaussian

    m = 3
    n = m * m

    def idx(r, c):
        return f"{r * m + c}"

    graph = {}
    for r in range(m):
        for c in range(m):
            nb = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < m:
                    nb.append(idx(rr, cc))
            graph[idx(r, c)] = nb
    regions = [idx(r, c) for r in range(m) for c in range(m)]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"region": regions, "y": rng.standard_normal(n)})
    tau, phi = 1.5, 0.5

    def as_model(block):
        return CompiledLGM(
            y=frame["y"].to_numpy(dtype=float),
            observed=np.ones(n, dtype=bool),
            offset=np.zeros(n),
            design=csr_matrix(block.design),
            precision=csr_matrix(block.precision),
            constraints=np.asarray(block.constraints, dtype=float),
            labels=tuple(f"{block.name}:{label}" for label in block.labels),
            likelihood=CompiledGaussian(0.1),
            blocks=(block,),
        )

    # dense path (constant at default 1024) then augmented path (constant forced to 1)
    dense_block = build_bym2(frame, "region", "region", graph, tau, phi)
    monkeypatch.setattr(bym2_mod, "_BYM2_AUGMENT_NODES", 1)
    aug_block = build_bym2(frame, "region", "region", graph, tau, phi)

    dense_var = np.diag(_fit_dense(as_model(dense_block)).covariance)
    aug_var = sparse_constrained_gaussian(as_model(aug_block)).posterior.marginal_variances()
    assert np.allclose(aug_var[:n], dense_var, atol=1e-6)


def test_bym2_augmented_reports_x_and_structured_subblocks(monkeypatch):
    """The augmented BYM2 fit exposes two named latent sub-blocks: ``region``
    (the x-effect, n values matching the dense build -- not the raw 2n block) and
    ``region.structured`` (the n structured-component u* marginals)."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    import pylgm.effects.bym2 as bym2_mod
    from pylgm.effects.bym2 import build_bym2
    from pylgm.inference.gaussian import _fit_dense, _fit_sparse
    from pylgm.ir.model import CompiledLGM
    from pylgm.likelihoods import CompiledGaussian

    m = 3
    n = m * m

    def idx(r, c):
        return f"{r * m + c}"

    graph = {}
    for r in range(m):
        for c in range(m):
            nb = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < m and 0 <= cc < m:
                    nb.append(idx(rr, cc))
            graph[idx(r, c)] = nb
    regions = [idx(r, c) for r in range(m) for c in range(m)]
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"region": regions, "y": rng.standard_normal(n)})
    tau, phi = 1.5, 0.5

    def as_model(block):
        return CompiledLGM(
            y=frame["y"].to_numpy(dtype=float),
            observed=np.ones(n, dtype=bool),
            offset=np.zeros(n),
            design=csr_matrix(block.design),
            precision=csr_matrix(block.precision),
            constraints=np.asarray(block.constraints, dtype=float),
            labels=tuple(f"{block.name}:{label}" for label in block.labels),
            likelihood=CompiledGaussian(0.1),
            blocks=(block,),
        )

    dense_block = build_bym2(frame, "region", "region", graph, tau, phi)
    monkeypatch.setattr(bym2_mod, "_BYM2_AUGMENT_NODES", 1)
    aug_res = _fit_sparse(as_model(build_bym2(frame, "region", "region", graph, tau, phi)))
    dense_region = _fit_dense(as_model(dense_block)).latent_marginals("region")

    region = aug_res.latent_marginals("region")
    structured = aug_res.latent_marginals("region.structured")
    assert region.mean.shape == (n,) and structured.mean.shape == (n,)
    # region (x) matches the dense x-marginals; structured is a separate quantity.
    assert np.allclose(region.mean, dense_region.mean, atol=1e-6)
    assert np.allclose(region.variance, dense_region.variance, atol=1e-6)
    assert np.all(np.isfinite(structured.mean)) and np.all(structured.variance > 0)


def test_allow_large_dense_forces_dense_above_threshold(proper_car_with_intercept_model):
    model = proper_car_with_intercept_model
    import pylgm.inference.gaussian as g
    original = g._MAX_DENSE_LATENT_DIMENSION
    try:
        g._MAX_DENSE_LATENT_DIMENSION = 1
        forced = fit_gaussian(model, allow_large_dense=True)   # must NOT route sparse
    finally:
        g._MAX_DENSE_LATENT_DIMENSION = original
    assert forced.covariance.shape[0] == len(model.labels)     # real dense covariance


def test_prior_logdet_cofactor_matches_dense_reference():
    """Connected-ring intrinsic block: the cofactor fast-path in _prior_logdet
    must equal a dense null_space reference to ~1e-9."""
    import numpy as np
    from scipy.linalg import null_space
    from scipy.sparse import lil_matrix
    from pylgm.inference.sparse import _is_connected_intrinsic

    n = 40
    Q = lil_matrix((n, n))
    for i in range(n):
        Q[i, i] = 2.0
        Q[i, (i + 1) % n] = -1.0
        Q[i, (i - 1) % n] = -1.0
    Q = Q.tocsr()
    rows = np.ones((1, n))
    assert _is_connected_intrinsic(rows, Q) is True

    # dense reference
    V = null_space(rows)
    _, dense_logdet = np.linalg.slogdet(V.T @ Q.toarray() @ V)
    # cofactor
    from pylgm.inference.sparse import SparseSpdFactor
    cofactor_logdet = np.log(n) + SparseSpdFactor(Q.tocsr()[1:, 1:], "t").logdet
    assert abs(cofactor_logdet - dense_logdet) < 1e-9

    # precondition rejects full-rank IID and multi-row constraints
    from scipy.sparse import identity
    assert _is_connected_intrinsic(rows, (2.0 * identity(n)).tocsr()) is False
    assert _is_connected_intrinsic(np.vstack([np.ones(n), np.r_[np.ones(20), -np.ones(20)]]), Q) is False
    mixed = np.array([[1.0, -1.0] * (n // 2)])
    assert _is_connected_intrinsic(mixed, Q) is False



def test_selected_inverse_on_a_factor_with_cancelled_fill():
    """Takahashi needs the symbolic fill pattern. This A_ss (a grounded Knorr-Held
    posterior precision; anchors from Linux LAPACK's pivoted QR) has L entries
    that cancel to exactly zero, which SuperLU drops; reading Sigma as 0 off the
    stored pattern gave entries wrong by 0.75 and marginal sds wrong by up to 7x."""
    from pathlib import Path

    from scipy.sparse import load_npz

    from pylgm.inference.sparse import selected_inverse

    matrix = load_npz(Path(__file__).parents[1] / "data" / "selinv_cancelled_fill.npz").tocsr()
    exact = np.linalg.inv(matrix.toarray())
    selected = selected_inverse(matrix).tocoo()
    np.testing.assert_allclose(selected.data, exact[selected.row, selected.col], atol=1e-10)


def test_prior_logdet_full_rank_block_matches_dense_reference():
    """`_full_rank_reduced_logdet` must match a dense null_space reference for a
    full-rank AR1-like block with confined constraint rows, fall back (return
    None) on an intrinsic (rank-deficient) block, and stay accurate on an
    ill-conditioned SPD block."""
    from pylgm.inference.sparse import _full_rank_reduced_logdet

    n = 60
    rho = 0.8

    def ar1_precision(n, rho, scale=1.0):
        q = np.zeros((n, n))
        for i in range(n):
            q[i, i] = (1.0 + rho**2) if 0 < i < n - 1 else 1.0
            if i + 1 < n:
                q[i, i + 1] = -rho
                q[i + 1, i] = -rho
        return csr_matrix(q * scale)

    q_full = ar1_precision(n, rho, scale=3.0)
    rng = np.random.default_rng(0)
    dense_rows = rng.standard_normal((4, n))
    rows = np.vstack([dense_rows, np.ones((1, n))])

    got = _full_rank_reduced_logdet(rows, q_full, "ar1")
    basis = null_space(rows)
    _, want = np.linalg.slogdet(basis.T @ q_full.toarray() @ basis)
    assert got is not None
    assert abs(got - want) < 1e-9

    # Ill-conditioned SPD (RW1 ring + tiny ridge) with an all-ones row.
    def rw1_ring(n, ridge):
        q = np.zeros((n, n))
        for i in range(n):
            q[i, i] = 2.0
            q[i, (i + 1) % n] += -1.0
            q[(i + 1) % n, i] += -1.0
        return q + ridge * np.eye(n)

    q_ill = csr_matrix(rw1_ring(n, 1e-6))
    rows_ones = np.ones((1, n))
    got_ill = _full_rank_reduced_logdet(rows_ones, q_ill, "seasonal")
    basis_ill = null_space(rows_ones)
    _, want_ill = np.linalg.slogdet(basis_ill.T @ q_ill.toarray() @ basis_ill)
    assert got_ill is not None
    assert abs(got_ill - want_ill) / abs(want_ill) < 1e-6

    # Intrinsic (singular) RW1 ring -> not SPD -> helper returns None. Note:
    # SuperLU's LU on this exactly-singular ring only reliably surfaces the
    # zero pivot (raising NumericalError, which the helper turns into None)
    # once the ring is a few hundred nodes; below that COLAMD's ordering can
    # happen to land the ~1e-16 rounding-noise "pivot" above zero. Use n=300,
    # confirmed to raise deterministically for this construction.
    n_intrinsic = 300
    q_intrinsic = csr_matrix(rw1_ring(n_intrinsic, 0.0))
    rows_intrinsic = np.ones((1, n_intrinsic))
    assert _full_rank_reduced_logdet(rows_intrinsic, q_intrinsic, "intrinsic") is None


def test_intrinsic_field_against_a_vague_intercept_keeps_sparse_variances_exact(monkeypatch):
    """An RW1 whose constant only a vague intercept pins leaves H with cond ~1e13;
    the sparse path used to solve it as is and lose ~1e-4 on the variances (then
    amplified by update() cancellations). Regularising every intrinsic block by
    its constraint rows -- exact on the constraint set -- keeps them to ~1e-7."""
    import pandas as pd

    import pylgm.inference.gaussian as gaussian_engine
    from pylgm import IID, LGM, RW1, Fixed, Gaussian

    rng = np.random.default_rng(0)
    regions, periods = 200, 60
    frame = pd.DataFrame([(f"r{r}", t) for r in range(regions) for t in range(periods)],
                         columns=["region", "t"])
    frame["y"] = (np.cumsum(rng.normal(scale=0.2, size=periods))[frame["t"]]
                  + rng.normal(size=regions)[frame["region"].str[1:].astype(int)]
                  + rng.normal(scale=0.5, size=len(frame)))
    frame.loc[frame["t"] >= 40, "y"] = np.nan                    # an unobserved tail
    model = LGM(response="y", likelihood=Gaussian(0.5),
                predictor=Fixed("1") + RW1("trend", index="t", precision=20.0)
                + IID("level", index="region", precision=1.0))
    dense = model.fit(frame)
    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    monkeypatch.setattr(gaussian_engine, "_SPARSE_LML_GRADIENT", True)
    sparse = model.fit(frame)
    assert sparse._sparse_posterior is not None
    np.testing.assert_allclose(
        sparse.latent_marginals().variance, np.diag(dense.covariance), rtol=1e-6
    )


# --- analytic LML gradient on the sparse path -------------------------------------

_CHAIN = {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j < 6] for i in range(6)}


def _gradient_frame():
    import pandas as pd

    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        [(str(r), t, str(g)) for r in range(6) for t in range(5) for g in range(3)],
        columns=["region", "t", "g"],
    )
    frame["y"] = rng.normal(size=len(frame))
    frame.loc[rng.random(len(frame)) < 0.2, "y"] = np.nan
    return frame


def _compile(model):
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data import CanonicalPanel

    prepared = _gradient_frame()
    prepared["__pylgm_row__"] = np.arange(len(prepared), dtype=np.int64)
    panel = CanonicalPanel.from_frame(
        prepared, DataConfig(time="__pylgm_row__", response="y"), require_observed=False
    )
    return compile_lgm(model, panel)


def _gradient_model(kind, theta):
    """theta = (field precision, field rho, iid precision, variance)."""
    from pylgm import IID, LGM, RW1, Besag, Fixed, Gaussian, ProperCAR

    prec, rho, iid, var = map(float, theta)
    field = {
        "rw1": RW1("f", index="t", precision=prec),
        "besag": Besag("f", index="region", graph=_CHAIN, precision=prec),
        "car": ProperCAR("f", index="region", graph=_CHAIN, rho=rho, precision=prec),
    }[kind]
    return _compile(LGM(
        response="y", likelihood=Gaussian(float(np.sqrt(var))),
        predictor=Fixed("1") + field + IID("g", index="g", precision=iid),
    ))


def _with_extra_row(model, extra):
    """Append a nonzero-rhs row on the field's columns (a data row if ``extra == "data"``)."""
    from pylgm.ir.model import CompiledLGM

    start = [i for i, label in enumerate(model.labels) if label.startswith("f:")][0]
    row = np.zeros((1, model.precision.shape[0]))
    row[0, start + 1] = 1.0
    row[0, start + 2] = 0.5
    return CompiledLGM(
        y=model.y, observed=model.observed, offset=model.offset, design=model.design,
        precision=model.precision, constraints=np.vstack([model.constraints, row]),
        labels=model.labels, likelihood=model.likelihood, blocks=model.blocks,
        extra_constraints=row, extra_constraint_rhs=np.array([0.8]),
        data_constraint_count=int(extra == "data"),
    )


@pytest.fixture
def force_sparse(monkeypatch):
    import pylgm.inference.gaussian as gaussian_engine

    monkeypatch.setattr(gaussian_engine, "_exceeds_dense_threshold", lambda model: True)
    monkeypatch.setattr(gaussian_engine, "_SPARSE_LML_GRADIENT", True)


@pytest.mark.parametrize("kind", ["rw1", "besag", "car"])
@pytest.mark.parametrize("extra", [None, "rhs", "data"])
def test_sparse_lml_gradient_matches_central_difference_and_dense(force_sparse, kind, extra):
    theta0 = np.array([1.3, 0.6, 0.8, 0.7])
    h = 1e-3

    def build(theta):
        model = _gradient_model(kind, theta)
        return model if extra is None else _with_extra_row(model, extra)

    def lml(theta):
        # Central difference of the dense LML. A vague intercept (precision 1e-6)
        # against an intrinsic field leaves both engines' LML ~1e-8 of rounding
        # noise, so the step is large and the tolerance 2e-5; the exact sparse-vs-
        # dense agreement (rtol 1e-8) below is the sharp check.
        return fit_gaussian(
            build(theta), predictive_variances=False, allow_large_dense=True
        ).log_marginal_likelihood

    index = {"prec": 0, "rho": 1, "iid": 2, "var": 3}
    names = ["prec", "rho", "iid", "var"] if kind == "car" else ["prec", "iid", "var"]
    directions, fd = [], []
    for name in names:
        e = np.zeros(4)
        e[index[name]] = 1.0
        fd.append((lml(theta0 + h * e) - lml(theta0 - h * e)) / (2 * h))
        if name == "var":
            directions.append((None, 1.0))
        else:  # Q is linear in each of these
            dq = (build(theta0 + h * e).precision - build(theta0 - h * e).precision) / (2 * h)
            directions.append((csr_matrix(dq), 0.0))
    model = build(theta0)
    sparse = fit_gaussian(model, lml_directions=tuple(directions))
    assert sparse._sparse_posterior is not None
    np.testing.assert_allclose(
        sparse.log_marginal_likelihood, lml(theta0), rtol=1e-8
    )
    grad = sparse.diagnostics["lml_gradient"]
    np.testing.assert_allclose(grad, fd, rtol=2e-5)
    dense = fit_gaussian(model, allow_large_dense=True, lml_directions=tuple(directions))
    assert dense._sparse_posterior is None
    np.testing.assert_allclose(grad, dense.diagnostics["lml_gradient"], rtol=1e-8)


def test_sparse_lml_gradient_iid_diagonal_and_absent_key(force_sparse):
    from pylgm import IID, LGM, Fixed, Gaussian

    def build(prec):
        return _compile(LGM(
            response="y", likelihood=Gaussian(0.9),
            predictor=Fixed("1") + IID("g", index="g", precision=prec),
        ))

    def lml(prec):
        return fit_gaussian(build(prec), predictive_variances=False).log_marginal_likelihood

    h = 1e-5
    dq = csr_matrix((build(1 + h).precision - build(1 - h).precision) / (2 * h))
    (grad,) = fit_gaussian(build(1.0), lml_directions=((dq, 0.0),)).diagnostics["lml_gradient"]
    np.testing.assert_allclose(grad, (lml(1 + h) - lml(1 - h)) / (2 * h), rtol=1e-6)
    assert "lml_gradient" not in fit_gaussian(build(1.0)).diagnostics


# Reference: the per-column Takahashi sweep selected_inverse replaced.
def _reference_symbolic_fill(strict_lower) -> list:
    """Row indices below the diagonal of each column of the Cholesky factor's
    symbolic pattern: struct(L_j) = struct(A_j) + the structs of j's children
    in the elimination tree, minus j. ``strict_lower`` is csc."""
    n = strict_lower.shape[0]
    children: list[list[int]] = [[] for _ in range(n)]
    structure: list = [None] * n
    for j in range(n):
        rows = set(strict_lower.indices[strict_lower.indptr[j]:strict_lower.indptr[j + 1]].tolist())
        for child in children[j]:
            rows |= structure[child]
        rows.discard(j)
        structure[j] = rows
        if rows:
            children[min(rows)].append(j)
    return [np.fromiter(sorted(rows), dtype=int, count=len(rows)) for rows in structure]


def _selected_inverse_reference(matrix) -> csr_matrix:
    """Entries of ``matrix⁻¹`` on the fill pattern of its factor (``L + Lᵀ``),
    for a sparse SPD matrix — the exact selected inverse via Takahashi
    recursion on the SuperLU fill pattern. Entries off that pattern are not
    stored and are *not* zero in ``matrix⁻¹``; the pattern contains ``matrix``'s own.

    Symmetric-mode ``splu`` (``COLAMD`` + ``SymmetricMode`` +
    ``diag_pivot_thresh=0.0``, as ``SparseSpdFactor``) yields ``perm_r == perm_c``
    and a unit-lower ``L`` with ``U == D·Lᵀ``. Not ``MMD_AT_PLUS_A``: on a
    bipartite network precision (borrowers x a few hub lenders) it finds the
    same fill but its ordering time grows quadratically (4.6s against 0.04s at
    20 000 borrowers), and this one call dominated the whole fit. A reverse column sweep over the *symbolic* fill pattern
    (``_symbolic_fill``) reconstructs the selected inverse exactly; SuperLU's
    stored pattern is not enough, because it drops entries that cancel to zero.
    Result is in the original ordering.
    """
    q_csc = matrix.tocsc()
    n = q_csc.shape[0]
    lu = splu(
        q_csc,
        permc_spec="COLAMD",
        options=dict(SymmetricMode=True),
        diag_pivot_thresh=0.0,
    )
    pc = lu.perm_c
    if not np.array_equal(lu.perm_r, pc):
        raise NumericalError("selected inversion expected a symmetric factorization")
    diag_u = lu.U.diagonal().astype(float)
    original = np.argsort(pc)  # permuted index p is original index original[p]
    # Takahashi needs the SYMBOLIC fill pattern: SuperLU drops L entries that
    # cancel to exactly zero (integer Laplacians do this), and reading Sigma as 0
    # off the stored pattern is wrong. Build it from the elimination tree of the
    # permuted matrix; L's value is 0 exactly where SuperLU dropped it.
    below = _reference_symbolic_fill(tril(q_csc[original][:, original], k=-1, format="csc"))
    lower = lu.L.tocsc()
    stored = np.repeat(np.arange(n), np.diff(lower.indptr)) * n + lower.indices
    order = np.argsort(stored)
    stored, stored_values = stored[order], lower.data.astype(float)[order]

    def l_values(column: int, rows: np.ndarray) -> np.ndarray:
        wanted = column * n + rows
        position = np.minimum(np.searchsorted(stored, wanted), stored.size - 1)
        return np.where(stored[position] == wanted, stored_values[position], 0.0)

    # Sigma lives on the lower symbolic pattern plus the diagonal, keyed col * n + row:
    # sorted keys make every sub-block gather one searchsorted per column.
    keys = np.sort(np.r_[
        np.concatenate([i * n + rows for i, rows in enumerate(below)] or [np.zeros(0, int)]),
        np.arange(n) * (n + 1),
    ])
    sig = np.zeros(keys.size)
    diagonal = np.searchsorted(keys, np.arange(n) * (n + 1))
    for i in range(n - 1, -1, -1):
        rows = below[i]
        if not rows.size:
            sig[diagonal[i]] = 1.0 / diag_u[i]
            continue
        values = l_values(i, rows)
        wanted = np.minimum.outer(rows, rows) * n + np.maximum.outer(rows, rows)
        # Every pair of a column's below-rows is on the symbolic pattern (closure).
        sub = sig[np.searchsorted(keys, wanted)]
        sig_below = -sub @ values
        sig[np.searchsorted(keys, i * n + rows)] = sig_below
        sig[diagonal[i]] = 1.0 / diag_u[i] - values @ sig_below
    rows, cols = original[keys % n], original[keys // n]
    off = rows != cols
    return coo_matrix(
        (np.r_[sig, sig[off]], (np.r_[rows, cols[off]], np.r_[cols, rows[off]])),
        shape=(n, n),
    ).tocsr()


def _spd_patterns():
    import scipy.sparse as sp

    rng = np.random.default_rng(7)

    def laplacian(adj, ridge):
        adj = ((adj + adj.T) > 0).astype(float)
        adj.setdiag(0)
        adj.eliminate_zeros()
        return (sp.diags(np.asarray(adj.sum(axis=1)).ravel() + ridge) - adj).tocsc()

    n = 400
    random = sp.random(n, n, density=4 / n, random_state=3, format="csr")
    random.data[:] = 1.0
    nb, nl = 300, 6
    lenders = np.vstack([rng.choice(nl, 3, replace=False) for _ in range(nb)])
    inc = sp.csr_matrix(
        (np.ones(3 * nb), (np.repeat(np.arange(nb), 3), lenders.ravel())), shape=(nb, nl)
    )
    hub = sp.bmat([[sp.identity(nb), inc], [inc.T, sp.identity(nl) * 3.0]])
    path = sp.diags([1.0, 1.0], [-1, 1], shape=(200, 200))
    side = sp.diags([1.0, 1.0], [-1, 1], shape=(18, 18))
    grid = sp.kron(side, sp.identity(18)) + sp.kron(sp.identity(18), side)
    yield "chain", laplacian(path, 0.5)
    yield "grid", laplacian(grid, 0.1)
    yield "hub", (hub.T @ hub + sp.identity(nb + nl)).tocsc()
    yield "random", laplacian(random, 0.3)
    yield "n1", sp.csc_matrix([[2.0]])
    yield "diagonal", sp.diags(np.arange(1.0, 9.0)).tocsc()
    # integer Laplacian: exact cancellation drops fill that the symbolic pattern keeps
    yield "integer-grid", laplacian(grid, 1.0)


@pytest.mark.parametrize("name,matrix", list(_spd_patterns()), ids=lambda v: v if isinstance(v, str) else "")
def test_selected_inverse_matches_reference(name, matrix):
    from pylgm.inference.sparse import selected_inverse

    want = _selected_inverse_reference(matrix).tocoo()
    got = selected_inverse(matrix).tocsr()
    np.testing.assert_allclose(np.asarray(got[want.row, want.col]).ravel(), want.data, rtol=1e-10, atol=1e-12)


def test_selected_inverse_matches_reference_on_cancelled_fill():
    from pathlib import Path

    from scipy.sparse import load_npz

    from pylgm.inference.sparse import selected_inverse

    matrix = load_npz(Path(__file__).parents[1] / "data" / "selinv_cancelled_fill.npz").tocsc()
    want = _selected_inverse_reference(matrix).tocoo()
    got = selected_inverse(matrix).tocsr()
    np.testing.assert_allclose(np.asarray(got[want.row, want.col]).ravel(), want.data, rtol=1e-10, atol=1e-12)

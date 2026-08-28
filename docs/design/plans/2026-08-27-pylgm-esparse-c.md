# E-sparse-C — Sparse Posterior Uncertainty (Full C, incl. off-diagonal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the full posterior-uncertainty surface (marginal variances, predictive variances, linear-combination variances, `predict`, constrained corrections, Sørbye-Rue scaling, augmented BYM2, and INLA integration) at network scale on the sparse solver — removing the `pending E-sparse-C` sentinels — deterministically and with no new dependency.

**Architecture:** A validated Takahashi selected-inversion primitive gives `diag(A⁻¹)` for the sparse field factor. A new `SparsePosterior` captures the factors from the Schur solve and composes them into `diag(Σ)`, `diag(MΣMᵀ)`, and the Rue-Held constrained correction — never forming the full `n×n` covariance above the dense guard. `GaussianResult`/`INLAResult` route their scoped accessors through the posterior when the dense covariance is absent. BYM2 gains an augmented `(x, u*)` sparse joint that reproduces the dense x-marginal exactly and scales.

**Tech Stack:** NumPy, SciPy (`scipy.sparse`, `scipy.sparse.linalg.splu`, `scipy.linalg.cho_solve`), pandas. Python 3.12+.

## Global Constraints

- **Git identity is `Ardea00 <Ardea00@users.noreply.github.com>` ONLY.** Never `--author`. Commit form: `git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "..."`. Before reporting a task DONE, run `git log -1 --format='%h | %an <%ae> | %cn <%ce>'` and confirm BOTH author and committer are Ardea00.
- **No new runtime dependency.** NumPy / SciPy / pandas / stdlib only. Deterministic, no MCMC.
- **Ponytail:** reuse before building, shortest correct diff, mark deliberate simplifications with a `ponytail:` comment naming the ceiling and the upgrade path, leave one runnable check per non-trivial piece.
- **Preserve A+B behavior.** The A+B sparse equivalence suite (`tests/inference/test_sparse.py`) and the full existing test suite are the regression gate. The sparse *mean / lml / predictive_mean* path from A+B must not change. This slice only *adds* the uncertainty surface and fills the pending-C seams.
- **Dense-equivalence oracle.** Every new uncertainty quantity is gated against the dense reference (`_fit_dense` / dense eigh / dense `null_space`) at ~1e-7 (variances) / ~1e-6 (lml) on a small model where both paths run.

---

## File Structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `src/pylgm/inference/sparse.py` | Takahashi primitive; `SparsePosterior`; posterior capture in `sparse_constrained_gaussian`; BYM2 generalized-cofactor prior logdet | 1,2,3,4,9 |
| `src/pylgm/inference/gaussian.py` | `_fit_sparse` populates posterior + predictive variance | 5 |
| `src/pylgm/inference/result.py` | `GaussianResult._sparse_posterior`; variance-vector accessor helpers; scoped-accessor routing; covariance scale-guard | 5,6 |
| `src/pylgm/inference/prediction.py` | `predict_from_sparse` | 6 |
| `src/pylgm/model.py` | `_rebuild_result` carries `_sparse_posterior` | 7 |
| `src/pylgm/effects/scaling.py` | sparse Sørbye-Rue null_dim==1 fast path | 8 |
| `src/pylgm/effects/bym2.py` | augmented `(x,u*)` sparse builder + large-graph branch | 9 |
| `src/pylgm/compiler.py` | wire augmented BYM2 through plain + parametric paths | 9 |
| `src/pylgm/optimization/inla.py` | diagonal integration (Gaussian + simplified-Laplace); full-Laplace guarded | 10 |
| `tests/inference/test_sparse.py`, `test_sparse_result.py`, `tests/effects/…`, `tests/optimization/…` | oracles & regression | all |
| `docs/roadmap.md` | shipped/next notes | 11 |

---

## Task 1: Takahashi selected-inverse diagonal primitive

**Files:**
- Modify: `src/pylgm/inference/sparse.py` (replace `selected_inverse_diagonal`, ~lines 193-201)
- Test: `tests/inference/test_sparse.py` (replace the pending-C raise test with a correctness test)

**Interfaces:**
- Consumes: `splu` (already imported), `np` (already imported).
- Produces: `selected_inverse_diagonal(matrix) -> np.ndarray` — the full diagonal of `matrix⁻¹` for a sparse SPD `matrix` (csr/csc). **Signature change:** the A+B stub `(factor, indices)` is replaced by `(matrix)`; there are no real callers yet (only the raise-test).

- [ ] **Step 1: Replace the raise-test with a correctness test** in `tests/inference/test_sparse.py`. Find the existing test that asserts `selected_inverse_diagonal` raises `NotImplementedError(match="E-sparse-C")` and replace it with:

```python
def test_selected_inverse_diagonal_matches_dense_inverse():
    """Takahashi selected inversion must equal diag(inv(Q)) on ring and grid GMRFs."""
    import numpy as np
    from scipy.sparse import lil_matrix
    from pylgm.inference.sparse import selected_inverse_diagonal

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
        idx = lambda r, c: r * m + c
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
        got = selected_inverse_diagonal(mat)
        want = np.diag(np.linalg.inv(mat.toarray()))
        assert np.max(np.abs(got - want)) < 1e-9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/inference/test_sparse.py::test_selected_inverse_diagonal_matches_dense_inverse -v`
Expected: FAIL (`NotImplementedError: ...pending E-sparse-C`).

- [ ] **Step 3: Replace the `selected_inverse_diagonal` body** in `src/pylgm/inference/sparse.py`. Replace the whole function (the stub currently raising) with the validated Takahashi recursion:

```python
def selected_inverse_diagonal(matrix) -> np.ndarray:
    """Diagonal of ``matrix⁻¹`` for a sparse SPD matrix — the exact selected
    inverse via Takahashi recursion on the SuperLU fill pattern.

    Symmetric-mode ``splu`` (``MMD_AT_PLUS_A`` + ``SymmetricMode`` +
    ``diag_pivot_thresh=0.0``) yields ``perm_r == perm_c`` and a unit-lower ``L``
    with ``U == D·Lᵀ``. A reverse column sweep on the below-diagonal fill pattern
    reconstructs exactly the selected-inverse entries needed for the diagonal.
    Off-pattern sub-block entries are true zeros (SuperLU drops only numerical
    zeros), so ``Sig.get(key, 0.0)`` is exact. Result is in the original ordering.
    """
    q_csc = matrix.tocsc()
    n = q_csc.shape[0]
    lu = splu(
        q_csc,
        permc_spec="MMD_AT_PLUS_A",
        options=dict(SymmetricMode=True),
        diag_pivot_thresh=0.0,
    )
    pc = lu.perm_c
    if not np.array_equal(lu.perm_r, pc):
        raise NumericalError("selected inversion expected a symmetric factorization")
    lower = lu.L.tocsc()
    diag_u = lu.U.diagonal().astype(float)
    indptr, indices, data = lower.indptr, lower.indices, lower.data.astype(float)
    below_rows: list = [None] * n
    below_l: list = [None] * n
    sig: dict = {}
    for i in range(n):
        seg = indices[indptr[i]:indptr[i + 1]]
        val = data[indptr[i]:indptr[i + 1]]
        mask = seg > i
        below_rows[i] = seg[mask]
        below_l[i] = val[mask]
    for i in range(n - 1, -1, -1):
        below = below_rows[i]
        below_vals = below_l[i]
        if len(below):
            k = len(below)
            sub = np.empty((k, k))
            for a in range(k):
                for b in range(k):
                    ra, rb = below[a], below[b]
                    lo, hi = (ra, rb) if ra <= rb else (rb, ra)
                    sub[a, b] = sig.get((hi, lo), 0.0)
            sig_below = -sub @ below_vals
            for a in range(k):
                sig[(below[a], i)] = sig_below[a]
            sig[(i, i)] = 1.0 / diag_u[i] - below_vals @ sig_below
        else:
            sig[(i, i)] = 1.0 / diag_u[i]
    return np.array([sig[(pc[i], pc[i])] for i in range(n)])
```

Also update the module's `NumericalError` import — it is already imported at the top of `sparse.py` (line 9), so no new import is needed.

- [ ] **Step 4: Run the correctness test**

Run: `python -m pytest tests/inference/test_sparse.py::test_selected_inverse_diagonal_matches_dense_inverse -v`
Expected: PASS.

- [ ] **Step 5: Report wall-time (no committed timing assert).** Run this one-liner and paste the output into your report (it is the scaling evidence, not a test):

```bash
python -c "
import time, numpy as np
from scipy.sparse import lil_matrix
from pylgm.inference.sparse import selected_inverse_diagonal
def ring(n,ridge=0.4):
    q=lil_matrix((n,n))
    for i in range(n):
        q[i,i]=2.0+ridge; q[i,(i+1)%n]=-1.0; q[i,(i-1)%n]=-1.0
    return q.tocsc()
for n in (5000,20000):
    t=time.time(); selected_inverse_diagonal(ring(n)); print(f'ring {n}: {time.time()-t:.2f}s')
"
```

Expected order of magnitude: ring 5000 ≈ 0.03 s, ring 20000 ≈ 0.15 s.

- [ ] **Step 6: Run the full sparse suite + ruff, then commit**

Run: `python -m pytest tests/inference/test_sparse.py -v -W error::UserWarning` (expect all green) and `python -m ruff check src tests` (expect `All checks passed!`).

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): Takahashi selected-inverse diagonal primitive"
```

---

## Task 2: `SparsePosterior` — capture factors + `apply_inverse`

**Files:**
- Modify: `src/pylgm/inference/sparse.py` (add `SparsePosterior` dataclass; add `posterior` field to `SparseFit`; populate it in `sparse_constrained_gaussian`)
- Test: `tests/inference/test_sparse.py` (one test: `posterior.apply_inverse` reproduces the mean)

**Interfaces:**
- Consumes: `selected_inverse_diagonal` (Task 1), `SparseSpdFactor`, `cho_solve` (already imported), the factors built in `sparse_constrained_gaussian`.
- Produces:
  - `SparsePosterior` frozen dataclass with fields `latent_size:int`, `sparse_index:np.ndarray`, `dense_index:np.ndarray`, `a_ss:SparseSpdFactor|None`, `a_ss_matrix`(csr)`|None`, `b:np.ndarray|None`, `schur_factor:object|None`, `d_factor:object|None`, `w_constraint:np.ndarray|None`, `cap_factor:object|None`.
  - `SparsePosterior.apply_inverse(rhs) -> np.ndarray`.
  - `SparseFit.posterior: SparsePosterior` (new field, last).
  - Marginal/predictive methods are added in Tasks 3-4 (do not add them here).

- [ ] **Step 1: Write the failing test** in `tests/inference/test_sparse.py`:

```python
def test_sparse_posterior_apply_inverse_reproduces_mean(besag_gaussian_model):
    """The captured posterior's apply_inverse must reproduce the fit mean from the score."""
    import numpy as np
    from pylgm.inference.sparse import sparse_constrained_gaussian

    fit = sparse_constrained_gaussian(besag_gaussian_model)
    posterior = fit.posterior
    # rebuild the score g = Z^T r / sigma^2 and check apply_inverse(g) == unconstrained mean
    model = besag_gaussian_model
    variance = float(model.likelihood.variance)
    observed = model.observed
    design = model.design[observed]
    residual = model.y[observed] - model.offset[observed]
    g = np.asarray(design.T @ residual).reshape(-1) / variance
    reconstructed = posterior.apply_inverse(g)
    # apply_inverse is the UNCONSTRAINED solve; compare to a direct dense solve of P.
    p = (model.precision + (design.T @ design) / variance).toarray()
    assert np.allclose(reconstructed, np.linalg.solve(p, g), atol=1e-7)
```

> Use whatever existing fixture the file already has for a small Besag Gaussian model (search the file for the fixture the A+B equivalence tests use, e.g. `test_sparse_matches_dense_besag`, and reuse its model-construction). If it is inline rather than a fixture, construct the model the same way inline.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/inference/test_sparse.py::test_sparse_posterior_apply_inverse_reproduces_mean -v`
Expected: FAIL (`AttributeError: 'SparseFit' object has no attribute 'posterior'`).

- [ ] **Step 3: Add the `SparsePosterior` dataclass** immediately below `SparseFit` in `src/pylgm/inference/sparse.py`:

```python
@dataclass(frozen=True)
class SparsePosterior:
    """Posterior precision factors from the Schur solve, exposed as an inverse
    operator and (Tasks 3-4) as diagonal-variance composers. Never forms the full
    n×n covariance above the dense guard.
    """

    latent_size: int
    sparse_index: np.ndarray
    dense_index: np.ndarray
    a_ss: "SparseSpdFactor | None"
    a_ss_matrix: "csr_matrix | None"   # A_ss = Q_ss + Z_s^T Z_s / sigma^2 (for Takahashi)
    b: "np.ndarray | None"             # B = Z_s^T Z_d / sigma^2  (n_s x m)
    schur_factor: object | None        # cho_factor of S = D - B^T A_ss^-1 B
    d_factor: object | None            # cho_factor of D (dense-only case)
    w_constraint: "np.ndarray | None"  # Sigma A_c^T  (latent x c), kriging basis
    cap_factor: object | None          # cho_factor of A_c Sigma A_c^T

    def apply_inverse(self, rhs: np.ndarray) -> np.ndarray:
        """``P_post^-1 @ rhs`` via the same Schur solve as the mean; 1-D or 2-D."""
        rhs = np.asarray(rhs, dtype=float)
        out = np.zeros_like(rhs)
        s, d = self.sparse_index, self.dense_index
        if s.size and d.size:
            v_s, v_d = rhs[s], rhs[d]
            x_d = cho_solve(self.schur_factor, v_d - self.b.T @ self.a_ss.solve(v_s))
            out[s] = self.a_ss.solve(v_s - self.b @ x_d)
            out[d] = x_d
        elif s.size:
            out[s] = self.a_ss.solve(rhs[s])
        else:
            out[d] = cho_solve(self.d_factor, rhs[d])
        return out
```

- [ ] **Step 4: Add `posterior` to `SparseFit`.** Append one field to the `SparseFit` dataclass:

```python
@dataclass(frozen=True)
class SparseFit:
    mean: np.ndarray
    log_marginal_likelihood: float
    predictive_mean: np.ndarray
    block_slices: Mapping[str, slice]
    diagnostics: dict[str, object]
    posterior: "SparsePosterior"
```

- [ ] **Step 5: Populate the posterior in `sparse_constrained_gaussian`.** Two edits in that function:

(a) Keep a reference to the sparse posterior-precision matrix when the factor is built. Change the `if n_s:` block so `a_ss_matrix` is captured:

```python
    a_ss_matrix = None
    if n_s:
        q_ss = q[sparse_index][:, sparse_index]
        a_ss_matrix = (q_ss + (z_s.T @ z_s) / variance).tocsr()
        a_s = SparseSpdFactor(a_ss_matrix, "sparse posterior precision")
        logdet_posterior += a_s.logdet
```

(b) In the constrained branch, `w` and `cap_factor` are already computed (`w = apply_inverse(a.T)`, `capacitance`, `cap_factor`). After the constrained/unconstrained branches, before building `SparseFit`, set the two kriging references (they stay `None` when there are no constraints). Introduce them where `w`/`cap_factor` are in scope: initialise `w_constraint = None; cap_factor_ref = None` just before `if constraint_count:`, and inside that branch after `cap_factor` is created add `w_constraint = w; cap_factor_ref = cap_factor`.

Then construct the posterior and pass it to `SparseFit`:

```python
    posterior = SparsePosterior(
        latent_size=latent_size,
        sparse_index=sparse_index,
        dense_index=dense_index,
        a_ss=a_s,
        a_ss_matrix=a_ss_matrix,
        b=b,
        schur_factor=schur_factor,
        d_factor=d_factor,
        w_constraint=w_constraint,
        cap_factor=cap_factor_ref,
    )
    return SparseFit(
        mean=mean,
        log_marginal_likelihood=log_marginal_likelihood,
        predictive_mean=predictive_mean,
        block_slices=_block_slices(model),
        diagnostics={
            "latent_dimension": int(latent_size),
            "observed_count": n_observed,
            "constraint_count": int(constraint_count),
            "sparse_dimension": int(n_s),
            "dense_dimension": int(m),
        },
        posterior=posterior,
    )
```

> Note `a_s`, `schur_factor`, `d_factor`, `b` are already local variables initialised to `None` at line ~242 (`a_s = schur_factor = d_factor = b = None`). Reuse them; do not re-declare.

- [ ] **Step 6: Run the test + full sparse suite**

Run: `python -m pytest tests/inference/test_sparse.py -v -W error::UserWarning`
Expected: PASS (existing A+B equivalence tests still green — mean/lml/predictive_mean unchanged).

- [ ] **Step 7: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): capture SparsePosterior (factors + apply_inverse) from the Schur solve"
```

---

## Task 3: `SparsePosterior.marginal_variances` — diagonal composition + constrained correction

**Files:**
- Modify: `src/pylgm/inference/sparse.py` (add two methods to `SparsePosterior`)
- Test: `tests/inference/test_sparse.py` (dense-equivalence oracle for `diag(Σ)`)

**Interfaces:**
- Consumes: `selected_inverse_diagonal` (Task 1), the fields captured in Task 2, `cho_solve`, `np`.
- Produces:
  - `SparsePosterior._unconstrained_marginal() -> np.ndarray` (diag of `P⁻¹`, full latent length).
  - `SparsePosterior.marginal_variances() -> np.ndarray` (diag of the *constrained* posterior `Σ_c`, full latent length, non-negative).

Composition math (validated): `Σ_ss` diagonal is `diag(A_ss⁻¹) + diag(W S⁻¹ Wᵀ)` with `W = A_ss⁻¹ B`, `S = D − Bᵀ A_ss⁻¹ B`; `Σ_dd` diagonal is `diag(S⁻¹)`. Constrained (Rue-Held): `diag(Σ_c) = diag(Σ) − diag(w cap⁻¹ wᵀ)`, `w = Σ A_cᵀ`, `cap = A_c Σ A_cᵀ`.

- [ ] **Step 1: Write the failing test** in `tests/inference/test_sparse.py`. This compares the sparse marginal variances to `_fit_dense`'s covariance diagonal on a small model where both paths run. Add an **unconstrained** case (IID + Besag, no sum-to-zero on the fixed block) *and* a **constrained** case (Besag component constraint):

```python
def test_sparse_marginal_variances_match_dense_besag(besag_gaussian_model):
    """diag(Σ) from the sparse posterior must equal _fit_dense's covariance diagonal."""
    import numpy as np
    from pylgm.inference.gaussian import _fit_dense
    from pylgm.inference.sparse import sparse_constrained_gaussian

    dense = _fit_dense(besag_gaussian_model)
    posterior = sparse_constrained_gaussian(besag_gaussian_model).posterior
    got = posterior.marginal_variances()
    assert np.allclose(got, np.diag(dense.covariance), atol=1e-7)
```

> Reuse the same small model the A+B `test_sparse_matches_dense_besag` uses — it carries the Besag component (constrained) block, so it exercises the Rue-Held correction. If a purely-unconstrained fixture is also present in the file, add a second identical assertion against it to cover the `w_constraint is None` branch.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/inference/test_sparse.py::test_sparse_marginal_variances_match_dense_besag -v`
Expected: FAIL (`AttributeError: ... has no attribute 'marginal_variances'`).

- [ ] **Step 3: Add the two methods** to `SparsePosterior` (below `apply_inverse`):

```python
    def _unconstrained_marginal(self) -> np.ndarray:
        """diag(P_post^-1) — the unconstrained posterior variance, full length."""
        diag = np.zeros(self.latent_size)
        s, d = self.sparse_index, self.dense_index
        if s.size:
            diag_ss = selected_inverse_diagonal(self.a_ss_matrix)   # diag(A_ss^-1)
            if d.size:
                w = self.a_ss.solve(self.b)                          # A_ss^-1 B  (n_s x m)
                sinv_wt = cho_solve(self.schur_factor, w.T)          # S^-1 W^T   (m x n_s)
                diag_ss = diag_ss + np.einsum("ij,ji->i", w, sinv_wt)
            diag[s] = diag_ss
        if d.size:
            m = d.size
            factor = self.schur_factor if s.size else self.d_factor
            diag[d] = np.diag(cho_solve(factor, np.eye(m)))
        return diag

    def marginal_variances(self) -> np.ndarray:
        """diag(Σ_c) — constrained posterior marginal variances, full length."""
        diag = self._unconstrained_marginal()
        if self.w_constraint is not None:
            w = self.w_constraint                                    # latent x c
            cw = cho_solve(self.cap_factor, w.T)                     # c x latent
            diag = diag - np.einsum("ij,ji->i", w, cw)
        return np.clip(diag, 0.0, None)
```

- [ ] **Step 4: Run the oracle**

Run: `python -m pytest tests/inference/test_sparse.py::test_sparse_marginal_variances_match_dense_besag -v`
Expected: PASS (match ~1e-7).

- [ ] **Step 5: Full sparse suite + ruff + commit**

Run: `python -m pytest tests/inference/test_sparse.py -v -W error::UserWarning` and `python -m ruff check src tests`.

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): marginal_variances via Takahashi + Schur + Rue-Held correction"
```

---

## Task 4: predictive / linear-combination variances + dense escape hatch

**Files:**
- Modify: `src/pylgm/inference/sparse.py` (three methods on `SparsePosterior`)
- Test: `tests/inference/test_sparse.py` (dense-equivalence for `diag(design Σ designᵀ)` and `covariance_dense`)

**Interfaces:**
- Consumes: `apply_inverse`, `w_constraint`, `cap_factor`, `cho_solve`, `np`.
- Produces:
  - `SparsePosterior.predictive_variances(design) -> np.ndarray` — `diag(design Σ_c designᵀ)`, one entry per row of `design`.
  - `SparsePosterior.linear_combination_variances(weights) -> np.ndarray` — identical quadratic form for a weight matrix (delegates to `predictive_variances`).
  - `SparsePosterior.covariance_dense() -> np.ndarray` — full `Σ_c` (O(n²), opt-in escape hatch).

- [ ] **Step 1: Write the failing test** in `tests/inference/test_sparse.py`:

```python
def test_sparse_predictive_and_covariance_match_dense(besag_gaussian_model):
    """Predictive variances and the dense escape hatch must equal _fit_dense."""
    import numpy as np
    from pylgm.inference.gaussian import _fit_dense
    from pylgm.inference.sparse import sparse_constrained_gaussian

    model = besag_gaussian_model
    dense = _fit_dense(model)
    posterior = sparse_constrained_gaussian(model).posterior

    design = model.design.toarray()
    want_pv = np.einsum("ij,jk,ik->i", design, dense.covariance, design)
    got_pv = posterior.predictive_variances(model.design)
    assert np.allclose(got_pv, want_pv, atol=1e-7)

    # linear_combination_variances shares the quadratic form
    weights = design[:5]
    want_lc = np.einsum("ij,jk,ik->i", weights, dense.covariance, weights)
    assert np.allclose(posterior.linear_combination_variances(weights), want_lc, atol=1e-7)

    # dense escape hatch reproduces the full covariance
    assert np.allclose(posterior.covariance_dense(), dense.covariance, atol=1e-7)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/inference/test_sparse.py::test_sparse_predictive_and_covariance_match_dense -v`
Expected: FAIL (`AttributeError: ... 'predictive_variances'`).

- [ ] **Step 3: Add the three methods** to `SparsePosterior`:

```python
    def predictive_variances(self, design) -> np.ndarray:
        """diag(design Σ_c designᵀ) — one variance per row of ``design``."""
        dense = design.toarray() if hasattr(design, "toarray") else np.asarray(design, float)
        # ponytail: apply_inverse densifies designᵀ to (latent x n_rows). At very
        # large latent × request width this is the memory ceiling — batch the
        # columns of designᵀ if it ever bites; fine at current network scale.
        cov_dt = self.apply_inverse(dense.T)                        # Σ designᵀ (latent x n_rows)
        var = np.einsum("ij,ji->i", dense, cov_dt)
        if self.w_constraint is not None:
            mw = dense @ self.w_constraint                          # n_rows x c
            cw = cho_solve(self.cap_factor, mw.T)                   # c x n_rows
            var = var - np.einsum("ij,ji->i", mw, cw)
        return np.clip(var, 0.0, None)

    def linear_combination_variances(self, weights) -> np.ndarray:
        """diag(M Σ_c Mᵀ) — same quadratic form as predictive_variances."""
        return self.predictive_variances(weights)

    def covariance_dense(self) -> np.ndarray:
        """Full constrained covariance Σ_c (O(n²) — opt-in escape hatch)."""
        sigma = self.apply_inverse(np.eye(self.latent_size))
        if self.w_constraint is not None:
            w = self.w_constraint
            sigma = sigma - w @ cho_solve(self.cap_factor, w.T)
        return 0.5 * (sigma + sigma.T)
```

- [ ] **Step 4: Run the oracle**

Run: `python -m pytest tests/inference/test_sparse.py::test_sparse_predictive_and_covariance_match_dense -v`
Expected: PASS.

- [ ] **Step 5: Full sparse suite + ruff + commit**

Run: `python -m pytest tests/inference/test_sparse.py -v -W error::UserWarning` and `python -m ruff check src tests`.

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): predictive / linear-combination variances + dense escape hatch"
```

---

## Task 5: route `GaussianResult` accessors through the posterior; guard covariance

**Files:**
- Modify: `src/pylgm/inference/gaussian.py` (`_fit_sparse`)
- Modify: `src/pylgm/inference/result.py` (`GaussianResult` gains `_sparse_posterior`; variance-vector helpers; accessor routing; covariance scale-guard)
- Test: `tests/inference/test_sparse_result.py`

**Interfaces:**
- Consumes: `SparsePosterior` (Tasks 2-4), `SparseFit.posterior`, existing `latent_marginals_from` / `linear_combinations_from` (result.py:175-230), `GaussianMarginals`.
- Produces:
  - `GaussianResult(..., sparse_posterior: SparsePosterior | None = None)` stored as `_sparse_posterior`.
  - `latent_marginals_from_variances(mean, variances, block_slices, block) -> GaussianMarginals`.
  - `linear_combinations_from_variances(mean, variances, weights) -> GaussianMarginals`.
  - `GaussianResult.latent_marginals` / `.linear_combinations` route to the posterior when `_covariance is None and _sparse_posterior is not None`.
  - `GaussianResult.covariance` raises a *scale* error (not pending-C) when covariance is absent.

- [ ] **Step 1: Write failing tests** in `tests/inference/test_sparse_result.py`. A model past the dense guard: `latent_marginals` and `linear_combinations` and `predictive_variance` now work; `covariance` raises a scale-limit error; and the numbers match a dense fit of the same (small, forced) model.

```python
def test_sparse_result_marginals_match_dense_but_covariance_guarded(small_model_forced_sparse):
    """Past the guard: scoped accessors work via the posterior; covariance raises."""
    import numpy as np
    import pytest
    from pylgm.inference.gaussian import _fit_dense, _fit_sparse

    model = small_model_forced_sparse
    dense = _fit_dense(model)
    sparse = _fit_sparse(model)

    # marginal variances match
    got = sparse.latent_marginals().variance
    assert np.allclose(got, dense.latent_marginals().variance, atol=1e-7)
    # predictive variance materialised on the result
    assert np.allclose(sparse.predictive_variance, dense.predictive_variance, atol=1e-7)
    # covariance is guarded (scale error, NOT pending-C)
    with pytest.raises(Exception) as exc:
        _ = sparse.covariance
    assert "E-sparse-C" not in str(exc.value)
```

> `small_model_forced_sparse`: build a small Besag Gaussian `CompiledLGM` (as in `test_sparse.py`) and call `_fit_sparse(model)` directly — `_fit_sparse` does not consult the guard, so any size works for the test. Add this fixture (or inline construction) to the test file.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/inference/test_sparse_result.py::test_sparse_result_marginals_match_dense_but_covariance_guarded -v`
Expected: FAIL (`_fit_sparse` still returns `covariance=None, predictive_variance=None`, and `latent_marginals` raises pending-C).

- [ ] **Step 3: Populate the posterior + predictive variance in `_fit_sparse`** (`gaussian.py`). Replace the current `_fit_sparse` body's return with:

```python
def _fit_sparse(model: CompiledLGM) -> GaussianResult:
    from pylgm.inference.sparse import sparse_constrained_gaussian

    variance = float(model.likelihood.variance)
    fit = sparse_constrained_gaussian(model)
    predictive_variance = fit.posterior.predictive_variances(model.design)
    _require_finite("posterior mean", fit.mean)
    _require_finite("log marginal likelihood", fit.log_marginal_likelihood)
    _require_finite("predictive mean", fit.predictive_mean)
    _require_finite("predictive variance", predictive_variance)
    return GaussianResult(
        labels=model.labels,
        mean=fit.mean,
        covariance=None,                       # never materialised above the guard
        log_marginal_likelihood=fit.log_marginal_likelihood,
        predictive_mean=fit.predictive_mean,
        predictive_variance=predictive_variance,
        observation_variance=variance,
        block_slices=fit.block_slices,
        diagnostics=fit.diagnostics,
        sparse_posterior=fit.posterior,
    )
```

- [ ] **Step 4: Add the variance-vector helpers** in `result.py` (immediately after `linear_combinations_from`, line ~230):

```python
def latent_marginals_from_variances(
    mean: np.ndarray,
    variances: np.ndarray,
    block_slices: Mapping[str, slice],
    block: str | None = None,
) -> "GaussianMarginals":
    """Same selection semantics as latent_marginals_from, but from a precomputed
    variance vector (sparse path — no full covariance)."""
    if block is None:
        selection = slice(None)
    else:
        try:
            selection = block_slices[block]
        except (KeyError, TypeError) as error:
            raise KeyError(f"unknown latent block {block!r}") from error
    return GaussianMarginals(mean[selection], variances[selection])


def linear_combinations_from_variances(
    mean: np.ndarray,
    variances: np.ndarray,
    weights: csr_matrix | np.ndarray,
) -> "GaussianMarginals":
    """Mean projection with a precomputed variance vector (sparse path)."""
    result_mean = np.asarray(weights @ mean).reshape(-1)
    return GaussianMarginals(result_mean, np.asarray(variances, float))
```

- [ ] **Step 5: Add `_sparse_posterior` to `GaussianResult` and route the accessors.** In `result.py`:

(a) `GaussianResult.__init__` (the `_init_common`-based constructor) gains a keyword `sparse_posterior=None` stored as `self._sparse_posterior = sparse_posterior`. Follow the existing pattern used for `observation_variance` (a GaussianResult-specific field alongside the `_BaseResult` common fields). Add `_sparse_posterior: "SparsePosterior | None"` to the field annotations and import `SparsePosterior` lazily inside a `TYPE_CHECKING` block to avoid a circular import (result.py must not import sparse.py at runtime).

(b) In `_BaseResult` (or `GaussianResult` if the accessors are overridden there), change the covariance property and the scoped accessors so an absent covariance with a present posterior routes to the posterior instead of raising pending-C:

```python
    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            if getattr(self, "_sparse_posterior", None) is not None:
                raise DenseReferenceLimitError(
                    "posterior covariance is not materialised at this scale; use "
                    "latent_marginals()/linear_combinations()/predict() for the "
                    "marginal and projected variances, or sparse_posterior."
                    "covariance_dense() to force the full matrix"
                )
            raise NotImplementedError(...)  # keep whatever non-sparse absent-cov behaviour existed
        return self._covariance
```

For `latent_marginals`:

```python
    def latent_marginals(self, block: str | None = None) -> "GaussianMarginals":
        if self._covariance is None and getattr(self, "_sparse_posterior", None) is not None:
            variances = self._sparse_posterior.marginal_variances()
            return latent_marginals_from_variances(self.mean, variances, self.block_slices, block)
        return latent_marginals_from(self.mean, self.covariance, self.block_slices, block)
```

For `linear_combinations`:

```python
    def linear_combinations(self, weights) -> "GaussianMarginals":
        if self._covariance is None and getattr(self, "_sparse_posterior", None) is not None:
            variances = self._sparse_posterior.linear_combination_variances(weights)
            return linear_combinations_from_variances(self.mean, variances, weights)
        return linear_combinations_from(self.mean, self.covariance, weights)
```

> Match the exact current method signatures/names in `result.py` (they may differ slightly, e.g. `linear_combinations(self, weights)` vs `(self, matrix)`); preserve them. Import `DenseReferenceLimitError` from `pylgm.exceptions` (already used across the inference layer).

(c) `_rebuild_result` in result.py is a sibling docstring only; the reconstruction lives in `model.py` (Task 7). Do not change model.py here.

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/inference/test_sparse_result.py tests/inference/test_result.py -v -W error::UserWarning`
Expected: PASS (new sparse-result test green; existing dense result tests unaffected — dense results still have `_covariance` and never hit the sparse branch).

- [ ] **Step 7: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/gaussian.py src/pylgm/inference/result.py tests/inference/test_sparse_result.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): route GaussianResult uncertainty accessors through the posterior"
```

---

## Task 6: sparse `predict`

**Files:**
- Modify: `src/pylgm/inference/prediction.py` (add `predict_from_sparse`)
- Modify: `src/pylgm/inference/result.py` (`GaussianResult.predict` routes to it when covariance is absent)
- Test: `tests/inference/test_sparse_result.py`

**Interfaces:**
- Consumes: existing `_design_for`, `_offset_for`, `Prediction`, `predict_from` (prediction.py); `SparsePosterior.predictive_variances`.
- Produces: `predict_from_sparse(context, mean, sparse_posterior, new_data) -> Prediction`.

- [ ] **Step 1: Write the failing test** (predict variances match dense on the forced-sparse small model):

```python
def test_sparse_predict_matches_dense(small_model_forced_sparse, prediction_new_data):
    import numpy as np
    from pylgm.inference.gaussian import _fit_dense, _fit_sparse

    dense = _fit_dense(small_model_forced_sparse)
    sparse = _fit_sparse(small_model_forced_sparse)
    dp = dense.predict(prediction_new_data)
    sp = sparse.predict(prediction_new_data)
    assert np.allclose(sp.predictive_variance, dp.predictive_variance, atol=1e-7)
    assert np.allclose(sp.response_prediction, dp.response_prediction, atol=1e-7)
```

> `prediction_new_data`: reuse whatever new-data frame the existing predict tests build for this model shape; the result must carry a `prediction_context` (set via `_rebuild_result(..., prediction_context=...)`) — construct the fit through the public `LGM.fit`/`.predict` path if that is how existing predict tests set the context, or set the context the same way they do.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/inference/test_sparse_result.py::test_sparse_predict_matches_dense -v`
Expected: FAIL (predict raises pending-C on the sparse result).

- [ ] **Step 3: Add `predict_from_sparse`** in `prediction.py`, mirroring `predict_from` (lines 231-258) but sourcing the variance from the posterior:

```python
def predict_from_sparse(context, mean, sparse_posterior, new_data):
    """Like predict_from, but predictive variance comes from the sparse posterior
    (diag(design Σ_c designᵀ)) instead of a dense covariance."""
    design = _design_for(context, new_data)
    offset = _offset_for(context, new_data)
    eta = np.asarray(design @ mean).reshape(-1) + offset
    predictive_variance = sparse_posterior.predictive_variances(design)
    response_prediction = context.link.inverse(eta) if context.link is not None else eta
    return Prediction(...)  # populate exactly as predict_from does (eta, variance, response)
```

> Copy `predict_from`'s exact `Prediction(...)` construction and response-scale handling (link inverse, offset, any response_prediction fields). The ONLY substantive change is `predictive_variance` sourced from `sparse_posterior.predictive_variances(design)` instead of the einsum over a dense covariance. Reuse `_design_for`/`_offset_for` verbatim.

- [ ] **Step 4: Route `GaussianResult.predict`** in `result.py`:

```python
    def predict(self, new_data):
        if self._covariance is None and getattr(self, "_sparse_posterior", None) is not None:
            return predict_from_sparse(
                self.prediction_context, self.mean, self._sparse_posterior, new_data
            )
        return predict_from(self.prediction_context, self.mean, self.covariance, new_data)
```

Import `predict_from_sparse` alongside the existing `predict_from` import in result.py.

- [ ] **Step 5: Run the test**

Run: `python -m pytest tests/inference/test_sparse_result.py::test_sparse_predict_matches_dense -v`
Expected: PASS.

- [ ] **Step 6: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/prediction.py src/pylgm/inference/result.py tests/inference/test_sparse_result.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): predict from the sparse posterior"
```

---

## Task 7: `_rebuild_result` carries the sparse posterior

**Files:**
- Modify: `src/pylgm/model.py` (`_rebuild_result`, lines 145-207)
- Test: `tests/inference/test_sparse_result.py`

**Interfaces:**
- Consumes: `GaussianResult._sparse_posterior` (Task 5).
- Produces: reordering/aligning a sparse `GaussianResult` preserves `_sparse_posterior` (latent space is *not* reordered by `caller_order`, which only permutes prediction rows — so the carry is a straight pass-through).

- [ ] **Step 1: Write the failing test**:

```python
def test_rebuild_preserves_sparse_posterior(small_model_forced_sparse):
    import numpy as np
    from pylgm.inference.gaussian import _fit_sparse
    from pylgm.model import _rebuild_result

    sparse = _fit_sparse(small_model_forced_sparse)
    order = np.arange(sparse.predictive_mean.size)[::-1]
    rebuilt = _rebuild_result(sparse, caller_order=order)
    assert rebuilt._sparse_posterior is sparse._sparse_posterior
    # latent marginals still computable after rebuild
    assert np.all(np.isfinite(rebuilt.latent_marginals().variance))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/inference/test_sparse_result.py::test_rebuild_preserves_sparse_posterior -v`
Expected: FAIL (`GaussianResult(...)` in `_rebuild_result` does not pass `sparse_posterior`, so `rebuilt._sparse_posterior is None`).

- [ ] **Step 3: Carry the field in `_rebuild_result`.** The `GaussianResult` return (line 207) is:

```python
    return GaussianResult(observation_variance=result.observation_variance, **common)
```

Change it to:

```python
    return GaussianResult(
        observation_variance=result.observation_variance,
        sparse_posterior=getattr(result, "_sparse_posterior", None),
        **common,
    )
```

> `getattr(..., None)` is defensive: `result` may be a `LaplaceResult` here too, which has no `_sparse_posterior`. `caller_order` permutes only `predictive_mean`/`predictive_variance` (prediction rows); `mean`, `block_slices`, and the posterior's latent index space are untouched, so no reindexing of the posterior is needed. Add a one-line `ponytail:` comment saying exactly that.

- [ ] **Step 4: Run the test + inference suite**

Run: `python -m pytest tests/inference/test_sparse_result.py -v -W error::UserWarning` and `python -m pytest tests/inference -q`
Expected: PASS.

- [ ] **Step 5: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/model.py tests/inference/test_sparse_result.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): carry sparse posterior through _rebuild_result"
```

---

## Task 8: sparse Sørbye-Rue scaling (null_dim == 1 fast path)

**Files:**
- Modify: `src/pylgm/effects/scaling.py` (`sorbye_rue_scale` gains a sparse null_dim==1 branch; add `_sorbye_sparse_diag` helper)
- Test: `tests/effects/test_scaling.py`

**Interfaces:**
- Consumes: `selected_inverse_diagonal` (Task 1), `SparseSpdFactor` (both from `pylgm.inference.sparse`), `scipy.sparse`.
- Produces:
  - `_sorbye_sparse_diag(structure) -> np.ndarray` — `diag(structure⁺)` for a connected intrinsic sparse `structure` with null = span(1), via double-centering of the grounded reduced inverse.
  - `sorbye_rue_scale(structure, null_dim)` returns a **sparse** scaled structure (`s * structure`) when `structure` is sparse *and* `null_dim == 1`; otherwise the existing dense eigh path, byte-identical, returning dense.

Math (validated): delete node 0 → SPD `Q_r`; `K = Q_r⁻¹`; `diag(K)` via Takahashi; `s = K·1` (one solve); `T = 1ᵀs`; then `diag(Q⁺)[0] = T/n²`, `diag(Q⁺)[1:] = diag(K) − (2/n)s + T/n²`. Scale multiplier `= exp(mean(log diag(Q⁺)))` (unchanged definition), so `R* = scale · structure`.

- [ ] **Step 1: Write the failing test** in `tests/effects/test_scaling.py`:

```python
def test_sorbye_sparse_matches_dense_eigh():
    """Sparse null_dim==1 Sørbye scale must equal the dense eigh reference."""
    import numpy as np
    from scipy.sparse import lil_matrix
    from pylgm.effects.scaling import sorbye_rue_scale

    def ring(n):
        q = lil_matrix((n, n))
        for i in range(n):
            q[i, i] = 2.0
            q[i, (i + 1) % n] = -1.0
            q[i, (i - 1) % n] = -1.0
        return q

    for n in (50, 200):
        sparse_struct = ring(n).tocsc()
        dense_struct = sparse_struct.toarray()
        got = sorbye_rue_scale(sparse_struct, null_dim=1)          # sparse in -> sparse out
        want = sorbye_rue_scale(dense_struct, null_dim=1)          # dense eigh reference
        assert np.allclose(got.toarray(), want, atol=1e-9)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_scaling.py::test_sorbye_sparse_matches_dense_eigh -v`
Expected: FAIL (current `sorbye_rue_scale` calls `np.linalg.eigh` on the sparse input → error or wrong type).

- [ ] **Step 3: Add the sparse helper + branch** in `scaling.py`:

```python
def _sorbye_sparse_diag(structure) -> np.ndarray:
    """diag(structure⁺) for a connected intrinsic structure (null = span(1)),
    via double-centering of the grounded reduced inverse. All sparse, near-linear."""
    from pylgm.inference.sparse import SparseSpdFactor, selected_inverse_diagonal

    q = structure.tocsc()
    n = q.shape[0]
    q_r = q[1:, 1:].tocsc()                       # delete node 0 -> SPD
    diag_k = selected_inverse_diagonal(q_r)        # diag(K), size n-1
    s = SparseSpdFactor(q_r.tocsr(), "sorbye reduced").solve(np.ones(n - 1))
    t = float(s.sum())
    diag = np.empty(n)
    diag[0] = t / n**2
    diag[1:] = diag_k - (2.0 / n) * s + t / n**2
    return diag
```

Then, at the top of `sorbye_rue_scale`, add the fast-path branch (before the dense eigh body):

```python
def sorbye_rue_scale(structure, null_dim):
    from scipy.sparse import issparse
    if null_dim == 1 and issparse(structure):
        variances = _sorbye_sparse_diag(structure)
        scale = float(np.exp(np.mean(np.log(variances))))
        return scale * structure                   # sparse in -> sparse out
    # ... existing dense eigh path unchanged (dense in -> dense ndarray out) ...
```

> Keep the existing dense body verbatim below the branch. The dense path (used by `besag._scaled_structure`, which passes dense per-component blocks) is unchanged, so Besag behavior is byte-identical. `scipy.sparse.issparse` guards against a dense ndarray taking the sparse branch.

- [ ] **Step 4: Run the test + effects suite**

Run: `python -m pytest tests/effects/test_scaling.py -v` and `python -m pytest tests/effects -q -W error::UserWarning`
Expected: PASS (existing Besag / scaling tests green — dense path untouched).

- [ ] **Step 5: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/effects/scaling.py tests/effects/test_scaling.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(effects): sparse Sørbye-Rue scale for connected intrinsic fields"
```

---

## Task 9: augmented BYM2 (sparse `(x, u*)` joint) + generalized-cofactor prior logdet

**Files:**
- Modify: `src/pylgm/effects/bym2.py` (add `_build_bym2_augmented`; branch `build_bym2` on graph size)
- Modify: `src/pylgm/inference/sparse.py` (`_prior_logdet` gains a BYM2-augmented generalized-cofactor branch; add `_bym2_augmented_logdet` helper)
- Modify: `src/pylgm/compiler.py` (the parametric BYM2 build closure assembles the augmented joint sparsely when the graph is large)
- Test: `tests/effects/test_bym2.py`, `tests/inference/test_sparse.py`

**DESIGN DECISION (flag at pre-flight review):** `build_bym2` keeps the existing dense spectral `n`-block for small graphs (existing behavior byte-identical) and switches to the augmented `2n`-block only when `len(nodes) > _BYM2_AUGMENT_NODES` (a module constant, set to `1024`). The augmented block reports `2n` labels — the region labels for the `x`-half and `f"{node}__u"` for the `u*`-half; users read the spatial effect via the region (x) labels. This changes the *public latent width for large BYM2 only*, which the spec (§6.7) accepts as the structurally invasive part. The augmented path requires a **single connected component**; a large multi-component BYM2 raises `NotImplementedError` (per-component augmentation is out of this slice — the dense path still covers small multi-component graphs).

**Interfaces:**
- Consumes: `normalize_graph`, `design_from_graph` (graph.py); `sorbye_rue_scale` (Task 8, sparse null_dim==1); `LatentBlock`; `scipy.sparse` (`identity`, `bmat`, `hstack`, `diags`, `csr_matrix`, `csgraph.connected_components`); `SparseSpdFactor` (sparse.py); the validated augmented forms.
- Produces:
  - `_build_bym2_augmented(frame, name, index, graph, precision, phi) -> LatentBlock` (2n block).
  - `_prior_logdet` handles the augmented block via `_bym2_augmented_logdet(rows, q_b) -> float | None` (returns `None` when the block is not the augmented pattern → dense fallback).

Validated joint (authoritative; supersedes the spec §6.7 √-transcription): `a=1/(1−φ)`, `b=−√φ/(1−φ)`, `d=φ/(1−φ)`; `Q_joint = τ·bmat([[a·I, b·I],[b·I, R*+d·I]])` over `(x, u*)`, sum-to-zero on `u*` only. Reproduces the dense BYM2 x-marginal exactly.

- [ ] **Step 1: Write the failing tests.**

(a) In `tests/effects/test_bym2.py` — augmented x-marginal equals the dense BYM2 x-marginal (force the augmented builder directly on a small grid so both paths run):

```python
def test_bym2_augmented_x_marginal_matches_dense():
    import numpy as np
    from scipy.linalg import null_space
    from pylgm.effects.bym2 import _build_bym2_augmented, bym2_spectrum, bym2_precision
    # small connected 4x4 grid graph
    m = 4
    graph = {}
    idx = lambda r, c: f"{r*m+c}"
    for r in range(m):
        for c in range(m):
            nb = []
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < m and 0 <= cc < m:
                    nb.append(idx(rr, cc))
            graph[idx(r, c)] = nb
    n = m * m
    import pandas as pd
    frame = pd.DataFrame({"region": [idx(r, c) for r in range(m) for c in range(m)]})
    tau, phi = 1.3, 0.6

    aug = _build_bym2_augmented(frame, "s", "region", graph, tau, phi)
    # dense reference x-marginal covariance
    vectors, values = bym2_spectrum(graph)
    q_dense = bym2_precision(vectors, values, tau, phi).toarray()
    dense_x_cov = np.linalg.inv(q_dense)  # BYM2 x-block prior covariance (full-rank build)

    # augmented: reduce joint under sum-to-zero on u*, invert, take x-block
    qj = aug.precision.toarray()
    c = np.zeros((1, 2 * n)); c[0, n:] = 1.0
    basis = null_space(c)
    sig = basis @ np.linalg.inv(basis.T @ qj @ basis) @ basis.T
    aug_x_cov = sig[:n, :n]
    assert np.allclose(np.diag(aug_x_cov), np.diag(dense_x_cov), atol=1e-9)
```

(b) In `tests/inference/test_sparse.py` — the generalized cofactor logdet equals the dense `null_space` constrained logdet:

```python
def test_bym2_augmented_prior_logdet_matches_dense():
    import numpy as np
    from scipy.linalg import null_space
    from scipy.sparse import identity, bmat, csr_matrix, lil_matrix
    from pylgm.inference.sparse import _bym2_augmented_logdet

    m = 5; n = m * m
    # scaled ICAR R* (dense build is fine at this size for the test)
    r = lil_matrix((n, n)); idx = lambda a, b: a * m + b
    for a in range(m):
        for b in range(m):
            i = idx(a, b); deg = 0
            for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                aa, bb = a+da, b+db
                if 0 <= aa < m and 0 <= bb < m:
                    r[i, idx(aa, bb)] = -1.0; deg += 1
            r[i, i] = deg
    r = r.tocsr()
    w, v = np.linalg.eigh(r.toarray()); inv = np.zeros_like(w); inv[1:] = 1/w[1:]
    var = np.einsum("ij,j,ij->i", v, inv, v); s = np.exp(np.mean(np.log(var)))
    rstar = csr_matrix(r.toarray() * s)
    tau, phi = 1.3, 0.6
    a_ = 1/(1-phi); b_ = -np.sqrt(phi)/(1-phi); d_ = phi/(1-phi)
    ident = identity(n, format="csr")
    qj = (tau * bmat([[a_*ident, b_*ident], [b_*ident, rstar + d_*ident]], format="csr"))
    rows = np.zeros((1, 2*n)); rows[0, n:] = 1.0

    got = _bym2_augmented_logdet(rows, qj)
    basis = null_space(rows)
    _, want = np.linalg.slogdet(basis.T @ qj.toarray() @ basis)
    assert got is not None and abs(got - want) < 1e-7
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/effects/test_bym2.py::test_bym2_augmented_x_marginal_matches_dense tests/inference/test_sparse.py::test_bym2_augmented_prior_logdet_matches_dense -v`
Expected: FAIL (`_build_bym2_augmented` / `_bym2_augmented_logdet` do not exist).

- [ ] **Step 3: Add `_build_bym2_augmented` + the size branch** in `bym2.py`. Update imports at the top:

```python
from scipy.sparse import bmat, csr_matrix, diags, hstack, identity
from scipy.sparse.csgraph import connected_components

from pylgm.effects.scaling import sorbye_rue_scale
```

Add the constant and builder:

```python
_BYM2_AUGMENT_NODES = 1024


def _build_bym2_augmented(
    frame: pd.DataFrame,
    name: str,
    index: str,
    graph: Mapping,
    precision: float,
    phi: float,
) -> LatentBlock:
    """Augmented (x, u*) BYM2 block: 2n latent, sparse joint precision, sum-to-zero
    on u* only. Reproduces the dense BYM2 x-marginal exactly and scales (no
    eigendecomposition). Selected for large connected graphs.
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must lie in [0, 1); got {phi}")
    nodes, w = normalize_graph(graph)
    n = len(nodes)
    n_components, _ = connected_components(w, directed=False)
    if n_components != 1:
        # ponytail: augmented path assumes null = (√φ·1, 1) over one component.
        # Per-component augmentation is out of this slice; small multi-component
        # BYM2 still fits via the dense spectral path.
        raise NotImplementedError(
            "augmented BYM2 requires a single connected graph component"
        )
    degree = np.asarray(w.sum(axis=1)).ravel()
    rstar = sorbye_rue_scale((diags(degree) - w).tocsc(), null_dim=1)   # sparse R*
    a = 1.0 / (1.0 - phi)
    b = -np.sqrt(phi) / (1.0 - phi)
    d = phi / (1.0 - phi)
    ident = identity(n, format="csr")
    joint = (precision * bmat([[a * ident, b * ident],
                               [b * ident, rstar + d * ident]], format="csr")).tocsr()
    x_design = design_from_graph(nodes, frame, index)                  # n_obs x n
    design = hstack([x_design, csr_matrix((x_design.shape[0], n))], format="csr")
    labels = tuple(nodes) + tuple(f"{node}__u" for node in nodes)
    constraints = np.zeros((1, 2 * n))
    constraints[0, n:] = 1.0
    return LatentBlock(name, labels, design, joint, constraints)
```

Branch `build_bym2`:

```python
def build_bym2(frame, name, index, graph, precision, phi):
    nodes, _ = normalize_graph(graph)
    if len(nodes) > _BYM2_AUGMENT_NODES:
        return _build_bym2_augmented(frame, name, index, graph, precision, phi)
    # existing dense spectral path (unchanged)
    design = design_from_graph(nodes, frame, index)
    vectors, values = bym2_spectrum(graph)
    precision_matrix = bym2_precision(vectors, values, precision, phi)
    constraints = np.empty((0, len(nodes)))
    return LatentBlock(name, nodes, design, precision_matrix, constraints)
```

- [ ] **Step 4: Add `_bym2_augmented_logdet` + wire it into `_prior_logdet`** (sparse.py). Add the helper above `_prior_logdet`:

```python
def _bym2_augmented_logdet(rows: np.ndarray, precision) -> float | None:
    """logdet(Vᵀ Q V) for a BYM2 augmented block (null g=(√φ·1, 1), pinned by
    sum-to-zero on the u* half), via the generalized matrix-tree cofactor.
    Returns None when the block is not the augmented pattern (caller falls back to
    the dense reduction).
    """
    width = precision.shape[0]
    if width % 2 or rows.shape[0] != 1:
        return None
    n = width // 2
    r = np.asarray(rows, dtype=float).ravel()
    if np.any(np.abs(r[:n]) > 0) or r[n] == 0 or not np.allclose(r[n:], r[n]):
        return None
    q = precision.tocsr()
    q00 = q[0, 0]
    q0n = q[0, n]
    if q00 == 0.0 or q0n >= 0.0:          # b = -√φ/(1-φ) < 0 for φ in (0,1)
        return None
    sqrt_phi = -q0n / q00
    g = np.concatenate([sqrt_phi * np.ones(n), np.ones(n)])
    scale = abs(q).max()
    if not np.allclose(np.asarray(q @ g).ravel(), 0.0, atol=1e-8 * max(1.0, scale)):
        return None                        # g is not the null vector -> not augmented
    i = width - 1                          # delete a u* index (g_i = 1)
    keep = np.arange(width) != i
    q_sub = q[keep][:, keep].tocsr()
    logdet_star = (
        np.log(g @ g) - 2.0 * np.log(abs(g[i]))
        + SparseSpdFactor(q_sub, "bym2 augmented prior").logdet
    )
    c = np.concatenate([np.zeros(n), np.ones(n)])   # canonical sum-to-zero on u*
    return logdet_star + np.log((c @ g) ** 2) - np.log(g @ g) - np.log(c @ c)
```

In `_prior_logdet`, insert the augmented branch between the `_is_connected_intrinsic` fast-path and the dense fallback:

```python
        if confined.any():
            rows = a[confined, start:stop]
            if _is_connected_intrinsic(rows, q_b):
                n_b = stop - start
                reduced = q_b.tocsr()[1:, 1:]
                logdet = np.log(n_b) + SparseSpdFactor(
                    reduced, f"intrinsic prior [{block.name}]"
                ).logdet
            else:
                bym2_logdet = _bym2_augmented_logdet(rows, q_b)
                if bym2_logdet is not None:
                    logdet = bym2_logdet
                else:
                    # ponytail: dense reduction, O(n^3), residual cases only
                    # (RW2, weighted / multi-row extra-constraints), small in practice.
                    basis_b = null_space(rows)
                    reduced = basis_b.T @ q_b.toarray() @ basis_b
                    _, logdet = _factor_positive_definite(reduced, f"reduced prior [{block.name}]")
        else:
            logdet = SparseSpdFactor(q_b.tocsr(), f"prior [{block.name}]").logdet
        total += logdet
```

- [ ] **Step 5: Wire the parametric (phi-hyperparameter / optimized-tau) path** in `compiler.py`. The BYM2 branch at ~653-664 uses `_compiled_block(effect.name, build_bym2, ...)`; since `build_bym2` now self-branches on size, the plain and fixed-phi paths already build augmented blocks for large graphs — **no change needed there**. For the `phi_is_hp` ParametricBlock path (~665-onward), the current `build` closure reassembles `bym2_precision(vectors, spectrum, tau, phi)` (dense). Add a size check: when `len(nodes) > _BYM2_AUGMENT_NODES`, build the ParametricBlock from a sparse-augmented template and a `build` closure that reassembles the sparse joint from a precomputed sparse `R*` and `(tau, phi)`:

```python
            from pylgm.effects.bym2 import _BYM2_AUGMENT_NODES, _build_bym2_augmented
            nodes_bym2, w_bym2 = normalize_graph(dict(effect.graph))
            if len(nodes_bym2) > _BYM2_AUGMENT_NODES:
                template = _compiled_block(
                    effect.name, _build_bym2_augmented,
                    frame, effect.name, effect.index, dict(effect.graph),
                    value, float(effect.phi.initial),
                )
                degree = np.asarray(w_bym2.sum(axis=1)).ravel()
                rstar = sorbye_rue_scale((diags(degree) - w_bym2).tocsc(), null_dim=1)
                ident = identity(len(nodes_bym2), format="csr")

                def build(values_map, rstar=rstar, ident=ident, n=len(nodes_bym2),
                          tau_name=tau_name, tau_fixed=tau_fixed, phi_name=phi_name):
                    tau = values_map[tau_name] if tau_name else tau_fixed
                    phi = values_map[phi_name]
                    a_ = 1.0/(1.0-phi); b_ = -np.sqrt(phi)/(1.0-phi); d_ = phi/(1.0-phi)
                    return (tau * bmat([[a_*ident, b_*ident],
                                        [b_*ident, rstar + d_*ident]], format="csr")).tocsr()

                params = tuple(name for name in (tau_name, phi_name) if name)
                scalable.append(ParametricBlock(template, params, build))
                # parameter registration identical to the dense branch below
                if optimized:
                    parameter_names.append(precision.name)
                    parameter_bounds[precision.name] = _log_bounds(precision)
                parameter_names.append(phi_name)
                parameter_bounds[phi_name] = phi_bounds
                continue
```

> Insert this block *before* the existing dense ParametricBlock assembly (the `vectors, values_ = bym2_spectrum(...)` line and its `build`), guarded so the dense path runs only for small graphs. Match the exact `tau_name`/`tau_fixed`/`phi_name`/`phi_bounds`/`ParametricBlock`/registration names already in scope there (read the surrounding lines 665-700 and mirror them). Add the imports `from scipy.sparse import bmat, diags, identity` and `from pylgm.effects.graph import normalize_graph` and `from pylgm.effects.scaling import sorbye_rue_scale` to compiler.py if not already present.

- [ ] **Step 6: End-to-end augmented BYM2 fit test** in `tests/inference/test_sparse.py` — a small graph with `_BYM2_AUGMENT_NODES` monkeypatched low so the augmented path runs, fitted through the public API, and its x-block marginal variances compared to a dense BYM2 fit:

```python
def test_bym2_augmented_end_to_end_matches_dense(monkeypatch):
    """Public BYM2 fit via the augmented path: x-block marginals match the dense build."""
    import numpy as np
    import pandas as pd
    import pylgm.effects.bym2 as bym2_mod
    # ... build a small connected-grid BYM2 LGM (as other effect tests do),
    # fit once with _BYM2_AUGMENT_NODES patched below n (augmented path) and once
    # above n (dense path); compare the region (x) marginal variances.
    # monkeypatch.setattr(bym2_mod, "_BYM2_AUGMENT_NODES", 1)  # force augmented
    # assert np.allclose(aug_x_var, dense_x_var, atol=1e-6)
```

> Fill this in using the file's existing BYM2 model-construction helper. The comparison is x-half marginal variances (the augmented result exposes `2n` labels; slice the first `n` / the region labels). If the file has no BYM2 fit helper, construct the model the same way `tests/effects/test_bym2.py` does and fit via the public `LGM.fit`.

- [ ] **Step 7: Run BYM2 + sparse + effects suites**

Run: `python -m pytest tests/effects/test_bym2.py tests/inference/test_sparse.py -v -W error::UserWarning` and `python -m pytest tests/effects tests/inference -q`
Expected: PASS. Existing small-graph BYM2 tests are unchanged (dense path). If any existing BYM2 test breaks, the size branch is wrong — STOP and report.

- [ ] **Step 8: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/effects/bym2.py src/pylgm/inference/sparse.py src/pylgm/compiler.py tests/effects/test_bym2.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(bym2): augmented sparse (x,u*) joint + generalized-cofactor prior logdet"
```

---

## Task 10: INLA integration — diagonal accumulation (Gaussian + simplified-Laplace)

**Files:**
- Modify: `src/pylgm/optimization/inla.py` (integration loop; `_model_criteria`; `_simplified_laplace_marginals`; guard `_full_laplace_marginals`)
- Modify: `src/pylgm/inference/result.py` (`INLAResult._latent_variances`; `latent_marginals`/`covariance` route to it)
- Test: `tests/optimization/test_inla.py`

**Interfaces:**
- Consumes: `GaussianResult._sparse_posterior` (Task 5) on the conditional fits; `SparsePosterior.marginal_variances` / `predictive_variances`.
- Produces:
  - `_conditional_latent_variances(fit) -> np.ndarray` — `diag(Σ)` from a conditional fit (dense: `np.diag(fit.covariance)`; sparse: `fit._sparse_posterior.marginal_variances()`).
  - `_conditional_predictive_variances(fit, design) -> np.ndarray` — `diag(design Σ designᵀ)` (dense einsum, or `fit._sparse_posterior.predictive_variances(design)`).
  - `INLAResult(..., latent_variances=None)` stored as `_latent_variances`; `INLAResult.latent_marginals()` uses it (Gaussian strategy); `INLAResult.covariance` guarded when absent.

- [ ] **Step 1: Write the failing test** — a large-enough Gaussian INLA model that routes conditionals to the sparse path still returns finite latent + predictive variances, and (on a small model forced through the same helper) the diagonal integration matches the dense integration:

```python
def test_inla_sparse_conditional_diagonal_integration(small_inla_gaussian_model):
    """INLA integrates diagonal latent + predictive variances when conditionals are sparse."""
    import numpy as np
    from pylgm.optimization.inla import _conditional_latent_variances
    from pylgm.inference.gaussian import _fit_dense, _fit_sparse

    model = small_inla_gaussian_model
    dense = _fit_dense(model)
    sparse = _fit_sparse(model)
    assert np.allclose(
        _conditional_latent_variances(sparse),
        np.diag(dense.covariance),
        atol=1e-7,
    )
```

> Also add an end-to-end assertion: fit a small Gaussian LGM with `integrate_inla` while forcing the sparse conditional path (patch the guard low, as in Task 9), and assert `result.latent_marginals().variance` and `result.predictive_variance` are finite and match a dense INLA fit to ~1e-6.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/optimization/test_inla.py::test_inla_sparse_conditional_diagonal_integration -v`
Expected: FAIL (`_conditional_latent_variances` does not exist; and the integration loop currently reads `cond.covariance`, which raises on a sparse conditional).

- [ ] **Step 3: Add the two helpers** near the top of `inla.py`:

```python
def _conditional_latent_variances(fit) -> np.ndarray:
    """diag(Σ) of a conditional fit, dense or sparse."""
    posterior = getattr(fit, "_sparse_posterior", None)
    if posterior is not None and fit._covariance is None:
        return posterior.marginal_variances()
    return np.diag(np.asarray(fit.covariance, float))


def _conditional_predictive_variances(fit, dense_design) -> np.ndarray:
    """diag(design Σ designᵀ) of a conditional fit, dense or sparse."""
    posterior = getattr(fit, "_sparse_posterior", None)
    if posterior is not None and fit._covariance is None:
        return posterior.predictive_variances(dense_design)
    cov = np.asarray(fit.covariance, float)
    return np.einsum("ij,jk,ik->i", dense_design, cov, dense_design)
```

- [ ] **Step 4: Replace the dense covariance accumulation in the integration loop** (lines 313-342). Swap the full-matrix `cov_acc` for a diagonal `var_acc`:

```python
    mean_acc = np.zeros_like(reference.mean)
    var_acc = np.zeros_like(reference.mean)          # diag accumulation, not cov_acc
    pm_acc = np.zeros_like(reference.predictive_mean)
    pv_acc = np.zeros_like(reference.predictive_variance)
    ...
    for (_, _, cond, theta, _), w in zip(kept, weights, strict=True):
        m = cond.mean
        pm = cond.predictive_mean
        mean_acc += w * m
        var_acc += w * (_conditional_latent_variances(cond) + m * m)
        pm_acc += w * pm
        pv_acc += w * (cond.predictive_variance + pm * pm)
        ...
    mean = mean_acc
    latent_variance = var_acc - mean * mean           # diagonal posterior variance
    predictive_mean = pm_acc
    predictive_variance = pv_acc - pm_acc * pm_acc
```

Then, in the finite-check loop (368-372) drop `covariance` from the checked pairs (there is no dense covariance above the guard) and check `latent_variance` instead:

```python
    for label, value in (("mean", mean), ("latent variance", latent_variance),
                         ("predictive mean", predictive_mean),
                         ("predictive variance", predictive_variance)):
        if not np.isfinite(value).all():
            raise NumericalError(f"INLA produced non-finite {label}")
```

In the `INLAResult(...)` return, pass `covariance=None` and `latent_variances=latent_variance` (see Step 6 for the field). Keep `predictive_mean`/`predictive_variance` as computed.

> `cond.predictive_variance` is already the diagonal on every conditional fit (Task 5 materialises it for sparse conditionals; dense fits have always had it), so `pv_acc` needs no helper.

- [ ] **Step 5: Fix `_model_criteria` (line 454) and `_simplified_laplace_marginals` (82-87)** to use the predictive-variance helper instead of the dense einsum:

In `_model_criteria`, replace line 454:

```python
        v = np.clip(_conditional_predictive_variances(fit, dense), 0.0, None)
```

In `_simplified_laplace_marginals`, replace the covariance reads (lines 82-87):

```python
        sigma = np.sqrt(np.clip(_conditional_latent_variances(fit), 0.0, None))
        ...
        eta_var = np.clip(_conditional_predictive_variances(fit, dense), 0.0, None)
```

> Read the exact surrounding code at 82-87 and 452-454 and substitute only the covariance-derived quantities (`np.diag(cov)` → `_conditional_latent_variances(fit)`; `einsum(... cov ...)` → `_conditional_predictive_variances(fit, dense)`). Leave the rest of both functions untouched.

Guard `_full_laplace_marginals` (117+): it reads `f.covariance[i, i]` / `fit.covariance` (lines 146, 152) and has no diagonal-only formulation in this slice. At the call site (inla.py ~406-416, the `latent_strategy == "laplace"` branch) the model is constrained-free; add a guard that raises a clear error when the conditional is sparse:

```python
    elif latent_strategy == "laplace":
        if getattr(kept[0][2], "_sparse_posterior", None) is not None:
            raise UnsupportedEngineError(
                "full Laplace latent marginals are not available above the sparse "
                "guard; use latent_strategy='gaussian' or 'simplified_laplace'"
            )
        ... # existing full-laplace body unchanged
```

- [ ] **Step 6: Add `_latent_variances` to `INLAResult` and route its accessors** (result.py):

(a) `INLAResult.__init__` gains `latent_variances=None` stored as `self._latent_variances`. Add to `_rebuild_result`'s INLA branch (model.py) the carry `latent_variances=result._latent_variances` (it is a full-latent diagonal, not row-indexed, so it is *not* permuted by `caller_order`).

(b) `INLAResult.latent_marginals` (994-1006): when `_latent_marginal_table` is None (the Gaussian strategy) and `_covariance is None`, build marginals from `_latent_variances`:

```python
    def latent_marginals(self, block=None):
        if self._latent_marginal_table is not None:
            ... # existing skew-normal / tabulated path unchanged
        if self._covariance is None and self._latent_variances is not None:
            return latent_marginals_from_variances(
                self.mean, self._latent_variances, self.block_slices, block
            )
        return latent_marginals_from(self.mean, self.covariance, self.block_slices, block)
```

(c) `INLAResult.covariance`: when `_covariance is None and _latent_variances is not None`, raise the same scale-limit `DenseReferenceLimitError` as `GaussianResult` (directing to `latent_marginals`), not pending-C.

- [ ] **Step 7: Run INLA + result suites**

Run: `python -m pytest tests/optimization/test_inla.py tests/inference/test_result.py -v -W error::UserWarning` and `python -m pytest tests/optimization -q`
Expected: PASS. The existing dense INLA tests must stay green — dense conditionals still have `_covariance`, so both helpers take the dense branch and the numbers are identical (the loop now tracks the diagonal of what `cov_acc` used to hold; `latent_variance == np.diag(old covariance)` by construction).

> **Regression note for the reviewer:** the old code exposed the full `INLAResult.covariance`. Above the guard it is now `None` (diagonal only). Any existing test asserting `INLAResult.covariance` off-diagonals on a *large* model would now be testing unsupported behavior; confirm no such test exists on a guard-crossing model (small-model dense INLA still returns the full covariance because those conditionals are dense). If a dense-model test reads `.covariance`, it still works — dense conditionals keep `_covariance`. **Decision point:** if a dense INLA model currently returns a full covariance and a test depends on it, that path is preserved; only the sparse (large) path drops to diagonal.

- [ ] **Step 8: ruff + commit**

Run: `python -m ruff check src tests`

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/optimization/inla.py src/pylgm/inference/result.py src/pylgm/model.py tests/optimization/test_inla.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(inla): diagonal latent/predictive integration for sparse conditionals"
```

---

## Task 11: large-model uncertainty smoke + full-suite regression + roadmap

**Files:**
- Modify: `tests/inference/test_sparse_large_model.py` (extend the existing 5000-region ring Besag test to exercise the uncertainty surface)
- Modify: `docs/roadmap.md`
- Test: the whole suite

**Interfaces:**
- Consumes: everything above, via the public `LGM.fit` API past the dense guard.

- [ ] **Step 1: Extend the large-model test.** In `tests/inference/test_sparse_large_model.py`, after the existing 5000-region ring Besag fit, add assertions that the uncertainty accessors now work past the guard (no pending-C), are finite, and are self-consistent (no dense oracle at this size — that is Task 3's job on a small model):

```python
def test_large_besag_uncertainty_surface(large_ring_besag_lgm):
    """Past the dense guard: latent/predictive variances + predict are finite."""
    import numpy as np
    result = large_ring_besag_lgm  # a fitted 5000-region ring Besag result
    marg = result.latent_marginals()
    assert marg.variance.shape[0] == result.mean.shape[0]
    assert np.all(np.isfinite(marg.variance)) and np.all(marg.variance >= 0)
    assert np.all(np.isfinite(result.predictive_variance))
    # covariance is guarded, not pending-C
    import pytest
    with pytest.raises(Exception) as exc:
        _ = result.covariance
    assert "E-sparse-C" not in str(exc.value)
```

> Reuse the file's existing fixture/fit for the 5000-region ring Besag. Keep the wall-time expectation informal (report it; no committed timing assert). Do not add a dense oracle at 5000 regions.

- [ ] **Step 2: Run the large-model test (report wall-time)**

Run: `python -m pytest tests/inference/test_sparse_large_model.py -v`
Expected: PASS in well under a minute per fit; paste the wall-time into the report.

- [ ] **Step 3: Full-suite regression**

Run: `python -m pytest -q`
Expected: the pre-existing environment failures only (per the A+B ledger: `test_package.py` Spark `JAVA_GATEWAY_EXITED` + console-entrypoint PATH, and any `graphify`/Spark-gated errors). Confirm this slice added no new failures. Paste the tally. If anything *other* than those known-environment items fails, STOP and report.

- [ ] **Step 4: ruff over the whole tree**

Run: `python -m ruff check src tests`
Expected: `All checks passed!`.

- [ ] **Step 5: Roadmap note.** In `docs/roadmap.md`, add a Shipped bullet for E-sparse-C (full posterior-uncertainty surface at network scale: marginal/predictive/linear-combination variances, constrained corrections, sparse Sørbye-Rue, augmented BYM2, diagonal INLA integration; no new dependency; deterministic) and remove/tick the corresponding Next item. Match the file's existing bullet phrasing/format.

- [ ] **Step 6: Commit**

```bash
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add tests/inference/test_sparse_large_model.py docs/roadmap.md
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "test(sparse): large-model uncertainty smoke; docs: E-sparse-C shipped"
```

---

## Self-Review

**1. Spec coverage** (against `docs/design/specs/2026-08-27-pylgm-esparse-c-design.md`):
- §5 Takahashi primitive → Task 1. ✓
- §4 `SparsePosterior` (`apply_inverse`/`marginal_variances`/`predictive_variances`/`linear_combination_variances`/`covariance_dense`) → Tasks 2-4. ✓
- §6.1 latent_marginals, §6.4 linear_combinations → Task 5. ✓
- §6.2 predictive_variance → Tasks 4-5. ✓
- §6.3 predict → Task 6. ✓
- §6.5 covariance guarded → Task 5. ✓
- §6.6 Sørbye-Rue sparse → Task 8. ✓
- §6.7 BYM2 augmented → Task 9 (validated joint + generalized-cofactor logdet; **spec √-transcription corrected to `b=−√φ/(1−φ)`**). ✓
- §6.8 INLA integration → Task 10 (Gaussian + simplified-Laplace diagonal; full-Laplace guarded). ✓
- §7 files touched → File Structure map. ✓
- §8 dense-equivalence oracles → Tasks 1,3,4,9,10. ✓
- model.py rebuild carry → Task 7. ✓

**2. Placeholder scan:** All code steps carry concrete code. Three steps (Task 6 `Prediction(...)`, Task 9 Step 6, Task 10 fill-ins) explicitly instruct "mirror the existing construction / reuse the file's model-construction helper" rather than inventing new interfaces — these are reuse directives against code the implementer will read, not TBDs. Test fixtures are named and their construction is pointed at the exact existing tests to copy.

**3. Type consistency:** `selected_inverse_diagonal(matrix)` (Task 1) is consumed by `_unconstrained_marginal` (Task 3) and `_sorbye_sparse_diag` (Task 8) with the single-matrix signature throughout. `SparsePosterior` field names (Task 2) are referenced identically in Tasks 3-4. `_sparse_posterior` (result field), `sparse_posterior` (constructor kw), `posterior` (SparseFit field) are used consistently across Tasks 2,5,6,7,10. `_conditional_latent_variances`/`_conditional_predictive_variances` (Task 10) match their call sites. `_BYM2_AUGMENT_NODES` and `_build_bym2_augmented` (Task 9) match compiler wiring and tests.

**Pre-flight items to batch to the human before Task 1:**
1. **BYM2 public-width change (Task 9):** large BYM2 switches to a `2n` augmented block reporting `2n` labels (region `x` + `{node}__u`). Confirm this is the accepted behavior (spec §6.7 accepts it) vs. an always-`n`-reporting alternative (would need IR support to hide `u*`).
2. **`_BYM2_AUGMENT_NODES = 1024` threshold** — confirm the cutoff (tie to the dense guard vs. a fixed node count).
3. **Large multi-component BYM2 raises** `NotImplementedError` on the augmented path (per-component augmentation deferred). Confirm acceptable.

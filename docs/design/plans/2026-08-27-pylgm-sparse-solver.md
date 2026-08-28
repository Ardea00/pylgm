# E-sparse (A+B) — Sparse Constrained Gaussian Solver + Sparse Spatial Builders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let already-shipped network / space-time models build and fit past the dense preflight guard by adding a sparse constrained-Gaussian solve that returns posterior mean, marginal likelihood, and point predictions, with posterior uncertainty carved off behind a single E-sparse-C seam.

**Architecture:** A new `inference/sparse.py` holds a `splu`-based SPD factor, a fixed-effect/field block partition, and a partitioned-Schur + conditioning-by-kriging solve that reproduces the dense engine's mean and log-marginal-likelihood without ever forming the dense null-space basis. `GaussianResult` gains an optional (absent) covariance so a sparse fit is a first-class result whose uncertainty accessors raise a clear "pending E-sparse-C" error. Routing turns the dense guard into a switch: below threshold stays dense (unchanged); above threshold routes sparse; `allow_large_dense=True` forces dense.

**Tech Stack:** numpy, scipy (`scipy.sparse.linalg.splu`/`factorized`/`eigsh` — already a dependency, currently unused), pandas. No new runtime dependency.

## Global Constraints

- No new runtime dependency; numpy / scipy / pandas / stdlib only. `scipy.sparse.linalg` is already available via the existing scipy dependency.
- Deterministic, no MCMC. The marginal likelihood must be exact (no stochastic logdet).
- Full suite green; `python -m ruff check src tests` clean — **no unused imports** (the recurring F401 trap in this codebase's tests).
- New tests pass under `-W error::UserWarning`.
- The dense path is **behavior-compatible below the threshold** — unchanged for every existing small-model test. The sparse path is exercised only above the threshold or when explicitly forced in a test.
- Git identity is **Ardea00 only**. Commit with `git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit`. Never pass `--author`; never commit as iongroup/AndreaPanozzo.
- Mark deliberate simplifications / known ceilings with a `ponytail:` comment naming the ceiling and upgrade path.

## Deliverable boundary (what A+B does and does not do)

A+B makes a large spatial model **fit and predict-in-mean** at scale. Concretely, on a past-guard model:

- Available: `result.mean`, `result.log_marginal_likelihood`, `result.hyperparameters`, `result.predictive_mean`.
- Deferred to E-sparse-C (raise a clear error): `result.covariance`, `result.predictive_variance`, `result.latent_marginals()`, `result.linear_combinations()`, `result.predict()`.
- `hyperparameters="optimize"` (empirical Bayes, the default) works at scale — its inner loop reads only `log_marginal_likelihood`. `hyperparameters="integrate"` (full INLA grid integration) is **not** supported at scale: `integrate_inla` accumulates per-grid `cond.covariance` (`optimization/inla.py:314,329`), so it surfaces the same pending-C error. This is expected and documented, not a bug.

This is a deliberate deviation from the design spec's "`predict` finite" line: `predict()` and all variances need the posterior inverse, which is E-sparse-C. See findings 1–3 in the accompanying commit message / session notes.

---

## File Structure

- **Add** `src/pylgm/inference/sparse.py` — the whole sparse engine: `SparseSpdFactor` (splu + logdet), `_partition_blocks`, `sparse_constrained_gaussian` (mean + lml + predictive_mean), `selected_inverse_diagonal` (the C seam). One module so `gaussian.py` stays a thin router.
- **Modify** `src/pylgm/inference/gaussian.py` — turn the guard into a switch; add `_fit_sparse` that wraps `sparse_constrained_gaussian` into a covariance-less `GaussianResult`.
- **Modify** `src/pylgm/inference/result.py` — allow `covariance`/`predictive_variance` to be absent; uncertainty accessors raise a clear pending-C error.
- **Modify** `src/pylgm/model.py` — `_rebuild_result` / `_align_predictions_with_source_rows` tolerate an absent covariance / predictive_variance.
- **Modify** `src/pylgm/effects/proper_car.py` — `_validity_interval` uses `eigsh` extreme eigenvalues instead of dense `eigvalsh`.
- **Add** `tests/inference/test_sparse.py`, `tests/inference/test_sparse_result.py`, `tests/effects/test_proper_car_eigsh.py`, and a large-model end-to-end test (path per Task 10).

---

## Task 1: proper_car ρ-interval via `eigsh`

Independent of the solve; smallest deliverable. Replaces the dense `eigvalsh` of the normalized adjacency with two sparse extreme-eigenvalue solves.

**Files:**
- Modify: `src/pylgm/effects/proper_car.py` (`_validity_interval`)
- Test: `tests/effects/test_proper_car_eigsh.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_validity_interval(w, degree)` unchanged signature and return `(lower, upper, mu_min, mu_max)`.

- [ ] **Step 1: Read the current implementation.** Open `src/pylgm/effects/proper_car.py`, read `_validity_interval` and `car_rho_interval`. Note the existing `ponytail:` comment flagging the eigsh upgrade, the normalized-adjacency construction, and the isolated-node error path. Preserve every error and the exact return tuple.

- [ ] **Step 2: Write the failing test.**

```python
# tests/effects/test_proper_car_eigsh.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.effects.proper_car import _validity_interval, car_rho_interval


def _ring_adjacency(n: int) -> csr_matrix:
    rows, cols = [], []
    for i in range(n):
        rows += [i, i]
        cols += [(i - 1) % n, (i + 1) % n]
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_eigsh_interval_matches_dense_eigvalsh():
    w = _ring_adjacency(12)
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    # Dense reference for the same extreme eigenvalues of the normalized adjacency.
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = (w.toarray() * d_inv_sqrt).T * d_inv_sqrt
    eigenvalues = np.linalg.eigvalsh(normalized)
    lower, upper, mu_min, mu_max = _validity_interval(w, degree)
    assert np.isclose(mu_min, eigenvalues[0], atol=1e-8)
    assert np.isclose(mu_max, eigenvalues[-1], atol=1e-8)
    # ρ interval is the reciprocal of the extreme eigenvalues, unchanged by this task.
    assert lower < 0 < upper
    assert np.isclose(lower, 1.0 / eigenvalues[0], atol=1e-8)
    assert np.isclose(upper, 1.0 / eigenvalues[-1], atol=1e-8)


def test_isolated_node_still_raises():
    # A node with zero degree must still raise the existing isolated-node error.
    w = csr_matrix((np.array([1.0, 1.0]), (np.array([0, 1]), np.array([1, 0]))), shape=(3, 3))
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    with pytest.raises(Exception):
        _validity_interval(w, degree)
```

- [ ] **Step 3: Run it, expect failure/mismatch.** `python -m pytest tests/effects/test_proper_car_eigsh.py -v`. The interval-match test may already pass if the dense path is numerically identical; it exists to lock behavior. If `_validity_interval` return names differ, adapt the test to the real tuple names before implementing.

- [ ] **Step 4: Replace the dense eigenvalues with `eigsh`.** In `_validity_interval`, build the normalized adjacency as a sparse operator (keep it sparse — do not `.toarray()`), then:

```python
from scipy.sparse.linalg import eigsh

# normalized is a symmetric sparse matrix (D^-1/2 W D^-1/2).
# k=1 extreme eigenvalues; 'SA' = smallest algebraic, 'LA' = largest algebraic.
mu_min = float(eigsh(normalized, k=1, which="SA", return_eigenvectors=False)[0])
mu_max = float(eigsh(normalized, k=1, which="LA", return_eigenvectors=False)[0])
```

Keep the isolated-node guard **before** the eigsh call (a zero-degree node makes `D^-1/2` non-finite — that must still raise the existing clear error, not an eigsh convergence failure). Leave the ρ-interval reciprocal math and the positive-definiteness gate exactly as they were. Update/remove the stale `ponytail:` comment. For very small graphs `eigsh` needs `k < n-1`; guard: if `n <= 2`, fall back to dense `eigvalsh` (`ponytail: eigsh needs k<n-1; tiny graphs go dense — trivially cheap`).

- [ ] **Step 5: Run tests.** `python -m pytest tests/effects/test_proper_car_eigsh.py -v` → PASS. Then the existing proper_car suite: `python -m pytest tests/ -k proper_car -v` → all PASS (no behavior change).

- [ ] **Step 6: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/effects/proper_car.py tests/effects/test_proper_car_eigsh.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(proper_car): sparse eigsh extreme eigenvalues for rho interval"
```

---

## Task 2: `SparseSpdFactor` — sparse factor with exact logdet

**Files:**
- Create: `src/pylgm/inference/sparse.py`
- Test: `tests/inference/test_sparse.py`

**Interfaces:**
- Produces:
  - `class SparseSpdFactor` with `__init__(self, matrix: csr_matrix, name: str)`, `.solve(self, b: np.ndarray) -> np.ndarray` (b may be 1-D or 2-D; columns solved together), and `.logdet: float`.
  - Raises `pylgm.exceptions.NumericalError(f"{name} must be positive definite")` on factorization failure or non-positive pivots, mirroring `_factor_positive_definite`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/inference/test_sparse.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.exceptions import NumericalError
from pylgm.inference.sparse import SparseSpdFactor


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
```

- [ ] **Step 2: Run it, expect ImportError.** `python -m pytest tests/inference/test_sparse.py -v` → FAIL (`sparse` module / `SparseSpdFactor` not defined).

- [ ] **Step 3: Implement `SparseSpdFactor`.**

```python
# src/pylgm/inference/sparse.py  (new file — header)
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from pylgm.exceptions import NumericalError


class SparseSpdFactor:
    """SuperLU factor of a sparse SPD matrix, with an exact log-determinant.

    Uses ``splu`` (not a sparse Cholesky, which SciPy does not ship). For an
    SPD matrix ``logdet = sum(log(diag(U)))`` from the LU factor; the row/column
    permutations contribute a determinant of +/-1, which cancels for the SPD
    magnitude. A non-positive product of diagonal entries means the matrix was
    not positive definite -- surfaced as NumericalError to match the dense path.
    """

    def __init__(self, matrix: csr_matrix, name: str) -> None:
        self._name = name
        try:
            # permc_spec chosen for fill-in reduction on GMRF precisions.
            self._lu = splu(matrix.tocsc(), permc_spec="COLAMD")
        except (RuntimeError, ValueError) as error:
            raise NumericalError(f"{name} must be positive definite") from error
        diag_u = self._lu.U.diagonal()
        if not np.all(diag_u > 0) or not np.isfinite(diag_u).all():
            raise NumericalError(f"{name} must be positive definite")
        self._logdet = float(np.sum(np.log(diag_u)))

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self._lu.solve(np.asarray(b, dtype=float))

    @property
    def logdet(self) -> float:
        return self._logdet
```

Note on the diagonal-positivity check: `splu`'s L has unit diagonal, so `U`'s diagonal carries the pivots; for a truly SPD matrix with COLAMD ordering they are all positive. `ponytail: relies on SPD pivots staying positive under COLAMD; a near-singular field could trip this and raise NumericalError, which is the correct signal, not a silent bad logdet.`

- [ ] **Step 4: Run tests.** `python -m pytest tests/inference/test_sparse.py -v` → PASS.

- [ ] **Step 5: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): SparseSpdFactor with splu-based exact logdet"
```

---

## Task 3: Block partition (fixed-effect dense block vs field sparse block)

**Files:**
- Modify: `src/pylgm/inference/sparse.py`
- Test: `tests/inference/test_sparse.py`

**Interfaces:**
- Produces: `_partition_blocks(model: CompiledLGM) -> tuple[np.ndarray, np.ndarray]` returning `(sparse_index, dense_index)` — int arrays of latent-column indices. `dense_index` = columns belonging to blocks whose prior precision has no nonzeros (improper/flat — `Fixed`, intercept). `sparse_index` = the rest (structured GMRF fields).

- [ ] **Step 1: Write the failing test.**

```python
# add to tests/inference/test_sparse.py
from pylgm.inference.sparse import _partition_blocks


def test_partition_splits_fixed_from_field(besag_with_intercept_model):
    # Fixture builds Fixed("1") + Besag(...) compiled model (see conftest).
    sparse_index, dense_index = _partition_blocks(besag_with_intercept_model)
    labels = besag_with_intercept_model.labels
    # The intercept (Fixed, zero prior precision) lands in the dense block.
    dense_labels = {labels[i] for i in dense_index}
    assert any(label.endswith(":1") for label in dense_labels)
    # Every Besag column (nonzero prior precision) lands in the sparse block.
    assert len(sparse_index) + len(dense_index) == len(labels)
    assert set(sparse_index).isdisjoint(dense_index)
```

Add a fixture to `tests/inference/conftest.py` (or the test file) that compiles `Fixed("1") + Besag(...)` on a tiny connected graph into a `CompiledLGM`. Reuse an existing compile helper if the test suite already has one (grep `tests/` for `compile_lgm(` usage patterns and copy the smallest).

- [ ] **Step 2: Run it, expect failure.** `python -m pytest tests/inference/test_sparse.py::test_partition_splits_fixed_from_field -v` → FAIL.

- [ ] **Step 3: Implement `_partition_blocks`.**

```python
# add to src/pylgm/inference/sparse.py
from pylgm.ir.model import CompiledLGM


def _partition_blocks(model: CompiledLGM) -> tuple[np.ndarray, np.ndarray]:
    """Split latent columns into the sparse field block and the dense fixed block.

    A block with an all-zero prior precision is an improper/flat effect (a
    ``Fixed`` covariate or intercept). Its design column couples to many
    observations, so ``Z^T Z`` would densify a joint sparse factor -- it is
    quarantined into the small dense block. Every structured GMRF field
    (IID / RW / Besag / proper_car / AR1 / space-time) carries a nonzero prior
    precision and an incidence-like design, so it stays sparse.

    ponytail: criterion is precision.nnz==0, which is exact for the in-scope
    spatial models. A structured block with a *dense* design (e.g. MIDAS lag
    columns) would be misrouted into the sparse block and densify A_s. MIDAS at
    scale is out of scope (E-sparse targets spatial models); such a model still
    fits via the dense path under allow_large_dense. Upgrade path: also test
    design column density here and raise a clear unsupported error.
    """
    sparse_cols: list[int] = []
    dense_cols: list[int] = []
    start = 0
    for block in model.blocks:
        width = block.design.shape[1]
        columns = range(start, start + width)
        if block.precision.nnz == 0:
            dense_cols.extend(columns)
        else:
            sparse_cols.extend(columns)
        start += width
    return np.asarray(sparse_cols, dtype=int), np.asarray(dense_cols, dtype=int)
```

- [ ] **Step 4: Run tests.** `python -m pytest tests/inference/test_sparse.py -v` → PASS.

- [ ] **Step 5: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py tests/inference/conftest.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): partition latent columns into field and fixed blocks"
```

---

## Task 4a: Unconstrained sparse solve — mean + lml + predictive_mean (no constraints)

> **Model:** dispatch this task on the most capable available model. It is a test-driven numerical derivation, not transcription: the exact code is settled by making the dense-equivalence assertion pass, with the identities below as the intended route.

**Files:**
- Modify: `src/pylgm/inference/sparse.py`
- Test: `tests/inference/test_sparse.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class SparseFit` with fields `mean: np.ndarray`, `log_marginal_likelihood: float`, `predictive_mean: np.ndarray`, `block_slices: dict[str, slice]`, `diagnostics: dict[str, object]`.
  - `sparse_constrained_gaussian(model: CompiledLGM) -> SparseFit`. In this task it handles the **no-constraint** case (`model.constraints.shape[0] == 0`); constraints are added in Task 4b. Raise `NotImplementedError` for the constrained case so 4b has a clear seam.

**The math this task must reproduce** (matching `gaussian.py:_fit_dense`, `allow_large_dense`-forced dense reference):

Let `σ² = model.likelihood.variance`, `Z = model.design[observed]`, `r = y[observed] − offset[observed]`, `Q = model.precision`. Partition columns via `_partition_blocks` into `s` (sparse field) and `d` (dense fixed). The posterior precision is

```
Q_post = Q + Zᵀ Z / σ²   =  [[A_s, B ], [Bᵀ, D]]
A_s = Q_ss + Z_sᵀ Z_s / σ²   (sparse, factor with SparseSpdFactor)
D   = Q_dd + Z_dᵀ Z_d / σ²   (dense m×m, m = len(dense_index))
B   = Q_sd + Z_sᵀ Z_d / σ²   (n_s × m; Q_sd is zero for block-diagonal Q, so B = Z_sᵀ Z_d / σ²)
```

Score `g = Zᵀ r / σ²`, split `g_s, g_d`. Solve by Schur complement on the small block:

```
W  = A_s.solve(B)                      # n_s × m
S  = D - Bᵀ W                          # m × m dense Schur complement
x_d = solve(S, g_d - Bᵀ A_s.solve(g_s))
x_s = A_s.solve(g_s - B x_d)
mean = scatter(x_s, x_d) into full latent order
logdet_posterior = A_s.logdet + logdet(S)     # via _factor_positive_definite(S)
```

Log marginal likelihood (unconstrained, matches the dense formula with empty constraints so `basis = I`, `logdet_prior = logdet(Q_prior)` on the sparse block plus the improper flat block contributing 0):

```
quadratic = rᵀ r / σ²  -  meanᵀ Zᵀ r / σ²        # = rᵀr/σ² - meanᵀ g   (Gaussian identity)
logdet_prior = A_s(Q_ss).logdet                  # dense/flat block has zero prior precision -> excluded
lml = -0.5 * ( n_obs·log(2π σ²) - logdet_prior + logdet_posterior + quadratic )
predictive_mean = offset + model.design @ mean   # sparse matvec, no densify
```

`ponytail: logdet_prior of the sparse field block uses SparseSpdFactor on Q_ss. For an INTRINSIC field (Besag/space-time) Q_ss is rank-deficient and this factor fails — that case is deferred to Task 4b, which computes the constrained prior determinant. Task 4a covers proper/full-rank fields (proper_car, IID, AR1) plus fixed effects, which is the no-constraint deliverable.`

- [ ] **Step 1: Write the dense-equivalence test** (the contract). Use a model with a Fixed intercept + a **proper_car** field (full-rank prior, no intrinsic constraints), small enough to run under the dense guard, forced through the sparse path.

```python
# add to tests/inference/test_sparse.py
from pylgm.inference.gaussian import _fit_dense
from pylgm.inference.sparse import sparse_constrained_gaussian


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
```

Add the fixture: compile `Fixed("1") + proper_car(...)` with a fixed ρ inside the validity interval and a fixed precision, on a small connected graph, with a handful of observations (some rows unobserved to exercise the `observed` mask).

- [ ] **Step 2: Run it, expect failure.** → FAIL (`sparse_constrained_gaussian` not defined / `NotImplementedError`).

- [ ] **Step 3: Implement `SparseFit` and the unconstrained `sparse_constrained_gaussian`** following the math block above. Densify only `D`, `B` (n_s × m, m small), and `S` (m × m) — never the field block. Use `_block_slices`-equivalent logic (copy the tiny helper from `gaussian.py` or import it) for `block_slices`. Guard the empty-field and empty-fixed edge cases (m == 0 → pure sparse solve, no Schur; n_s == 0 → pure dense, delegate to the dense helpers). Raise `NotImplementedError("constrained sparse solve arrives in Task 4b")` when `model.constraints.shape[0] > 0`.

- [ ] **Step 4: Iterate against the equivalence test** until `mean`, `lml`, and `predictive_mean` all match. This is the derivation loop — expect to reconcile sign conventions and the `observed`-mask handling with `_fit_dense`. Re-read `gaussian.py:108-176` line by line to match `residual`, `score`, `quadratic`, and the lml sign exactly.

- [ ] **Step 5: Run the full sparse test file.** `python -m pytest tests/inference/test_sparse.py -v` → PASS.

- [ ] **Step 6: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py tests/inference/conftest.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): partitioned Schur solve for mean, lml, predictive_mean (no constraints)"
```

---

## Task 4b: Constrained sparse solve — intrinsic fields via conditioning by kriging

> **Model:** most capable available. Test-driven derivation. This task carries a **validation spike** (Step 1) because the intrinsic-GMRF constrained determinant is the make-or-break identity.

**Files:**
- Modify: `src/pylgm/inference/sparse.py`
- Test: `tests/inference/test_sparse.py`

**Interfaces:**
- Extends `sparse_constrained_gaussian` to handle `model.constraints.shape[0] > 0` (block sum-to-zero constraints + model-level `extra_constraints` with nonzero `constraint_rhs`). Same `SparseFit` return.

**The math** (conditioning by kriging, Rue & Held 2005 §2.3.3; constrained marginal likelihood §2.4). Let `A = model.constraints` (c × n), `e = model.constraint_rhs`. Let `Q_post` be the partitioned posterior precision from 4a with an `apply_inverse(v) -> Q_post⁻¹ v` built from the Schur solve (so `A Q_post⁻¹ Aᵀ` is a c × c dense matrix from c inverse-applications). Then:

```
# Unconstrained posterior mean μ and score as in 4a.
M   = A @ apply_inverse(Aᵀ)                 # c × c dense
rhs = A @ μ - e
μ*  = μ - apply_inverse(Aᵀ) @ solve(M, rhs) # constrained (kriged) mean
```

Constrained log-determinants via the SPD identity (posterior is SPD, `basis` orthonormal so `det(basis)=1`):

```
logdet( basisᵀ Q_post basis ) = logdet(Q_post) + logdet( A Q_post⁻¹ Aᵀ ) - logdet( A Aᵀ )
```

with `logdet(Q_post) = A_s.logdet + logdet(S)` from 4a. The **prior** determinant `logdet(basisᵀ Q_prior basis)` is the subtle one: `Q_prior` is rank-deficient for intrinsic fields. Split by Sørbye–Rue scaling `Q_prior = τ · structure` per block:

```
logdet(basisᵀ Q_prior basis) = Σ_blocks (rank_b)·log(τ_b) + base_const
```

where `rank_b = width_b - null_dim_b` and `base_const = logdet(basisᵀ structure basis)` is θ-independent. Compute `base_const` once from the block structure's nonzero eigenvalues — the **same eigendecomposition Sørbye–Rue already performs** (`effects/scaling.py`). `ponytail: base_const reuses the one-time-dense structure eigendecomposition (the Sørbye–Rue ceiling the spec already documents); E-sparse-C makes this sparse. For a non-intrinsic field (proper_car/IID/AR1) the field prior is full-rank and 4a's SparseSpdFactor logdet is used directly instead.`

- [ ] **Step 1 (SPIKE): validate the identities on tiny intrinsic models before touching the engine.** Write `tests/inference/test_sparse.py::test_kriging_identities_spike` that, on a small **Besag** model (intrinsic, one sum-to-zero constraint), asserts each identity in isolation against dense references computed with `scipy.linalg.null_space`:
  - `μ*` equals the dense constrained mean (`gaussian.py` dense path result `.mean`).
  - `logdet(basisᵀ Q_post basis)` from the SPD identity equals `np.linalg.slogdet(basis.T @ Q_post_dense @ basis)[1]`.
  - `logdet(basisᵀ Q_prior basis)` from the τ-split equals `np.linalg.slogdet(basis.T @ (Q_prior_dense) @ basis)[1]` (drop the flat/fixed block, which contributes a zero row/col to the reduced prior — confirm the dense reference restricts to the field block's null space consistently with `_fit_dense`'s `logdet_prior`).

  Run it; iterate the identities until each matches. **This spike is the gate for the rest of 4b** — if an identity cannot be made to match, stop and escalate (the determinant convention needs rework) rather than proceeding.

- [ ] **Step 2: Write the end-to-end constrained equivalence test.**

```python
def test_sparse_matches_dense_besag(besag_with_intercept_model):
    model = besag_with_intercept_model            # intrinsic + sum-to-zero
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
```

Add fixtures: `besag_with_intercept_model` (`Fixed("1") + Besag(...)`), `besag_extraconstr_model` (same + an `LGM(..., constraints=[({"region:oslo": 1.0}, 2.5)])`-style nonzero-rhs extra constraint compiled in).

- [ ] **Step 3: Run, expect failure.** → FAIL.

- [ ] **Step 4: Implement the constrained branch** using the validated spike identities: build `apply_inverse` from the 4a Schur solve, form the kriged mean, and assemble `logdet_prior` (τ-split for intrinsic blocks, direct sparse logdet for full-rank blocks) and `logdet_posterior` (SPD identity). Reuse `_factor_positive_definite` for the small `M`, `S`, and `A Aᵀ` dense pieces. Remove the Task-4a `NotImplementedError` guard.

- [ ] **Step 5: Iterate against Step 2's tests** until mean and lml match on both the intrinsic sum-to-zero and the nonzero-rhs extraconstr models.

- [ ] **Step 6: Run full sparse file + existing inference suite.** `python -m pytest tests/inference/ -v` → PASS (dense path untouched).

- [ ] **Step 7: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py tests/inference/conftest.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): constrained solve via conditioning by kriging with intrinsic-field determinants"
```

---

## Task 5: `selected_inverse_diagonal` — the E-sparse-C seam

**Files:**
- Modify: `src/pylgm/inference/sparse.py`
- Test: `tests/inference/test_sparse.py`

**Interfaces:**
- Produces: `selected_inverse_diagonal(factor: SparseSpdFactor, indices: np.ndarray) -> np.ndarray` — the single plug-point for E-sparse-C's Takahashi recursion. Raises `NotImplementedError` in this slice.

- [ ] **Step 1: Write the failing test.**

```python
def test_selected_inverse_diagonal_raises():
    from scipy.sparse import csr_matrix
    from pylgm.inference.sparse import SparseSpdFactor, selected_inverse_diagonal

    factor = SparseSpdFactor(csr_matrix(np.eye(3)), "id")
    with pytest.raises(NotImplementedError, match="E-sparse-C"):
        selected_inverse_diagonal(factor, np.array([0, 1, 2]))
```

- [ ] **Step 2: Run, expect failure.** → FAIL.

- [ ] **Step 3: Implement the seam.**

```python
def selected_inverse_diagonal(factor: "SparseSpdFactor", indices: np.ndarray) -> np.ndarray:
    """Diagonal of the inverse at ``indices`` -- posterior marginal variances.

    The single plug-point for E-sparse-C: the Takahashi selected-inversion
    recursion replaces this body and nothing else in the call path changes.
    """
    raise NotImplementedError(
        "sparse posterior variances (selected inversion) pending E-sparse-C"
    )
```

- [ ] **Step 4: Run, PASS. ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/sparse.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(sparse): selected_inverse_diagonal seam for E-sparse-C"
```

---

## Task 6: `result.py` — optional (absent) covariance with pending-C accessors

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Test: `tests/inference/test_sparse_result.py`

**Interfaces:**
- `_BaseResult._init_common` accepts `covariance=None` and `predictive_variance=None`: stores `_covariance=None` / `_predictive_variance=None` and skips the finite/dtype validation for whichever is None.
- `covariance`, `predictive_variance`, `latent_marginals`, `linear_combinations`, `predict` raise `NotImplementedError` with a message naming E-sparse-C when covariance is absent.
- `mean`, `predictive_mean`, `log_marginal_likelihood`, `hyperparameters`, `labels`, `block_slices` work as before.

- [ ] **Step 1: Write the failing test.**

```python
# tests/inference/test_sparse_result.py
import numpy as np
import pytest

from pylgm.inference.result import GaussianResult


def _covless_result():
    return GaussianResult(
        labels=("a:1", "a:2"),
        mean=np.array([1.0, 2.0]),
        covariance=None,
        log_marginal_likelihood=-3.0,
        predictive_mean=np.array([0.5, 1.5]),
        predictive_variance=None,
        observation_variance=1.0,
        block_slices={"a": slice(0, 2)},
    )


def test_covless_result_exposes_mean_and_lml():
    result = _covless_result()
    assert np.allclose(result.mean, [1.0, 2.0])
    assert result.log_marginal_likelihood == -3.0
    assert np.allclose(result.predictive_mean, [0.5, 1.5])


def test_covless_result_raises_on_uncertainty():
    result = _covless_result()
    for accessor in (
        lambda: result.covariance,
        lambda: result.predictive_variance,
        lambda: result.latent_marginals(),
        lambda: result.linear_combinations(np.eye(2)),
    ):
        with pytest.raises(NotImplementedError, match="E-sparse-C"):
            accessor()
```

- [ ] **Step 2: Run, expect failure.** `python -m pytest tests/inference/test_sparse_result.py -v` → FAIL (`_init_common` rejects None covariance at `np.asarray(None)` / finite check).

- [ ] **Step 3: Edit `_init_common`** (`result.py:544-604`). Replace the covariance block:

```python
        _validate_prediction_keys(prediction_keys, predictive_mean)
        if covariance is None:
            object.__setattr__(self, "_covariance", None)
        else:
            covariance = np.asarray(covariance)
            if not np.issubdtype(covariance.dtype, np.number) or not np.isrealobj(covariance):
                raise TypeError("covariance must have a real numeric dtype")
            if not np.isfinite(covariance).all():
                raise ValueError("covariance must be finite")
            object.__setattr__(self, "_covariance", _readonly_array(covariance))
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "_mean", _readonly_array(mean))
        object.__setattr__(self, "log_marginal_likelihood", float(log_marginal_likelihood))
        object.__setattr__(self, "_predictive_mean", _readonly_array(predictive_mean))
        if predictive_variance is None:
            object.__setattr__(self, "_predictive_variance", None)
        else:
            object.__setattr__(self, "_predictive_variance", _readonly_array(predictive_variance))
```

(Delete the old unconditional `_covariance`/`_predictive_variance` sets. Keep the rest of `_init_common` — `extra_store`, keys, diagnostics, hyperparameters — unchanged. Note `_validate_prediction_keys` uses `predictive_mean` size, still real.)

- [ ] **Step 4: Add a pending-C guard and update the accessors.** Near the top of `result.py`, add:

```python
_PENDING_C = (
    "posterior covariance is unavailable on the sparse path "
    "(pending E-sparse-C); only mean, marginal likelihood, and predictive_mean "
    "are available for large models fitted through the sparse solver"
)
```

Update the properties/methods on `_BaseResult`:

```python
    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise NotImplementedError(_PENDING_C)
        return _readonly_array(self._covariance)

    @property
    def predictive_variance(self) -> np.ndarray:
        if self._predictive_variance is None:
            raise NotImplementedError(_PENDING_C)
        return _readonly_array(self._predictive_variance)
```

In `latent_marginals`, `linear_combinations`, and `predict`, add a guard `if self._covariance is None: raise NotImplementedError(_PENDING_C)` before the covariance is used (`predict` already reads `self.covariance`, which now raises — but add the explicit guard so the message is the pending-C one, not a stack from `predict_from`). Keep the `INLAResult.latent_marginals` table branch first (it never has a None covariance in practice; guard the fallback branch).

- [ ] **Step 5: Run the new test + the full existing result suite.** `python -m pytest tests/inference/test_sparse_result.py tests/ -k result -v` → PASS. The docstring on `predictive_variance` (result.py:619-628) stays valid for the real-covariance case.

- [ ] **Step 6: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/result.py tests/inference/test_sparse_result.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(result): optional absent covariance with pending-E-sparse-C accessors"
```

---

## Task 7: `model.py` — rebuild/reorder tolerate an absent covariance

**Files:**
- Modify: `src/pylgm/model.py` (`_rebuild_result`, `_align_predictions_with_source_rows`)
- Test: `tests/inference/test_sparse_result.py`

**Interfaces:**
- Consumes: covariance-less `GaussianResult` from Task 6.
- `_rebuild_result` reads the raw `_covariance` / `_predictive_variance` (None-tolerant) instead of the raising public properties, and skips reindexing `predictive_variance` when it is None.

- [ ] **Step 1: Write the failing test.**

```python
# add to tests/inference/test_sparse_result.py
def test_rebuild_preserves_absent_covariance():
    from pylgm.model import _rebuild_result

    result = _covless_result()
    rebuilt = _rebuild_result(result, hyperparameters={"tau": 1.0})
    assert rebuilt._covariance is None
    assert rebuilt._predictive_variance is None
    assert np.allclose(rebuilt.mean, [1.0, 2.0])
    assert rebuilt.hyperparameters["tau"] == 1.0


def test_align_reorders_without_covariance():
    from pylgm.model import _align_predictions_with_source_rows

    result = _covless_result()
    aligned = _align_predictions_with_source_rows(result, np.array([1, 0]))
    assert aligned._covariance is None
    assert aligned._predictive_variance is None
```

- [ ] **Step 2: Run, expect failure.** → FAIL (`_rebuild_result` reads `result.covariance` → raises; and `predictive_variance[caller_order]` on None → TypeError).

- [ ] **Step 3: Edit `_rebuild_result`** (`model.py:145-203`). Change the covariance/predictive_variance reads to the raw None-tolerant attributes and guard the reorder:

```python
    predictive_mean = result.predictive_mean
    predictive_variance = result._predictive_variance   # raw; None on the sparse path
    if caller_order is not None:
        predictive_mean = predictive_mean[caller_order]
        if predictive_variance is not None:
            predictive_variance = predictive_variance[caller_order]
    common = dict(
        labels=result.labels,
        mean=result.mean,
        covariance=result._covariance,                  # raw; None on the sparse path
        log_marginal_likelihood=result.log_marginal_likelihood,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        ...
    )
```

Leave the `LaplaceResult` / `INLAResult` branches unchanged (they always carry a real covariance; `_covariance` is a real array there, passed through fine). `ponytail: reading result._covariance/_predictive_variance (private) here is deliberate — _rebuild_result reconstructs the object, so it needs the raw stored value, not the pending-C-raising public property.`

- [ ] **Step 4: Run tests.** `python -m pytest tests/inference/test_sparse_result.py -v` → PASS.

- [ ] **Step 5: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/model.py tests/inference/test_sparse_result.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(model): rebuild and reorder tolerate an absent sparse covariance"
```

---

## Task 8: Routing — dense guard becomes a switch to the sparse path

**Files:**
- Modify: `src/pylgm/inference/gaussian.py`
- Test: `tests/inference/test_sparse.py`

**Interfaces:**
- Consumes: `sparse_constrained_gaussian` (Task 4), covariance-less `GaussianResult` (Task 6).
- `fit_gaussian(model, *, allow_large_dense=False)`: below threshold → `_fit_dense` (unchanged); above threshold and not `allow_large_dense` → new `_fit_sparse`; `allow_large_dense=True` → `_fit_dense` regardless (forces dense).
- Produces: `_fit_sparse(model: CompiledLGM) -> GaussianResult` wrapping the `SparseFit` into a covariance-less result.

- [ ] **Step 1: Write the failing tests.**

```python
# add to tests/inference/test_sparse.py
from pylgm.inference.gaussian import fit_gaussian


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
    with pytest.raises(NotImplementedError, match="E-sparse-C"):
        _ = sparse.covariance


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
```

- [ ] **Step 2: Run, expect failure.** → FAIL (`fit_gaussian` still raises `DenseReferenceLimitError` above threshold).

- [ ] **Step 3: Add `_fit_sparse` and reroute `fit_gaussian`.**

```python
# gaussian.py — new import at top
from pylgm.inference.sparse import sparse_constrained_gaussian

# new function
def _fit_sparse(model: CompiledLGM) -> GaussianResult:
    variance = float(model.likelihood.variance)
    if not np.isfinite(variance) or variance <= 0:
        raise NumericalError("sigma squared must be finite and positive")
    fit = sparse_constrained_gaussian(model)
    _require_finite("posterior mean", fit.mean)
    _require_finite("log marginal likelihood", fit.log_marginal_likelihood)
    _require_finite("predictive mean", fit.predictive_mean)
    return GaussianResult(
        labels=model.labels,
        mean=fit.mean,
        covariance=None,                       # pending E-sparse-C
        log_marginal_likelihood=fit.log_marginal_likelihood,
        predictive_mean=fit.predictive_mean,
        predictive_variance=None,              # pending E-sparse-C
        observation_variance=variance,
        block_slices=fit.block_slices,
        diagnostics=fit.diagnostics,
    )
```

Rewrite `fit_gaussian` so the guard routes instead of only raising:

```python
def fit_gaussian(model: CompiledLGM, *, allow_large_dense: bool = False) -> GaussianResult:
    if not isinstance(model.likelihood, CompiledGaussian):
        raise UnsupportedEngineError("exact Gaussian inference requires a Gaussian likelihood")
    if type(allow_large_dense) is not bool:
        raise TypeError("allow_large_dense must be a boolean")
    if not allow_large_dense and _exceeds_dense_threshold(model):
        return _fit_sparse(model)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _fit_dense(model)
    except FloatingPointError as error:
        raise NumericalError("exact Gaussian numerical calculation was non-finite") from error
```

Extract the threshold test from `preflight_dense_reference` into a predicate so both share it:

```python
def _exceeds_dense_threshold(model: CompiledLGM) -> bool:
    latent_size = model.precision.shape[0]
    if latent_size > _MAX_DENSE_LATENT_DIMENSION:
        return True
    return _estimated_dense_bytes(model.design.shape[0], latent_size) > _MAX_DENSE_BYTES
```

Keep `preflight_dense_reference` (still called elsewhere / by `laplace.py`); have it delegate to `_exceeds_dense_threshold` for the raise. `ponytail: fit_gaussian no longer calls preflight to raise — it switches. preflight stays for the Laplace path (E-sparse-D) which still has no sparse route.` Verify `laplace.py` still imports/behaves unchanged.

- [ ] **Step 4: Run tests.** `python -m pytest tests/inference/test_sparse.py -v` → PASS. Then `python -m pytest tests/ -k "gaussian or laplace or preflight" -v` → PASS (dense + laplace behavior intact).

- [ ] **Step 5: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add src/pylgm/inference/gaussian.py tests/inference/test_sparse.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "feat(gaussian): route past-threshold fits to the sparse solver"
```

---

## Task 9: Large-model end-to-end fit (the deliverable)

**Files:**
- Test: `tests/inference/test_sparse_large_model.py`

**Interfaces:**
- Consumes: the whole stack via `LGM.fit`.

- [ ] **Step 1: Write the deliverable test.** A proper-CAR (or Besag) model whose latent dimension is genuinely past `_MAX_DENSE_LATENT_DIMENSION` (e.g. a graph with > 4096 nodes — a large ring or grid is fine), fit through the public API with a `Hyperparameter` precision so empirical Bayes runs.

```python
# tests/inference/test_sparse_large_model.py
import numpy as np
import pandas as pd
import pytest

# Build a >4096-node spatial model, fit it, and assert the A+B deliverable surface.
def test_large_spatial_model_fits_mean_and_lml(large_besag_frame_and_model):
    model, frame = large_besag_frame_and_model          # LGM + panel; > 4096 latent dim
    result = model.fit(frame)                            # default: optimize (EB)
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.log_marginal_likelihood)
    assert result.hyperparameters                        # estimated, non-empty
    assert all(np.isfinite(v) for v in result.hyperparameters.values())
    assert np.isfinite(result.predictive_mean).all()
    # Uncertainty is pending E-sparse-C.
    for accessor in (lambda: result.covariance,
                     lambda: result.predictive_variance,
                     lambda: result.latent_marginals(),
                     lambda: result.predict(frame)):
        with pytest.raises(NotImplementedError, match="E-sparse-C"):
            accessor()


def test_large_model_integrate_is_unsupported(large_besag_frame_and_model):
    model, frame = large_besag_frame_and_model
    with pytest.raises(NotImplementedError, match="E-sparse-C"):
        model.fit(frame, hyperparameters="integrate")
```

Build the fixture from the smallest existing Besag/proper_car example helper, scaled past 4096 nodes. Keep the observation count modest so the fit is fast; the point is the latent dimension crosses the guard.

- [ ] **Step 2: Run, expect failure** (before the stack is wired this raised `DenseReferenceLimitError`; after Tasks 1–8 it should reach the assertions). Iterate only if a fixture/plumbing detail is off — the engine is already proven by Task 4's equivalence tests.

- [ ] **Step 3: Confirm timing.** The fit should complete in seconds, not minutes. If it is slow, check that no `.toarray()` slipped into the field path (grep `sparse.py` for `toarray`/`todense` — only `D`, `B`, `S`, `M`, `A Aᵀ` small dense pieces are allowed).

- [ ] **Step 4: ruff + commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add tests/inference/test_sparse_large_model.py
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "test(sparse): large spatial model fits mean+lml end-to-end past the dense guard"
```

---

## Task 10: Full-suite regression + roadmap note

**Files:**
- Modify: `docs/roadmap.md` (mark E-sparse A+B shipped, uncertainty pending C)

- [ ] **Step 1: Run the full non-Spark suite** under the warning gate:

```bash
python -m pytest tests -W error::UserWarning -q
```

Expected: all green (the pre-existing Windows symlink skip is fine). No regressions in the dense path.

- [ ] **Step 2: ruff on the whole tree.** `python -m ruff check src tests` → clean.

- [ ] **Step 3: Roadmap note.** Add a short additive line under the appropriate section of `docs/roadmap.md`: E-sparse (A+B) shipped — network/space-time models fit past the dense guard for posterior mean, marginal likelihood, and point predictions; posterior uncertainty (variances, `predict`, full INLA integration) pending E-sparse-C. Do not restructure the file.

- [ ] **Step 4: commit.**

```bash
python -m ruff check src tests
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' add docs/roadmap.md
git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com' commit -m "docs(roadmap): mark E-sparse A+B shipped (uncertainty pending C)"
```

---

## Self-Review

**1. Spec coverage.**
- Sparse constrained-Gaussian solve (mean, logdet, quadratic/lml) → Tasks 4a/4b. ✓
- Constraints + fixed effects via small dense Schur block → Tasks 3, 4a, 4b. ✓
- proper_car `eigsh` ρ-interval → Task 1. ✓
- Routing: guard becomes a switch; `allow_large_dense` forces dense → Task 8. ✓
- E-sparse-C seam raising a clear error → Task 5 (`selected_inverse_diagonal`) + Task 6 (result accessors). ✓
- besag / space-time already-sparse precisions consumed, not rebuilt → Tasks 4a/4b (no builder change; `sparse.py` reads `model.precision`). ✓ Space-time is covered by the same intrinsic-constraint path as Besag (its interaction constraints are sum-to-zero rows in `model.constraints`); Task 4b's Besag/extraconstr equivalence tests exercise that path. If a space-time-specific equivalence test is cheap to add, add it alongside Task 4b's.
- Testing: equivalence anchor (4a/4b), large-model fit (9), routing crossover (8), variance seam (5,6), proper_car eigsh (1). ✓
- **Deviation from spec, flagged:** "`predict` finite" is replaced by "predictive_mean finite; predict/variances raise pending-C" (predict needs the inverse). Documented in "Deliverable boundary" and Task 9. BYM2 untouched (deferred to C) — no task, correct.

**2. Placeholder scan.** Tasks 1, 2, 3, 5, 6, 7, 8 carry complete code. Tasks 4a/4b are explicitly test-driven numerical derivations with the identities stated and the dense-equivalence assertion as the contract — this is intentional and labeled, not a hidden placeholder (writing correct-but-unvalidated determinant code as if it were transcription would be worse). Task 4b Step 1 is a validation spike that gates the rest. Fixtures ("compile a small `Fixed + Besag` model") are described rather than coded because the repo's compile helpers must be reused, not reinvented — the first fixture-building step should grep `tests/` for the existing `compile_lgm(`/panel helpers and copy the smallest.

**3. Type consistency.** `SparseSpdFactor` (Task 2) → used in 4a/4b/5. `_partition_blocks` returns `(sparse_index, dense_index)` (Task 3) → consumed in 4a. `SparseFit(mean, log_marginal_likelihood, predictive_mean, block_slices, diagnostics)` (Task 4a) → consumed by `_fit_sparse` (Task 8). `selected_inverse_diagonal(factor, indices)` (Task 5) matches the spec's seam name. `covariance=None`/`predictive_variance=None` path (Task 6) consumed by `_fit_sparse` (Task 8) and `_rebuild_result` (Task 7). `_exceeds_dense_threshold` shared by `fit_gaussian` and `preflight_dense_reference` (Task 8). Consistent.

**Open risk (surface to the human if it bites):** Task 4b's intrinsic-prior determinant (`base_const` from the structure eigendecomposition) is the one genuinely uncertain identity. The Step-1 spike is the gate; if it cannot be made to match the dense `logdet_prior` for Besag/space-time, stop and escalate — the fallback is to narrow this slice to full-rank fields (proper_car/IID/AR1) now and fold intrinsic-field lml into E-sparse-C with the selected inversion, rather than force a wrong determinant.

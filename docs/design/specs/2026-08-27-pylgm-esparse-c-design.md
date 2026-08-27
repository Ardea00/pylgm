# E-sparse-C: Sparse posterior uncertainty at scale (Full C)

**Status:** design
**Date:** 2026-08-27
**Branch:** `feat/sparse-solver` (stacked on `feat/hybrid-nowcast`), built before merge
**Predecessor:** `2026-08-27-pylgm-sparse-solver-design.md` (E-sparse A+B — sparse
constrained-Gaussian mean/lml/predictive_mean, already shipped on this branch)

## 1. Goal

A+B made the sparse solver produce the posterior **mean**, **log marginal
likelihood**, and **predictive mean** at network scale (200k-node prior
determinant in 87 ms; 5000-region Besag fit in ~70 s). Every posterior
**uncertainty** accessor was deferred behind a `NotImplementedError(_PENDING_C)`
sentinel: `covariance`, `predictive_variance`, `latent_marginals`,
`linear_combinations`, `predict`, plus the effects-layer dense fallbacks
(Sørbye–Rue scaling, BYM2) and INLA hyperparameter integration.

E-sparse-C removes that sentinel. It delivers the **full** posterior-uncertainty
surface on the sparse path — including off-diagonal cross-terms — at scale,
with **no new runtime dependency** (NumPy/SciPy/stdlib only) and deterministically
(no MCMC).

## 2. The design constraint that shapes everything: diagonal vs. off-diagonal

Posterior uncertainty splits into two classes with very different cost at scale:

- **Diagonal quantities** — `latent_marginals` (diag Σ over the `p` latent nodes),
  `predictive_variance` (diag ZΣZᵀ over observations), the Sørbye–Rue scaling
  factor (diag of a pseudo-inverse). These need one number per node/row. We cannot
  afford `p` linear solves, and we cannot materialize the dense `p×p` Σ. They need
  a **selected inverse** — the diagonal (and on-pattern entries) of Q⁻¹ computed
  directly from the factor.

- **Off-diagonal, request-scoped quantities** — `predict` on `k` new rows,
  `linear_combinations` for `k` weight vectors. Each is `z Σ zᵀ` for specific rows,
  which equals solving the posterior precision against just those rows. Cost scales
  with **the request width `k`, not with `p`** — no dense `p×p`.

- **The full `p×p` covariance itself** and any code that stores it (the INLA
  integrator's per-config `cond.covariance`) genuinely cannot scale. Per the scope
  decision (§3), these are **guarded**: available as a dense array only under the
  existing dense-size guard; above it a clear error steers to the scoped accessors.

SciPy ships **no sparse Cholesky** — only `splu` (SuperLU LU). The selected inverse
(Takahashi recursion) needs a *symmetric* factor; SuperLU pivots. §5 resolves this.

## 3. Scope (fixed with the project owner)

**Full C, including off-diagonal**, with the "scoped-exact + guarded full" policy:

| Quantity | Sparse-path behaviour after C |
|---|---|
| `latent_marginals`, `predictive_variance` | exact, at scale (selected inverse + low-rank corrections) |
| `predict`, `linear_combinations` | exact, at scale (targeted factor solves, cost ∝ request width) |
| `covariance` (full `p×p`) | dense array under the dense-size guard; above it, a clear error → scoped accessors |
| Sørbye–Rue scaling | sparse selected inverse on `R*` (replaces dense `eigh`) |
| BYM2 | augmented-Riebler sparse reformulation (replaces dense `pinv`+`eigh`) |
| INLA integration (Gaussian latent) | diagonal accumulation via law of total variance; no dense `p×p` stored |
| INLA non-Gaussian latent *density tables* | selected/per-node solves where terms lie on the fill pattern; guard-bounded fallback otherwise (§6.7) |

## 4. Architecture at a glance

One genuinely new primitive; everything else is composition and wiring.

```
sparse_constrained_gaussian(model)
   ├─ builds a_s (sparse field factor), Schur pieces (b, schur_factor),
   │  kriging pieces (w, cap_factor)            ← already exist in A+B
   └─ returns SparseFit{..., posterior: SparsePosterior}   ← NEW field

SparsePosterior (NEW, sparse.py) — carries the factors, exposes:
   ├─ apply_inverse(rhs)               (moved out of the solver closure; reused)
   ├─ marginal_variances()   → diag(Σ_c) over all p nodes   [selected inverse + corrections]
   ├─ predictive_variances(design)     → diag(design Σ_c designᵀ)  [solves]
   ├─ linear_combination_variances(W)  → diag(W Σ_c Wᵀ)           [solves]
   └─ covariance_dense()               → full p×p Σ_c   [guarded; small models / INLA]

selected_inverse_diagonal(a_s, indices)   ← the existing C seam, body now implemented (Takahashi)

_fit_sparse  stores the SparsePosterior on GaussianResult._sparse_posterior (NEW optional field)

result.py accessors: when _covariance is None but _sparse_posterior is set,
   route to the SparsePosterior methods instead of raising _PENDING_C.
```

The pieces `SparsePosterior` needs — `a_s`, `b`, `schur_factor`, `w`, `cap_factor`,
`sparse_index`, `dense_index`, the constraint matrix `a`, `latent_size` — are all
already computed inside `sparse_constrained_gaussian`. C surfaces them; it does not
recompute them.

## 5. The core primitive: Takahashi selected inversion from a symmetric-mode SuperLU factor

**Validated by spike** (small ring-Besag+ridge SPD precision, `n=60`):

- `splu(Q.tocsc(), permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True),
  diag_pivot_thresh=0.0)` gives `perm_r == perm_c` (a **symmetric** reordering) and
  a unit-lower `L` with `U == D·Lᵀ` to `5e-17` — i.e. a genuine symmetric `LDLᵀ`.
- The Takahashi recursion off `(L, D)` reproduced the inverse **diagonal to `5e-16`**
  against a dense `inv(Q)` reference.

So `SparseSpdFactor` gains a symmetric-mode factorization path (or a second factor
built with these options) exposing `L`, `diag(U)`, and `perm_c`. The Takahashi
recursion computes Σ on the fill pattern of `L`:

```
Given A' = L D Lᵀ (A' = Q permuted symmetrically by perm_c, L unit-lower):
for i = n-1 .. 0 (reverse):
    for j in pattern(column i of L), j >= i (descending):
        Σ'[i,j] = (1/D[i] if i==j else 0) − Σ_{k>i, L[k,i]≠0} L[k,i] Σ'[k,j]
        Σ'[j,i] = Σ'[i,j]
un-permute: Σ = Σ'[inv(perm_c)][:, inv(perm_c)]  (selected entries only)
```

Implemented as a **column sweep vectorized with SciPy sparse ops** over the fill
pattern (not the dense double loop the spike used to prove the math). The result is
the selected inverse of `a_s` on its fill pattern; `selected_inverse_diagonal` returns
its diagonal at requested `indices`.

**Scale ceiling (honest):** the recursion runs in pure NumPy/SciPy — no C kernel.
It is near-linear in `nnz(L)` for the bounded-fill patterns these spatial GMRF
precisions produce, but a pure-Python sweep has a constant-factor ceiling a C
implementation would not. Marked with a `ponytail:` comment naming the ceiling and
the upgrade path (a `scikit-sparse`/CHOLMOD selected inverse *iff* a sparse-Cholesky
dependency is ever admitted). **Task 1 is a spike** that both proves correctness
against a dense reference AND reports wall-time at 5k / 20k nodes, so the ceiling is
sized with real numbers before the rest of the plan commits.

## 6. Composition — every uncertainty quantity from the primitive + reused factors

Let the unconstrained posterior precision be partitioned into the sparse field block
`A = A_ss` (factored by `a_s`), dense fixed block `D` (`m×m`, small), and coupling
`B` (`n_s×m`). Schur complement `S = D − Bᵀ A⁻¹ B` (this is the code's `schur`,
factored as `schur_factor`), and `W = A⁻¹ B` (`m` solves through `a_s`).

Standard partitioned inverse (`Σ = P⁻¹`):

```
Σ_dd = S⁻¹                            (m×m, dense, small)
Σ_ss = A⁻¹ + W S⁻¹ Wᵀ                 (field block)
diag(Σ_ss) = selected_diag(A⁻¹) + diag(W S⁻¹ Wᵀ)
           = Takahashi(a_s)          + einsum over W, S⁻¹      ← m-column low-rank add
diag(Σ_dd) = diag(S⁻¹)
```

**Constrained (Rue–Held) correction.** With constraints `A_c x = e`,
`Σ_c = Σ − w cap⁻¹ wᵀ`, where `w = Σ A_cᵀ` (the solver's existing `w =
apply_inverse(a.T)`) and `cap = A_c Σ A_cᵀ` (the existing `capacitance`, factored as
`cap_factor`):

```
diag(Σ_c) = diag(Σ) − diag(w cap⁻¹ wᵀ)          ← c-column low-rank subtract
```

So **`marginal_variances()`** = `Takahashi(a_s)` placed at `sparse_index`, `diag(S⁻¹)`
placed at `dense_index`, plus the `W S⁻¹ Wᵀ` add on the field block, minus the
`w cap⁻¹ wᵀ` correction. Every term but the Takahashi call reuses a factor the solver
already built.

### 6.1 `latent_marginals(block=None)` (sparse path)
Route to `SparsePosterior.marginal_variances()`, slice by `block_slices`, return
`GaussianMarginals(mean[sel], variances[sel])`. Mirrors `latent_marginals_from` but
sourced from the selected inverse instead of a dense `covariance`.

### 6.2 `predictive_variance` (sparse path)
`diag(Z Σ_c Zᵀ)` over observed rows. `Z` here is the model design (each structured-field
row loads one cell → sparse). Computed via `predictive_variances(Z)`:
`ΣZᵀ = apply_inverse(Z.T)` (through the Schur machinery — handles both blocks exactly),
then `pv_i = Σ_k Z[i,k] (ΣZᵀ)[k,i]`, minus the constraint correction
`diag((Z w) cap⁻¹ (Z w)ᵀ)`. Stored on the `GaussianResult` at fit time (it is a fixed
per-observation vector), so `predictive_variance` returns a value rather than routing live.

### 6.3 `predict(new_data)` (sparse path)
`_BaseResult.predict` currently calls `predict_from(context, mean, covariance, new_data)`,
which does `einsum("ij,jk,ik->i", design, covariance, design)` — needs dense `covariance`.
Add a **sparse-aware branch**: when `_covariance is None` and `_sparse_posterior` is set,
build the design via the same `prediction._design_for(context, new_data)` and offset via
`_offset_for`, then compute `predictive_variance =
_sparse_posterior.predictive_variances(design)` (solve-based, `k` = new-row count).
`fitted_mean` and `predictive_mean` are unchanged (`response_prediction` on `eta`). The
cleanest shape: a `predict_from_sparse(context, mean, sparse_posterior, new_data)` in
`prediction.py` reusing `_design_for`/`_offset_for`/`Prediction`.

### 6.4 `linear_combinations(weights)` (sparse path)
`diag(W_c Σ_c W_cᵀ)` for the requested weight rows `W_c` (`k` rows). Route to
`SparsePosterior.linear_combination_variances(weights)`: `Σ W_cᵀ = apply_inverse(W_cᵀ)`,
`var_i = Σ_k W_c[i,k] (Σ W_cᵀ)[k,i]`, minus `diag((W_c w) cap⁻¹ (W_c w)ᵀ)`, with the
same negative-roundoff clamp `linear_combinations_from` already applies. Accepts the CSR
and dense weight forms the dense path accepts.

### 6.5 `covariance` (full `p×p`, guarded)
The guard decision is made **at fit time** in `_fit_sparse` (which already imports the A+B
guard `_exceeds_dense_threshold` / `_MAX_DENSE_LATENT_DIMENSION`), not live in `result.py`
(`result.py` must not import from `gaussian.py` — wrong layer direction):

- **Under the guard** (small model): `_fit_sparse` eagerly materializes `Σ_c` via
  `SparsePosterior.covariance_dense()` and stores it as the normal `_covariance`. The
  `covariance` property and all accessors then use the existing dense path unchanged — no
  sparse branch needed below the guard.
- **Above the guard** (large model): `_covariance` stays `None`, `_sparse_posterior` is set.
  The `covariance` property, seeing `_covariance is None and _sparse_posterior is not None`,
  raises a clear scale error naming the scoped accessors (`latent_marginals`,
  `predictive_variance`, `linear_combinations`, `predict`), which remain exact and scalable
  because they route to the `SparsePosterior` solve/selected-inverse paths regardless of size.

`covariance_dense()` builds `Σ_c` from `apply_inverse(np.eye(p))` + the kriging correction
(the same construction `_fit_dense` uses, through the sparse factor) — only ever called
under the guard, so `p²` is bounded.

Consequence: the scoped accessors (`marginal_variances`, `predict`, `linear_combinations`)
need their sparse routing in `result.py` only for the above-guard case; below the guard the
dense values are already present. Tests must force the above-guard path (patch the threshold)
to exercise the sparse routing directly.

### 6.6 Sørbye–Rue sparse scaling (`effects/scaling.py`)
`sorbye_rue_scale` currently does dense `eigh(structure)` then
`variances = einsum("ij,j,ij->i", V, 1/λ, V)` (diag of the generalized pseudo-inverse
dropping `null_dim` smallest eigenvalues), and returns `exp(mean(log variances)) *
structure`. For the dominant `null_dim == 1` connected-ICAR case, that diagonal is exactly
the selected inverse of the cofactor `R*₋₀₋₀` (matrix-tree reduction, same identity A+B
already uses for the intrinsic determinant): `diag(pinv(R*))ᵢ` obtained from
`selected_inverse_diagonal` on the sparse reduced factor. Add a `structure_sparse`/
`null_dim==1` fast path that computes the marginal variances sparsely; keep the dense
`eigh` as the fallback for `null_dim >= 2` (RW2) and small structures. Caller
`besag.py:37` is unchanged (same return contract: `factor * structure`).

### 6.7 BYM2 augmented-Riebler sparse reformulation (`effects/bym2.py`)
`bym2_spectrum` does dense `pinv`+`eigh` (O(n³) once at compile) and `bym2_precision`
reassembles a **dense** `n×n` precision per `(τ,φ)`. Replace with the **augmented Riebler
2n parameterization** `(x, u*)`: the model carries both the mixed field `x` and the
scaled structured component `u*`, with a coupled **sparse** precision

```
Q(τ,φ) = τ · [[ 1/(1−φ) I,        −√(φ/(1−φ)) I         ],
              [ −√(φ/(1−φ)) I,     I + φ/(1−φ) R*        ]]
```

(`R*` the Sørbye–Rue-scaled sparse ICAR structure), plus the sum-to-zero constraint on
`u*`. This is block-sparse (`R*` is the graph Laplacian pattern), fits the existing
sparse solver directly, and needs no dense `pinv`/`eigh`. The user-facing effect is the
marginal `x`; `u*` is an auxiliary block. `build_bym2` returns a `LatentBlock` (or a pair)
carrying the 2n design/precision/constraints. Keep the dense spectral path as the
small-graph fallback. **This is the most structurally invasive component** (changes the
latent dimension and block layout for BYM2) — sequenced last among the effects work and
gated by its own dense-equivalence test (marginal variances of `x` must match the dense
BYM2 to tolerance on a small graph).

### 6.8 INLA integration at scale (`optimization/inla.py`)
The integrator accumulates `cov_acc += w*(cond.covariance + outer(m,m))` — a dense `p×p`
per grid point. Reformulate the **latent** accumulation to diagonals via the law of total
variance:

```
var_acc += w * (diag(cond Σ_c) + m²)      # diag(cond Σ_c) = cond._sparse_posterior.marginal_variances()
latent_variance = var_acc − mean²
```

`predictive_variance` already accumulates as a vector (`pv_acc`) — unchanged, sourced from
the scalable per-config `predictive_variance`. `_model_criteria` (line 454) does
`einsum("ij,jk,ik->i", dense, fit.covariance, dense)` per config — replace with the
solve-based `predictive_variances(design)` on the sparse path. The integrated `INLAResult`
stores `covariance=None` and carries the integrated latent-marginal **variances** so
`latent_marginals` (Gaussian) works at scale (a new optional `_latent_variances` vector on
`INLAResult`, used when `_covariance is None` and `_latent_marginal_table is None`).

**Non-Gaussian latent density tables** (`_simplified_laplace_marginals`,
`_full_laplace_marginals`) build a per-node skew-normal / tabulated density and need
per-node cross-covariance terms. Where those terms lie on the fill pattern they come from
the selected inverse; the residual per-node conditional solves are `O(1)` each but `O(p)`
in number. Design decision: provide these tables via per-node selected/on-pattern solves,
and where a term needs an off-pattern entry, **fall back to the dense-guard path** (skew
tables only under the guard; Gaussian latent marginals above it, with a diagnostic). This
keeps every *scaling* quantity exact and confines the one true `O(p)`-solve ceiling to the
non-Gaussian *shape* tables. (Flagged for the owner's confirmation at review — this is the
only place "Full C" meets a genuine per-node cost wall.)

## 7. Files touched

- **Modify** `src/pylgm/inference/sparse.py` — `SparseSpdFactor` symmetric-mode factor +
  Takahashi; implement `selected_inverse_diagonal`; add `SparsePosterior`; add its field to
  `SparseFit`; move `apply_inverse` onto `SparsePosterior`.
- **Modify** `src/pylgm/inference/gaussian.py` — `_fit_sparse` stores `_sparse_posterior`
  and the computed `predictive_variance`; reuse the dense guard for the covariance property.
- **Modify** `src/pylgm/inference/result.py` — `_sparse_posterior` optional field on
  `GaussianResult` (thread through `_init_common` or `GaussianResult.__init__`); route the
  five accessors to it when `_covariance is None`; `INLAResult._latent_variances`.
- **Modify** `src/pylgm/inference/prediction.py` — `predict_from_sparse` reusing the design
  builders.
- **Modify** `src/pylgm/effects/scaling.py` — sparse selected-inverse fast path.
- **Modify** `src/pylgm/effects/bym2.py` — augmented sparse reformulation.
- **Modify** `src/pylgm/optimization/inla.py` — diagonal accumulation; criteria/table paths.
- **Modify** `src/pylgm/inference/model.py` — `_rebuild_result` reorder/align must carry
  `_sparse_posterior`/`predictive_variance` (currently tolerates their absence).
- **Tests** under `tests/inference/`, `tests/effects/`, `tests/optimization/`.

## 8. Testing strategy

Correctness gate throughout is **dense equivalence**: every sparse uncertainty quantity
matches `_fit_dense` / the existing dense INLA path on a small model to tolerance
(`~1e-7` mean/variance, `~1e-6` lml), the same oracle A+B used.

- Takahashi: selected-inverse diagonal vs dense `inv(a_s)` diagonal (`~1e-9`); plus the
  Task-1 spike wall-time report at 5k/20k nodes.
- `marginal_variances` / `predictive_variance`: vs `_fit_dense` on a constrained Besag +
  fixed-effect model (exercises both blocks and the kriging correction).
- `predict` / `linear_combinations`: sparse vs dense on the same model and weights.
- `covariance`: dense-equivalent under the guard; scale error raised above it.
- Sørbye–Rue: sparse vs dense `eigh` factor (`~1e-9`).
- BYM2: marginal variances of `x` from the augmented sparse model vs the dense BYM2.
- INLA: integrated latent/predictive variances sparse vs dense on a small grid.
- Large-model smoke: the 5000-region fit now also returns finite `latent_marginals` /
  `predictive_variance` (no time assertion committed).

## 9. Global constraints (binding on every task)

- **Git identity is `Ardea00 <Ardea00@users.noreply.github.com>` ONLY.** Never `--author`.
  Commit form: `git -c user.name='Ardea00' -c user.email='Ardea00@users.noreply.github.com'
  commit -m "..."`. Verify author AND committer before reporting DONE.
- **No new runtime dependency** — NumPy/SciPy/stdlib only. Deterministic, no MCMC.
- **Ponytail:** reuse the factors the solver already builds; shortest correct diff; every
  deliberate ceiling (pure-Python Takahashi, guarded full covariance, non-Gaussian-table
  fallback) marked with a `ponytail:` comment naming the ceiling and upgrade path; one
  runnable check per non-trivial piece.
- **Preserve A+B behaviour** — mean/lml/predictive_mean and the intrinsic-determinant
  fast path are unchanged; C only fills the deferred uncertainty surface.

## 10. Out of scope / deferred

- A C/CHOLMOD selected-inverse kernel (would need a new dependency; the pure-Python sweep
  is the deliberate ceiling).
- Off-pattern non-Gaussian latent density tables above the dense guard (§6.7 fallback).
- `proper_car` ρ-interval `eigsh` scaling (a separate known limitation, untouched here).
- Cross-block constraints (A+B already raises `NotImplementedError` for these).
```

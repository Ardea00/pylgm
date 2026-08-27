# E-sparse (A+B) — Sparse Constrained Gaussian Solver + Sparse Spatial Builders (Design Spec)

**Slice:** E-sparse, bundle A+B, of the economic-context expansion
(`docs/design/plans/2026-08-26-pylgm-economic-expansion.md`, build-order step 5).

**Goal:** Let every network / space-time model already shipped **build and fit
past the dense preflight guard** (~4096 latent dim / 512 MB) by adding a sparse
constrained-Gaussian solve path and keeping the spatial precisions sparse
through it. The fit — posterior mean, log-determinant, predictions — runs fully
sparse. Posterior *variances* are deferred to E-sparse-C behind a single carved
seam.

## Motivation

`LatentBlock.precision` is already `scipy.sparse.csr_matrix`, and the compiler
assembles a block-diagonal `csr` precision. That sparsity is discarded at two
`.toarray()` calls in the solve (`inference/gaussian.py:113`, `:169`), after
which everything is dense `cho_factor` / `cho_solve`. The dense reference guard
(`preflight_dense_reference`, `gaussian.py:27-46`) then *raises*
`DenseReferenceLimitError` for any model above the threshold — so proper CAR,
Besag, and Knorr-Held space-time models cannot fit at realistic size. This
slice makes the solve consume the sparse precision it is already handed, and
turns the guard from a wall into a switch.

## Scope

In scope:
- A sparse constrained-Gaussian solve path in the exact-Gaussian engine:
  posterior mean, log-determinant, and the empirical-Bayes quadratic form,
  computed from a sparse factorization with **no new runtime dependency**.
- Constraints (sum-to-zero) and fixed effects handled via a small dense Schur
  block on top of the sparse factor (Approach A below).
- `proper_car` ρ-interval via a sparse extreme-eigenvalue solve (`eigsh`) so
  compile stops densifying the adjacency.
- Routing: past the dense threshold, route to the sparse path instead of
  raising; `allow_large_dense=True` still forces dense.
- A carved seam for E-sparse-C: `latent_marginals` variance requests on the
  sparse path raise a clear "pending E-sparse-C" error from a single
  selected-inversion plug-point.

Out of scope (explicitly sequenced):
- **E-sparse-C** (next slice): selected inversion (Takahashi) for sparse
  marginal variances; the sparse Sørbye–Rue scalar; the BYM2 augmented sparse
  reformulation. All three are the same diag-of-inverse math.
- **E-sparse-D** (later): sparse Laplace / GLM path for non-Gaussian
  likelihoods.
- BYM2 at scale — its precision is dense by construction
  (`bym2.py:38` reassembles an `n×n` matrix per `(τ, φ)`); untouched here.

## Approach: partitioned solve (Schur complement)

The constrained solve today reduces the whole system through a **dense**
null-space basis — `reduced_precision = basis.T @ precision @ basis`
(`gaussian.py:125`) — so a sparse factorization is not a `cho_factor`→sparse
drop-in; the solve is restructured. Three options were considered:

- **A. Partitioned solve — big sparse block + small dense block (chosen).**
  Partition the latent field into the structured part `x_s` (fields — large,
  sparse precision) and the dense part `x_d` (fixed effects / intercept —
  small, `m ≪ n`):

  ```
  Q_post = [[ A_s   B  ],     A_s = Q_ss + Z_sᵀ Z_s / σ²   (sparse)
            [ Bᵀ    D  ]]     D   = Q_dd + Z_dᵀ Z_d / σ²   (dense, m×m)
  ```

  Factor `A_s` sparse once. Form the `m×m` Schur complement
  `S = D − Bᵀ A_s⁻¹ B` densely (`m` small) and solve. Sum-to-zero constraints
  fold into the same small dense block as additional low-rank rows
  (conditioning by kriging, Rue & Held 2005 §2.3.3). Log-determinant splits:
  `logdet(Q_post) = logdet(A_s)_sparse + logdet(S)_dense`.

- **B. Factor the whole `Q_post` sparse, constraints post-hoc.** Rejected: the
  intercept (a column of ones) makes `Z_dᵀ Z_d` and its cross terms dense rows,
  so factorization fill-in silently densifies. Every model has an intercept.
  Approach A quarantines exactly that block.

- **C. Iterative (CG + stochastic logdet).** Rejected: the log-determinant
  becomes stochastic/approximate, breaking the deterministic exact
  marginal-likelihood the optimizer requires (global no-MCMC constraint).

**Chosen: A.** Standard INLA structure, exact, deterministic. Honest ceiling:
the dense Schur block scales with the **constraint + fixed-effect count**
(`O(S+T)` for space-time Type IV), not with `n` — far below the latent
dimension it replaces.

## Sparse solve mechanics

- Factorize `A_s` with `scipy.sparse.linalg.splu` / `factorized` (SuperLU).
  This honors the "numpy / scipy / pandas only, no new runtime dependency"
  constraint — `scipy.sparse.linalg` is currently unused in the codebase but is
  already a dependency. Log-determinant from `sum(log|diag(U)|)` (exact for a
  positive-definite system; permutation determinants are ±1 and do not affect
  the magnitude).
- Assemble and factor the small dense Schur block (dense Cholesky, reusing the
  existing `_factor_positive_definite` for the `m×m` piece) for fixed effects
  and constraints.
- Return posterior **mean**, **logdet**, and the quadratic form. This is
  everything the empirical-Bayes inner loop consumes; the fit runs fully sparse
  with no selected inversion.

## Spatial builder changes (A)

Precisions for proper_car / besag / space-time are already sparse `csr` — Sørbye–Rue
only scalar-multiplies the sparse structure (`scaling.py:24` returns
`factor * structure`). So the builder work is narrow:

- **proper_car** (`effects/proper_car.py`): `_validity_interval` replaces the
  dense `np.linalg.eigvalsh` of the normalized adjacency with
  `scipy.sparse.linalg.eigsh(k=1, which="SA")` and `eigsh(k=1, which="LA")` for
  the two extreme eigenvalues. Behavior (the returned `(lower, upper, mu_min,
  mu_max)` and the positive-definiteness gate) is unchanged; only the
  eigenvalue computation stops densifying.
- **besag / space-time**: no precision change — they already emit sparse `csr`.
  The Sørbye–Rue **scalar** stays one-time-dense at compile. Documented ceiling
  (structure eigendecomposition up to a few thousand nodes per component);
  existing `scale=False` opt-out remains the escape hatch for extreme graphs.
- **bym2**: untouched (deferred to C).

## Routing

Turn the dense guard into a switch. Where `preflight_dense_reference` currently
raises `DenseReferenceLimitError` above the threshold, route to the sparse
solve instead. `allow_large_dense=True` still forces the dense path (for
exact-equivalence testing and small-model debugging). No new `engine` value and
no new user-facing flag: a network / space-time model that used to raise now
simply fits.

The dense path remains the default *below* the threshold, so all existing
small-model behavior and tests are unchanged.

## Variance seam for E-sparse-C

The sparse fit returns mean + logdet + predictions. A single function —
`selected_inverse_diagonal(factor, indices)` in the new sparse helper module —
is the plug-point for C. In this slice it raises
`NotImplementedError("sparse posterior variances pending E-sparse-C")`.
`latent_marginals` on a sparse fit surfaces that as a clear error rather than a
silent or half-correct number. C replaces the body with the Takahashi recursion
and nothing else in the call path changes.

## Files

- Modify: `src/pylgm/inference/gaussian.py` — sparse solve path + routing at
  the guard.
- Add: `src/pylgm/inference/sparse.py` — sparse factorization, log-determinant,
  Schur-complement constrained solve (fixed effects + kriging constraints), and
  the `selected_inverse_diagonal` seam. Keeps `gaussian.py` focused.
- Modify: `src/pylgm/effects/proper_car.py` — `eigsh` extreme eigenvalues in
  `_validity_interval`.
- Test: `tests/` — see Testing.

No change to the `LatentBlock` / `CompiledLGM` precision contract.

## Testing

- **Equivalence anchor (correctness).** On a small model *under* the guard,
  forced through the sparse path, the posterior mean and log-determinant match
  the dense path to numerical tolerance (`np.allclose`, tol chosen with margin).
  This is what proves the sparse path correct, not merely fast. Run for a model
  with constraints (Besag / space-time) and one with fixed effects + a field,
  so the Schur/kriging block is exercised.
- **Large-model fit (the deliverable).** A proper-CAR (or space-time) model
  whose latent dimension is past the old dense threshold fits end-to-end
  (`fit` returns, hyperparameters finite, `predict` finite). This is the model
  that raised `DenseReferenceLimitError` before this slice.
- **Routing crossover.** A model just below the threshold takes the dense path;
  the same model scaled just above takes the sparse path; `allow_large_dense=
  True` forces dense above the threshold.
- **Variance seam.** `latent_marginals` on a sparse fit raises the
  "pending E-sparse-C" error; the mean/prediction path does not.
- **proper_car eigsh.** The `eigsh` ρ-interval matches the previous dense
  `eigvalsh` interval to tolerance on an existing small graph (no behavior
  change), and the isolated-node / invalid-ρ errors still raise.

## Global constraints (inherited)

- No new runtime dependency; numpy / scipy / pandas / stdlib only. Deterministic,
  no MCMC. (`scipy.sparse.linalg` is already available via the existing scipy
  dependency.)
- Full suite green; `ruff check src tests` clean (no unused imports — the
  recurring F401 trap in this codebase's tests).
- New tests pass under `-W error::UserWarning`.
- Existing effects, engines, and the optimise/integrate paths stay
  behavior-compatible: the dense path is unchanged below the threshold, and the
  sparse path is exercised only above it (or when explicitly forced in tests).

## Ceilings

- Sørbye–Rue **scalar** left one-time-dense at compile (→ C makes it sparse);
  `scale=False` opts out for extreme graphs.
- Schur block scales with the **constraint + fixed-effect count**, not `n`
  (`O(S+T)` for space-time Type IV) — dense but bounded well below the latent
  dimension.
- Space-time constraint-basis SVD (`_interaction_constraints`) stays
  one-time-dense at compile; a documented ceiling for very large `S·T`.
- Posterior **variances** unavailable on the sparse path until E-sparse-C.
- BYM2 not sparse until E-sparse-C (augmented Riebler reformulation).

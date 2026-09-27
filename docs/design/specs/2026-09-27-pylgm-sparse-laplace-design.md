# E-sparse-D — Sparse Laplace Engine (Design Spec)

**Slice:** E-sparse-D, the non-Gaussian follow-up the sparse-solver spec
(`2026-08-27-pylgm-sparse-solver-design.md`) deferred: "sparse Laplace / GLM
path for non-Gaussian likelihoods".

**Goal:** Fit every non-Gaussian model (Poisson, Binomial, NegBin, survival,
zero-inflated, joint mixtures) **past the dense guard** and at sparse-GMRF cost,
by running the Laplace Newton iteration on the partitioned Schur solver the
exact-Gaussian engine already uses. Exact parity with the dense Laplace engine
below the guard; no new runtime dependency.

## Motivation

The Laplace engine is dense end to end: `_fit_laplace_dense`
(`inference/laplace.py:57`) calls `model.precision.toarray()`, reduces onto a
dense constraint null space (`basis.T @ precision @ basis`) and Cholesky-factors
a dense `p x p` Hessian every Newton step. `fit_laplace` then calls
`preflight_dense_reference`, which **raises** above ~4096 latent nodes instead of
routing anywhere.

Measured against pyINLA (`benchmarks/pyinla/run.py`, Poisson + scaled Besag on
an `L x L` lattice, PC(1, 0.01) prior, `hyperparameters="integrate"`, median of
3; pyINLA 0.1.10):

| areas | pyLGM | pyINLA | ratio |
|---|---|---|---|
| 100 | 0.30 s | 0.20 s | 1.5x |
| 900 | 4.77 s | 0.26 s | 18x |
| 2 500 | 56.1 s | 0.43 s | 130x |
| 4 900 | fails (`DenseReferenceLimitError`) | ok | -- |

The ratio grows like `p^3`: this is the dense factorisation, not the method.
Accuracy is already at parity (hyperparameter posterior means agree to 4
significant figures, fixed-effect sds within ~1%). The Gaussian engine went
through exactly this transition in E-sparse A+B/C and is now at or ahead of
pyINLA on Gaussian panels; this slice gives the non-Gaussian path the same
treatment.

## Scope

In scope:

- A sparse Laplace fit, `_fit_laplace_sparse`, routed to above the dense
  threshold exactly as `_fit_sparse` is for the Gaussian engine
  (`_exceeds_dense_threshold`); `allow_large_dense=True` still forces dense.
- The partitioned Schur solver generalised from a scalar noise variance to
  **per-row weights** -- the one seam the Newton iteration needs.
- Structural constraints (sum-to-zero, label constraints), nonzero rhs, and
  trailing data constraints (`LinearConstraint`), with the same log marginal
  likelihood identities as the dense engine.
- `mean_correction=True` (variational mean shift) on the sparse path.
- `LaplaceResult` carrying a `SparsePosterior`: latent marginals, `predict`,
  `linear_combinations`, `sample` all work unchanged through the existing
  `getattr(self, "_sparse_posterior")` seams.
- Newton **warm start** from a caller-supplied mode, used by the INLA grid and
  the empirical-Bayes search.

Out of scope (sequenced after this slice):

- Simplified-Laplace and full-Laplace latent strategies above the guard. They
  need `cov(x_i, eta_j)` off-diagonals (`inla.py` already guards them with
  `UnsupportedEngineError`); unchanged here.
- Symbolic-factorisation reuse across Newton steps and grid points. SciPy's
  `splu` exposes no analyse/factor split; see *Future work*.
- Cross-block constraint rows (already `NotImplementedError` on the sparse
  Gaussian path, `_block_column_confinement`).

## Approach

Three options were considered.

- **A. Newton on the partitioned Schur solver, with row weights (chosen).**
  One Newton step for the Laplace mode is a heteroscedastic Gaussian solve:
  with `W = diag(w(eta))` the working weights and `g(eta)` the likelihood score,

  ```
  H(x)  = Q + A^T W A
  x_new = argmin_{C x = e}  1/2 (x - x)^T H (x - x) + grad f(x)^T (x - x)
  ```

  i.e. `x_N = H^-1 A^T (W A x + g)` projected onto `C x = e`: the
  `sparse_constrained_gaussian` solve with `1/sigma^2` replaced by `W` and the
  score `Z^T r / sigma^2` replaced by `A^T (W A x + g)`. The Schur partition (big sparse field
  block, small dense fixed/constraint block), kriging for constraints, the
  prior log-determinant decomposition and the selected-inverse variances are
  all reused unchanged.

- **B. Keep the dense reduced Newton, go sparse only for the final
  covariance.** Rejected: the cost is the per-iteration factorisation, which
  stays dense.

- **C. Add CHOLMOD (`scikit-sparse`) as the factoriser.** Rejected *for this
  slice*: a new compiled dependency, and not needed for correctness. It is the
  natural home for symbolic reuse and rank-k updates later (*Future work*).

**Chosen: A.** Exact (same fixed point, same approximation as the dense
engine), deterministic, and it moves the whole non-Gaussian path onto code the
Gaussian engine already exercises.

## The weights seam

`sparse_constrained_gaussian(model)` reads a scalar
`variance = model.likelihood.variance` in five places: `A_ss = Q_ss + Z_s^T
Z_s / var`, `D`, `B`, the score `Z^T r / var`, and the Gaussian quadratic /
normalising terms. Split it:

```
_sparse_posterior(model, weights, score) -> (SparsePosterior, mean,
                                             logdet_posterior_reduced,
                                             logdet_prior, kriging state)
sparse_constrained_gaussian(model)  = _sparse_posterior(model, 1/var, Z^T (y - offset) / var)
                                      + the Gaussian lml terms
```

with `Z^T W Z` formed as `Z.T @ Z.multiply(w[:, None])` (still sparse). The seam
takes the **score**, not a working residual: the IRLS form `r = A x + g / w`
divides by `w`, which underflows to 0 on rows the likelihood has saturated (a
Poisson row with very negative `eta`); `A^T (W A x + g)` never divides. The Gaussian engine calls it with a constant
vector, so its numerics are untouched; this is a pure refactor, verified by the
existing Gaussian suite before any Laplace code lands.

`SparsePosterior` already describes any `Q + Z^T W Z` (it stores `a_ss`,
`a_ss_matrix`, `b`, Schur and kriging factors), so no change there. Its
`predictive_variances` (selected inverse + Schur + vertex-cover fallback) is
what both the mean correction and the reported variances use.

## Newton iteration

Mirrors `_fit_laplace_dense` step for step, so both engines stop at the same
mode:

1. Start at `x0` (warm start) or `0`, projected onto `C x = e` by kriging.
2. At `x`: `eta = A x + offset`, `g = lk.gradient`, `w = lk.working_weights`.
   Build the weighted posterior at `x` via `_sparse_posterior(model, w,
   A^T (w * (A x) + g))`, which returns the constrained Newton target `x_N`;
   `step = x_N - x`.
3. Backtracking Armijo line search on the same objective
   `f(x) = -log p(y | x) + 1/2 x^T Q x`, same constants (`1e-4`, halving, 50
   tries), same `NumericalError` on failure.
4. Convergence: the same max-abs reduced-gradient test and the same
   Newton-decrement rescue on the failure path (`newton_decrement` reported in
   diagnostics), so a mode the dense engine accepts is accepted here.

A non-positive-definite `H` (a non-log-concave likelihood far from the mode)
surfaces as `NumericalError` from `SparseSpdFactor`, exactly as the dense
engine's `_factor_positive_definite` does today.

## Log marginal likelihood at the mode

```
log p(y | theta) ~= log p(y | x*) - 1/2 (x* - nu)^T Q (x* - nu)
                   + 1/2 logdet*(Q | C) - 1/2 logdet*(H(x*) | C)
                   [+ log p(e_D | y)  for data constraints]
                   + normalisation
```

Every term already exists on the sparse Gaussian path: `_prior_logdet`
(block-separable, matrix-tree cofactor for intrinsic fields), the reduced
posterior logdet via `logdet(H) + logdet(C H^-1 C^T) - logdet(C C^T)` from the
QR-factored capacitance, `nu` for nonzero rhs, and the data-constraint Schur
term. Only `log p(y | x*)` is new, and it is the likelihood's own
`log_likelihood` at the mode.

## Results, variances, sampling

- `LaplaceResult` gains `sparse_posterior=` (mirroring `GaussianResult`);
  `covariance=None` above the guard. `_rebuild_result` passes it through.
- `predictive_variance` = `SparsePosterior.predictive_variances(prediction_design)`;
  `fitted_mean` = `likelihood.response_prediction(eta, var)` as today.
- `mean_correction`: `d = 0.5 H^-1 A^T (sigma_eta^2 g3)` needs `sigma_eta^2`
  on observed rows (selected inverse) and one `apply_inverse`: no dense
  covariance.
- `sample()`: `GridSampler(posterior=SparsePosterior)`, as for sparse Gaussian
  fits; draws satisfy every constraint via the existing kriging.

## Warm start

`fit_laplace(model, *, initial_mode=None)` starts Newton at the given latent
vector. Callers:

- `integrate_inla.evaluate` passes the mode of the nearest already-evaluated
  grid point (the grid is explored outward from `theta*`, so this is its
  parent).
- `optimize_empirical_bayes` passes the mode of the previous evaluation.

A warm start changes only the iteration count, never the fixed point; it is
therefore safe to enable by default. This is the largest constant-factor win
after sparsity (INLA does the same).

## Routing

```
fit_laplace(model, allow_large_dense=False)
    allow_large_dense                                  -> dense (explicit opt-in)
    _exceeds_dense_threshold(model) or latent > 1000   -> _fit_laplace_sparse
    otherwise                                          -> _fit_laplace_dense
```

*Revised during implementation.* The first draft routed by the Gaussian
engine's memory guard (~4 096 latents). Measured on Poisson + Besag (both paths
agree to ~3e-11):

| latents | dense | sparse |
|---|---|---|
| 400 | 0.35 s | 0.70 s |
| 900 | 2.44 s | 1.99 s |
| 2 500 | 64.3 s | 4.14 s |

Laplace refactors at every Newton step and grid point, so the crossover is far
below the memory guard; routing at ~1 000 latents recovers the 15x. Consequence:
a Laplace fit above 1 000 latents keeps no dense `covariance` (the property
raises and names the sparse alternatives); marginals, `predict`,
`linear_combinations` and `sample` are unchanged. Joint models
(`joint.py:388, 419`) route through `fit_laplace` and inherit the switch.

## Singular `A_ss`: confounded intrinsic effects (found in implementation)

Two intrinsic blocks whose null spaces are confounded through the design
(Besag + RW1: `v = (1_s, -1_t)` has `Q v = 0` and `Z v = 0`) make
`A_ss = Q_ss + Z_s^T W Z_s` singular *before* the constraints. The dense
engines reduce onto `null(C)` first and are fine; the sparse path is not. This
predates the slice (the sparse Gaussian path raised a pivot `NumericalError`
by round-off luck) and the weights refactor turned it silent -- a mean
violating both sum-to-zero rows. Now: a structural check
(`_require_identified_field`, from the blocks' null-space constraint rows)
raises `UnsupportedEngineError`, with a post-kriging constraint-residual
backstop. Supporting these models sparsely needs a grounded KKT solve (one
anchor node per intrinsic block moved into the dense Schur block, constraint
rows as multipliers) and a sampler that does not need `H^1/2`: a follow-up
slice, **E-sparse-D2**.

### E-sparse-D2 (implemented)

Built differently from the KKT sketch above, reusing every kriging and
determinant identity unchanged:

- **Exact regulariser.** `H_hat = H + U U^T`, with `U^T` the involved blocks'
  own constraint rows. On `{C x = e}`, `x^T U U^T x = ||e_U||^2` is constant, so
  the constrained posterior is unchanged; `N^T U U^T N = 0`, so the reduced
  logdet is unchanged; and `H_hat` is SPD, so kriging applies as is. The null
  basis comes from `_confounded_null_space` (the SVD of `W^1/2 Z_s V`, `V` the
  blocks' null-space constraint rows).
- **Grounding.** `U U^T` is dense on the field, so `k` anchor nodes (pivoted QR
  on the null basis) move to the dense Schur block, leaving `A_s's'` SPD and
  sparse; the prior's `Q_sd`, zero for a block-granular partition, is now
  included in `B`.
- **Woodbury.** The rank-`k` remainder `U_s' U_s'^T` rides on the sparse factor
  (`_LowRankSpdFactor`): SMW solves, determinant-lemma logdet, an exact square
  root `F (I + g g^T)^1/2` for the capacitance QR and sampling, and a
  `w M^-1 w^T` correction to the selected-inverse variances.

Oracle: dense vs sparse on Besag + RW1 (Gaussian and Poisson), RW2 + RW1 on one
index, and the full Knorr-Held type IV model. At scale: Poisson, Besag on 4 900
areas + RW1, 19 600 rows, both precisions estimated by EB: 10 s, constraints
met to 1e-16.

## Testing

Oracles, not smoke tests; every one runs on both paths via the existing
`monkeypatch _exceeds_dense_threshold` fixture pattern.

1. **Sparse vs dense Laplace, below the guard.** Mode, log marginal likelihood,
   latent sds, predictive mean/variance to `1e-8` (sds `rtol 1e-6`, the
   selected-inverse round-off already documented): Poisson + Besag, Binomial +
   RW1, NegBin + IID, Weibull survival, zero-inflated Poisson, a
   `LinearConstraint` data row, a nonzero-rhs label constraint, a two-outcome
   `Joint` with `Shared`.
2. **Gaussian anchor.** A Gaussian likelihood through `_fit_laplace_sparse`
   equals `fit_gaussian` (mode = mean, lml exact): the correctness anchor
   `fit_laplace`'s docstring already promises.
3. **Weights-seam refactor.** The full Gaussian suite unchanged before any
   Laplace code lands.
4. **Warm start.** Same mode and lml from a warm and a cold start; strictly
   fewer Newton iterations from the warm one.
5. **Hyperparameter effectiveness.** A row in
   `tests/test_hyperparameter_effectiveness.py` for a sparse-Laplace fit (the
   "registered, optimised, inert hyperparameter" failure mode this project has
   shipped six times).
6. **Past the guard.** Poisson + Besag on a 70 x 70 lattice fits (it raises
   today) and agrees with pyINLA within the tolerances of the benchmark
   (`benchmarks/pyinla/run.py`).

## Acceptance

- Every test above green on dense and sparse paths.
- `benchmarks/pyinla/run.py`, `poisson_besag`: 4 900 areas fits; 2 500 areas
  within **5x** of pyINLA (from 130x); hyperparameter means within `1e-3`
  relative of pyINLA as today.

## Risks

- **Fill-in from the fixed effects.** Handled as on the Gaussian path: fixed
  columns go to the dense Schur block (`_partition_blocks`), never into `A_ss`.
- **Re-analysis every Newton step.** `splu` redoes the ordering and symbolic
  factorisation each call. At the target sizes that is milliseconds; it is the
  ceiling to watch (`ponytail:` comment at the call site).
- **Near-singular `H` at extreme `theta`.** Same failure surface as the dense
  engine (`NumericalError` -> invalid point in the EB search); the EB message
  now names the root cause (`615d877`).

## Open question (investigate in this slice)

pyLGM and pyINLA fixed-effect **means** on the Poisson + Besag benchmark differ
by 0.007-0.02 (sds agree within 1%, hyperparameters to 4 s.f.) at every size,
so it is not a sparsity artefact. Candidates: INLA's default
simplified-Laplace latent marginals vs our Gaussian ones, or the intercept /
sum-to-zero interaction. Check with `latent_strategy="simplified_laplace"`
(dense, small lattice) before and independently of the sparse work.

## Future work

- **Symbolic reuse / rank-k updates via CHOLMOD** (optional `scikit-sparse`
  extra): analyse once per model, refactor numerically per Newton step and grid
  point; also gives `GaussianResult.update` a refactor path when a vintage is
  as large as the latent field.
- **Sparse simplified/full Laplace strategies**: column-wise `Sigma A^T` via the
  posterior factor instead of the dense covariance.

## Implementation slices

1. Weights seam in `sparse_constrained_gaussian` (pure refactor; Gaussian suite
   is the oracle).
2. `_fit_laplace_sparse` + `LaplaceResult.sparse_posterior` + routing; oracle
   tests 1, 2, 5.
3. Data constraints, nonzero rhs, mean correction, `Joint`; oracle test 1 rows.
4. Warm start through INLA and EB; test 4; rerun the benchmark and record it in
   `docs/comparison.md`.

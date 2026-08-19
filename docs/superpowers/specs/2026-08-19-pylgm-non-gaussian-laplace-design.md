# pyLGM Non-Gaussian Laplace Design

**Status:** Approved 2026-08-19

## Purpose

This milestone adds the first non-Gaussian inference to pyLGM: `Poisson` and
`Bernoulli` likelihoods fit by a Laplace approximation at fixed (plug-in)
hyperparameters. It is the non-Gaussian analogue of the 0.3 conditional
exact-Gaussian engine and the second vertical slice of the
[Latte-parity architecture](2026-08-08-pylgm-latte-pyspark-architecture.md).

It reuses the existing intermediate representation (`CompiledLGM`), canonical
panel, effect builders, constraint handling, and dense factorization helpers. It
does not add distributed inference, hyperparameter optimization, or a new model
language.

## Scope

### Included

- `Poisson` likelihood with a canonical log link.
- `Bernoulli` likelihood with a canonical logit link.
- `LogLink` and `LogitLink` link objects.
- A model-level offset column: a known per-row additive term on the linear
  predictor.
- A Laplace inference engine (`engine="laplace"`): damped Newton to the latent
  posterior mode, the Laplace-Gaussian posterior, and a Laplace log marginal
  likelihood.
- A `LaplaceResult` sharing the `GaussianResult` posterior surface, plus
  response-scale fitted values.
- The YAML frontend for the new families and offset.
- A runnable count-data example.
- Spark parity: `model.fit(spark_df, engine="laplace")` works through the
  existing adapter with a parity test and no new adapter code.

### Excluded (explicit later slices)

- Empirical-Bayes / any hyperparameter optimization over the Laplace marginal.
- Binomial(n) trials and observation weights.
- Non-canonical links (e.g. identity-link Poisson).
- INLA-style hyperparameter integration and marginal corrections.
- `result.predict(new_data)`.
- A unified `LGMResult` type (Gaussian and Laplace results stay parallel for
  now; unification is a later refactor slice).

The design must not preclude these. See **Extensibility** below.

## Public Contract

```python
from pylgm import LGM, Poisson, Bernoulli, Fixed, IID, RW1

model = LGM(
    response="count",
    likelihood=Poisson(),                 # canonical log link
    predictor=Fixed("1 + x") + IID("region_effect", index="region", precision=2.0),
    panel=("region",),
    time="time",
    offset="log_exposure",                # optional; known additive term on eta
)
result = model.fit(frame, engine="laplace")

result.latent_marginals("region_effect")     # exact Gaussian marginals on x
result.linear_combinations(weights)           # as for GaussianResult
result.log_marginal_likelihood                # Laplace approximation
result.predictive_mean                        # posterior mean of eta_i (linear scale)
result.predictive_variance                    # posterior variance of eta_i
result.fitted_mean                            # response scale (see below)
result.converged, result.diagnostics          # Newton iterations, final gradient norm
```

- `engine="laplace"` is required for non-Gaussian likelihoods and is rejected for
  a Gaussian likelihood; `engine="exact_gaussian"` is rejected for a non-Gaussian
  likelihood. There is no silent fallback between engines.
- Bernoulli responses must be `0/1` (or boolean) on observed rows; Poisson
  responses must be non-negative integers on observed rows. Unobserved rows
  (null response) are prediction targets, exactly as for the Gaussian engine.
- Predictions cover every canonical row; Pandas keeps caller order, Spark returns
  canonical order with `prediction_keys`, matching the Gaussian contract.

## Architecture

### Likelihood vocabulary and protocol

`Poisson` and `Bernoulli` are frozen declarative specs, parallel to `Gaussian`.
Each `.materialize(values)` returns a compiled likelihood. To keep the engine
independent of any specific likelihood and to make new families cheap, every
compiled non-Gaussian likelihood implements one small, stable protocol evaluated
on the **observed** linear predictor `eta` and observed response `y`:

```python
class CompiledLikelihood(Protocol):
    link: Link
    def log_likelihood(self, eta: np.ndarray, y: np.ndarray) -> float: ...
    def gradient(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:   # d loglik / d eta
        ...
    def working_weights(self, eta: np.ndarray, y: np.ndarray) -> np.ndarray:  # -d2 loglik / d eta2
        ...
    def response_mean(self, eta: np.ndarray) -> np.ndarray:             # g^{-1}(eta) = E[y|eta]
        ...
    def validate_response(self, y: np.ndarray) -> None: ...
```

- Poisson(log): `mu = exp(eta)`, `gradient = y - mu`, `working_weights = mu`,
  `response_mean = exp(eta)`.
- Bernoulli(logit): `p = sigmoid(eta)`, `gradient = y - p`,
  `working_weights = p*(1-p)`, `response_mean = p`.

Working weights are the canonical-link Fisher information and are `>= 0`, which
keeps the Newton Hessian positive semidefinite; combined with the strictly
positive-definite prior precision on the constrained latent space, the joint
Hessian is positive definite. `CompiledGaussian` is refactored to satisfy the
same protocol (identity link, `gradient = (y-eta)/sigma^2`,
`working_weights = 1/sigma^2`, `response_mean = eta`) so the Laplace kernel is
literally likelihood-agnostic and Gaussian is a checkable special case in tests.

### Links

`links.py` gains `LogLink` and `LogitLink`, each exposing `inverse(eta)` (and,
where needed, a numerically stable form). `IdentityLink` is unchanged. Links hold
no state; they are chosen by the likelihood (canonical only in this slice).

### Offset

`LGM.offset: str | None = None` names a data column. The compiler reads it into
the existing `CompiledLGM` offset vector (currently always zeros); it enters the
linear predictor as `eta = design @ x + offset`. Absent offset stays zeros, so
the Gaussian path is byte-for-byte unchanged. The offset column is added to the
canonical-panel required columns and to the Spark projection.

### Compiler path

A non-Gaussian compile path produces the same `CompiledLGM` structure as the
Gaussian path (shared block assembly, global design/precision/constraints) but
carries a compiled non-Gaussian likelihood and a populated offset. Effect
precisions are plug-in `.initial` values, exactly as today. Block assembly,
label qualification, and constraint stacking are shared with the Gaussian path;
only likelihood construction and offset population differ.

### Laplace engine (`inference/laplace.py`)

`fit_laplace(model: CompiledLGM, *, allow_large_dense=False, max_iterations=100,
tolerance=1e-8) -> LaplaceResult`.

Damped Newton on the latent field, reusing existing helpers:

1. Work in the constraint null space via `_constraint_null_space` (shared with
   the Gaussian engine), so the iteration is unconstrained in reduced
   coordinates `z`, with `x = mu + basis @ z`.
2. At each step, form `eta = design @ x + offset`, restrict to observed rows,
   and query the likelihood for `gradient` and `working_weights`.
3. Newton system in reduced coordinates:
   `H = reduced_precision + reduced_design_obs.T @ diag(W) @ reduced_design_obs`,
   `g = reduced_precision @ z - reduced_design_obs.T @ likelihood.gradient(...)`.
   Factor `H` with the shared `_factor_positive_definite`; step `dz = -H^{-1} g`.
4. Backtracking line search on the joint negative log posterior (prior quadratic
   minus log-likelihood) guarantees a decrease and tames Poisson `exp` overflow;
   all arithmetic under `np.errstate(over="raise", invalid="raise")` converted to
   `NumericalError`.
5. Converge when the reduced gradient infinity-norm `< tolerance`. Exceeding
   `max_iterations` raises `InferenceConvergenceError` carrying the iteration
   count and final gradient norm.
6. At the mode: posterior covariance in reduced coordinates is `H^{-1}`; map back
   with `basis`. Laplace log marginal likelihood:
   `loglik(mode) - 0.5 * z*.T reduced_precision z* + 0.5*logdet(reduced_precision)
   - 0.5*logdet(H)`, using the same Cholesky log-dets the Gaussian engine already
   computes.

Predictions (all canonical rows): `eta_i` posterior mean `= offset + design @ mean`
and variance `= diag(design @ covariance @ design.T)`, reusing the Gaussian
engine's `einsum` path. `fitted_mean` is response scale: for the log link the
exact posterior-predictive mean of a log-normal, `exp(mean_eta + 0.5*var_eta)`;
for the logit link the documented point estimate `sigmoid(mean_eta)` (no closed
form; exactness deferred with quadrature/INLA). The size preflight
(`preflight_dense_reference`) is shared unchanged.

### Result surface (`LaplaceResult`)

Mirrors `GaussianResult` (defensive-copy arrays, `latent_marginals`,
`linear_combinations`, `block_slices`, `log_marginal_likelihood`,
`prediction_keys`, immutable `diagnostics`), so downstream code and the Spark
result-key path are unchanged. Differences: `engine == "laplace"`; `converged`
reflects real Newton convergence; `diagnostics` includes `newton_iterations` and
`final_gradient_norm`; adds a read-only `fitted_mean` (response scale) and
records the link name. `predictive_mean`/`predictive_variance` are the linear
predictor (`eta`) posterior.

To avoid duplicating the substantial marginal/linear-combination logic, the
shared, likelihood-independent posterior surface is factored into a common base
(or shared mixin/helpers) that both `GaussianResult` and `LaplaceResult` reuse;
this is the minimal step toward the target unified result API without committing
to full unification now.

## Extensibility

The design is shaped so deferred work slots in without rework:

- **Hyperparameter optimization (empirical Bayes).** `fit_laplace` returns a
  Laplace `log_marginal_likelihood` and takes a fully materialized `CompiledLGM`,
  exactly like `fit_gaussian`. The existing `optimize_empirical_bayes` wraps
  `fit_gaussian` over a `CompiledGaussianFamily`; a later slice adds a
  non-Gaussian family whose `materialize(parameters)` returns a `CompiledLGM`, and
  reuses the same optimizer with `fit_laplace` as the inner fit. No engine change
  is needed. The likelihood protocol therefore takes a materialized likelihood,
  and the engine never reads hyperparameter metadata.
- **New likelihoods / links.** A new family is a spec + a compiled object
  implementing the five-method protocol; the engine and result are untouched.
  Non-canonical links become a `link=` argument on a likelihood plus a `Link`
  with a derivative; the protocol already routes all link use through the
  likelihood.
- **Weights / Binomial(n).** Add a fixed per-row weight/trials vector to the
  compiled likelihood closure; `gradient`/`working_weights` already return
  per-row arrays, so the engine is unchanged.
- **Unified result.** The shared posterior base is the seam for a future single
  `LGMResult`.

## Errors

- Wrong engine for the likelihood (`laplace` on Gaussian, `exact_gaussian` on
  non-Gaussian): `UnsupportedEngineError` naming the mismatch.
- Invalid response domain (negative/non-integer Poisson, non-`{0,1}` Bernoulli on
  observed rows): `DataContractError` naming the column and rule, raised at
  compile time via `validate_response`.
- Newton non-convergence within `max_iterations`:
  `InferenceConvergenceError(InferenceError)` carrying iterations and final
  gradient norm.
- Non-finite arithmetic / non-PD Hessian: `NumericalError`, as in the Gaussian
  engine.
- Dense size limits: existing `DenseReferenceLimitError` via the shared preflight.

No failed fit is returned as a successful result; `converged` is always truthful.

## Testing and Validation

1. **Likelihood units.** Analytic `gradient`/`working_weights`/`response_mean`
   for Poisson and Bernoulli at hand-checked points; response-domain validation.
2. **Gaussian-as-Laplace anchor.** Running the Laplace kernel on the
   Gaussian-protocol likelihood reproduces `fit_gaussian`'s mean, covariance, and
   log marginal likelihood to tight tolerance (Laplace is exact for a Gaussian
   likelihood). This directly validates the Newton/Laplace machinery.
3. **Mode self-consistency.** At the returned mode the reduced gradient is within
   tolerance; the mode satisfies the score/prior balance.
4. **Fixed-effects-only GLM oracle.** With a single fixed block and a near-flat
   prior, the Poisson/Bernoulli mode matches the GLM MLE. A tiny independent
   in-test Newton solver is the oracle; an optional `statsmodels` parity check is
   guarded by `importorskip` and never added to package dependencies.
5. **Predictions.** Log-link `fitted_mean` equals `exp(mean_eta + var_eta/2)`;
   unobserved rows still receive predictions.
6. **Result contract.** `LaplaceResult` immutability, `latent_marginals`,
   `linear_combinations`, diagnostics, `converged`, and `prediction_keys`
   parity with the Gaussian surface.
7. **Dispatch and errors.** Engine/likelihood mismatch, invalid responses, and
   non-convergence (a pathological low `max_iterations`) raise the specified
   typed errors.
8. **YAML + Spark.** `family: poisson|bernoulli` and `offset:` load to the same
   model as the Python API; a Spark/Pandas Laplace parity test on one small panel.

## Acceptance Criteria

1. `LGM(likelihood=Poisson()/Bernoulli(), offset=...).fit(frame, engine="laplace")`
   returns a converged `LaplaceResult` for a small supported panel.
2. The Laplace kernel reproduces the exact Gaussian fit on a Gaussian likelihood
   within tolerance.
3. Fixed-effects-only Poisson/Bernoulli modes match an independent GLM solve.
4. Engine/likelihood mismatch, invalid responses, and non-convergence raise the
   specified typed errors; no failed fit is returned as success.
5. Pandas-only installs and tests need no new dependencies; Spark Laplace fits
   work through the existing adapter.
6. The likelihood protocol and `fit_laplace` marginal-likelihood surface are
   sufficient for a later empirical-Bayes slice to reuse `optimize_empirical_bayes`
   with no engine changes.

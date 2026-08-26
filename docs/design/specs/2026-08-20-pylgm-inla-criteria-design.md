# pyLGM INLA Model-Assessment Criteria Design

**Status:** Approved 2026-08-20

## Purpose

Add the INLA model-assessment criteria — **DIC, WAIC, CPO, and PIT** — to the
integrated fit produced by `LGM.fit(..., hyperparameters="integrate")`. These
quantify model fit/complexity (DIC, WAIC) and leave-one-out predictive
performance and calibration (CPO, PIT). They are computed from the integrated
hyperparameter posterior (slice 3a) and the per-observation linear-predictor
posteriors, and attached to the result as `result.criteria`.

This is INLA sub-slice **3b**. It builds directly on 3a's grid integration and
adds a per-observation quadrature over each observation's linear-predictor
posterior. It uses the **Gaussian latent strategy** already in place (3c's richer
latent corrections remain deferred).

## Scope

### Included
- Two new per-observation likelihood methods: `pointwise_log_density(eta, y)` and
  `cdf(eta, y)`, on `CompiledGaussian`/`CompiledPoisson`/`CompiledBernoulli`.
- A `ModelCriteria` result object with `dic`, `dic_effective_parameters`,
  `waic`, `waic_effective_parameters`, per-observation `cpo` and `pit` arrays, a
  `cpo_failures` count, and the leave-one-out log-score `log_cpo_sum`.
- A `_model_criteria` helper computing all four by Gauss–Hermite quadrature over
  each observation's η-posterior, folded into the INLA driver's grid loop.
- `INLAResult.criteria` (always populated when INLA runs), docs, and an example.

### Excluded (later sub-slices / non-goals)
- Criteria for the non-integrated (plug-in EB/MAP, or single Laplace/Gaussian)
  fits — this slice attaches criteria only to the INLA-integrated result.
- Randomized PIT for discrete responses (this slice uses the non-randomized
  `P(Y ≤ y)`; documented).
- Richer latent marginal strategies (3c), spatial effects, HMC-Laplace.
- Any new runtime dependency (SciPy is already available).

## Definitions

For observation `i`, the linear predictor `η_i = offset_i + (X x)_i` has, at grid
point `k` (weight `w_k`, conditional fit with latent mean `x̄_k`, covariance
`Σ_k`), a Gaussian posterior `η_i | θ_k, y ~ N(m_{ki}, v_{ki})` with
`m_{ki} = offset_i + (X x̄_k)_i` and `v_{ki} = (X Σ_k Xᵀ)_{ii}`. The full
posterior of `η_i` is the grid mixture `Σ_k w_k N(m_{ki}, v_{ki})`. Let
`p_i(η) = exp(pointwise_log_density(η, y_i))` be the likelihood of `y_i` at
predictor value `η` (using grid point `k`'s materialized likelihood).

Per grid point `k` and observation `i`, integrate over `N(m_{ki}, v_{ki})` by
**Gauss–Hermite quadrature** (nodes/weights `(z_j, ω_j)`,
`η_{kij} = m_{ki} + sqrt(2 v_{ki}) z_j`, normalized weights
`ω̃_j = ω_j / sqrt(π)`):

- `D_{ki} = Σ_j ω̃_j p_i(η_{kij})`            (predictive density)
- `L_{ki} = Σ_j ω̃_j log p_i(η_{kij})`         (mean log-density)
- `Q_{ki} = Σ_j ω̃_j (log p_i(η_{kij}))²`      (mean squared log-density)
- `R_{ki} = Σ_j ω̃_j / p_i(η_{kij})`           (mean reciprocal density)
- `F_{ki} = Σ_j ω̃_j cdf_i(η_{kij}) / p_i(η_{kij})`   (reciprocal-weighted CDF)

Aggregate over the grid (weights `w_k`, and `m̄_i = Σ_k w_k m_{ki}`):

- `lppd_i = log Σ_k w_k D_{ki}`
- `Elog_i = Σ_k w_k L_{ki}`, `Elog2_i = Σ_k w_k Q_{ki}`,
  `Var_i = max(Elog2_i − Elog_i², 0)`
- `Einv_i = Σ_k w_k R_{ki}`, `Ecdf_over_p_i = Σ_k w_k F_{ki}`

**WAIC** (Watanabe–Akaike): `p_WAIC = Σ_i Var_i`,
`WAIC = −2 Σ_i (lppd_i − Var_i)`.

**DIC** (Deviance Information Criterion): posterior mean deviance
`D̄ = −2 Σ_i Elog_i`; deviance at the posterior mean predictor
`D(m̄) = −2 Σ_i pointwise_log_density(m̄_i, y_i)`; effective parameters
`p_D = D̄ − D(m̄)`; `DIC = D̄ + p_D`.

**CPO** (Conditional Predictive Ordinate, leave-one-out): by the harmonic
identity `CPO_i = p(y_i | y_{−i}) = 1 / E_{η_i|y}[1/p_i(η_i)] = 1 / Einv_i`.
`log_cpo_sum = Σ_i log CPO_i` is the LOO log predictive score.

**PIT** (Probability Integral Transform, leave-one-out):
`PIT_i = P(Y_i^{new} ≤ y_i | y_{−i}) = E_{η_i|y}[cdf_i(η_i)/p_i(η_i)] / E_{η_i|y}[1/p_i(η_i)] = Ecdf_over_p_i / Einv_i`.
For discrete responses `cdf_i` is `P(Y ≤ y_i)` (non-randomized PIT; documented).

**CPO reliability.** The reciprocal-density expectation `Einv_i` can be
dominated by a few extreme quadrature contributions (heavy right tail of `1/p`),
making `CPO_i`/`PIT_i` unreliable. Flag observation `i` as a failure when the
single largest `(k, j)` contribution to `Einv_i` exceeds a fraction
`cpo_failure_threshold` (default `0.5`) of the total. `cpo_failures` counts these;
flagged entries still return their (untrustworthy) numeric value.

## Likelihood methods (`likelihoods.py`)

Add to each compiled likelihood:

- `pointwise_log_density(eta, y) -> np.ndarray`: the per-observation summands of
  the existing `log_likelihood`.
  - Gaussian: `−0.5 (log(2π σ²) + (y − η)² / σ²)`.
  - Poisson: `y η − exp(η) − gammaln(y + 1)`.
  - Bernoulli: `y η − softplus(η)`.
- `cdf(eta, y) -> np.ndarray`: `P(Y_i ≤ y_i | η_i)`.
  - Gaussian: `0.5 (1 + erf((y − η) / (σ sqrt 2)))`.
  - Poisson: `scipy.special.gammaincc(floor(y) + 1, exp(η))` (the regularized
    upper incomplete gamma = Poisson CDF).
  - Bernoulli: `where(y ≥ 1, 1.0, 1 − sigmoid(η))` (i.e. `1` at `y=1`, `1−p` at
    `y=0`).

`CompiledGaussian` reuses its `variance`; Poisson/Bernoulli reuse `self.link`.
No new dependency (SciPy `erf`, `gammaincc`, `gammaln` already used).

## Architecture

### Criteria helper (`optimization/inla.py`)

`_model_criteria(design, offset, y, grid) -> ModelCriteria`, where `grid` is the
kept grid points as `(weight, conditional_fit, compiled_likelihood)`. It computes
`m_{ki}`/`v_{ki}` from `design @ fit.mean + offset` and
`diag(design @ fit.covariance @ designᵀ)`, runs the fixed Gauss–Hermite quadrature
(`n_nodes` default `21`, from `numpy.polynomial.hermite_e` or `numpy.polynomial.hermite`),
and accumulates the aggregates above. `design`, `offset`, `y` are constant across
the grid (only the likelihood's parameters and block precisions vary), so they are
extracted once. All arithmetic is guarded by the engine's `np.errstate`
discipline; a non-finite aggregate raises `NumericalError`.

### Driver integration (`optimization/inla.py`)

`integrate_inla`'s `evaluate(u)` additionally returns the materialized
`CompiledLGM` (so the grid loop has each point's `design`/`offset`/`y`/likelihood).
After the mixture accumulation, `_model_criteria` is called once with the kept
points and the constant `design`/`offset`/`y` (taken from the first kept
`CompiledLGM`). The `ModelCriteria` is passed into `INLAResult`.

### Result surface (`inference/result.py`)

`ModelCriteria` is a frozen dataclass (read-only per-observation arrays via
`_readonly_array`). `INLAResult` gains a required `criteria: ModelCriteria`
constructor argument and a `criteria` property returning it. `_rebuild_result`'s
`INLAResult` branch carries `criteria` through (it is not row-dependent, so it is
not reordered). Export `ModelCriteria` from `pylgm.inference`.

### Dispatch (`model.py`)

No `LGM.fit` signature change — `hyperparameters="integrate"` always attaches
`result.criteria`. `_run_inla` is unchanged beyond receiving the criteria-bearing
`INLAResult` from `integrate_inla`.

## Errors

- Non-finite criteria aggregates: `NumericalError`.
- All existing INLA/engine/dispatch errors unchanged.

## Testing and Validation

1. **Likelihood methods.** `pointwise_log_density` sums to the existing
   `log_likelihood`; `cdf` matches `scipy.stats` (`norm.cdf`, `poisson.cdf`,
   and the Bernoulli closed form) at hand-checked points.
2. **Single-grid Gaussian anchor (the crux).** For a one-grid-point (plug-in)
   Gaussian case, the DIC/WAIC ingredients have clean closed forms (log-density
   is quadratic in η) that `_model_criteria` must reproduce:
   `lppd_i = log N(y_i; m_i, v_i + σ²)` (E[p] is the Gaussian convolution);
   `Elog_i = −0.5(log(2πσ²) + ((y_i−m_i)² + v_i)/σ²)`;
   `Var_i = (v_i² + 2 v_i (y_i−m_i)²) / (2σ⁴)`. DIC/WAIC assembled from these.
   CPO and PIT are the harmonic/reweighted leave-one-out quantities
   (`1/E[1/p]`, `E[cdf/p]/E[1/p]`), which are NOT the arithmetic-mean predictive
   `N(y_i; m_i, v_i+σ²)`/`Φ(·)`; anchor them instead against an **independent
   fine-node** Gauss–Hermite reference computed in the test (require `v_i < σ²`
   so `E[1/p]` converges). Both the helper (default nodes) and the fine reference
   use the same harmonic/reweighting definitions, so agreement validates the
   implementation, not a re-derivation.
3. **Quadrature convergence.** Increasing `n_nodes` converges to the DIC/WAIC
   closed forms and to the fine-node CPO/PIT reference; the default `n_nodes`
   matches within tolerance.
4. **Poisson/Bernoulli sanity.** On a small fixture, DIC/WAIC are finite, CPO ∈
   (0, 1] density-scaled values are positive, PIT ∈ [0, 1]; a deliberately
   mis-scaled model has worse (larger) WAIC/DIC than the true one.
5. **Failure flag.** A fixture with `v_i ≫ σ²` (uninformative latent) flags CPO
   failures; a well-behaved fixture flags none.
6. **End-to-end.** `LGM.fit(..., hyperparameters="integrate").criteria` is
   populated for both engines with finite DIC/WAIC and per-observation
   CPO/PIT arrays of length `n_observed`.
7. **Result contract.** `ModelCriteria` immutability/defensive copies;
   `_rebuild_result` carries `criteria` through Pandas realignment and Spark keys.

## Acceptance Criteria

1. `LGM.fit(..., hyperparameters="integrate").criteria` returns a `ModelCriteria`
   with DIC, WAIC (and their effective-parameter counts), per-observation CPO and
   PIT, `cpo_failures`, and `log_cpo_sum`, for both engines.
2. On a single-grid Gaussian case the computed DIC/WAIC/CPO/PIT match the closed
   forms within tolerance; increasing quadrature nodes converges to them.
3. PIT for discrete responses is the documented non-randomized `P(Y ≤ y)`; CPO
   uses the leave-one-out harmonic identity with a documented reliability flag.
4. No new runtime dependencies; full suite green; the `optimize` path, the
   non-criteria result types, and Pandas/Spark contracts are unchanged.
5. CPO/PIT randomization and criteria for non-integrated fits remain deferred and
   recorded.

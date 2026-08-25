# pyLGM Laplace Newton Stall Rescue Design

**Status:** Approved 2026-08-24

## Purpose

Fix the Laplace inner solver's spurious non-convergence, which currently makes
`hyperparameters="integrate"` unusable with a non-Gaussian likelihood for
**every** structured effect. This unblocks integrated inference for Poisson and
Bernoulli models — the case applied count nowcasting needs.

## The defect

`_fit_laplace_dense` declares convergence when `max|∇f| < tolerance` with an
**absolute** `tolerance = 1e-8`. The gradient's scale is set by the data (for a
Poisson model its components are of order the counts), so on many perfectly
well-behaved problems the Newton iteration reaches the mode and then stalls
just above that threshold — taking tiny Armijo-accepted steps that never push
`max|∇f|` below `1e-8`. After `max_iterations` it raises
`InferenceConvergenceError` despite sitting at the optimum.

Under `integrate` this is amplified: a single failing grid point aborts the
whole integration. Measured failure rates over 10 seeds, Poisson + integrate,
on `main`:

| effect | failures |
| --- | --- |
| RW1 | 3/10 |
| IID | 5/10 |
| ProperCAR | 6/10 |
| AR1 | 8/10 |
| BYM2 | 8/10 |

The AR1 slice pinned this as an `xfail`
(`tests/test_ar1_fit.py::test_poisson_integrate_is_currently_unreliable`).

## The fix: the Newton decrement as a stall rescue

The loop already computes, per iteration, `step = H⁻¹(−∇f)` and
`slope = ∇fᵀ·step = −∇fᵀH⁻¹∇f`. The **Newton decrement** `λ² = −slope` is the
textbook scale-invariant convergence measure for Newton's method: it bounds the
suboptimality, `f(z) − f* ≈ λ²/2`. Unlike `max|∇f|` it does not scale with the
data.

**It is applied only where the solver would otherwise raise.** The iteration
loop and its `max|∇f| < tolerance` test are left exactly as they are; at the
point of failure — after the iteration budget is exhausted and the existing
gradient re-check has not rescued it — the decrement is computed at the final
iterate and convergence is accepted when `λ²/2 < tolerance`, i.e. when the
point is demonstrably within `tolerance` of optimal in *objective* terms.

This placement is the whole design, and it was chosen from evidence. Using the
decrement as a *primary* criterion (breaking on it inside the loop) also cures
the stalls, but it stops earlier than the gradient test on well-scaled problems
and so relocates the mode slightly: two existing precision tests
(`test_laplace_poisson_mode_solves_the_score_equation`,
`test_laplace_bernoulli_intercept_matches_logit_of_rate`) fail under that
variant. Confining it to the failure path means **every inner Newton solve that
converges today stops at the identical iterate**, and only genuine stalls change
outcome.

That identity holds at the *solver*; it does not mean user-visible results are
unchanged. `optimize` never surfaced a stall either — `empirical_bayes.py`
catches `InferenceError` and substitutes an invalid objective, so a stalled
hyperparameter was silently treated as infeasible and searched around. With the
rescue those points return real fits, so empirical-Bayes estimates move too.
Measured over a 189-fit matrix: 179 were comparable (10 previously raised), of
which 160 are byte-identical and 19 change — the largest by 4.9 nats of log
marginal likelihood, and one AR1 `rho` posterior mean correcting from −0.9998 to
+0.91 on data simulated with `rho = 0.8`. Where comparable the new fit attains an
equal or higher marginal likelihood, so these are corrections, not regressions.

Verified with the change in place:

- failure rates become **0/10** for all five effects in the table above;
- the full suite is green and the AR1 `xfail` becomes an `XPASS`;
- the two precision tests continue to pass, as they do not reach the rescue.

The extra Hessian factorization the rescue needs is paid only on the path that
was about to raise, so the happy path is unchanged in cost.

## Scope

### Included
- The decrement rescue in `_fit_laplace_dense`'s post-loop failure branch.
- Convert `test_poisson_integrate_is_currently_unreliable` from an `xfail` into
  a real passing test (it is the regression test for this fix).
- A direct unit test that a stalling problem now converges, and that the
  rescued fit really is at the mode (score equation satisfied to a sensible
  tolerance) rather than merely declared converged.
- Docs: remove the "integrate is unreliable for non-Gaussian likelihoods"
  caveats added by the AR1 slice (README AR1 section, the AR1 spec's acceptance
  criterion), and record the fix.

### Excluded (deferred / roadmap)
- **INLA grid robustness** (dropping a failed grid point and renormalising).
  Deliberately not added: with the root cause fixed no grid point fails in
  measurement, and silently dropping points could mask a genuinely broken
  model. Revisit only if failures reappear.
- Relative/adaptive gradient tolerances as the primary criterion, and any
  change to `max_iterations` or the line search. Note a stalled solve still
  burns the full iteration budget before the rescue fires (~26% of inner solves
  in a stalling integrate run); a decrement check gated on gradient stagnation
  *inside* the loop would cut that several-fold without relocating any mode.
- Accepting a stalled *line search* as convergence — a different failure mode,
  still a `NumericalError`.

## Architecture
- `inference/laplace.py`: `_fit_laplace_dense` only. The failure branch gains a
  Hessian factorization, a decrement computation, and the acceptance test; the
  loop body and the convergence semantics of everything that already works are
  untouched.
- No other module changes. `InferenceConvergenceError` keeps its signature and
  is still raised when the point is genuinely not optimal.

## Errors
- A point that is neither gradient-converged nor decrement-converged still
  raises `InferenceConvergenceError(iterations, gradient_norm)`, unchanged.
- A non-positive-definite Hessian at the rescue point surfaces as the existing
  `NumericalError` from `_factor_positive_definite`.

## Testing and validation
1. **The regression case:** the AR1 Poisson + integrate test converts from
   `xfail` to a passing test.
2. **Breadth:** Poisson + integrate succeeds across the structured effects that
   previously failed (RW1, IID, AR1, Besag, BYM2) — at least a representative
   subset in the suite, with the measured before/after rates recorded.
3. **The rescue lands at the mode:** on a problem that stalls, assert the
   returned mode satisfies the score equation to a tolerance appropriate to the
   data scale — i.e. it converged in substance, not just by relabelling.
4. **No behaviour change on the happy path:** the full existing suite passes
   unchanged, including the two precision tests that the primary-criterion
   variant broke.
5. **Genuine non-convergence still raises:** a model given an unreachable
   iteration budget (e.g. `max_iterations=1` on a problem far from its mode)
   still raises `InferenceConvergenceError`.

## Acceptance criteria
1. Poisson/Bernoulli models fit under `hyperparameters="integrate"` reliably;
   the previously-failing cases succeed.
2. Every inner Newton solve that converges today converges to the identical
   iterate — the full suite passes with no test tolerance loosened. User-visible
   non-Gaussian results may change where a stall previously poisoned the
   empirical-Bayes surface or aborted an integration; that is documented.
3. The AR1 `xfail` becomes a real passing test; the docs no longer warn that
   non-Gaussian integrate is unreliable.
4. A genuinely non-converged fit still raises `InferenceConvergenceError`.
5. No new runtime dependency; no changes outside `inference/laplace.py` and the
   tests/docs.

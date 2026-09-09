# pyLGM F4: calibration validation (SBC via PIT)

**Status:** Proposed 2026-09-09
**Branch:** `research-tier` (research-grade; see [research status](../../research-status.md))
**Atlas ref:** F4 — *Validation & benchmarking harness (SBC + cross-framework)*, Tier 1 rank 4
**Refs:** Talts et al., arXiv:1804.06788 · Modrák et al., arXiv:2211.02383 · Rue, Martino & Chopin (2009) §5

## Purpose

Every modifier in `docs/research-status.md` carries the line *"no validation
against published results on real data"*. That is the gap keeping this work off
`main`, and no amount of further feature work closes it. F4 is the gate.

The atlas states F4 as two halves. They are not in the same state:

- **Cross-framework** is *partly shipped*. `tests/integration/test_mcmc_crosscheck.py`
  already establishes the convention: a frozen NUTS reference committed as JSON
  (`examples/joint_mcmc_crosscheck/nuts_reference.json`), PyMC needed only to
  regenerate it, and — importantly — a written statement of what is and is not
  asserted, because pyLGM reports a joint mode where MCMC reports marginal means.
- **SBC does not exist.** Nothing in the repo checks that a reported posterior is
  *calibrated* rather than merely *plausible*.

This spec covers the missing half. It reuses the crosscheck's conventions rather
than introducing a second validation framework.

## The reframing: PIT, not ranks

Talts et al. define SBC for *sampling* algorithms: draw `θ̃ ~ p(θ)`, draw
`ỹ ~ p(y|θ̃)`, fit, and compute the rank of `θ̃` among `L` posterior draws. Under
exact inference that rank is Discrete-Uniform on `{0,…,L}`.

pyLGM does not produce draws. It produces marginal densities, and every marginal
representation on the result surface already exposes a CDF or can trivially be
given one. So use the probability integral transform directly:

    u_r = F_r(x_r^true),    F_r = the reported posterior marginal CDF for replicate r

Under exact inference with `x^true` drawn from the prior, `u ~ Uniform(0,1)`
exactly. This is the `L → ∞` limit of the SBC rank statistic, and for a
deterministic engine it is strictly better:

- no Monte-Carlo noise enters the statistic, so no `L` posterior draws to take
  and no binning of ranks — a factor-`L` saving and a sharper test;
- continuous uniformity admits an exact one-sample KS test instead of a
  chi-square on rank bins;
- it is reproducible bit-for-bit, which matters because this harness is meant to
  *gate* approximation choices across releases.

**Honest limitation.** Uniformity of a scalar marginal PIT is necessary, not
sufficient, for joint correctness: a posterior can be marginally calibrated and
jointly wrong. This tests each reported marginal, which is what pyLGM actually
reports and what users actually read. Joint calibration is out of scope
(see *Out of scope*).

## Independence structure (the part easy to get wrong)

Within one replicate the `p` latent components are *dependent* — they share one
dataset. Pooling all `R × p` PIT values into a single KS test with `R·p` degrees
of freedom overstates the evidence by roughly the within-replicate correlation
and will manufacture significance from a correctly calibrated engine.

PIT values are independent **across replicates, at a fixed component index**.
So the harness tests per index:

    for each tracked latent index i:  KS({u_{r,i} : r = 1..R}, Uniform(0,1))

with a Bonferroni (or Benjamini–Hochberg) correction over the tracked indices.
Track a small fixed set of indices — the intercept, the first, middle and last
field node — rather than all `p`. This keeps `R` as the only sample-size knob
and keeps the multiplicity correction mild.

## Prerequisites found in the code

Three things block a naive implementation. All were verified against the source,
not assumed.

### 1. `GaussianMarginals` has no `cdf`

`src/pylgm/inference/result.py`:

| class | `quantile` | `cdf` |
|---|---|---|
| `GaussianMarginals` (:278) | yes (:306) | **missing** |
| `SkewNormalMarginals` (:313) | yes (:396) | yes (:390) |
| `TabulatedMarginals` (:419) | yes (:514) | yes (:506) |

The `LatentMarginals` protocol (:170) declares `mean/variance/std/quantile` and
omits `cdf` — so the CDF is available on two of three implementations by
accident rather than by contract.

**And the two that have it disagree.** `SkewNormalMarginals.cdf(x)` is
*elementwise*, returning `F_i(x_i)` with shape `(p,)`; `TabulatedMarginals.cdf(x)`
returns the *cross product* `F_i(x_j)` with shape `(p, len(x))`. Verified by
running both. Generic code written against the protocol will be silently wrong
against one of them. This spec does **not** fix it — that is a behaviour change to
a shipped, tested class and belongs in its own slice — but the new
`GaussianMarginals.cdf` adopts the elementwise convention (2 of 3), and the
harness normalises the shapes in one place with the inconsistency named in a
comment.

**Change:** add `cdf` to `GaussianMarginals` (`norm.cdf((x - mean)/std)`, mirroring
`quantile`'s validation) and add `def cdf(self, x: np.ndarray) -> np.ndarray: ...`
to the protocol. Roughly four lines plus a protocol row. This is the only change
this spec makes to shipped inference code, and it is additive.

### 2. Priors have no sampler

`src/pylgm/priors.py` gives `GaussianPrior`, `PCPrecision` and `PCBYM2Phi` a
`logpdf` and nothing else. Phase 2 needs `sample(rng)`:

- `GaussianPrior` — `rng.normal`, trivial.
- `PCPrecision` — the PC construction is `Exponential(λ)` on the KLD distance
  `d = τ^{-1/2}`, with the rate in closed form in `logpdf` (:93),
  `λ = −log(α)/upper_sd`. Inverting gives `τ = (−log U / λ)^{-2}` directly; the
  existing `logpdf` is exactly `Exp(λ)` on `d` times the Jacobian `½τ^{-3/2}`,
  which is worth asserting in a test since it is the identity being relied on.
- `PCBYM2Phi` — no closed form. `_BoundPCBYM2Phi` solves its rate numerically
  (`_solve_pc_rate`, :199) and exposes `distance` (:171) and
  `distance_derivative` (:174), so inverse-CDF by Newton on the existing
  derivative, bisection-guarded. This is the only non-trivial one.

Phase 1 does not need any of these (see below), so this work is deferred to the
slice that needs it.

### 3. Hyperparameter marginals are collapsed to a Gaussian on the natural scale

`src/pylgm/optimization/inla.py` accumulates only the first two moments over the
integration grid (:365–366) and reports

```python
hyper_marginals = {name: GaussianMarginals(mean, variance) for name in names}
```

with `theta[name]` on the **natural** scale (`from_internal`, :275). The weighted
grid points themselves are discarded.

For a precision `τ > 0` under a PC prior the posterior is strongly right-skewed;
a Gaussian moment-match on the natural scale puts mass on `τ < 0` and mis-states
both tails. Its PIT will be visibly non-uniform **even if the INLA integration is
perfect** — a reporting artefact, not an inference error. Running hyperparameter
SBC before fixing this produces a red result that says nothing about the engine.

Two candidate fixes, both larger than they look:

1. *Retain the grid.* The accumulation loop already has `(theta, w)` per point;
   keeping them and exposing a weighted/tabulated θ marginal is a few lines
   there — but `hyperparameter_marginals()` is typed `Mapping[str, GaussianMarginals]`
   and enforced at runtime (`result.py:998`), so it is a public typed break.
2. *Report on the internal scale*, where Gaussianity is far more defensible.
   Cheaper, but changes the meaning of a published number.

Neither is sufficient on its own: `_build_grid` (:241) lays out `O(3^d)` points
sized for *integration accuracy of the latent marginals*, not for CDF resolution
of θ. A three-points-per-dimension weighted CDF is far too coarse to feed a KS
test.

**Consequence for the roadmap:** F4's hyperparameter half depends on F5 (smart
hyperparameter grid / adaptive integration), or on a dedicated 1-D refinement of
the θ marginal. The atlas lists F4 and F5 as independent; they are not. This is
the main finding of this spec and the reason for the phasing below.

## Scope, in three phases

### Phase 1 — latent calibration at fixed θ *(this slice)*

Fix `θ` at a known value, so the target is the conditional posterior `p(x | y, θ)`
and the *only* thing under test is the latent approximation:

1. compile the model once via `compile_lgm(model, panel)`;
2. draw `x ~ N(0, Q(θ)^{-1})` from the compiled block precisions;
3. form `η = A x`, draw `y ~ p(y | η)` from the likelihood;
4. fit with `hyperparameters="optimize"` and every hyperparameter pinned;
5. `u = result.latent_marginals().cdf(x_true)` at the tracked indices.

This needs no prior sampler and does not touch the θ-marginal problem. It is
also the phase with the most diagnostic value per unit of work, because it
isolates one approximation at a time:

- **Gaussian likelihood + `latent_strategy="gaussian"`** — the posterior is exact,
  so the PIT is uniform *to machine precision*. This is the harness's own
  self-test: if the simulator, the indexing, or the CDF is wrong, this case
  fails, and it cannot fail for statistical reasons. Build it first.
- **Poisson / Binomial** across `latent_strategy ∈ {gaussian, simplified_laplace, laplace}` —
  this measures what the Laplace family actually buys, which is precisely the
  "natural gate for every approximation choice" the atlas asks for. Expect
  `gaussian` to be detectably miscalibrated in the tails at small counts and
  `simplified_laplace` to be better; recording *how much* better is the point.

### Phase 1b — intrinsic and constrained effects *(done)*

Delivered, and more cheaply than this spec assumed. The spec proposed sampling
the proper part and then conditioning by kriging with a pseudo-inverse. That is
unnecessary: **both engines already reparametrise onto the null space of the
constraint matrix** (`inference/gaussian.py:155`, `inference/laplace.py:40`),
setting `x = x_p + B z` with `B` orthonormal, giving `z` the precision `BᵀQB` and,
for a nonzero right-hand side, the prior mean `−(BᵀQB)⁻¹BᵀQx_p`.

Sampling `z` from exactly that and mapping back makes the simulator agree with
the engine *by construction* rather than by argument — which is the property that
matters, since any disagreement would surface as miscalibration with no way to
attribute it. It also collapses the two cases into one code path: unconstrained,
`B` is the identity and the expression is the ordinary `N(0, Q⁻¹)`.

`BᵀQB` is positive definite exactly when the constraints span `Q`'s null space,
which is the library-wide invariant the M1 slice-4 spec established; when they do
not, the Cholesky fails and the prior genuinely is improper, which is the error
reported. User-supplied `extraconstr` rows, nonzero rhs included, ride the same
path.

One correction to the effect classification: **`BYM2` is not intrinsic.** Its
phi-mixture of a scaled Besag and an IID part is proper by construction — full
rank, zero constraints — so it was already sampleable in phase 1. `RW1`, `RW2`
and `Besag` are the genuinely rank-deficient ones.

### Phase 2 — full SBC over the joint prior *(done)*

`hyperparameters="integrate"`, `θ̃` drawn from the declared priors, latent PIT
only. This is honest SBC: under empirical Bayes the latent posterior is
conditional on `θ̂` rather than marginalised, so the joint-prior version is
*expected* to fail under `"optimize"` — EB is not Bayes, and reporting that as a
calibration defect would be a category error.

**Prerequisite 2 dissolved.** The spec called for a `sample(rng)` on each of the
three prior classes, with `PCBYM2Phi` needing Newton on its own
`distance_derivative`. None of that was written. A numeric inverse CDF over
`logpdf`, laid out on the parameter's *internal* scale with the Jacobian
included, is shorter than one bespoke sampler and strictly better:

- it truncates to the declared `lower`/`upper` for free, which is required for
  correctness — the engine only ever searches inside the bounds, so the prior it
  uses *is* the truncated one, and drawing from the untruncated prior would put
  mass where the posterior cannot follow;
- it works for any prior with a `logpdf`, including a user's own and the
  graph-bound `PCBYM2Phi` that reaches the family already bound.

`priors.py` is untouched. The grid is built once per hyperparameter rather than
per draw — `logpdf` is scalar, and rebuilding a 4001-point grid inside the
replicate loop costs more than every fit in that loop.

**The finding.** Phase 2 immediately caught a real, attributable miscalibration:
under `hyperparameters="integrate"`, the true latent marginal is a mixture over
the θ grid, and `latent_strategy="gaussian"` reports only its first two moments.
A moment-matched Gaussian has the right variance and the wrong shape, and the
dispersion statistic sees it while the location statistic does not. At 512
replicates on a Gaussian-likelihood IID model, worst p(dispersion): `gaussian`
4e-10, `simplified_laplace` 9e-07, `laplace` 7e-07.

Attributed rather than asserted, by two controls:

1. **Tightening the hyperparameter's bounds** so the mixture collapses to
   essentially one component restores calibration — which exonerates the prior
   draw, the per-replicate rebuild of `Q(θ)`, and the integrate path, since a bug
   in any of those would not care how wide the prior is.
2. **Refining the integration grid** (`grid_step` 1.0 → 0.4, `radius` 3 → 8)
   changes nothing: 2.6e-05 → 1.3e-05 for `gaussian`, 0.0021 → 0.0021 for
   `simplified_laplace`. So the residual is the marginal *representation*, not
   integration accuracy.

Control 2 matters for the roadmap: this result does **not** argue for F5. A
smarter hyperparameter grid would not have fixed it.

### Phase 3 — hyperparameter PIT

Blocked on prerequisite 3 **and** on F5. Specified here so it is not rediscovered;
not scheduled.

Phase 2's finding sharpens the case for it. The same Gaussian-moment collapse
that miscalibrates the *latent* marginals under `latent_strategy="gaussian"` is
applied unconditionally to the *hyperparameter* marginals, on the natural scale,
where the skew is worse. Phase 2 measured the size of that error for the latent
field; there is no reason to expect the hyperparameter version to be smaller, and
currently no way to check it.

## Architecture

One module and one test file. No new dependency — numpy, scipy and pytest cover it.

```
src/pylgm/validation.py             # flat module; a package can wait for phase 2
tests/validation/test_sbc.py
```

The module is small and deliberately so:

```
simulate_latent(compiled, rng)      -> x            # N(0, Q^-1) from block precisions
simulate_response(compiled, x, rng) -> y            # draw from the likelihood at eta = A x
pit(marginals, x_true, indices)     -> u            # F(x_true) at tracked indices
calibration_report(u, indices)      -> per-index KS statistic, p-value, corrected verdict
```

`calibration_report` returns a dataclass; it does not assert. Tests assert, and
the reporting path is reusable from a notebook for the "how much better is
simplified_laplace" question, which is a measurement rather than a pass/fail.

**Where the module lives.** Shipped in `src/`, not kept in `tests/`: letting users
calibrate *their own* model is how the atlas pitches F4 ("buys credibility"), and
a validation tool nobody outside the test suite can run buys none. The cost is an
API commitment, which is paid down by keeping the surface to one obvious entry
point — `calibrate(model, frame)` — with the pieces exported for anyone who wants
to assemble the loop themselves.

## What this cannot detect

The simulator draws from the **compiled IR** — the same `Q` and `A` the engine
then conditions on. That is deliberate: a separately hand-written generator that
disagreed with the IR would show up as miscalibration with no way to tell which
side was wrong, which is the classic way SBC harnesses waste a week.

The cost is that **an error in IR assembly is invisible here**: if `Q` is built
wrong, both sides are wrong identically and the PIT stays uniform. That is
covered elsewhere and must stay covered — the effect oracle tests
(`tests/test_grouped_spacetime_oracle.py`, `tests/effects/test_sorbye_scaling.py`,
and the Kronecker null-space identity from the M1 slice-4 spec) pin `Q` against
closed-form or R-INLA-derived values. Clean division of labour: **oracles pin the
model, SBC pins the inference given the model.** Neither substitutes for the other,
and the docstring must say so — `test_mcmc_crosscheck.py` sets the precedent for
stating non-assertions explicitly.

The handoff's standing warning applies with full force here: a calibration test
that passes because it is vacuous is worse than none. Every phase-1 case must be
verified by **mutating the implementation and confirming the test fails** —
inflate a reported variance by 10%, shift a mean by 0.1σ, and confirm the KS test
catches both at the chosen `R`.

## Testing and CI budget

SBC cost is `R` fits per case, and `R` drives the power of the KS test. Budget:

- `R = 256` for the CI tier, with small `n` per replicate (`n ≈ 30–60`) — these are
  calibration tests, not scale tests. The whole file runs in ~5s.
- `R = 512+` for the recorded runs behind any documented number.

**Two statistics, not one — measured, not assumed.** An earlier draft of this spec
claimed `R = 128` catches a 10% variance error "with high probability". That is
false, and the implementation measured it: against a 10% inflated reported SD,
plain KS has ≈0.01 power even at `R = 1024`. A dispersion error leaves the PIT
median at 0.5 and pushes mass symmetrically into both tails; KS is a
maximum-CDF-deviation statistic and barely sees it. Cramér–von Mises is no better
(≈0.01 at `R = 1024`).

Folding the PIT about its centre — testing `|u − 0.5| · 2`, which is Uniform(0,1)
whenever `u` is — converts that symmetric deviation into the one-sided kind KS
detects (≈0.51 power at `R = 1024`). So each component gets **both**: KS on `u`
for location, KS on the folded values for dispersion, Bonferroni over `2 ×
n_components`. Measured behaviour at `R = 256`:

| corruption | p(location) | p(dispersion) |
|---|---|---|
| none (control) | 0.031 | 0.044 |
| mean +0.1σ | **0.0004** | 0.039 |
| SD ×0.85 | 0.015 | **2.4e-06** |
| SD ×1.15 | 0.0065 | **0.0008** |

Each failure mode is caught by exactly one of the two, which is why both ship.

Seeds are fixed and the per-case seed recorded in the report, so a failure is
reproducible in one line. Follow the crosscheck's precedent for anything
expensive: commit the resulting PIT table as JSON under `examples/` and let CI
re-run only the cheap tier.

Assertions are on the KS p-value against a fixed threshold, *not* on the PIT
values themselves — the latter would be a snapshot test of numerical noise and
would break on every LAPACK change, which `3342150` already had to fix once
elsewhere in this repo.

## Found while implementing phase 1

Three things the spec did not anticipate, all of which changed the code:

1. **Empirical Bayes silently invalidates the check.** A model declaring a
   `Hyperparameter` simulates its latent field from `Q` at that parameter's
   `initial`, then re-estimates it from each simulated dataset — so the PIT
   describes a posterior conditional on `θ̂(y)`, not on the `θ` that generated
   the data. It does not blow up; it returns `PASS` with unremarkable p-values
   while measuring a different quantity. `calibrate` now rejects a declared
   `Hyperparameter` outright. This is the phase-1/phase-2 boundary made concrete,
   and it is the kind of green-for-the-wrong-reason result the handoff warns about.
2. **Two tests, not one** — see the CI budget section. Plain KS cannot see a
   dispersion error.
3. **The `cdf` convention is inconsistent between shipped classes** — see the
   prerequisites section.

The exact-Gaussian self-test earned its place: it is what caught the shape
mismatch between the elementwise and cross-product `cdf` conventions, because it
is the one case whose correct answer is known a priori.

## Rejections

- **Rank-based SBC with posterior draws.** Requires inventing a sampler for a
  library whose entire point is not sampling, adds `L`× cost, and adds Monte-Carlo
  noise to the statistic. PIT is the same test in the limit.
- **A general benchmarking framework.** The atlas phrases F4 as "harness", which
  invites a plugin architecture for models × engines × frameworks. Not built:
  phase 1 is a parametrised pytest case, and a parametrised pytest case remains
  the right answer until there is a second consumer.
- **Comparing against R-INLA as the primary reference.** Already argued and
  rejected in `test_mcmc_crosscheck.py`'s docstring: agreeing with R-INLA shows
  only that pyLGM reproduces another Laplace approximation of the same family.
  SBC has no reference implementation at all, which is why it is worth more here.
- **Fixing the hyperparameter marginal in this slice.** It is a public typed break
  that does not deliver a working hyperparameter PIT anyway without F5.

## Out of scope

- Joint (multivariate) calibration; only scalar marginals are tested.
- Posterior predictive calibration on the response scale (would need randomised
  PIT for discrete `y`).
- `Joint` models.
- Any change to `hyperparameter_marginals()` or to the INLA grid.

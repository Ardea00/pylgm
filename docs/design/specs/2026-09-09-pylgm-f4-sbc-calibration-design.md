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

**And the two that had it disagreed.** `SkewNormalMarginals.cdf(x)` was
*elementwise*, returning `F_i(x_i)` with shape `(p,)`; `TabulatedMarginals.cdf(x)`
returned the *cross product* `F_i(x_j)` with shape `(p, len(x))`. Generic code
written against the protocol would be silently wrong against one of them — it
would read component `j`'s CDF where component `i`'s was meant.

**Since reconciled.** `TabulatedMarginals.pdf` and `.cdf` are now elementwise
too, so all three representations are interchangeable behind the protocol, and
`pit` calls `cdf` directly instead of normalising shapes. The elementwise
convention won because two of three classes already used it, because
`SkewNormalMarginals.quantile` depends on it internally, and because it is what
the rest of the protocol implies — `mean`, `variance`, `std` and `quantile` are
all per-component `(p,)`.

The cross product had no callers in `src/` or `tests/`; for the plotting case it
served, `TabulatedMarginals` exposes `.x` and `.density` directly, which *is* the
tabulation. Mismatched input is now a `ValueError` rather than a silently
differently-shaped result.

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

### Phase 3 — hyperparameter PIT *(done, and it fails)*

Delivered: the drawn `θ̃` is PIT'd against its own reported marginal, listed as
`hyper:<name>`. Two things had to change first.

**Prerequisite 3, resolved.** `hyperparameter_marginals()` no longer collapses
the θ posterior to two moments. With one hyperparameter the grid is a line in
`u`, and `s(u) = log p(y|θ) + log π(θ)` is the unnormalised log posterior
evaluated on it, so splining that log density and mapping it through the
transform gives the marginal itself as a `TabulatedMarginals`. Every grid point
feeds it, not only the ones the integration weights retain — they are all
evaluated by the time the marginals are built, and the `log_density_drop` filter
would truncate the tails at ≈2.2σ instead of the grid's own 3. With more than one
hyperparameter the lattice is rotated onto the whitened Hessian's directions, so
single-axis projections scatter and no marginal can be read off it; those keep
the moment match. The runtime type check widened from `GaussianMarginals` to the
`LatentMarginals` protocol, which is what callers actually use.

Verified against a brute-force reference — the exact log posterior on a dense θ
grid, no INLA in the loop. Median relative quantile error across datasets:

| | grid covers (5/30) | grid truncates (25/30) |
|---|---|---|
| tabulated | **0.038** | **0.445** |
| moment-matched | 0.505 | 1.899 |

Better in both regimes, by 13× and 4×. On a single well-identified dataset the
2.5% credible bound moves from 0.238 to 0.471 against a reference of 0.468 — the
old collapse was wrong by a factor of two at the lower end, because it assumes
away a posterior skewness of 1.08.

**The finding: hyperparameter marginals are not calibrated, and the cause is the
grid, not the summary.** PIT mean 0.61 rather than 0.5, KS p ≈ 1e-11. Attributed
in three steps rather than asserted:

1. **The harness is exonerated.** Re-running the same SBC loop against a
   brute-force reference posterior — same prior draws, same simulated data, exact
   posterior in place of the engine's — gives a uniform PIT (mean 0.480, p 0.15).
   So the prior sampler, the truncation to bounds, and the simulation are right.
   *(This needed care: a first attempt integrated the reference over a wider
   support than the sampler's bounds and produced a spurious deviation. The
   reference must use the same truncated prior the engine does.)*
2. **The direction is one-sided.** The grid's upper edge reaches only 0.139× the
   reference's 97.5th percentile and fails to cover it in 85% of datasets, while
   covering the lower tail comfortably (0% failure). The reported mean is 0.28×
   the reference mean.
3. **The summary is not the culprit.** Swapping the tabulated marginal back for
   the moment match barely moves the PIT (mean 0.60 vs 0.62; both p < 1e-8).
   Fixing the shape cannot fix missing mass.

The mechanism: the grid is centred on the empirical-Bayes mode and scaled by the
curvature of `s` there, spanning ±3 of those units. For a weakly identified
precision the log posterior has a long right tail that a mode-centred, curvature-
scaled grid does not reach.

**This does argue for F5** — and corrects what phase 2 concluded. There, refining
the grid changed nothing, so the latent-marginal error was attributed to the
summary and F5 ruled out. That holds for the *latent* marginals and does not
generalise: for the hyperparameter marginal the grid *is* the representation, and
its coverage is the whole error. An adaptive integration scheme is the fix;
widening `radius` is not, since the cost is `(2r+1)^d`.

**Since fixed — see the F5 slice below.**

## F5 (first slice): adaptive grid extent

The grid built the full `(2r+1)^d` lattice at a fixed `radius=3`, evaluated every
point, then discarded whatever fell below `log_density_drop`. Both halves are
wrong: it truncates a posterior wider than three curvature units, *and* it pays a
conditional fit for every point it then throws away.

Replaced with the exploration of Rue, Martino & Chopin (2009, §3.1): step outward
along each whitened direction from the mode and stop when the log density has
fallen `explore_drop` below it, so the extent is set by the posterior rather than
by a guess. Axis probes are cached and reused as grid points.

Measured on the 10-group IID model that exposed the problem (30 datasets against
brute-force reference posteriors):

| | fixed radius | adaptive |
|---|---|---|
| grid edge ÷ true 97.5th pct | 0.18 | **1.37** |
| covers the true 97.5th pct | 17% | **63%** |
| reported mean ÷ true mean | 0.32 | **0.86** |
| median relative quantile error | 0.43 | **0.067** |
| calibration PIT mean | 0.62 | **0.53** |
| calibration p(location) | 9e-09 | **0.41** |

**Phase 3's check now passes**, and its test asserts calibration rather than
documenting the failure — it is a regression test for the grid.

**Depth is gated by what consumes it.** Integration weights drop everything below
`log_density_drop`, so exploring past that buys them nothing, and at `d > 1` the
extra depth would cost `(2r+1)^d` fits for points immediately discarded. The one
consumer that wants the tails is the tabulated hyperparameter marginal, which
exists only for a single hyperparameter. So the depth is `explore_drop` at `d = 1`
and `log_density_drop` beyond it. Conditional fits per `integrate` fit:

| hyperparameters | fixed radius | adaptive |
|---|---|---|
| 1 | 30 | 27 |
| 2 | 60 | 85 |
| 3 | 360 | 307 |

Comparable, and wider only where the posterior genuinely is.

**Integration is untouched.** The `kept` set at `log_density_drop` is identical,
so weights, latent marginals, `log_marginal_likelihood` and criteria do not move:
the surface baseline changes 20 leaves, all of them `hyperparameter_marginals`
and the `repr` that quotes them.

### F5, second part: prune the box to an ellipsoid

Adaptive extent fits the box to the posterior but leaves its `(2r+1)^d` shape, and
that shape is nearly all corners. Whitening makes the local Gaussian isotropic, so
lattice point `z` has a *predicted* log-density drop of exactly
`0.5·grid_step²·‖z‖²` — and a `d`-dimensional corner sits `√d` further out than an
axis point with the same per-axis index. Points whose predicted drop clears
`log_density_drop` by a margin are skipped before evaluation: the integration
weights would have discarded them anyway.

**This is a cost optimisation, not an approximation, and is tested as one** — the
integrated mean, covariance and `log_marginal_likelihood` come out *bit-identical*
to the unpruned grid at every dimension tried.

| hyperparameters | box | pruned | conditional fits before → after |
|---|---|---|---|
| 2 | 49 | 45 | 79 → 75 |
| 3 | 343 | 203 | 394 → 250 |
| 4 | 2401 | 873 | 2484 → 938 |
| 5 | 16807 | 3423 | **error → 3451** |

Five hyperparameters were previously not integrable at all: the box needed 16807
points against a `max_grid_points` of 4096, so `hyperparameters="integrate"`
raised rather than ran.

A point that has already been *measured* is never pruned. The axis probes keep
whatever the exploration found, because a heavier-than-Gaussian tail is precisely
the case where the prediction is wrong, and there the density is known rather than
assumed.

### F5, third part: stop filling a region (CCD)

Six or more hyperparameters still exceeded the cap after pruning, because pruning
changes the region's *shape* and not the fact that filling a region costs points
exponential in `d`. Past a handful of dimensions the only way out is to stop
filling and start **designing**, which is what R-INLA does
(`int.strategy="ccd"`; Rue, Martino & Chopin 2009, §6.5).

The design is a rotatable central composite:

- a factorial core taken from `d` columns of a **Sylvester Hadamard matrix**, so
  the columns are orthogonal and every row is a `±1` vector of norm `√d`. This is
  what keeps it `O(d)` — a full `2^d` factorial would defeat the purpose;
- `2d` axial points at `±√d` on each coordinate, sharing that norm;
- everything off-centre scaled by `f0`, putting the design on one sphere of radius
  `f0·√d` — rotatable, i.e. equally accurate in every direction.

Weights follow from requiring `Σᵢ wᵢ zᵢ zᵢᵀ = I`: orthogonality makes the
off-centre sum `f0²·n_p·I`, so those share `1/(f0²·n_p)` and the centre takes
`1 − 1/f0²`, positive exactly when `f0 > 1`.

**Two corrections the textbook design needs here.**

*The design must be centred on the density it integrates.* `u*` is the mode of `s`
alone, but the posterior in `u` is `exp(s + jacobian)`, and for the usual log
transform the Jacobian is `u` — a linear tilt that moves the mode a full standard
deviation. Centring on `s`'s mode left the design systematically off-target, and a
few dozen points cannot absorb that; it was the first version's dominant error. One
Newton step fixes it and costs no conditional fits, since the Jacobian is analytic
and a linear tilt leaves the curvature alone.

*The evaluated densities must still do work.* The design integrates the Gaussian
implied by the Hessian, so the weights carry an importance ratio — dividing by
that Gaussian, i.e. adding `½‖z‖²` in logs. Without it CCD would report the
Laplace approximation back to itself.

With both, the scheme is **exact to machine precision on an exactly Gaussian
target** — mean, covariance *and* log-marginal constant, in every dimension tried.
That test is what separates an implementation bug from the approximation CCD is
entitled to make, and it is what caught the centring error.

| hyperparameters | before | strategy | conditional fits |
|---|---|---|---|
| 4 | 2484 | grid | 956 |
| 5 | error | grid | 3528 |
| 6 | error | **ccd** | 170 |
| 8 | error | **ccd** | 268 |
| 12 | error | **ccd** | 533 |

Twelve hyperparameters now integrate in about two seconds; six was previously an
error. Cost past the design itself is dominated by the `O(d²)` finite-difference
Hessian, not by the design's `O(d)` points.

**CCD is chosen only where the grid cannot run.** `int_strategy="auto"` (the
default) predicts the pruned grid's size arithmetically — no conditional fits —
and keeps the grid whenever it fits `max_grid_points`, so nothing about existing
models changes. `"grid"` and `"ccd"` force the choice.

**The accuracy cost is real and should not be understated.** A second-order design
integrates quadratics in `u` exactly, and little else. Against the dense grid on
models where both run: latent means agree to ~1e-3–1e-1 relative, the
log-marginal likelihood to ~0.1–2 nats, and *natural-scale hyperparameter means*
only to tens of percent — worst of all, because `θ = e^u` is nowhere near
quadratic. So CCD is a fallback, not an upgrade: where the grid is affordable it
stays, and where CCD runs the alternative is not a better answer but no answer.

### F5, fourth part: a Korobov lattice, and a Smolyak rule that was rejected

Both candidates were implemented and measured. **The lattice ships; the sparse
grid does not.**

**Smolyak, rejected.** A level-2 sparse grid of Gauss-Hermite rules needs `2d+1`
points — fewer than a CCD's `~4d` — and is exact to degree five *per coordinate*,
which on an idealised log-gamma posterior beat CCD by one to two orders of
magnitude. On real models it did not: it tied CCD, and at four hyperparameters it
**crashed**, producing a non-finite CPO.

The crash is the disqualifying part and it is structural, not a bug to fix.
Smolyak subtracts lower-level rules, so its weights carry signs — the centre
weight is `1 − d/3`, negative from `d = 4` — and the model criteria treat the
integration weights as a probability mixture. A negative weight makes `CPO`
non-finite. Making it work would mean either clipping the weights, which destroys
the exactness that was the whole point, or redefining the criteria for signed
measures. The cancellation is real too: `Σ|w|` grows about like `d/3`, reaching 7
at twelve dimensions.

**Korobov, shipped.** A randomly shifted rank-1 lattice buys accuracy by
*equidistribution* rather than polynomial exactness: `count` points spread evenly
through the Gaussian, each with weight `1/count`. That makes it the only candidate
whose weights are all positive and equal, which matters for three reasons — the
criteria stay finite, there is no cancellation, and **accuracy is tuned by raising
`count`** rather than by moving to a fundamentally more expensive design. The
generating vector is Korobov's `(1, a, a², …) mod count`, with `a` chosen by a
small spectral search that runs on the lattice alone and costs no conditional
fits. The shift is seeded, so fits stay reproducible.

`int_strategy="auto"` now selects `korobov` where the grid will not fit, and
sixteen hyperparameters integrate in under five seconds:

| hyperparameters | before | strategy | fits |
|---|---|---|---|
| 5 | error | grid | 3528 |
| 6 | error | **korobov** | 277 |
| 12 | error | **korobov** | 623 |
| 16 | error | **korobov** | 929 |

Cost past the design is dominated by the `O(d²)` finite-difference Hessian, not by
the lattice's fixed `count`.

### A correction, and a larger finding

The CCD slice above reported accuracy "against the dense grid" and concluded CCD
was a fallback rather than an upgrade. **That comparison used a biased reference.**
The grid truncates its integration weights at `log_density_drop`, whose default of
2.5 keeps only the points within about 2.2 standard deviations of the mode — and
for a skewed hyperparameter posterior that discards enough mass to dominate every
other error. Against a *converged* reference (fine step, no truncation) the
ranking inverts:

| rule | rel. error, latent means (d=2 / d=3) | \|Δ lml\| |
|---|---|---|
| grid, shipped defaults | 0.22 / 0.60 | 0.55 / 1.03 |
| ccd | 0.043 / 0.074 | 0.062 / 0.019 |
| korobov, 128 points | 0.030 / 0.040 | 0.029 / 0.055 |
| korobov, 512 points | **0.020 / 0.031** | 0.023 / 0.049 |

The designed rules are five to twenty times *more* accurate than the shipped grid,
not less. Isolating the one variable confirms the cause — holding step and extent
fixed and varying only the truncation:

| `log_density_drop` | 2.5 | 5 | 8 | 12 | 20 |
|---|---|---|---|---|---|
| rel. latent error (d=3) | 0.60 | 0.19 | 0.072 | 0.025 | 0.024 |

**So `log_density_drop = 2.5` is the single largest accuracy lever in the
integrator**, worth one to two orders of magnitude, and it converges by about 12.

Note the reference itself only converges to about `7e-3`, so it ranks these rules
but does not resolve differences below roughly one percent.

### F5, fifth part: integrate deep enough to be right

`log_density_drop` rises from **2.5 to 12**. It sets how far down the log density
the integration weights reach, and 2.5 kept only what lay within about 2.2
standard deviations of the mode — for a skewed hyperparameter posterior, throwing
away enough mass to dominate every other error in the integrator.

Every surface scenario moves toward an integration converged by refinement, by one
to three orders of magnitude:

| scenario | latent mean, before → after | log-marginal likelihood |
|---|---|---|
| `gaussian_iid_integrate` | 3.5e-2 → 6.0e-4 | 5.6e-2 → 7.0e-4 |
| `gaussian_ar1_integrate_fixed_rho` | 7.3e-3 → **3.8e-6** | 8.6e-3 → 7.4e-7 |
| `poisson_iid_laplace_integrate_simplified_laplace` | 5.8e-1 → 8.8e-3 | 5.2e-1 → 8.9e-3 |
| `poisson_besag_laplace_integrate` | 7.3e-1 → 2.0e-2 | 5.6e-1 → 2.0e-2 |
| `bernoulli_ar1_laplace_integrate_full_laplace` | 7.3e-1 → 2.2e-2 | 5.9e-1 → 2.1e-2 |

Latent means on the count scenarios were previously wrong by up to **73%**.

**The cost is better than it looks, and at some sizes it is negative.** Raising the
depth makes the grid larger, which makes `auto` reach the point of switching to a
lattice sooner — and a lattice is both cheaper and more accurate than the
truncated grid it replaces. Conditional fits per `integrate` fit:

| hyperparameters | before | after |
|---|---|---|
| 1 | grid, 23 | grid, 23 |
| 2 | grid, 81 | grid, 123 |
| 3 | grid, 258 | grid, 717 |
| 4 | grid, 956 (1.7s) | **korobov, 209 (0.4s)** |
| 5 | grid, 3528 (7.2s) | **korobov, 233 (0.7s)** |
| 6+ | korobov | unchanged |

So the whole cost is two and three hyperparameters; four and five get *cheaper* and
more accurate at once. That is the earlier finding restated as a default: filling a
region and then discarding the mass that matters is worse than either filling it
properly or not filling it at all.

Exploration depth is now `max(log_density_drop, …)`, since a weighting that keeps
points to depth 12 must not be handed a grid explored only to 10.

The calibration harness — the gate this whole thread was built around — passes
before and after, with the worst latent p-value improving from 0.0015 to 0.0061.

`tests/inference/result_surface_baseline.json` moves 1004 leaves across the five
`integrate` scenarios, and the criteria snapshot in
`test_predictive_variance_convention.py` moves with it. Both were verified against
a converged run rather than accepted: those criteria were off by 0.42 (DIC), 0.53
(WAIC) and 0.29 (log-CPO), and are now off by 0.0063, 0.0080 and 0.0047.

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

# Research status

pyLGM keeps two levels of evidence, and they are not interchangeable.

**`main` carries only fully verified results.** A feature reaches `main` when its
correctness has been established against something outside pyLGM — published
results, an analytic solution, or an independent implementation — on the model
as users will actually run it.

**`research-tier` carries frontier work.** The code is tested, reviewed, and
believed correct, but its validation is internal or partial: agreement with an
independent optimisation of the same objective, agreement with MCMC on simulated
data, exact reduction to an already-verified path. That is real evidence. It is
not the same as reproducing a published result on real data, and this page does
not pretend otherwise.

If you are deciding whether to use something from `research-tier` for work you
will publish, read the entry for it below and treat the gaps as yours to close.

---

## Joint models (multi-likelihood stacking) — RESEARCH

Several `LGM` sub-models stacked into one, optionally sharing a latent field
with a per-sub-model scaling that may be estimated. See
[joint models](joint-models.md).

### What is verified

| Claim | Evidence |
|---|---|
| Stacking is exact | With no shared effect the joint likelihood factorises, and it does: joint log marginal likelihood equals the sum of the separate fits to a relative 5.6e-15, posterior means to 5.7e-13. |
| A degenerate joint is the ordinary model | A one-sub-model joint compiles to a `CompiledLGM` with the same design, precision and labels as the equivalent `LGM`. |
| The latent mean is the true posterior mode | Checked against an independent scipy optimisation of the exact log-posterior, written in plain numpy with no pyLGM in the loop: agreement to <1e-5, and the log-posterior is higher there than at the MCMC posterior mean. |
| Posterior curvature is right | Posterior SDs match NUTS with ratios in [0.949, 1.018]. |
| Joint models add no approximation error of their own | A plain single-response `LGM` shows the same mode-vs-mean gap: max abs z 0.813 against the joint's 1.058. |
| Prediction round-trips | Predicting on the fit rows reproduces the fitted means to 5.6e-17. |
| `delta` estimation is consistent | Mean estimate 1.471 / 1.538 / 1.517 against a true 1.6 at 40 / 150 / 600 districts, SD falling 0.299 -> 0.106 -> 0.101. No systematic bias. |

Reference posteriors and the measurement setup are in
[`examples/joint_mcmc_crosscheck/`](https://github.com/Ardea00/pylgm/tree/main/examples/joint_mcmc_crosscheck).

### What is NOT verified — the reason this is not on `main`

**No validation against published results on real data.** Everything above is
either internal consistency or agreement with MCMC on *simulated* data. The
original plan was to reproduce the Knorr-Held & Best (2001) shared-component
analysis of oral cavity and oesophageal cancer across 544 German districts. That
was not achievable, and the search is worth recording so nobody repeats it:

- The **oesophageal** counts Knorr-Held & Best used are not publicly available.
- `spam::Oral` and `INLA::Germany` both cover 544 German districts 1986-1990, and
  their documentation **contradicts itself** about which disease each holds:
  `spam` describes `INLA::Germany` as larynx cancer, while `INLA::Germany`
  describes itself as oral cavity. The counts differ (15,466 against 7,283).
  Building a two-disease model on data whose identity its own sources dispute
  would be a false validation, not a weak one.
- `INLAjoint`, the reference INLA package for joint models, ships **simulated**
  data with its examples.
- Adin et al. (2024) `INLA_groupCV` bundles no data; it references NHS England.
- `JM`, which held the classic joint longitudinal-survival datasets, is no
  longer in the CRAN listing.
- `SpatialEpi::pennLC` does give two genuine outcomes on one geography (male and
  female lung cancer over 67 Pennsylvania counties, with a smoking covariate),
  but no published shared-component posteriors exist for that pair, and it ships
  no adjacency graph.

The conclusion is not that the search was insufficient. Open data *and*
published posteriors for a joint or shared-component model is a combination that
is close to absent from the reproducible literature.

**`latent_strategy="laplace"` degrades on joint models.** Measured against NUTS:
mean abs z 0.337 against the Gaussian baseline's 0.072, worst case 1.14 against
0.40, with skewness estimates correlating only +0.23 with the truth. A
single-response control over the same data shows `laplace` slightly *improving*,
so this is specific to joint models. `simplified_laplace` behaves as designed
(skew correlation +0.87, worst-case error more than halved) and is the one to
prefer. Documented rather than pinned by a test, because a test would cement
behaviour we believe is wrong.

**Untested or unsupported surface.**

- `hyperparameters="integrate"` is covered, but only on small simulated joints.
- No YAML frontend for `Joint`; the declarative path does not reach joint models.
- `Joint.fit` takes only a pandas DataFrame; `LGM.fit` also takes Spark.
- `Joint.fit` drops NaN-response rows rather than holding them out. Deliberate —
  see [joint models](joint-models.md) — but it diverges from `LGM.fit`.
- A shared effect's own precision, rho or phi cannot be estimated; only the
  `Shared` scale may be a `Hyperparameter`. The exception is
  `Shared(Correlated(...))`, whose per-outcome precisions and correlations are
  estimable (see the network-reconstruction entry below).
- Off-block-diagonal coupling across outcomes exists only as
  `Shared(Correlated(...))`, i.e. correlated IID components; there is no
  structured (spatial/temporal) coregionalization.

### What would move this to `main`

Reproducing a published joint or shared-component analysis on real data, with
posterior summaries matching within a stated tolerance. Failing that, an
independent implementation of the same model fitted to the same real data, with
the comparison recorded the way the MCMC cross-check already is. Resolving the
`laplace` degradation, or establishing it as expected, is a prerequisite either
way.

---

## `Weighted` effects (spatially-varying coefficients) — RESEARCH

`Weighted(effect, by)` scales an indexed effect's design row-wise by a numeric
column, `diag(by) A`, so a covariate's slope can itself be a latent field. See
[Weighted effects](effects.md#weighted-effects).

### What is verified

| Claim | Evidence |
|---|---|
| The design is exactly `diag(by) A` | Checked column-for-column against a manually built weighted incidence matrix. |
| A constant weight reduces to the unweighted effect | With `by` all ones, the weighted fit's log marginal likelihood and posterior mean match the equivalent unweighted `IID` fit to a relative 1e-9 / 1e-7. |
| Prediction round-trips | Predicting on the fit rows reproduces the fitted means to machine precision. |
| A known spatially-varying coefficient is recovered | Simulated `u ~ N(0, 0.5²)` across 15 regions, `log mu = 0.5 + z*u_region`, Poisson response, 40 draws/region: fitted-vs-true correlation 0.979 (threshold 0.8). |

### What is NOT verified

- **No validation against published results on real data.** As with joint
  models, everything above is internal consistency or recovery on *simulated*
  data.
- **`Weighted` inside a `Joint`'s `shared=` is rejected, not merely untested.**
  `Shared.__post_init__` requires the wrapped effect to expose `.index`, which
  `Weighted` deliberately does not; `Shared(Weighted(...))` raises `TypeError`
  at construction. A shared field with a per-outcome spatially-varying weight
  cannot be built at all today.
- **`Weighted` inside a `Joint` sub-model has no test.** Unlike `shared=`,
  wrapping a sub-model's own effect in `Weighted` does compile and predict —
  it just has no test or example exercising it, so it belongs under neither
  heading above yet.
- **No YAML/config surface.** Every neighbouring effect in
  [effects.md](effects.md) has a YAML block; `Weighted` has none, and the
  config schema has no `weighted` effect type to parse one into.

---

## `Copy` effects (a field entering a predictor twice) — RESEARCH

`Copy(name, index, scale=1.0)` folds `scale * A_index` into an existing
field's design at a second index, so the field enters the predictor once
unscaled and once (rescaled) again. It produces no block of its own. See
[Copy](effects.md#copy).

### What is verified

| Claim | Evidence |
|---|---|
| The design is exactly the target's plus `scale * A_index` | Checked column-for-column against a manually built incidence matrix summed onto the target's design (`tests/test_copy_compile.py`, `tests/test_copy_model.py`). |
| A copy adds no block and no labels | `IID("u", ...) + Copy("u", ...)` produces the same `blocks` and `result.labels` as `IID("u", ...)` alone (`tests/test_copy_compile.py::test_a_copy_adds_no_block_of_its_own`). |
| Prediction round-trips (optimize path) | Predicting on the fit rows reproduces the fitted means to a relative and absolute 1e-12, with both a fixed scale and an estimated (`Hyperparameter`) scale, over a fixture whose copy index visits levels in an order that differs from the target's sorted label order (`tests/test_copy_predict.py`). |
| Prediction under `hyperparameters="integrate"` uses the INLA marginal mean, not the initial guess, but is a plug-in approximation | The point estimate for an estimated copy scale is read from `result.hyperparameter_marginals()` (`result.hyperparameters` is `None` on the integrate path), exactly as `midas_parametric` already does. Predicting on the fit rows lands within a max relative 0.40 of the fitted means -- the genuine plug-in-versus-mixture gap for a hyperparameter-dependent design, not a bug: a no-copy integrate baseline (hyperparameter-independent design) round-trips to machine precision (`tests/test_copy_predict.py`). |
| Multiple copies of one field accumulate | Two copies of the same field at two different indices both fold into the same columns, matching a hand-summed design (`tests/test_copy_compile.py::test_two_copies_of_the_same_field_at_different_indices_both_fold_in`). |
| A known copy scale is recovered | Simulated `u ~ N(0, 0.5²)` over 12 levels, `log mu = 0.3 + u_i + beta*u_j`, Poisson response, 600 rows, `beta` estimated as a `Hyperparameter`: fitted `beta = 1.69` against a true `1.8` (relative tolerance 0.4) (`tests/test_copy_model.py::test_the_copy_scale_is_recovered`). |

### What is NOT verified

- **No validation against published results on real data.** As with `Joint`
  and `Weighted`, everything above is internal consistency or recovery on
  *simulated* data.
- **`Copy` inside a `Joint` is rejected, not merely untested.** `Joint`
  compiles each sub-model's effects independently and has no target block for
  a copy to fold into, so a sub-model containing a `Copy` fails to compile
  with `CompilationError: unsupported effect type: Copy` rather than being
  supported or silently ignored.
- **An estimated copy scale on a target whose precision is itself a
  `ParametricBlock` is rejected, not supported.** When the target's own
  precision is a function of hyperparameters (an estimated `rho` on `AR1`,
  `ProperCAR`, `SAR`, `phi` on `BYM2`, or an estimated precision on `Seasonal`
  or `MIDAS`, for example) *and* the copy's scale is also a `Hyperparameter`,
  compilation raises `CompilationError` rather than combining the two.
  Combining an estimated copy scale with an estimated structural parameter on
  the same target is a real modelling case that this slice does not cover.
- **No YAML/config surface.** As with `Weighted`, `Copy` has no YAML block in
  [effects.md](effects.md) and no config schema entry.

---

## `Replicated` effects (independent copies sharing hyperparameters) — RESEARCH

`Replicated(effect, over)` builds `R` independent copies of any indexed
effect, one per level of `over`: precision `I_R ⊗ Q`, design on
`(replicate, level)` pairs, one constraint per replicate. This is R-INLA's
`f(index, model=..., replicate=r)`. See [Replicated](effects.md#replicated).

### What is verified

| Claim | Evidence |
|---|---|
| The precision is exactly the Kronecker product | Checked against a hand-built `np.kron(I_R, Q)` (`tests/test_replicated_compile.py::test_precision_is_the_kronecker_product_of_identity_and_the_inner_structure`). |
| A constrained inner effect gets one constraint per replicate, full rank | `R` copies of the inner constraint, not one shared across all `R` — checked directly for count, placement and rank (`tests/test_replicated_compile.py::test_a_constrained_inner_effect_gets_one_constraint_per_replicate`). |
| A single replicate reduces to the unwrapped inner block | `Replicated(effect, over=...)` over a column with one level matches `effect` alone (`tests/test_replicated_compile.py::test_a_single_replicate_reduces_to_the_inner_block`). |
| It commutes with `Weighted` | `Replicated(Weighted(...))` and `Weighted(Replicated(...))` compile to the same design and precision (`tests/test_replicated_compile.py::test_replicated_commutes_with_weighted`). |
| Matches the shipped `AR1(group=)` bit for bit | `Replicated(AR1(...), over=...)` and the pre-existing `AR1(..., group=...)` produce identical labels, design, precision and constraints across `rho` in `{0.0, 0.3, -0.6, 0.9}`, and identical `log_marginal_likelihood`/posterior mean under a full Poisson/Laplace fit — the general machinery checked against an implementation that was correct before this slice existed (`tests/test_replicated_equivalence.py`). |
| Prediction round-trips, including when sorted and first-seen level order diverge | Predicting on the fit rows reproduces the fitted means to a relative/absolute 1e-12, on a fixture (`t1..t11`) chosen because lexicographic and first-seen order genuinely disagree — the class of bug this project has shipped twice before (`tests/test_replicated_predict.py::test_round_trip_holds_when_sorted_and_first_seen_level_order_differ`, plus the unseen-replicate/unseen-level and `Replicated(Weighted(...))` round-trip cases in the same file). |
| Every declared hyperparameter on a `Replicated` effect actually affects the fit | Covered by the project's structural cross-check for the "registered but dead" failure mode, for `Replicated(IID(tau))`, `Replicated(AR1(rho))`, and `Replicated(Weighted(IID(tau)))` (`tests/test_hyperparameter_effectiveness.py`). |

### What is NOT verified

- **No validation against published results on real data.** As with `Joint`,
  `Weighted` and `Copy`, everything above is internal consistency or exact
  agreement with an independent implementation, not recovery on real data.
- **`Replicated` inside a `Joint` is untested.** No test exercises a
  `Replicated` effect inside a `Joint` sub-model or as a `Shared` target; the
  `Joint`/`Shared` entry above already notes no `replicate` within a shared
  effect is supported.
- **A `ParametricDesignBlock` inner effect is rejected, not supported.** An
  effect whose design is itself a function of an estimated hyperparameter —
  today only `MIDASParametric` — cannot be replicated. In practice this is
  already unreachable through the public API (`MIDASParametric` has no
  `index`, so `Replicated` rejects it at construction), but the compiler
  keeps a second guard for the same case so a future design-varying effect
  fails loudly rather than silently replicating over the wrong row space.
- **No YAML/config surface.** As with `Weighted` and `Copy`, there is no
  `replicated` effect type in the config schema and no YAML block in
  [effects.md](effects.md); the Python API is the only way to declare one.
- **Not supported on Spark.** `_required_columns` in `data/spark.py` reads
  `effect.index` unconditionally, so a `Replicated` model raises a bare
  `AttributeError` there, and `over` is never added to the projection either.
  This slice did not introduce the gap: that helper is already blind to
  `MIDAS`, `MIDASParametric`, `SpaceTime` and `DynamicSpatialPanel`, and it
  drops even `AR1(replicate=)`'s replicate column. It is recorded here rather
  than fixed because the fix belongs to that helper, across all of them.

---

## `Grouped` effects (correlated copies with a between-group structure) — RESEARCH

`Grouped(effect, over, structure)` builds `R` *correlated* copies of any
indexed effect, one per level of `over`, tied by a between-group precision
`Q_S`: `Q_S ⊗ Q_E`. This is R-INLA's `f(index, model=..., group=g,
control.group=list(model=...))`. `Replicated` is the special case
`structure=IIDStructure()`. See [Grouped](effects.md#grouped).

### What is verified

| Claim | Evidence |
|---|---|
| The precision is exactly the Kronecker product of the structure's and the inner effect's | Checked against a hand-built `np.kron(structure, inner)` (`tests/test_grouped_compile.py::test_precision_is_the_kronecker_product_of_structure_and_inner`). |
| An `IIDStructure` reduces `Grouped` exactly to `Replicated` | Same labels, design, precision, and constraints, checked directly (`tests/test_grouped_compile.py::test_an_iid_structure_reduces_grouped_to_replicated`). |
| A single group level reduces to the bare inner effect | `tests/test_grouped_compile.py::test_a_single_group_level_reduces_to_the_bare_effect`. |
| An unobserved graph node still gets a cell | `BesagStructure`'s graph is the universe, not the observed levels, matching `Besag`/`build_spacetime` (`tests/test_grouped_compile.py::test_an_unobserved_graph_node_still_gets_a_cell`). |
| Constraints span the null space of the composed precision, not one per group | Checked by rank, since `null(Q_S) ⊗ R^E` overlaps any null space the inner effect already carries (`tests/test_grouped_compile.py::test_constraints_span_the_null_space_of_the_composed_precision`). |
| An integer index keeps its numeric level order, on both the plain and the family (estimated-hyperparameter) compile path | Regression guard for the ordering bug this project has shipped before; verified non-vacuous by mutation in this slice (`tests/test_grouped_compile.py`, two tests). |
| The family path composes against the structure's own precision, not an identity, on **both** its branches: the `ParametricBlock` rebuild closure (a structure hyperparameter, e.g. `AR1`'s `rho`) and the ordinary `ScalableBlock` branch (a plain inner precision, e.g. `IID`'s) | `ParametricBlock` branch: `tests/test_grouped_compile.py::test_grouped_family_rebuild_uses_the_structure_precision_not_identity`. `ScalableBlock` branch: `tests/test_grouped_compile.py::test_an_estimated_inner_precision_scales_every_group`, which pins the materialised precision against `np.kron(structure.precision(groups), inner_template)`; verified non-vacuous by mutation (`composed = grouped_block(...)` with `effect.structure` swapped for `IIDStructure()` — the most likely user path, and the project's signature failure mode — fails only this test). |
| An estimated inner precision scales every group's block by exactly that factor in the family path | `tau=1.0` vs. `tau=50.0` changes every nonzero precision entry by exactly `50.0`, added and verified non-vacuous in this slice (`tests/test_grouped_compile.py::test_an_estimated_inner_precision_scales_every_group`). |
| It commutes with `Weighted` | `tests/test_grouped_compile.py::test_grouped_and_weighted_commute`. |
| A full Poisson/Laplace fit runs end to end and returns a finite log marginal likelihood | `tests/test_grouped_compile.py::test_a_grouped_model_fits_end_to_end`. |
| Prediction round-trips on the fit rows, including with `Weighted` inside or outside the group and with a subset of groups | `tests/test_grouped_predict.py`. |
| `Grouped` reproduces all four Knorr-Held `SpaceTime` interaction types (I-IV) bit for bit on design, and on precision up to one pre-existing global scalar for the RW-based types (II, IV) — see below | `tests/test_grouped_spacetime_oracle.py`, checked against `build_spacetime`, an implementation that predates this slice. |
| Every declared hyperparameter on a `Grouped` effect actually affects the fit | Covered by the project's structural cross-check for the "registered but dead" failure mode, for `Grouped(IID(precision))`, `Grouped(AR1(rho))`, and `Grouped(Weighted(IID(precision)))`, each over a `BesagStructure` (`tests/test_hyperparameter_effectiveness.py`); verified non-vacuous by mutation in this slice (neutering `_effect_hyperparameters`'s `Grouped` delegation to `return []` makes these models declare zero hyperparameters and fail at collection, since `compile_family` then returns `None`). |

### What is NOT verified

- **No validation against published results on real data, and no
  known-parameter recovery test on simulated data either** (unlike `Weighted`
  and `Copy`, which each recover a known ground-truth coefficient). Everything
  above is internal consistency or exact agreement with an independent
  implementation (`build_spacetime`).
- **The RW scaling divergence — reconciled.** `RW1Structure`/`RW2Structure`
  used to be Sørbye-Rue scaled unconditionally while `RW1`/`RW2` default to
  unscaled, so `Grouped(RW1(...), structure=RW1Structure())` compiled to
  `kron(scaled, unscaled)` — two different matrices spelled RW1 in one call.
  The structures now take the same keyword-only `scale` flag with the same
  default (`False`), and `Grouped(RW1(..., scale=True), over, IIDStructure())`
  reproduces `SpaceTime` type II exactly. `tests/test_grouped_spacetime_oracle.py`
  pins both. Changed numerics: only `Grouped` fits using `RW*Structure()`
  without `scale=True`.
- **`group` and `replicate` together — supported** as
  `Replicated(Grouped(...), over=r)` (R-INLA's `f(idx, group=, replicate=)`,
  `I_R ⊗ Q_S ⊗ Q_E`). Verified exactly: with the replicated term alone and a
  fixed noise, the fit decomposes into one `Grouped` fit per replicate (log
  marginal likelihood and predictions to 1e-8, `tests/test_replicated_grouped.py`).
- **A `Hyperparameter` on a structure's own parameters is not supported.**
  `AR1Structure(rho)` takes a fixed float only; passing a `Hyperparameter`
  raises `TypeError` at construction. Only the *inner* effect's own
  hyperparameters are estimated.


---

## Network reconstruction (censored registers, margins, correlated effects) — RESEARCH

`CensoredHurdle` (edges absent from a register that reports amounts at or above
a threshold), `scale="below_threshold"` margins on the mass below it, exact
non-Gaussian `LinearConstraint`s, and the `Correlated` effect (sender/receiver,
reciprocity, cross-outcome). See [joint models](joint-models.md#censored-registers-censoredhurdle)
and [Correlated](effects.md#correlated-components).

### What is verified

| Claim | Evidence |
|---|---|
| A non-Gaussian `LinearConstraint` gives the exact constrained mode | Poisson with `scale="log"` aggregates against SLSQP on the exact log-posterior (1e-5); the old conditioning of the unconstrained Laplace posterior was only first-order (`tests/test_non_gaussian_linear_inputs.py`). |
| The constraint evidence `log p(e) + log p(y \| e)` is exact for a Gaussian likelihood | Both Laplace engines reproduce the exact engine's evidence to 1e-8 (`tests/inference/test_sparse_laplace.py`). |
| The censored hurdle's mode and evidence | Against an independent numpy optimisation of the exact posterior (mode 1e-5) and a finite-difference Laplace evidence (1e-4); dense and sparse engines agree to 1e-8 (`tests/test_censored_hurdle.py`). |
| Below-threshold margins enter exactly | Same oracle, with the margins term; the map's value and Jacobian against its definition and finite differences. Under Gauss-Newton curvature the evidence was off by 1.1 nats on this case. |
| `Correlated` is exact | Gaussian closed form for the posterior mean (1e-9) and evidence (1e-8), in one model and shared across two outcomes (`tests/test_correlated_effect.py`). |
| `Correlated` hyperparameters are identified | EB recovers a simulated correlation (0.6), reciprocity (0.8) and a cross-outcome correlation through a Bernoulli outcome. |
| The CPC parametrization gives LKJ | Sampling CPCs from their Beta laws gives correlation marginals `Beta(k/2, k/2)` for `k = 2, 3, 4`. |
| It scales | 20 000 borrowers × 20 lenders (400 000 candidate edges, 40 000 latents): 7.9 s without margins, 38 s with one margin per borrower, on a laptop. |

### What is NOT verified

- **No real data.** Everything above is exactness against closed forms or
  independent optimisations, and recovery on simulated data. The planned
  validations are a mask-and-reconstruct study inside a credit register (hold
  out borrowers or raise the reporting threshold, reconstruct, score with
  `pylgm.evaluation`'s network metrics) and the public OECD ICIO tables.
- **Identification of the hurdle.** With only reported and censored edges,
  "no link" and "a link below the threshold" are separated by the parametric
  amount tail alone; known non-edges or margins are what identify the link
  intercept. No sensitivity study to the lognormal assumption exists.
- **Exact constraints do not scale to one per node**: they need dense
  `c x latent` workspace. Use soft margins (`LinearObservation`) at scale.
- **A nonlinear constraint keeps Gauss-Newton curvature**, so its evidence is
  approximate when it binds away from the prior.
- **Unsupported with a censored hurdle or a nonlinear aggregate**:
  `mean_correction`, `simplified_laplace`/`laplace` latent strategies, per-row
  criteria, `update()`/`news()` of coupled rows — each refused, not approximated.

### What would move this to `main`

A mask-and-reconstruct validation on a real register (or ICIO) with the
reconstruction scored against the held-out truth and compared with the
standard baselines (maximum entropy / RAS, the density-corrected gravity
model), recorded the way the joint models' MCMC cross-check is.

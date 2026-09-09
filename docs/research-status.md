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
  `Shared` scale may be a `Hyperparameter`.
- No off-block-diagonal precision coupling, so no coregionalization; no `copy`
  or `replicate` within a single sub-model.

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
- **The RW scaling divergence — the most important gap here.**
  `build_spacetime` always builds its RW time factor Sørbye-Rue *scaled*
  (`rw_structure(T, order, scale=True)`); `Grouped`'s inner `RW1`/`RW2`
  compiles through the ordinary, *unscaled* `build_random_walk` — the same
  builder every standalone `RW1`/`RW2` effect in this library uses. The two
  differ by exactly one global scalar on every nonzero precision entry: for
  `T=5, order=1` the ratio (unscaled precision / scaled precision) is
  `1.36979319`, the reciprocal of `sorbye_rue_scale`'s factor. **The pyLGM
  facts, stated precisely rather than against another library's default:**
  a standalone `RW1(...)`/`RW2(...)` and `Grouped(RW1(...), ...)`'s inner
  factor are unscaled; `Grouped(..., structure=RW1Structure())`'s outer
  factor and every `SpaceTime` RW factor are Sørbye-Rue scaled (`RW1Structure`
  and `RW2Structure` are pinned against `rw_structure(n, order, scale=True)`
  directly — see `tests/test_structures.py`, added this slice). R-INLA's
  `rw1`/`rw2` take an explicit `scale.model` argument that does not scale
  unless the caller asks — pyLGM does not claim to match whatever a
  particular R-INLA version defaults `scale.model` to, so check your own R
  call: a user who wrote `scale.model=TRUE` there, or who compares against
  pyLGM's own `SpaceTime`, is the one who gets a different model under the
  same nominal `precision` from pyLGM's unscaled `RW1`/`RW2`.
  This is a pre-existing inconsistency between `RW1`/`RW2` and
  `build_spacetime`'s internal convention, not something this slice
  introduced, and it is recorded rather than fixed because reconciling it
  changes already-released numerics for both `RW1`/`RW2` and `SpaceTime`.
  `tests/test_grouped_spacetime_oracle.py` pins both drift directions (that
  the two disagree by this factor, and that the factor is never zero, i.e.
  the discrepancy is real).
- **The same divergence recurs *inside a single `Grouped` call*, between its
  own two factors.** `Grouped(RW1("u", index="t"), over="g",
  structure=RW1Structure())` compiles to exactly
  `kron(rw_structure(G, 1, scale=True), rw_structure(T, 1, scale=False))` —
  verified numerically in this slice. The outer `RW1Structure` factor is
  Sørbye-Rue scaled; the inner `RW1` factor, spelled with the same name in
  the same call, is not. This is not only a `Grouped`-versus-`SpaceTime`
  question: two things called "RW1" in one line of code are two different
  matrices.
- **No YAML/config surface.** Unlike `SpaceTime`, which has one, there is no
  `type: grouped` in the config schema and no YAML block in
  [effects.md](effects.md); the Python API is the only way to declare one.
- **Not supported on Spark.** `_required_columns` in `data/spark.py` reads
  `effect.index` unconditionally; a `Grouped` model raises a bare
  `AttributeError: 'Grouped' object has no attribute 'index'`, and `over` is
  never added to the projection either — so a future fix must add it or the
  group column is silently projected away and the failure moves from loud to
  silent. Pre-existing gap in a helper already blind to `MIDAS`, `SpaceTime`,
  `DynamicSpatialPanel`, and `AR1(replicate=)`'s replicate column.
- **`group` and `replicate` together are rejected**, which R-INLA permits on
  one `f()` term. `Grouped(Replicated(...), ...)`, `Replicated(Grouped(...),
  ...)`, and wrapping an effect that already declares its own `replicate=`
  all raise `TypeError` at construction. An f() parity gap, recorded rather
  than half-implemented, since the label scheme (`replicate@group@level`) and
  the predict path both assume exactly one pairing.
- **A `Hyperparameter` on a structure's own parameters is not supported.**
  `AR1Structure(rho)` takes a fixed float only; passing a `Hyperparameter`
  raises `TypeError` at construction. Only the *inner* effect's own
  hyperparameters are estimated.
- **`Grouped` inside a `Joint` is untested**, though it does work: a `Grouped`
  effect used as an ordinary (non-shared) effect inside one `Joint` sub-model
  compiles and fits without error. No test exercises the combination.
  `Shared(Grouped(...))` is rejected by design, not merely untested —
  `Grouped` exposes no `.index`, so it fails `Shared`'s "must be indexed"
  guard the same way `Weighted`, `Fixed`, `MIDAS`, `SpaceTime`, and
  `DynamicSpatialPanel` already do.

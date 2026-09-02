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
- **`Weighted` inside a `Joint`'s `shared=` is untested.** Both compile
  independently; their combination — a shared field with a per-outcome
  spatially-varying weight — has no test and no example.

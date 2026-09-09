# pyLGM Joint Models: Multi-Likelihood Stacking Design

**Status:** Approved 2026-09-01

## Purpose

`LGM` takes exactly one `response` column and one `likelihood`. Every joint and
multivariate model in the latent-Gaussian literature — shared-component disease
mapping, joint PD/LGD, longitudinal-plus-survival, coregionalization, dynamic
Nelson-Siegel — needs **stacked responses with a different likelihood per row
block**. That gap, not the copy primitive, is what actually blocks them.

This slice adds joint models: a `Joint` container over ordinary `LGM`
sub-models, a row-dispatching mixture likelihood, and a `Shared` declaration
that lets one latent field enter several sub-models with a per-sub-model
scaling. The scaling may be a `Hyperparameter`, which makes this slice a
superset of R-INLA's `copy` with an estimated `beta` for the shared-field case.

The reference model is Knorr-Held & Best (2001):

    log mu_1i = alpha_1 + delta * u_i + v_1i
    log mu_2i = alpha_2 + delta^-1 * u_i + v_2i

with `u` the shared spatial component, `v_k` the disease-specific components,
and `delta > 0` the scaling. The `delta, delta^-1` parameterisation keeps the
shared component's overall scale identified.

No new dependency, no new inference regime, no MCMC.

## What already exists and is reused unchanged

Three pieces of the machinery this needs are already shipped, which is why the
slice is smaller than it looks:

- **`ParametricDesignBlock`** (`ir/family.py`) — a latent block whose *design*
  is rebuilt as a function of hyperparameters, precision and constraints held
  fixed. Built for restricted MIDAS; it is exactly what an estimated `delta`
  needs. `MIDASParametric`'s registration in `compiler.py` is the template,
  including the short-circuit to a plain `ScalableBlock` when nothing is
  estimated.
- **The likelihood interface is row-separable.** `gradient`,
  `working_weights`, `third_derivative`, `pointwise_log_density`, `cdf`,
  `response_mean` and `response_prediction` all return per-row arrays, and
  `log_likelihood` is a sum over rows.
- **The Laplace engine dispatches through that interface only** — no
  `isinstance` checks in `inference/laplace.py`. A mixture likelihood therefore
  requires **no solver change**.

## Architecture

**A joint model compiles to an ordinary `CompiledLGM` with more rows.** The IR
does not change.

Responses stack as `y = (y^(1), ..., y^(K))`, sub-model `k` occupying rows
`R_k = [n_<k, n_<k + n_k)`. Then:

- **Both `CompiledLGM` invariants survive.** The design is still
  `hstack(blocks)` and the precision still `block_diag(blocks)`. Sharing is
  expressed on the *design* side, not as off-block-diagonal precision coupling,
  so the existing validation stays exactly as strict as it is today.
- **A sub-model-private block** is zero-padded outside `R_k`. Its precision,
  labels and constraints are untouched, so a Besag sum-to-zero still means what
  it meant.
- **A shared block** has nonzero rows in every slice it enters, scaled per
  slice.

Off-block-diagonal precision coupling — needed for coregionalization (atlas M2)
— is explicitly **not** part of this slice.

New module `src/pylgm/joint.py`. `model.py` is already 532 lines and this is a
distinct responsibility.

## Public API

```python
Joint(
    [LGM(response="oral",   likelihood=Poisson(), offset="log_E_oral",   predictor=...),
     LGM(response="larynx", likelihood=Poisson(), offset="log_E_larynx", predictor=...)],
    shared=[Shared(Besag("u", index="district"), scale=Hyperparameter("delta"))],
)
```

- `Joint(submodels, shared=())` — frozen dataclass. `.fit(frame, ...)` mirrors
  `LGM.fit`, minus `engine="exact_gaussian"` (see Rejections).
- `Shared(effect, scale)` — one latent field entering several sub-models.
  `scale` is a float, a `Hyperparameter`, or a per-sub-model tuple of either.
  A **scalar** `Hyperparameter` is shorthand for the KHB `(delta, delta^-1)`
  pairing and is therefore accepted **only when the joint has exactly two
  sub-models**; with three or more the pairing has no canonical meaning, so an
  explicit per-sub-model tuple is required. A tuple sets each slice's scale
  directly and is always allowed.
- Sub-models are identified by their `response` string, which must be unique
  across the joint.

## Components

### `CompiledMixture(_CompiledLikelihood)` — `likelihoods.py`

Holds `parts: tuple[(mask, likelihood), ...]` over row masks validated at
construction to be **disjoint and covering** — a gap would silently drop
observations from the likelihood, which is the failure mode most likely to pass
tests and give wrong answers.

- `log_likelihood` sums the parts.
- The per-row methods scatter each part's result into an output array.
- `for_observations(aux)` forwards each part its own aux slice, so Binomial
  `trials` and survival `event`/`entry` stay per-family.
- `validate_response` validates each part on its own rows, so Poisson's
  non-negative-integer check never sees Gaussian rows.

### `_pad_block_rows(block, before, after)` — `joint.py`

Vertical zero-pad of a `LatentBlock`'s design via `scipy.sparse.vstack`.
Precision, labels and constraints pass through untouched.

### `Shared` compilation

The shared effect is built once against the **union** of its index levels
across sub-models, then registered as a `ParametricDesignBlock` whose `build`
returns `vstack(delta_1 * A_1, ..., delta_K * A_K)`, with `A_k` the incidence of
the shared index on slice `k`.

`delta` is registered under a **log transform** (`delta > 0`). When every scale
is fixed the design bakes in and the block degrades to a plain `ScalableBlock`,
mirroring `MIDASParametric`.

### `compile_joint` / `compile_joint_family` — `compiler.py`

Mirror `compile_lgm` / `compile_family`. Each sub-model's blocks are built
against its **own sub-frame** by the existing effect builders, unchanged, then
padded into the stacked row space. `offset` and `observed` concatenate per
slice.

### Label and hyperparameter namespacing

`CompiledLGM` derives its labels as `f"{block.name}:{label}"`, so namespacing is
applied to the **block name**, not to the label string: a sub-model-private
block is renamed `"oral:fixed"`, giving the qualified label
`"oral:fixed:Intercept"` against `"larynx:fixed:Intercept"`. Shared blocks keep
their bare name, giving `"u:district_1"`. Required because `CompiledLGM` demands
globally unique labels and both sub-models have a `fixed` effect.

Hyperparameter names are **not** namespaced the way block/label names are: every
sub-model and every `Shared` scale draws from one flat name space, and the
compiler rejects a collision with `CompilationError` instead of prefixing
around it. Two Gaussian sub-models must therefore be given distinct `sigma`
names explicitly (e.g. `Gaussian(sigma=Hyperparameter("sigma_oral", ...))`), and
a `Shared` scale's name must not collide with any sub-model hyperparameter's
name. The one exception is deliberate reuse: the same `Hyperparameter` object
passed to more than one `Shared` entry (or produced by the `(delta,
delta^-1)` scalar shorthand) dedups rather than raising, since it is the same
declaration, not a collision.

## Data flow

```
frame --> per-sub-model CanonicalPanel   (each validates its own response/observed)
      --> per-sub-model blocks via existing builders (own sub-frame)
      --> _pad_block_rows into the stacked row space
      --> shared blocks as ParametricDesignBlock over the union index
      --> hstack / block_diag --> CompiledLGM(likelihood=CompiledMixture)
      --> Laplace engine, unchanged
      --> result with namespaced labels
```

## Prediction

`result.predict(new_data, outcome="oral")`. `outcome` is **required** on a joint
model and **rejected** on a single one. New rows are homogeneous; predicting
two outcomes means two calls.

One `PredictionContext` is built per sub-model at fit time, holding that
sub-model's own entries plus the shared entries. Because `eta = A x` is over the
*full* stacked latent, each context must place its blocks at the right column
offsets:

- `PredictionContext` gains `column_slices: tuple[tuple[int, int], ...]`,
  aligned with `entries`.
- `_design_for` scatters each entry's columns into a `width`-wide zero array at
  its slice instead of hstacking.
- For a single-outcome model the slices are contiguous and gapless, so this
  **reduces exactly to today's behaviour** — a generalisation, not a special
  case. Existing prediction tests must pass unchanged.

The shared entry is `("shared", (name, index, labels, scale_spec))` where
`scale_spec` is a hyperparameter name or a float, with the fitted `delta_k`
substituted at predict time — the same pattern `_midas_parametric_block`
already uses for `theta_spec`.

Per-outcome prediction uses that sub-model's **own** compiled likelihood, not
the mixture, so `response_prediction`, `trials` and survival aux behave as they
do today.

## Rejections (fail loud, matching the frontend's existing style)

- Compilation errors are prefixed with the outcome name — `failed to compile
  effect 'fixed' for outcome 'larynx'` — because two sub-models have a `fixed`
  effect and the bare message is useless.
- Duplicate `response` names across sub-models.
- A `Shared` effect naming an index column no sub-model has.
- A shared index whose level set differs between sub-models: the union is taken
  for the latent, but a level present in one sub-model and absent in another is
  reported, not silently zero-filled.
- A `scale` tuple whose length does not match the sub-model count.
- A scalar `Hyperparameter` `scale` on a joint with three or more sub-models.
- `predict()` without `outcome` on a joint model, or with an unknown outcome —
  the error lists the valid names.
- `engine="exact_gaussian"` on a `Joint`. The exact engine requires a
  `CompiledGaussian` (`inference/gaussian.py`); a mixture routes to Laplace,
  which is exact for an all-Gaussian stack anyway.
- `CompiledMixture` construction with masks that overlap or leave a row
  uncovered.

## Testing

- **Reduction.** A `Joint` with one sub-model and no shared effects compiles to
  a `CompiledLGM` numerically identical to the equivalent `LGM` — same design,
  precision, labels, marginal likelihood. The strongest cheap guard.
- **Factorisation.** Two stacked sub-models with no shared effect give the same
  posterior as fitting them separately, since the joint factorises.
- **Padding.** `_pad_block_rows` preserves precision, labels and constraints
  exactly.
- **Mixture dispatch.** A `CompiledMixture` over two masks agrees row-for-row
  with each part evaluated on its own slice, for every interface method.
- **Recovery.** Simulate from the KHB structure with known `u`, `v_k`, `delta`;
  check recovery within posterior uncertainty.
- **Reference.** The oral-cavity + larynx fit on the 544 German districts,
  cross-checked against the same model in R-INLA. Tolerances are recorded
  explicitly in the test rather than tightened until flaky.
- **Predict round-trip.** Predicting on the fit rows reproduces the fitted
  means per outcome.
- **No regression.** The existing prediction suite passes unchanged, which is
  what proves `column_slices` is a generalisation.

## Reference data

- **Oral cavity cancer**, 544 German districts, 1986-1990 — `spam::Oral`
  (CRAN), columns `Y`, `E`, `SMR`, with the district polygon/adjacency
  structure in the same package.
- **Larynx cancer**, same 544 districts and period — `INLA::Germany`, with a
  smoking covariate.

Both are public and redistributable. The **oesophageal** counts that
Knorr-Held & Best actually used as the second outcome could not be confirmed as
publicly available, so this reproduces the shared-component *method* on a real
two-disease pair on the identical geography, validated against R-INLA — not the
paper's exact posterior numbers. Oral cavity and larynx are both
smoking/alcohol driven, so the shared component remains substantively
meaningful.

## Out of scope

- **Off-block-diagonal precision coupling** (coregionalization, atlas M2). The
  `precision == block_diag(blocks)` invariant is preserved here.
- **`copy` between effects within a single `LGM`.** This slice covers sharing
  *across* sub-models. Within-model copy is the natural follow-on and reuses
  the same `ParametricDesignBlock` registration.
- **`replicate`** — conditionally independent copies sharing hyperparameters.
- **Latent fields in the likelihood's scale or precision** (atlas Theme 02 /
  A8). Effects still sum into the mean's linear predictor only.
- **A `Hyperparameter` on a `Shared` effect's own structural fields**
  (`precision`, `rho`, `phi`, `gamma`, `eta`, the MIDASParametric shapes).
  `ParametricDesignBlock` is a design-only mechanism; combining a
  Hyperparameter-driven precision with a Hyperparameter-driven design on one
  block is a genuine extension. The compiler rejects it with
  `CompilationError` at compile time -- only the `Shared.scale` may be a
  `Hyperparameter` today, and the effect's own fields must be fixed values.
- **YAML frontend for `Joint`.** The Python API lands first; the declarative
  surface follows once the shape is settled.
- **Mixed-outcome `predict` in a single call.**

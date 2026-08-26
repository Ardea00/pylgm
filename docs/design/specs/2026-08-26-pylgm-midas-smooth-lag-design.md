# MIDAS Smooth-Lag (S1) Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a `MIDAS` latent effect that renders a mixed-frequency
distributed-lag regressor the LGM-native way: expand a high-frequency (HF)
covariate into `K` lag columns aligned to each low-frequency (LF) target row,
then put a **random-walk smoothness penalty over the lag index** so the lag
weights form a penalized smooth curve (the U-MIDAS + RW2 rendering of the
restricted MIDAS lag polynomial). Second slice of the
[economic-expansion plan](../plans/2026-08-26-pylgm-economic-expansion.md);
delivers the Pillar-1 core (nowcast quarterly GDP / credit risk from
daily/monthly financial series).

**Architecture:** a new effect that reuses machinery wholesale.
- The **design** is dense: row `t` (one LF target row) × column `k` = the HF
  covariate at lag `k`. The latent field `β = (β₀…β_{K-1})` are the lag
  weights; the predictor contribution is `design @ β = Σₖ βₖ·xₜ₋ₖ`.
- The **precision** is a random-walk curvature penalty over the lag ordering
  plus a fixed diffuse ridge on the penalty's null space:
  `Q(τ) = τ·DᵀD + δ·P₀`, `D` the order-`m` difference operator (`m=1`→RW1,
  `m=2`→RW2, default 2), `P₀` the orthogonal projector onto `null(D)` (the
  `T = [1, k]` span for `m=2`, `[1]` for `m=1`), `δ` a small fixed precision.
  `DᵀD` penalizes curvature of the lag curve; its null space (`m=2`: constant +
  linear in lag; `m=1`: constant) is the lag weights' **level and slope — the
  covariate's actual effect — which MIDAS must NOT shrink with the smoothing
  strength.** `Q(τ)` diagonalizes exactly: curvature directions get precision
  `τ·λ` (penalized by `τ`), the level/slope directions get precision `δ`
  (**fixed, independent of `τ`**), because `P₀` and `DᵀD` act on orthogonal
  subspaces. So `τ` smooths the curve while the covariate's aggregate
  effect/drift stays free — the mixed-model "unpenalized polynomial + proper
  smooth" split, rendered as one block in lag coordinates. Unlike `RW2`, MIDAS
  carries **no sum-to-zero constraint**; `δ` (not a constraint) is what makes
  `Q` proper (PD). This is the standard mixed-model reparameterization of a
  penalized spline, chosen over a uniform ridge `τ·(DᵀD+εI)` precisely so the
  smoothing precision never touches level/slope.
- Because the `δ·P₀` term does **not** scale with `τ`, `Q(τ)` is not `τ·M`, so
  when `τ` is estimated MIDAS is a `ParametricBlock` (rebuild `τ·DᵀD + δ·P₀`
  per `τ`) — architecturally identical to how `AR1`/`ProperCAR`/`BYM2` already
  carry a parametric precision, **reusing that seam, no new engine machinery.**
  When `τ` is a fixed float, `Q` is a constant matrix and the effect is a plain
  fixed-precision `LatentBlock` (`ScalableBlock(block, None, 1.0)` in the
  family). Because it is proper and unconstrained (like
  `IID`/`ProperCAR`/`BYM2`), it works under **all three latent strategies**,
  including full Laplace.

`IID`/`ProperCAR` already ship proper, unconstrained precisions with empty
constraints (`np.empty((0, n))`) and the engine handles both fixed and
parametric variants — MIDAS is the same block shape with a dense lag design, so
**no inference, IR, or constraint code changes.** The design stays in lag
coordinates (labels = column names), so reporting, labels, and prediction are
untouched — no basis reconstruction.

**Tech stack:** numpy / scipy (`csr_matrix`, `diags`) / pandas. No new runtime
dependency.

## Scope

- **U-MIDAS only** (linear in the lag weights). The parametric restricted-MIDAS
  weight shapes (exp-Almon / Beta), which are nonlinear in 1–2 shape parameters
  and need a per-θ design rebuild, are **S2** (the one new IR seam) and out of
  scope here.
- **The K lag columns are supplied by the caller** as named frame columns in
  lag order (lag 0 … lag K−1). Building them from a raw HF series aligned to LF
  targets is one `pandas` `shift`/merge upstream and is the user's job (shown in
  the example); the builder assembles the *design matrix* from the named
  columns, it does not do HF→LF temporal alignment. (S2 reuses this design
  code.)
- **YAML frontend:** out of scope for S1 (Python API first), noted in the
  roadmap. The hybrid `MIDAS + BYM2 + AR1` nowcasting example + integration test
  is **S3**, not here.

## Effect API

```python
MIDAS(name, columns, precision=1.0, order=2, ridge=1e-6)
```

| field | meaning |
|---|---|
| `name` | effect name (unique in the predictor) |
| `columns` | `tuple[str, ...]` of frame columns in lag order (lag 0 first); `len > order` |
| `precision` | `float | Hyperparameter` — the smoothness precision `τ` |
| `order` | `1` or `2` — RW1 or RW2 penalty over the lag index (default `2`) |
| `ridge` | small positive float — the fixed precision `δ` on the penalty null space (level/slope), `τ`-independent (default `1e-6`, matching `Fixed.prior_precision`) |

Frozen dataclass, mirrors the other specs (`_ComposableEffect`, `_non_empty_string`,
`_positive_precision`). `columns` validated as a non-empty tuple of non-empty
strings with `len(columns) > order`; `order ∈ {1,2}`; `ridge` a finite positive
real (the fixed null-space precision `δ`). Labels = the column names (identify
each lag coefficient).

## Global constraints

- New effect returns a `LatentBlock` and wires through **both**
  `compiler.compile_lgm` (fixed-hyperparam) and `compiler.compile_family`
  (optimisable) branches, plus `effects/__init__.py` + `pylgm/__init__.py`
  exports and the `EffectSpec` union / `Predictor` isinstance tuples — the
  standard 5-site tax. Prediction context is a 6th site (see Task 2).
- Existing effects, engines, latent strategies, and the optimise/integrate
  paths stay behaviour-compatible; full suite green; `ruff check src tests`
  clean.
- Reuse the difference operator: extract the RW difference-operator
  construction from `build_random_walk` into a shared
  `difference_operator(n, order) -> csr_matrix` in `effects/random_walk.py` and
  call it from both. Pure refactor, covered by existing RW tests.
- One source of truth for the penalty pieces: a `midas_penalty(n, order) ->
  (DtD, P0)` helper in `effects/midas.py` returns `DᵀD` and the null-space
  projector `P₀ = T (TᵀT)⁻¹ Tᵀ` (`T = [1, k]` for order 2, `[1]` for order 1),
  used by both `build_midas` (fixed `τ`) and the `compile_family` parametric
  branch (`build: τ·DtD + δ·P0`).

---

### Task 1: `MIDAS` spec + `build_midas` builder

**Files:**
- Modify: `src/pylgm/effects/spec.py` (new `MIDAS` dataclass; add to `EffectSpec`
  and both `Predictor` isinstance tuples; `__all__`).
- Modify: `src/pylgm/effects/random_walk.py` (extract `difference_operator`).
- Create: `src/pylgm/effects/midas.py` (`build_midas`).
- Modify: `src/pylgm/effects/__init__.py`, `src/pylgm/__init__.py` (exports).
- Test: `tests/test_midas_builder.py`.

**`build_midas(frame, name, columns, precision, order, ridge) -> LatentBlock`:**
1. Validate `len(columns) > order`; every column present in `frame`; the
   selected values all finite (a NaN lag means the HF alignment left a gap —
   raise, naming the column).
2. `design = csr_matrix(frame[list(columns)].to_numpy(dtype=float))` — shape
   `(len(frame), K)`.
3. `DtD, P0 = midas_penalty(K, order)`;
   `precision_matrix = csr_matrix(precision * DtD + ridge * P0)`.
4. `constraints = np.empty((0, K))` (none — see architecture).
5. `return LatentBlock(name, tuple(columns), design, precision_matrix, constraints)`.

- [ ] **Step 1: failing tests** — builder + `midas_penalty` unit tests:
  - design shape `(n, K)` and equals the stacked columns.
  - `order=2` precision equals `τ·D₂ᵀD₂ + δ·P₀` for a hand-built `D₂` and
    `P₀ = T (TᵀT)⁻¹ Tᵀ`, `T = column_stack([ones(K), arange(K)])`; symmetric,
    PD (all eigenvalues `> 0`).
  - **`τ`-independence of the null space:** for a linear lag curve
    `β = a + b·k` (in `null(D₂)`), `β' Q(τ) β = δ·‖β‖²` for *any* `τ` — the
    curvature penalty contributes nothing and `δ` does not scale with `τ`. This
    is the load-bearing B property; assert it at two different `τ`.
  - constraints are `(0, K)` (unconstrained).
  - `len(columns) <= order` raises; a NaN in a lag column raises naming it;
    a missing column raises.
  - `order=1` uses the first-difference operator and `P₀ = (1/K)·11ᵀ`.
- [ ] **Step 2: implement** `difference_operator` extraction + `midas_penalty`
  + `build_midas` + spec + exports.
- [ ] **Step 3:** tests green; `ruff check`.

### Task 2: Compiler wiring + hyperparameter path + prediction

**Files:**
- Modify: `src/pylgm/compiler.py` (`compile_lgm` branch + `compile_family` branch).
- Verify/modify: `src/pylgm/inference/prediction.py` +
  `compiler.build_prediction_context` (MIDAS has no `.index`; the structured
  reconstruction path is one-hot-by-index and will not fit a dense lag design).
- Test: `tests/test_midas_effect.py`.

- [ ] **Step 1: failing tests**
  - **Fixed τ, end-to-end:** `Fixed("1") + MIDAS("lag", columns=(…), precision=1.0)`
    on a small Gaussian frame fits and `latent_marginals("lag").mean` has shape
    `(K,)`.
  - **Estimate τ:** `precision=Hyperparameter("lag.precision", …)` under
    `fit(...)` (empirical Bayes) and `hyperparameters="integrate"` both run and
    return a `lag.precision` estimate/marginal — confirms the `ParametricBlock`
    path (`τ·DᵀD + δ·P₀` rebuilt per `τ`, `_log_bounds` on `τ`, no bespoke
    hyperparameter).
  - **All strategies:** the fixed-τ fit succeeds under
    `latent_strategy="laplace"` (proof it is unconstrained), unlike `RW2`.
  - **Predict:** `result.predict(new_frame)` with the same lag columns on new
    rows returns predictions of the right length (rebuilds `design =
    new_frame[columns]`).
- [ ] **Step 2: implement**
  - `compile_lgm`: `elif isinstance(effect, MIDAS): precision =
    _resolved_precision(effect.precision); block = build_midas(frame,
    effect.name, effect.columns, precision, effect.order, effect.ridge);
    precisions[effect.name] = precision`.
  - `compile_family`: a dedicated MIDAS branch beside `AR1`/`ProperCAR`.
    Build the template once (`build_midas` at `τ`-initial for the parametric
    case, or the fixed `τ` otherwise) and `DtD, P0 = midas_penalty(K, order)`.
    - `τ` a `Hyperparameter` → `ParametricBlock(template, (τ.name,), build)`
      where `build(values) = csr_matrix(values[τ.name]*DtD + effect.ridge*P0)`;
      `parameter_names.append(τ.name)`, `parameter_bounds[τ.name] =
      _log_bounds(τ)`. Mirror the `AR1` closure (default-arg capture of
      `DtD`/`P0`/`ridge`/`τ.name`).
    - `τ` fixed float (another effect carries the hyperparameter) → precision is
      constant: `ScalableBlock(build_midas(..., τ, ...), None, 1.0)`.
  - Prediction: add a `("midas", (name, columns))` entry in
    `build_prediction_context` and a reconstruction arm in `prediction.py` that
    builds `design = csr_matrix(new_frame[list(columns)].to_numpy(float))`.
    **Fallback** if that arm is more than a few lines: raise a clear
    "predict on new rows is not yet supported for MIDAS" from the prediction
    path and keep in-sample `result` working — record it as a follow-up rather
    than expanding scope. Confirm which, don't assume.
- [ ] **Step 3:** tests green; full suite green; `ruff check`.

### Task 3: Statistical recovery test + docs + example

**Files:**
- Test: `tests/test_midas_effect.py` (extend).
- Docs: `docs/effects.md` (new MIDAS section) + `docs/roadmap.md`.
- Example: `examples/midas_nowcast/` (a runnable smooth-lag fit).

- [ ] **Step 1: recovery test** — simulate `y = Σₖ wₖ·xₜ₋ₖ + noise` from a known
  smooth decaying lag kernel `wₖ` (e.g. an exp-Almon shape) over `K≈12` lags;
  fit `MIDAS(order=2)` with estimated `τ`; assert the fitted lag curve (a) has
  much smaller total squared second difference than the unpenalised OLS lag
  coefficients (it is smoother), and (b) correlates strongly with the true `wₖ`.
  This is the "penalized lag curve recovers the MIDAS polynomial" claim.
- [ ] **Step 2: docs** — a MIDAS section in `docs/effects.md`: the U-MIDAS lag
  design, the RW-over-lag penalty, why it is unconstrained (level/slope are the
  effect), the `ridge` knob, works under all strategies, `τ` estimable; note
  the HF→LF alignment is the caller's (one `pandas` `shift`). Add a roadmap
  "Shipped/Next" line (MIDAS smooth-lag shipped; parametric exp-Almon/Beta =
  S2; hybrid nowcast = S3; YAML MIDAS later).
- [ ] **Step 3: example** — `examples/midas_nowcast/run.py`: build `K` monthly
  lag columns from a synthetic monthly indicator aligned to quarterly targets
  via `pandas` `shift`, fit `Fixed("1") + MIDAS(...)`, print the recovered lag
  curve vs. the simulating kernel. Self-contained (numpy/pandas/pylgm only).
- [ ] **Step 4:** full suite green; `ruff check src tests examples` clean.

---

## Design decisions (considered & chosen)

- **Mixed-model split (chosen) vs. uniform ridge.** The lag curve must smooth
  its *curvature* while leaving its *level and slope* (the covariate's actual
  effect) free of the smoothing precision `τ`. Two ways to make the improper RW
  penalty proper:
  - *Uniform ridge* `Q = τ·(DᵀD + ε·I)` — one `ScalableBlock`, simplest, but the
    ridge floor `τε` **scales with `τ`**: a large estimated `τ` also regularises
    level/slope, coupling the smoothing strength to the aggregate effect. A
    documented flaw, negligible only at realistic `τ`.
  - *Mixed-model split* `Q(τ) = τ·DᵀD + δ·P₀` (**chosen**) — `P₀` the null-space
    projector, `δ` a fixed precision. Diagonalizes so `τ` acts only on
    curvature and `δ` (τ-independent) only on level/slope. This is the
    unpenalized-polynomial + proper-smooth reparameterization, but rendered as
    **one block in lag coordinates** via the projector — so it gets the exact
    statistics (no `τ`↔level/slope coupling) without the two-coupled-blocks +
    cross-block reconstruction the naive split needs. Cost over the ridge: a
    `ParametricBlock` when `τ` is estimated instead of a `ScalableBlock`, which
    the engine already supports for `AR1`/`ProperCAR`/`BYM2`. `δ` defaults to
    `1e-6`, matching `Fixed.prior_precision` (the same diffuse prior every
    unpenalized coefficient gets).
- **Caller supplies lag columns vs. builder computes lags.** Computing lags
  needs HF↔LF frequency alignment, which pyLGM's LF-row-oriented data layer does
  not model; doing it in the builder would be fragile and duplicate pandas.
  U-MIDAS = "lags as regressors", so the caller (or a one-line `shift`) provides
  them. Chosen for S1; a `midas_lags` helper is YAGNI until S2/S3 show a shared
  need.

## Out of scope (later slices)

- **Parametric restricted MIDAS** (exp-Almon / Beta weights, per-θ design
  rebuild) — S2 (new `ParametricDesignBlock` IR seam).
- **Hybrid `MIDAS + BYM2 + AR1` nowcast example + integration test** — S3.
- **YAML `midas` effect type** — later; `columns` is a plain list, easy to add.
- **Automatic HF→LF alignment / a lag-construction helper** — pull in with S3 if
  a shared need appears.

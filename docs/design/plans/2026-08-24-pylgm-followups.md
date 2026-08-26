# Deferred Follow-Ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clear the tech-debt and performance follow-ups recorded across the last six slices, plus one deliberate breaking change.

**Architecture:** Five independent tasks, each in its own commit so a regression is attributable. Tasks 1, 2, 4 are behaviour-preserving; Task 3 is a **declared breaking change** (dataclass field order). Nothing here adds a feature.

**Tech Stack:** Python, NumPy. No new runtime dependency.

## Global Constraints

- `tests/inference/test_result_surface.py` is a golden baseline of the public result surface. Tasks 1, 2 and 4 must pass it with `result_surface_baseline.json` **unmodified**. Task 3 changes it deliberately, and its diff must contain **only** `repr` entries — no numeric key may move.
- The three result types must remain **siblings** under `_BaseResult` (never one inheriting another). `model._rebuild_result` dispatches `isinstance(LaplaceResult)` before `INLAResult`; `optimization/inla.py` tests `isinstance(reference, LaplaceResult)`. `test_result_types_are_siblings` pins it.
- Baseline regeneration is gated: `python tests/inference/test_result_surface.py --regenerate`.
- No new runtime dependency. Full suite green (770 currently); `ruff check src tests` clean.

---

### Task 1: Extract the bounded-parameter bounds block

**Files:**
- Modify: `src/pylgm/compiler.py`
- Test: `tests/test_compiler.py` (append)

**Context.** `compile_family` repeats the same eight-line dance for `ProperCAR.rho` (line ~409), `BYM2.phi` (~473) and `AR1.rho` (~531): enforce `transform == "logit"`, inset the interval, intersect any user `lower`/`upper`, check `initial` is inside, and build `OptimizationBounds(..., LogitTransform(a, b))`. They differ only in the interval source, the inset formula (`1e-6 * (b - a)` for proper CAR, a flat `1e-6` for the other two), and the error wording.

**Interfaces:**
- Produces: `_bounded_parameter(hyperparameter, lower, upper, *, label, inset) -> OptimizationBounds`, raising `CompilationError` for a non-logit transform or an out-of-interval `initial`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compiler.py` a test that, for each of the three effects, a non-logit hyperparameter raises `CompilationError` naming `transform='logit'` **and** the effect, and an out-of-interval `initial` raises naming the interval. Build the models through `LGM.fit` so the real path is exercised. These pass today — they are the safety net for the extraction, so run them BEFORE changing anything and confirm they pass, then keep them green.

- [ ] **Step 2: Extract the helper**

```python
def _bounded_parameter(
    hyperparameter: Hyperparameter,
    lower: float,
    upper: float,
    *,
    label: str,
    inset: float,
) -> OptimizationBounds:
    """Bounds for a hyperparameter confined to an open interval.

    Shared by proper CAR's rho, BYM2's phi and AR1's rho: all three are bounded
    parameters inferred on a logit scale, and all three must reject a log
    transform, which would silently inherit positive-only default bounds and
    confine the parameter to a wrong, one-sided interval.
    """
    if hyperparameter.transform != "logit":
        raise CompilationError(
            f"{label} {hyperparameter.name!r} must be declared with transform='logit' "
            f"(it is bounded to ({lower:.6g}, {upper:.6g})); got "
            f"transform={hyperparameter.transform!r}"
        )
    low = lower + inset if hyperparameter.lower is None else max(hyperparameter.lower, lower + inset)
    high = upper - inset if hyperparameter.upper is None else min(hyperparameter.upper, upper - inset)
    if not low <= hyperparameter.initial <= high:
        raise CompilationError(
            f"initial value for {label} {hyperparameter.name!r} must lie in "
            f"({low:.6g}, {high:.6g}); got {hyperparameter.initial}"
        )
    return OptimizationBounds(
        float(hyperparameter.initial), low, high, transform=LogitTransform(lower, upper)
    )
```

Call it from all three branches with `label="proper CAR rho"` / `"BYM2 phi"` / `"AR1 rho"`, `inset=1e-6 * (b - a)` for proper CAR and `inset=1e-6` for the other two. The helper returns the bounds; each branch still computes `low`/`high` for its `ParametricBlock` template — read them back off the returned object (`bounds.lower` / `bounds.upper`) rather than recomputing.

> The error **messages change slightly** (they become uniform). Update the three
> existing tests that match on them — grep `transform='logit'` and
> `must lie in` under `tests/`. Do not weaken the assertions; keep them matching
> the effect label and the interval.

- [ ] **Step 3: Verify**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
The golden baseline must be untouched (`git status`). Report the line count of `compiler.py` before and after.

- [ ] **Step 4: Commit**

```bash
git add src/pylgm/compiler.py tests/test_compiler.py
git commit -m "refactor: share the bounded-parameter bounds construction"
```

---

### Task 2: Stop a stalled Newton solve burning the whole iteration budget

**Files:**
- Modify: `src/pylgm/inference/laplace.py`
- Test: `tests/inference/test_laplace.py` (append)

**Context.** The decrement rescue added earlier fires only *after* `max_iterations`, so a stalled solve still runs the full 100 iterations before being accepted. Measured previously: ~26% of inner solves in a stalling `integrate` run burn the budget (36/138; 4344 Newton iterations against ~1093 if stalls stopped early).

**The constraint that shapes this.** Breaking on the decrement *unconditionally* inside the loop cures the stalls but stops earlier than the gradient test on well-scaled problems, relocating the mode — measured, it breaks two precision tests. So the in-loop check must fire **only when the iteration is no longer making progress**, leaving every currently-converging fit on exactly its present path.

**Interfaces:** no API change.

- [ ] **Step 1: Write the failing test**

```python
def test_a_stalled_solve_stops_early_instead_of_burning_the_budget():
    from pylgm.compiler import compile_lgm
    from pylgm.data import CanonicalPanel
    from pylgm.inference.laplace import fit_laplace

    model, frame = _stalling_poisson_model(0)
    panel = CanonicalPanel(frame, np.ones(len(frame), dtype=bool), ("t",), "y")
    compiled = compile_lgm(model, panel)

    result = fit_laplace(compiled, max_iterations=100)
    # It converges (the rescue or the early exit accepts it) ...
    assert result.diagnostics["newton_iterations"] < 100
    # ... and lands on the same mode as a long run.
    reference = fit_laplace(compiled, max_iterations=400)
    spread = np.sqrt(np.diag(reference.covariance))
    assert np.max(np.abs(result.mean - reference.mean) / spread) < 1e-4
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_laplace.py -q -k stalled`
Expected: FAIL — `newton_iterations` is currently 100.

- [ ] **Step 3: Add the stagnation-gated early exit**

Inside the iteration loop, after `slope` is computed and the line search has run, track whether the objective is still improving. When the improvement over the previous iterate falls below a small multiple of machine epsilon relative to the objective magnitude **and** the Newton decrement is below `tolerance`, accept convergence and break. Both conditions are required: stagnation alone is not optimality, and the decrement alone stops well-scaled fits too early.

Keep `max_iterations`, the line search, `_factor_positive_definite`, and the post-loop rescue exactly as they are — the rescue remains the backstop for anything this does not catch.

- [ ] **Step 4: Verify — the equivalence gate**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Every existing test must pass **with no tolerance loosened**, in particular
`test_laplace_poisson_mode_solves_the_score_equation` and
`test_laplace_bernoulli_intercept_matches_logit_of_rate`, which are precisely
the tests an over-eager in-loop exit breaks. The golden baseline must be
untouched.

Report the before/after `newton_iterations` for the stalling model and for a
normally-converging one (the latter must be unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference/laplace.py tests/inference/test_laplace.py
git commit -m "perf: exit a stalled Newton solve once it stops improving"
```

---

### Task 3: `_BaseResult` as a dataclass (BREAKING — field order)

**Files:**
- Modify: `src/pylgm/inference/result.py`
- Modify: `tests/inference/result_surface_baseline.json` (deliberately)

**Context.** Each result type still declares the same eleven shared fields and forwards them through an eleven-keyword `_init_common(...)` call — roughly 106 duplicated lines. (Outcome: only ~14 were actually removable — the bulk is the explicit `__init__` signatures and keyword forwarding, which the task's own constraints keep.) Making `_BaseResult` a dataclass lets the shared fields be declared once, with subclass fields appended after.

**This reorders `dataclasses.fields()` and therefore `repr`.** That is the breaking part, and the only thing that may change.

**Interfaces:** no change to constructor signatures or any public attribute.

- [ ] **Step 1: Convert**

Declare the eleven shared fields on `_BaseResult` (decorated `@dataclass(frozen=True, init=False)`), remove them from the three subclasses, keep each subclass's explicit `__init__` and its `_init_common(...)` call. The subclasses keep only their own fields.

> Keep `_init_common`'s `extra_validate`/`extra_store` hooks exactly as they are.
> They encode the discovered fact that the storage epilogue validates; removing
> them reintroduces the ordering bug a previous review caught.

- [ ] **Step 2: Confirm the blast radius is repr-only**

Run: `PYTHONPATH=src python -m pytest tests/inference/test_result_surface.py -q` — it will FAIL on `repr`.
Regenerate: `PYTHONPATH=src python tests/inference/test_result_surface.py --regenerate`
Then **walk the diff key by key** and confirm every changed path ends in `.repr`. If any numeric key, `isinstance` entry, `immutability` flag or error message moved, STOP — that is a bug, not the intended reorder. Report the changed key count and one before/after `repr` example.

- [ ] **Step 3: Verify**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`
Confirm the sibling invariant still holds (`test_result_types_are_siblings`), and report `result.py`'s line count before and after.

- [ ] **Step 4: Commit**

```bash
git add src/pylgm/inference/result.py tests/inference/result_surface_baseline.json
git commit -m "refactor: declare the shared result fields once"
```

---

### Task 4: Three small correctness/clarity fixes

**Files:**
- Modify: `src/pylgm/inference/result.py`, `src/pylgm/compiler.py`
- Test: `tests/inference/test_result_surface.py`, `tests/test_compiler.py`

- [ ] **Step 1: Pin the `_BACKING_FIELDS` pairing**

`_BaseResult._BACKING_FIELDS` hand-lists both the private and public name of every shared field, with nothing enforcing the pairing — a new shared field silently escapes the `__getattr__` guard if only one name is added. Add a test deriving the expected set from the class's properties and comparing, so the two cannot drift.

- [ ] **Step 2: Restore the integrated-predict distinction**

`INLAResult.predict`'s override was deleted (it was code-identical), taking with it the docstring noting the posterior is hyperparameter-integrated. Fold that distinction into `_BaseResult.predict`'s docstring — it should say the latent posterior is whatever the result carries, which for `INLAResult` is the integrated one — rather than restoring a duplicate method.

- [ ] **Step 3: Wrap effect-compilation errors in `compile_family` too**

`compile_lgm` wraps builder failures as `CompilationError(f"failed to compile effect {name!r}: ...")`; `compile_family` has no equivalent, so the same malformed effect (e.g. a single-level AR1) raises a bare `ValueError` when a hyperparameter is declared and a `CompilationError` when one is not. Wrap the builder calls in `compile_family` the same way, and add a test asserting both paths raise `CompilationError` naming the effect for a single-level AR1.

- [ ] **Step 4: Verify and commit**

Run: `PYTHONPATH=src python -m pytest -q && PYTHONPATH=src ruff check src tests`; the golden baseline must be untouched by Steps 1–2 (Step 3 does not touch result surfaces).

```bash
git add src/pylgm/ tests/
git commit -m "fix: pin the guard field pairing and unify effect-compilation errors"
```

---

### Task 5: Documentation

**Files:** `README.md`, `docs/superpowers/specs/2026-08-08-pylgm-latte-pyspark-architecture.md`

- [ ] **Step 1: Record the breaking change**

Note that result `repr` output changed (shared fields now print before type-specific ones) as a consequence of declaring them once. Values, attribute names and behaviour are unchanged. Keep it brief and accurate — do NOT claim any API rename or removal.

- [ ] **Step 2: Update the roadmap**

Mark done: the bounded-parameter bounds extraction, the in-loop Newton decrement, `_BaseResult` as a dataclass, and the small fixes. Keep recorded as still-deferred, with one line each on why: predictive quantiles/simulation, Spark `new_data`, config-file spatial/AR1 effect types, and irregular AR1 spacing — all four are features needing their own design, not cleanup.

- [ ] **Step 3: Verify and commit**

Run any touched README example verbatim; then `PYTHONPATH=src python -m pytest -q`.

```bash
git add README.md docs/
git commit -m "docs: record the follow-up cleanup and the repr change"
```

---

## Self-Review Notes

- **Independence:** the five tasks touch disjoint code (`compiler.py`, `laplace.py`, `result.py`, then docs) and are separately committed, so any regression is attributable. Task 4 Step 3 touches `compiler.py` after Task 1 — sequence them in order to avoid a conflict.
- **The golden baseline is the gate throughout**: untouched for Tasks 1, 2, 4; deliberately regenerated in Task 3 with a repr-only diff. That asymmetry is the whole reason Task 3 is separate.
- **Task 2's real risk** is stopping well-scaled fits early. The stagnation gate exists specifically to prevent that, and the two precision tests that an unconditional in-loop decrement breaks are named in the verification step as the canary.
- **Task 1 changes error messages** (they become uniform across the three effects); the plan says so and points at the tests to update, rather than letting an implementer discover it as a surprise failure.
- **Not in scope, and why:** predictive quantiles, Spark `new_data`, config-file effect types and irregular AR1 spacing are features with unresolved design questions; bundling them into an execution branch is how the riskiest defects in this project were introduced.

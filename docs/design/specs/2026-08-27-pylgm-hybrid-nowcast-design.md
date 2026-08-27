# S3 — Hybrid MIDAS + BYM2 + AR1 Nowcasting Example (Design Spec)

**Slice:** S3 of the economic-context expansion
(`docs/design/plans/2026-08-26-pylgm-economic-expansion.md`, Pillar 1).

**Goal:** Prove Pillar 1 end to end — that `+` composes a mixed-frequency
high-frequency (HF) term, a spatial-network term, and a temporal term into one
latent Gaussian field that fits and predicts. The deliverable is a runnable
example plus an integration test. **No new `src/` code**: this slice is pure
composition over effects that already exist (`MIDAS`, `MIDASParametric`,
`BYM2`, `AR1`, `Fixed`).

## Motivation

Pillar 1 of the expansion is "Hybrid MIDAS + BYM2 + temporal." S1 shipped
U-MIDAS, S2 shipped parametric MIDAS, and BYM2/AR1 already existed. The
"hybrid" is claimed to be free because `+` already sums effects into one
latent field. S3 is the proof: a regional-GDP nowcast whose target is driven
by an HF indicator (via a lag kernel), a spatial signal over a region graph,
and a quarterly temporal signal — recovered by a single composed model.

## Scope

In scope:
- One self-contained example directory building a synthetic regional-GDP panel
  and fitting **two** hybrid models on it (U-MIDAS and parametric MIDAS),
  sharing the same frame and graph.
- An integration test asserting both models fit, compose, and predict.
- A subprocess smoke-test entry so the example runs in CI.
- A roadmap update marking S3 / Pillar 1 shipped.

Out of scope:
- Any change to `src/pylgm/`. If the example cannot be expressed with the
  current effect/compiler APIs, that is a bug to file, not to patch inside S3.
- Real economic data or any new runtime dependency (data is generated inline).
- Out-of-sample prediction onto unseen regions/quarters — `predict()` is
  exercised as an in-sample round-trip over the training frame, which already
  drives the full prediction context (including the parametric-MIDAS
  fitted-weights path).

## Synthetic data

A regional-GDP nowcast keyed by `(region, quarter)`, generated inline (no
`data.csv`), deterministic under a fixed seed:

- `R ≈ 8` regions, each a string id, with an inline adjacency
  `{region: [neighbour, ...]}` (list form, undirected, connected).
- `T ≈ 40` quarters, integer index `0..T-1`.
- A monthly HF indicator aligned to each quarterly row via `pandas.Series.shift`
  (the same alignment `examples/midas_nowcast/run.py` uses), producing `K ≈ 12`
  lag columns `ind_lag0 … ind_lag{K-1}` per region-quarter row.
- Target:
  `y[r,t] = μ + Σ_k w_k · x[r,t,k] + s[r] + a[t] + ε`, where
  - `w_k` is a **known** smooth decaying lag kernel (exp-Almon-shaped), the
    same object both MIDAS variants must recover;
  - `s[r]` is a spatial signal that varies smoothly over the region graph
    (neighbouring regions similar);
  - `a[t]` is an AR1-correlated quarterly signal;
  - `ε` is small Gaussian noise.

The frame has one row per `(region, quarter)`, columns: `region` (str),
`quarter` (int), `ind_lag0..K-1` (float), `y` (float).

## Models

Response is continuous → Gaussian likelihood, `engine="exact_gaussian"`.

Shared spatial + temporal terms:

```python
common = (
    BYM2("region", index="region", graph=graph,
         precision=Hyperparameter("region.precision", initial=1.0), phi=0.5)
    + AR1("quarter", index="quarter",
          precision=Hyperparameter("quarter.precision", initial=1.0), rho=0.6)
)
```

Fit A — U-MIDAS (S1):

```python
Fixed("1") + MIDAS("lag", columns=cols,
                   precision=Hyperparameter("lag.precision", initial=1.0)) + common
```

Fit B — parametric MIDAS (S2):

```python
Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon") + common
```

**Fixed mixing parameters.** BYM2 `phi` and AR1 `rho` are passed as fixed
floats, not `Hyperparameter`s. Only the precisions and the MIDAS
smoothing/shape parameters are estimated. Rationale: BYM2 `φ` is weakly
identified on short panels (an expansion-plan ceiling), and fixing the two
mixing parameters keeps the example fast and the integration test
deterministic. This is a deliberate simplification of the example, not a
limitation of the effects — both `phi` and `rho` accept `Hyperparameter`s and
are exercised as such by their own slices' tests. The example's `run.py`
carries a `ponytail:` comment stating this and the upgrade path (pass
`Hyperparameter(...)` to estimate them).

## Integration test

`tests/integration/test_hybrid_nowcast.py`. Build the synthetic panel once;
for **each** of the two fits assert:

1. `model.fit(frame)` returns without raising.
2. `result.labels` contain a label from every composed block — the fixed
   intercept, the MIDAS/parametric term, the `region` block, and the `quarter`
   block — proving the four effects composed through the compiler into one
   field.
3. `result.predict(frame).predictive_mean` is all finite.
4. `corr(predictive_mean, y) > 0.8` — the composed field tracks the target.
5. The recovered lag weights correlate with the true kernel `> 0.85`:
   - U-MIDAS: `result.latent_marginals("lag").mean` vs the true `w_k`.
   - parametric: the fitted weight vector `midas_weights("exp_almon", K, θ̂)`
     (θ̂ read from `result.hyperparameters`) vs the true `w_k`.

Thresholds (`0.8`, `0.85`) are chosen with margin against the fixed seed so the
test is not flaky; they assert real recovery, not a tautology.

## Example script + README

- `examples/hybrid_nowcast/run.py`: a `main()` that builds the panel, fits both
  models, and prints — per fit — the estimated hyperparameters, the
  `corr(predictive_mean, y)`, and the recovered-vs-true kernel correlation. It
  runs from the repo root via `PYTHONPATH=src python examples/hybrid_nowcast/run.py`
  and exits 0. Style mirrors `examples/midas_nowcast/run.py`.
- `examples/hybrid_nowcast/README.md`: explains the panel, shows both composed
  model expressions, and contrasts the two MIDAS renderings (U-MIDAS spends a
  smoothed coefficient per lag; parametric spends two shape parameters), linking
  the effects guide.

## CI wiring

Add a subprocess entry for `examples/hybrid_nowcast/run.py` to the example
smoke tests in `tests/test_package.py`, matching the existing pattern
(run under the current interpreter with `PYTHONPATH=src`, assert returncode 0).

## Files

- Create: `examples/hybrid_nowcast/run.py`
- Create: `examples/hybrid_nowcast/README.md`
- Create: `tests/integration/test_hybrid_nowcast.py`
- Modify: `tests/test_package.py` (add the smoke entry)
- Modify: `docs/roadmap.md` (mark S3 / Pillar-1 hybrid shipped)

No `src/` changes.

## Global constraints (inherited)

- No new runtime dependency; numpy / pandas / stdlib only. Deterministic, no
  MCMC.
- Full suite green; `ruff check src tests` clean (no unused imports — the
  recurring F401 trap in this codebase's tests).
- New tests pass under `-W error::UserWarning`.
- Existing effects, engines, and the optimise/integrate paths stay
  behavior-compatible (this slice adds no `src/` code, so this holds by
  construction).

## Ceilings

- The synthetic panel is small (dense solve well under the preflight guard);
  scaling the hybrid to realistic firm/region counts is E-sparse's job, not
  S3's.
- `φ`/`ρ` fixed in the example (see above); estimating them is a one-line change
  a reader can make, documented in the README.

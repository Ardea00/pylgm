# S3 — Hybrid MIDAS + BYM2 + AR1 Nowcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Pillar 1 — that `+` composes an HF MIDAS term, a spatial BYM2 term, and a temporal AR1 term into one latent field that fits and predicts — with a runnable example and an integration test, and **no new `src/` code**.

**Architecture:** A synthetic regional-GDP panel (regions × quarters) whose target is driven by a known exp-Almon lag kernel, a smooth spatial signal over a ring graph, and an AR1 temporal signal. Two hybrid models — U-MIDAS (`MIDAS`) and parametric (`MIDASParametric`) — share the same `BYM2 + AR1` common term and are fit on the same frame. This is pure composition over effects that already ship.

**Tech Stack:** Python, numpy, pandas, pytest. `pylgm` exact-Gaussian engine. No new dependency.

## Global Constraints

- No new runtime dependency; numpy / pandas / stdlib only. Deterministic, no MCMC. (spec §"Global constraints")
- No changes to `src/pylgm/`. If the example cannot be expressed with current APIs, that is a bug to file, not to patch inside S3. (spec §Scope)
- `ruff check src tests` must be clean — **no unused imports** (the recurring F401 trap in this codebase's tests). (spec §"Global constraints")
- New tests pass under `-W error::UserWarning`.
- BYM2 `phi` and AR1 `rho` are passed as **fixed floats**, not `Hyperparameter`s — deliberate example simplification, carries a `ponytail:` comment with the upgrade path. (spec §Models)
- Recovery thresholds asserted with margin against the fixed seed: `corr(predictive_mean, y) > 0.8`, `corr(recovered kernel, true) > 0.85`. (spec §"Integration test")
- Panel shape: `R = 8` regions (string ids), `T = 40` quarters (int `0..T-1`), `K = 12` lag columns `ind_lag0..ind_lag11`; ring adjacency (list form); `SEED = 0`. (spec §"Synthetic data")

---

## File Structure

- `examples/hybrid_nowcast/run.py` — builds the panel, fits both models, prints per-fit hyperparameters + `corr(pred, y)` + recovered-vs-true kernel correlation. Standalone-runnable, mirrors `examples/midas_nowcast/run.py`.
- `examples/hybrid_nowcast/README.md` — explains the panel and the two composed model expressions.
- `tests/integration/test_hybrid_nowcast.py` — self-contained DGP + 5 assertions per fit (matches `test_nic_shaped_backtest.py`'s self-contained-DGP convention).
- `tests/test_package.py` — add one subprocess smoke entry for the example.
- `docs/roadmap.md` — mark hybrid composition demonstrated.

The example and the integration test each carry their own copy of the small DGP. That is the repo convention (`test_nic_shaped_backtest.py` owns `synthetic_nic_frame()`); two copies is rule-of-two, not rule-of-three — no shared home is warranted, and an example must stay standalone-runnable while a test must stay self-contained.

---

### Task 1: Hybrid example script, README, and CI smoke entry

**Files:**
- Create: `examples/hybrid_nowcast/run.py`
- Create: `examples/hybrid_nowcast/README.md`
- Modify: `tests/test_package.py` (add one smoke-test function)

**Interfaces:**
- Consumes (all already shipped): `from pylgm import AR1, BYM2, Fixed, Gaussian, Hyperparameter, LGM, MIDAS, MIDASParametric`; `from pylgm.effects.midas import midas_weights`.
- Produces: an executable `examples/hybrid_nowcast/run.py` whose `main()` exits 0 and prints lines containing `corr(pred,y)=` for both fits. Nothing imports from it.

- [ ] **Step 1: Write the example script**

Create `examples/hybrid_nowcast/run.py`:

```python
"""Hybrid nowcast: one latent field composed from a MIDAS lag term, a BYM2
spatial term, and an AR1 temporal term, fit two ways (U-MIDAS and parametric).

Run from the repo root:
    PYTHONPATH=src python examples/hybrid_nowcast/run.py
"""
import numpy as np
import pandas as pd

from pylgm import AR1, BYM2, Fixed, Gaussian, Hyperparameter, LGM, MIDAS, MIDASParametric
from pylgm.effects.midas import midas_weights

R, T, K = 8, 40, 12
SEED = 0


def build():
    """Synthetic regional-GDP panel keyed by (region, quarter).

    y = mu + sum_k w_k * x_lag_k + s[region] + a[quarter] + noise, where w_k is a
    known exp-Almon decay, s is smooth over the ring graph, a is AR1 over quarters.
    """
    rng = np.random.default_rng(SEED)
    regions = [str(i) for i in range(R)]
    graph = {str(i): [str((i - 1) % R), str((i + 1) % R)] for i in range(R)}  # ring
    w_true = midas_weights("exp_almon", K, (0.2, -0.05))
    s = 1.2 * np.sin(2 * np.pi * np.arange(R) / R)  # smooth spatial signal
    a = np.zeros(T)  # AR1 temporal signal
    for t in range(1, T):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    cols = tuple(f"ind_lag{k}" for k in range(K))
    rows = []
    for ri, reg in enumerate(regions):
        monthly = rng.normal(size=T + K)
        lagged = {name: pd.Series(monthly).shift(k) for k, name in enumerate(cols)}
        rdf = pd.DataFrame(lagged).iloc[K:].reset_index(drop=True)  # drop NaN warmup
        rdf = rdf.iloc[:T].reset_index(drop=True)
        hf = rdf[list(cols)].to_numpy() @ w_true
        rdf["region"] = reg
        rdf["quarter"] = np.arange(len(rdf))
        rdf["y"] = 0.5 + 3.0 * hf + s[ri] + a[: len(rdf)] + rng.normal(scale=0.3, size=len(rdf))
        rows.append(rdf)
    return pd.concat(rows, ignore_index=True), cols, graph, w_true


def main():
    frame, cols, graph, w_true = build()
    print(f"panel: {len(frame)} rows, {R} regions x {T} quarters, {K} lags")

    # ponytail: phi/rho fixed as floats -> deterministic, fast example. Both accept
    # Hyperparameter(...) to estimate them (weakly identified on short panels).
    common = (
        BYM2("region", index="region", graph=graph,
             precision=Hyperparameter("region.precision", initial=1.0), phi=0.5)
        + AR1("quarter", index="quarter",
              precision=Hyperparameter("quarter.precision", initial=1.0), rho=0.6)
    )

    # Fit A: U-MIDAS (smoothed coefficient per lag)
    mA = LGM(response="y", panel=("region",), time="quarter",
             predictor=Fixed("1")
             + MIDAS("lag", columns=cols, precision=Hyperparameter("lag.precision", initial=1.0))
             + common,
             likelihood=Gaussian(sigma=0.3))
    rA = mA.fit(frame)
    predA = rA.predict(frame).predictive_mean
    kernelA = rA.latent_marginals("lag").mean
    print(f"[A U-MIDAS] corr(pred,y)={np.corrcoef(predA, frame['y'])[0, 1]:.3f} "
          f"corr(kernel,true)={np.corrcoef(kernelA, w_true)[0, 1]:.3f} "
          f"hyper={ {k: round(v, 3) for k, v in rA.hyperparameters.items()} }")

    # Fit B: parametric MIDAS (two shape parameters)
    mB = LGM(response="y", panel=("region",), time="quarter",
             predictor=Fixed("1")
             + MIDASParametric("m", cols, kernel="exp_almon")
             + common,
             likelihood=Gaussian(sigma=0.3))
    rB = mB.fit(frame)
    predB = rB.predict(frame).predictive_mean
    theta_hat = (rB.hyperparameters["m.shape1"], rB.hyperparameters["m.shape2"])
    kernelB = midas_weights("exp_almon", K, theta_hat)
    print(f"[B param ] corr(pred,y)={np.corrcoef(predB, frame['y'])[0, 1]:.3f} "
          f"corr(kernel,true)={np.corrcoef(kernelB, w_true)[0, 1]:.3f} "
          f"theta_hat=({theta_hat[0]:.3f}, {theta_hat[1]:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example to verify it exits 0 and prints both fits**

Run: `PYTHONPATH=src python examples/hybrid_nowcast/run.py`
Expected: exit 0; two lines, `[A U-MIDAS] corr(pred,y)=0.98...` and `[B param ] corr(pred,y)=0.98...`, each with `corr(kernel,true)` ≈ 0.99. (Verified numbers from the prototype: A = 0.981 / 0.998, B = 0.980 / 0.999, theta_hat ≈ (0.207, -0.049).)

- [ ] **Step 3: Write the README**

Create `examples/hybrid_nowcast/README.md`:

```markdown
# Hybrid nowcast — MIDAS + BYM2 + AR1 in one latent field

This example proves the core composition claim: `+` sums a high-frequency
mixed-frequency term, a spatial-network term, and a temporal term into a
single latent Gaussian field that fits and predicts.

## The panel

`build()` generates a deterministic regional-GDP nowcast keyed by
`(region, quarter)`: `R = 8` regions on a ring graph, `T = 40` quarters, and a
monthly high-frequency indicator aligned into `K = 12` lag columns
`ind_lag0..ind_lag11` (via `pandas.Series.shift`, the same alignment
`examples/midas_nowcast` uses). The target is

```
y[r,t] = mu + sum_k w_k * x[r,t,k] + s[r] + a[t] + noise
```

with a known exp-Almon lag kernel `w_k`, a spatial signal `s[r]` that varies
smoothly over the ring, and an AR1 temporal signal `a[t]`.

## The two composed models

Both share the same spatial + temporal common term:

```python
common = (
    BYM2("region", index="region", graph=graph, precision=..., phi=0.5)
    + AR1("quarter", index="quarter", precision=..., rho=0.6)
)
```

- **A — U-MIDAS:** `Fixed("1") + MIDAS("lag", columns=cols, precision=...) + common`.
  Spends a smoothed coefficient per lag; recovers the kernel as
  `result.latent_marginals("lag").mean`.
- **B — parametric MIDAS:** `Fixed("1") + MIDASParametric("m", cols, kernel="exp_almon") + common`.
  Spends only two shape parameters; recovers the kernel as
  `midas_weights("exp_almon", K, theta_hat)` from the fitted
  `result.hyperparameters`.

Both recover the true kernel and track the target (`corr > 0.98` on the fixed
seed). See the [effects guide](../../docs/effects.md).

## Fixed mixing parameters

`phi` (BYM2) and `rho` (AR1) are passed as fixed floats to keep the example
fast and deterministic — both are weakly identified on a panel this short.
Both accept a `Hyperparameter(...)` instead to estimate them; that is a
one-line change per term.

## Run

```
PYTHONPATH=src python examples/hybrid_nowcast/run.py
```
```

- [ ] **Step 4: Add the CI smoke-test entry**

In `tests/test_package.py`, append (after the last example smoke test, before `test_installed_console_entrypoint_loads_and_reports_help`), matching the existing pattern exactly:

```python
def test_hybrid_nowcast_example_reports_correlation():
    root = Path(__file__).parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, str(root / "examples/hybrid_nowcast/run.py")],
        capture_output=True, check=False, text=True, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "corr(pred,y)=" in completed.stdout
```

- [ ] **Step 5: Run the smoke test**

Run: `PYTHONPATH=src python -m pytest tests/test_package.py::test_hybrid_nowcast_example_reports_correlation -q -W error::UserWarning`
Expected: PASS.

- [ ] **Step 6: Ruff check**

Run: `ruff check examples/hybrid_nowcast/run.py tests/test_package.py`
Expected: clean (no F401 / no unused imports).

- [ ] **Step 7: Commit**

```bash
git add examples/hybrid_nowcast/run.py examples/hybrid_nowcast/README.md tests/test_package.py
git commit -m "feat: hybrid MIDAS+BYM2+AR1 nowcast example + CI smoke test"
```

---

### Task 2: Integration test — both fits compose, predict, and recover

**Files:**
- Create: `tests/integration/test_hybrid_nowcast.py`

**Interfaces:**
- Consumes (all shipped): same imports as Task 1, plus `pytest`. Carries its own DGP (self-contained, per repo convention).
- Produces: a parametrized test asserting, for each of the two fits: fit succeeds, all four composed blocks resolve via `latent_marginals`, `predict` is finite, `corr(pred, y) > 0.8`, `corr(recovered kernel, true) > 0.85`.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_hybrid_nowcast.py`:

```python
import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, BYM2, Fixed, Gaussian, Hyperparameter, LGM, MIDAS, MIDASParametric
from pylgm.effects.midas import midas_weights

R, T, K = 8, 40, 12
SEED = 0


def _panel():
    """Deterministic regional-GDP panel; see examples/hybrid_nowcast for the DGP."""
    rng = np.random.default_rng(SEED)
    regions = [str(i) for i in range(R)]
    graph = {str(i): [str((i - 1) % R), str((i + 1) % R)] for i in range(R)}  # ring
    w_true = midas_weights("exp_almon", K, (0.2, -0.05))
    s = 1.2 * np.sin(2 * np.pi * np.arange(R) / R)
    a = np.zeros(T)
    for t in range(1, T):
        a[t] = 0.6 * a[t - 1] + rng.normal(scale=0.5)
    cols = tuple(f"ind_lag{k}" for k in range(K))
    rows = []
    for ri, reg in enumerate(regions):
        monthly = rng.normal(size=T + K)
        lagged = {name: pd.Series(monthly).shift(k) for k, name in enumerate(cols)}
        rdf = pd.DataFrame(lagged).iloc[K:].reset_index(drop=True)
        rdf = rdf.iloc[:T].reset_index(drop=True)
        hf = rdf[list(cols)].to_numpy() @ w_true
        rdf["region"] = reg
        rdf["quarter"] = np.arange(len(rdf))
        rdf["y"] = 0.5 + 3.0 * hf + s[ri] + a[: len(rdf)] + rng.normal(scale=0.3, size=len(rdf))
        rows.append(rdf)
    return pd.concat(rows, ignore_index=True), cols, graph, w_true


def _common(graph):
    # ponytail: phi/rho fixed floats -> deterministic test; both accept Hyperparameter.
    return (
        BYM2("region", index="region", graph=graph,
             precision=Hyperparameter("region.precision", initial=1.0), phi=0.5)
        + AR1("quarter", index="quarter",
              precision=Hyperparameter("quarter.precision", initial=1.0), rho=0.6)
    )


def _fit_umidas(frame, cols, graph):
    model = LGM(response="y", panel=("region",), time="quarter",
                predictor=Fixed("1")
                + MIDAS("lag", columns=cols, precision=Hyperparameter("lag.precision", initial=1.0))
                + _common(graph),
                likelihood=Gaussian(sigma=0.3))
    result = model.fit(frame)
    kernel = result.latent_marginals("lag").mean
    return result, "lag", kernel


def _fit_parametric(frame, cols, graph):
    model = LGM(response="y", panel=("region",), time="quarter",
                predictor=Fixed("1")
                + MIDASParametric("m", cols, kernel="exp_almon")
                + _common(graph),
                likelihood=Gaussian(sigma=0.3))
    result = model.fit(frame)
    theta_hat = (result.hyperparameters["m.shape1"], result.hyperparameters["m.shape2"])
    kernel = midas_weights("exp_almon", K, theta_hat)
    return result, "m", kernel


@pytest.mark.parametrize("fit", [_fit_umidas, _fit_parametric])
def test_hybrid_composes_predicts_and_recovers(fit):
    frame, cols, graph, w_true = _panel()

    result, midas_block, kernel = fit(frame, cols, graph)  # (1) fit succeeds

    # (2) every composed block resolves to finite latent marginals
    for block in ("fixed", midas_block, "region", "quarter"):
        assert np.isfinite(result.latent_marginals(block).mean).all()

    # (3) prediction is finite
    pred = result.predict(frame).predictive_mean
    assert np.isfinite(pred).all()

    # (4) the composed field tracks the target
    assert np.corrcoef(pred, frame["y"].to_numpy())[0, 1] > 0.8

    # (5) the recovered lag kernel matches the true kernel
    assert np.corrcoef(kernel, w_true)[0, 1] > 0.85
```

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=src python -m pytest tests/integration/test_hybrid_nowcast.py -v -W error::UserWarning`
Expected: 2 passed (both parametrizations). Prototype margins: `corr(pred,y)` ≈ 0.98 (> 0.8), `corr(kernel,true)` ≈ 0.998/0.999 (> 0.85).

- [ ] **Step 3: Ruff check**

Run: `ruff check tests/integration/test_hybrid_nowcast.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_hybrid_nowcast.py
git commit -m "test: hybrid nowcast integration test (compose + predict + recover, both MIDAS variants)"
```

---

### Task 3: Roadmap — mark hybrid composition demonstrated

**Files:**
- Modify: `docs/roadmap.md`

**Interfaces:**
- Consumes: nothing. Produces: nothing (docs only).

- [ ] **Step 1: Add the shipped bullet**

In `docs/roadmap.md`, in the `## Shipped in 0.3` list, immediately after the `**Effects**` bullet (the one ending "…the Knorr-Held `SpaceTime` interaction (Types I–IV). See [effects](effects.md)."), insert:

```markdown
- **Hybrid composition** — a mixed-frequency `MIDAS` term, a spatial `BYM2`
  term, and a temporal `AR1` term sum through `+` into one latent field that
  fits and predicts, demonstrated end to end in
  [`examples/hybrid_nowcast`](https://github.com/Ardea00/pylgm/tree/main/examples/hybrid_nowcast).
```

Leave `## Next` item 7 (the hybrid **frontend & config-file `midas` type**) unchanged — S3 ships the composition proof, not the frontend.

- [ ] **Step 2: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs: note hybrid MIDAS+BYM2+AR1 composition shipped (example)"
```

---

## Self-Review

**1. Spec coverage:**
- Synthetic panel (spec §"Synthetic data") → Task 1 `build()` / Task 2 `_panel()`. ✓ (R=8, T=40, K=12, ring graph, seed 0, exp-Almon kernel, spatial + AR1 signals)
- Two hybrid fits sharing frame + graph (spec §Models) → Tasks 1 & 2, U-MIDAS + parametric over shared `common`. ✓
- Fixed φ/ρ with ponytail comment + upgrade path (spec §Models) → `ponytail:` comment in run.py and test; README "Fixed mixing parameters" section. ✓
- Integration test, 5 assertions per fit (spec §"Integration test") → Task 2, parametrized over both fits. ✓
- CI smoke entry (spec §"CI wiring") → Task 1 Step 4. ✓
- Example + README (spec §"Example script + README") → Task 1. ✓
- Roadmap update (spec §Files) → Task 3. ✓
- No `src/` changes (spec §Scope) → no task touches `src/`. ✓

**2. Placeholder scan:** No TBD/TODO; every code step is complete and lifted from the verified prototype. ✓

**3. Type consistency:** `latent_marginals(block).mean` used identically in run.py and test; `hyperparameters["m.shape1"]/["m.shape2"]` and `midas_weights("exp_almon", K, theta_hat)` consistent across both files and the S2 tests; block names `("fixed", "lag"/"m", "region", "quarter")` match the effect `name=` arguments. ✓

**4. Duplication note:** DGP appears twice (example + test) — rule-of-two, repo convention (`test_nic_shaped_backtest.py`), no shared home; not extracted. ✓

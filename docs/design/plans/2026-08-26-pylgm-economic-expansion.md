# pyLGM Economic-Context Expansion Plan

> **Scope:** a multi-slice roadmap, not a single-feature implementation plan.
> Each slice below is independently shippable and gets its own detailed
> spec + task-by-task plan (in `docs/design/{specs,plans}`) when it is picked
> up. Build order is at the bottom; start from the enablers.

**Goal:** make pyLGM usable for economic modelling — nowcasting with
mixed-frequency data, and dependence over firm / bank / ownership /
supply-chain networks — by adding MIDAS, weighted & dynamic network structure,
and Knorr-Held space-time interaction, plus supporting likelihoods.

**Driving insight:** the CAR math is graph-agnostic. `Q = τ(D−W)` never cared
whether `W` is geography. Make `W` accept edge weights and vary over time and
the *entire* CAR family (`Besag`/`ProperCAR`/`BYM2`) becomes a firm/bank/
ownership/supply-chain model with no new effect family — including BYM2's `φ`,
which then reads as *"fraction of variance explained by network dependence vs.
idiosyncratic."* The network pillar is therefore mostly a change at one file
(`effects/graph.py`), not a new inference path.

**Tech stack:** numpy / scipy / pandas only — no new runtime dependency, no
MCMC. All slices reuse the existing `LatentBlock(design, precision,
constraints, labels)` contract and the `ScalableBlock` / `ParametricBlock`
hyperparameter seam except where explicitly noted (S2).

## Global constraints

- No new runtime dependency; deterministic-approximation-first (no MCMC).
- Every new effect returns a `LatentBlock` and wires through **both**
  `compiler.compile_lgm` (fixed-hyperparam) and `compiler.compile_family`
  (optimisable) branches, plus the `effects/__init__.py` + `pylgm/__init__.py`
  exports and the `EffectSpec` union / `Predictor` isinstance tuples.
- New bounded hyperparameters reuse `optimization/transforms.py`
  (`Log`/`Logit`) and participate in `optimize` and `integrate`.
- Existing effects, engines, latent strategies, priors, and the
  optimise/integrate paths stay behavior-compatible; full suite green;
  `ruff check src tests` clean.

---

## Cross-cutting enablers (do first)

### E0 — Effect-registration refactor (optional; do early if committing to the full pipeline)

Adding an effect today touches ~5 sites: the `EffectSpec` union + `Predictor`
isinstance tuples in `effects/spec.py`, a branch in **both**
`compiler.compile_lgm` and `compiler.compile_family`, and two `__init__`
exports. Six new effects × 5 sites is the biggest recurring tax in this plan. A
small registry (`spec_type → (builder, family_builder)`) collapses the
isinstance chains.

- **Files:** `effects/spec.py`, `compiler.py`, `effects/__init__.py`.
- **Ponytail:** pays off only because the pipeline below is long. Skip for a
  one-off effect. Do it before S8 if committing to the full pipeline.

### E-sparse — Sparse solver backend (roadmap #2; horizontal enabler)

Every solver calls `.toarray()`; the preflight guard rejects >4096 latent dims
/ 512 MB, and `Besag`/`BYM2` precision math (`pinv`, `eigh`) is dense too.
Economic networks (thousands of firms/banks) and space-time fields (S·T cells)
blow past that immediately. This slice adds no model — it makes the network and
space-time slices usable at real size.

- **Files:** `inference/gaussian.py`, `inference/laplace.py` (sparse
  Cholesky / `cholmod` path), spatial precision builders (sparse eigensolvers).
- **Dependency:** gates S7 and S8 at realistic size. Sequence before S7; before
  S8 at nontrivial S·T.

---

## Pillar 1 — Hybrid MIDAS + BYM2 + temporal

The "hybrid" is free: `+` already sums effects into one latent field. BYM2 and
AR1/RW already exist. The work is the MIDAS effect, split into a linear slice
and a parametric slice because MIDAS is nonlinear in its weight parameters.

### S1 — MIDAS smooth-lag (U-MIDAS + RW2 penalty). Pure LGM, no new machinery.

- **Idea:** expand a high-frequency covariate into `K` lag columns aligned to
  each low-frequency target row, then put an `RW2` (or `RW1`/`AR1`) smoothness
  prior **over the lag index**. The restricted MIDAS lag polynomial becomes a
  penalized latent curve over lag order — the LGM-native rendering of MIDAS.
- **Economic use:** nowcast quarterly GDP / credit risk from daily/monthly
  financial series.
- **New vs reused:** new `MIDAS` spec + a builder that constructs the lag
  design; **reuses** the RW2 precision and the Gaussian field wholesale. Fits
  the `LatentBlock` 4-tuple; it is a `ScalableBlock` (precision scales
  linearly).
- **Files:** `effects/spec.py`, new `effects/midas.py`, both compiler branches,
  exports. Test + `examples/` entry.

### S2 — MIDAS parametric (exponential-Almon / Beta weights). New IR seam.

- **Idea:** the classic restricted MIDAS — weight shape parameters (1–2 for
  exp-Almon, 2 for Beta) estimated by empirical Bayes / integrated by INLA.
- **New seam:** `ParametricBlock` rebuilds **precision** per θ; restricted
  MIDAS must rebuild the **design column** per θ (re-aggregate HF lags under
  current weights → one regressor). This is the one genuinely new IR concept in
  the plan — a `ParametricDesignBlock`. Depends on S1's lag-design code and
  reuses `transforms.py` bounds.
- **Files:** `ir/family.py` (new block type), `effects/midas.py`,
  `compiler.compile_family`, `inference/*` (materialize design per θ).

### S3 — Hybrid nowcasting example + integration test (proves the pillar).

`Fixed(...) + MIDAS(hf_financial) + BYM2(region_graph) + AR1(quarter)` on a
regional-GDP nowcast. No new code — an `examples/` script + an integration test
that the composed field fits and predicts. This is the Pillar-1 deliverable.

*(Optional S-seasonal: a cyclic-RW `Seasonal` effect — small, reuses RW
machinery, useful for economic seasonality; absent today. Slot near S1 if
seasonality matters.)*

---

## Pillar 2 — Weighted dynamic network (replaces geographic Besag)

### S4 — Weighted graphs. Smallest, highest-leverage change in the plan.

- **Idea:** generalize `normalize_graph` in `effects/graph.py` to accept
  `{node: {neighbour: weight}}` (list form → weight 1, back-compat). Weighted
  Laplacian `D−W`, `D` = weighted degree. Sørbye–Rue scaling already runs on
  the `R*=D−W` eigendecomposition, so it carries over untouched. This one change
  makes `Besag`/`ProperCAR`/`BYM2` weighted-network models — the static case of
  the whole pillar, at one file.
- **Directed networks** (ownership, exposure, supply): CAR needs symmetric `Q`
  → symmetrize `(W+Wᵀ)/2` with a documented policy (or use S5).
- **Files:** `effects/graph.py` (+ its tests); spatial builders unchanged
  because they already consume `(nodes, w)`.

### S5 — SAR effect for directed influence (network-native alternative).

- **Idea:** economic influence is directional; symmetrizing discards that. A
  simultaneous-autoregressive precision `Q = τ(I−ρW)ᵀ(I−ρW)` is PD for suitable
  ρ and handles asymmetric/directed `W` naturally — the right structure for
  supply-chain / ownership flow. New `NetworkSAR` effect, parametric in ρ
  (reuse the `ProperCAR` ρ machinery pattern).
- **Files:** `effects/spec.py`, new `effects/sar.py`, both compiler branches,
  `transforms.py` bound resolution.

### S6 — Economic-network graph constructors (ergonomics).

Thin helpers to build a weighted graph from an economic matrix: ownership
shares, interbank exposure, IO/Leontief coefficients, correlation/distance
thresholding, plus the symmetrization policy. Pure convenience on top of S4.

### S7 — Dynamic (time-varying) network `W_t`. Frontier.

- **Idea:** `graph={t: graph_t}`; spatial coupling in period `t` uses `W_t`.
  Time-varying `W` breaks the clean Kronecker structure, so the precision is
  block-structured: per-period spatial Laplacians coupled by a temporal term —
  a generalization of Knorr-Held Type II/IV with a per-period graph.
- **Dependency:** S4 + S8 (space-time machinery) + **E-sparse** (firms ×
  periods). Build last.

---

## Pillar 3 — Knorr-Held space-time interaction

### S8 — `SpaceTime` effect, Types I–IV. Kronecker precision + typed constraints.

- **Precision:** `Q = τ·(K_s ⊗ K_t)` via `scipy.sparse.kron`:
  - Type I — `I_s ⊗ I_t` (exchangeable interaction)
  - Type II — `I_s ⊗ R_t` (per-area independent temporal trend)
  - Type III — `R_s ⊗ I_t` (per-time independent spatial pattern)
  - Type IV — `R_s ⊗ R_t` (inseparable space-time)
  where `R_s` is the (weighted, from S4) Besag Laplacian and `R_t` an
  RW1/RW2/AR1 structure — **both structure functions already exist**
  (`besag._scaled_structure`, `random_walk`, `ar1.ar1_structure`).
- **Constraints:** the rank deficiency needs the type-specific sum-to-zero sets
  (Type IV: the S+T−1 Knorr-Held / Schrödle–Held rows). The existing null-space
  + kriging constraint machinery expresses these directly as `LatentBlock`
  constraint rows — no new inference code.
- **Dependency:** reuses temporal structures ✓, benefits from S4; gate at scale
  on **E-sparse** (`S·T` dense blows the guard fast). S8 is the machinery S7
  builds on.
- **Files:** `effects/spec.py`, new `effects/spacetime.py`, both compiler
  branches, exports.

---

## Pillar 4 — Other useful proposals (independent; pull by demand)

| Slice | What | Economic use | Cost |
|---|---|---|---|
| **S9** | Negative-Binomial + Binomial likelihoods (roadmap #4) | overdispersed counts: defaults, firm entry/exit | Low — duck-typed GLM protocol |
| **S10** | Gamma / log-normal / Beta likelihoods | positive continuous (loan/firm sizes, values); rates/shares | Low–med — same protocol |
| **S11** | Random slopes / correlated random coefficients | heterogeneous covariate effects across firms/sectors (panel econ) | Med — covariate-scaled design + (correlated) Gaussian prior |
| **S12** | First-class `forecast(h)` for temporal effects | forecasting horizon (today via NaN-response rows) | Low — ergonomics over existing mechanism |
| **S13** | Measurement-error / errors-in-variables covariate | noisy economic regressors | Med — latent covariate with own prior |
| **S14** | Shrinkage priors for wide fixed-effect sets (horseshoe-like) | high-dim nowcasting covariate banks | High/research — scale mixtures aren't Gaussian; approximate |

New likelihoods (S9/S10) implement the GLM-protocol duck interface in
`likelihoods.py` (`log_likelihood`, `gradient`, `working_weights`,
`third_derivative`, `response_mean`, `response_prediction`,
`pointwise_log_density`, `cdf`, `validate_response`) + a spec with
`.materialize()`, and wire into the Laplace path — no base class.

---

## Recommended one-by-one build order

Ordered so each slice is shippable and unblocks the next:

1. **S4 weighted graphs** — one file, unblocks the most, delivers static
   economic networks immediately.
2. **S1 MIDAS smooth-lag** — pure LGM, no new machinery, delivers the Pillar-1
   core.
3. **S8 Knorr-Held space-time** — reuses temporal + graph; introduces the
   Kronecker + constraint patterns S7 needs.
4. **S3 hybrid example** — cheap once S1 exists; proves Pillar 1.
5. **E-sparse solver** — unblocks realistic scale for every network/space-time
   slice.
6. **S2 MIDAS parametric** — adds the parametric-**design** seam; refines S1.
7. **S7 dynamic network** — frontier; needs S4 + S8 + E-sparse.
8. **S5 / S6, S9–S14** — as demand dictates.

*(E0 registry refactor: before step 3 if committing to the full pipeline; skip
if stopping after a couple of slices.)*

## Ceilings to respect

- **Dense solve ceiling is the real constraint.** Networks past a few hundred
  nodes and any nontrivial `S·T` need E-sparse — it gates S7/S8 at economic
  scale, not optional decoration.
- **BYM2 `φ` is weakly identified** (already documented); on economic networks
  with short panels expect boundary `φ̂` — keep the PC prior, read a boundary as
  "network share not identified."
- **Directed→symmetric loses information** in CAR; S5 (SAR) is the honest fix,
  so don't over-invest in symmetrized CAR for genuinely directional flows.
- **S2's parametric-design seam** is the only slice needing a new IR concept;
  everything else reuses `LatentBlock` / `ScalableBlock` / `ParametricBlock` /
  the constraint machinery as-is.

# Weighted Graphs (S4) Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** let the CAR family (`Besag`/`ProperCAR`/`BYM2`) run on a **weighted**
neighbour graph, so `W` can encode firm-ownership / interbank-exposure /
supply-chain / IO dependence strength rather than only geographic 0/1
adjacency. First enabler of the [economic-expansion plan](../plans/2026-08-26-pylgm-economic-expansion.md).

**Architecture:** the whole change lives in `effects/graph.py`. Every spatial
builder already derives structure from `(nodes, w)` via `degree = w.sum(axis=1)`
and `D − W` (`besag._scaled_structure`, `proper_car._validity_interval`,
`bym2._scaled_structure`→`pinv`, `_component_constraints` via
`connected_components(w)`) — all weight-agnostic. So `normalize_graph` learns to
read edge weights and return a weighted `csr_matrix W`; `canonical_graph` learns
to preserve them across the frozen-spec round-trip; **no builder, compiler, IR,
or inference code changes.** The ICAR density it feeds is the standard weighted
GMRF `exp(−τ/2 · Σ_{i~j} wᵢⱼ(xᵢ−xⱼ)²)`, which is a valid GMRF exactly when
`wᵢⱼ ≥ 0` — hence the nonnegative-weight requirement below (Rue & Held 2005).

**Tech stack:** numpy / scipy (`csr_matrix`) / pandas. No new runtime
dependency.

## Global Constraints

- **`effects/graph.py` only** for the model change (plus tests, docs, one
  example, and a YAML round-trip check). If a spatial builder, the compiler, or
  an inference file needs editing, stop — that means an assumption here is
  wrong; re-derive before proceeding.
- **Weights must be finite and strictly positive.** A weighted ICAR/CAR is a
  valid GMRF only for `wᵢⱼ ≥ 0`; a zero weight is "no edge" (omit it), and a
  non-positive or non-finite weight raises at graph-normalization time.
  Directed / signed economic matrices (exposure, correlations) are made
  symmetric-nonnegative **upstream** (S5 SAR for directed influence; S6
  constructors for the symmetrization policy) — `normalize_graph` stays a
  strict validator, it does not silently symmetrize.
- **The graph must be symmetric** (ICAR precondition): every edge `i→j` needs
  its reverse `j→i` present with an equal weight (within `rtol=1e-9,
  atol=1e-12`); the stored value is their mean, to kill float round-off. A
  missing reverse or a weight mismatch beyond tolerance raises, naming the pair.
- **Full back-compat.** An unweighted graph (`{node: [neighbours]}`) produces a
  `W` byte-identical to today (all weights `1.0`) and a `canonical_graph` output
  byte-identical to today (bare-label tuples), so every existing spatial test,
  YAML path, `.graph` file, and frozen-spec hash is unchanged.
- Self-loops, unknown neighbours, empty/`str`/`bytes` neighbour containers keep
  their current rejections. Isolated nodes (degree 0) keep being rejected by the
  builders, unchanged.
- Full suite green; `ruff check src tests` clean.

## Accepted input grammar (what `normalize_graph` reads)

Per node key, the neighbours value is one of:

| Form | Example | Meaning |
|---|---|---|
| bare-label sequence (legacy) | `{"a": ["b", "c"]}` | edges of weight `1.0` |
| mapping | `{"a": {"b": 2.0, "c": 1.0}}` | weighted edges (**documented user form**) |
| pair sequence | `{"a": [("b", 2.0), ("c", 1.0)]}` | weighted edges (round-trip form) |
| mixed sequence | `{"a": ["b", ("c", 2.0)]}` | `b`:1.0, `c`:2.0 |

Per-element detection inside a sequence: an element that is a non-`str`/`bytes`
sequence of length 2 whose second item is a real number is a `(label, weight)`
pair; anything else is a bare label with weight `1.0`. Labels are stringified
(as today).

## Canonical output (what `canonical_graph` freezes)

For each node (label-sorted), neighbours label-sorted. If **every** edge weight
in the graph is exactly `1.0`, emit today's bare tuple `(nbr, …)`
(back-compat); otherwise emit a pair tuple `((nbr, weight), …)`. `dict()` on the
result is re-accepted by `normalize_graph` in both cases.

---

### Task 1: Weighted `normalize_graph`

**Files:**
- Modify: `src/pylgm/effects/graph.py`
- Test: `tests/test_weighted_graph.py`

**Interfaces:**
- `normalize_graph(graph) -> (nodes: tuple[str, ...], w: csr_matrix)` —
  unchanged signature; `w` now carries edge weights (a symmetric nonnegative
  `csr_matrix`, all-ones for unweighted input). Accepts the four input forms
  above.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_weighted_graph.py
import numpy as np
import pytest

from pylgm.effects.graph import canonical_graph, normalize_graph


def _dense(graph):
    _, w = normalize_graph(graph)
    return w.toarray()


def test_unweighted_is_all_ones_and_symmetric():
    nodes, w = normalize_graph({"a": ["b"], "b": ["a"]})
    assert nodes == ("a", "b")
    assert np.array_equal(w.toarray(), np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_mapping_form_carries_weights():
    w = _dense({"a": {"b": 2.5}, "b": {"a": 2.5}})
    assert w[0, 1] == 2.5 and w[1, 0] == 2.5


def test_pair_and_mixed_sequence_forms_agree_with_mapping():
    m = _dense({"a": {"b": 2.0, "c": 1.0}, "b": {"a": 2.0}, "c": {"a": 1.0}})
    p = _dense({"a": [("b", 2.0), ("c", 1.0)], "b": [("a", 2.0)], "c": [("a", 1.0)]})
    x = _dense({"a": ["c", ("b", 2.0)], "b": [("a", 2.0)], "c": ["a"]})
    assert np.array_equal(m, p) and np.array_equal(m, x)


def test_weighted_degree_is_sum_of_weights():
    _, w = normalize_graph({"a": {"b": 2.0, "c": 3.0}, "b": {"a": 2.0}, "c": {"a": 3.0}})
    degree = np.asarray(w.sum(axis=1)).ravel()
    assert degree[0] == pytest.approx(5.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_nonpositive_or_nonfinite_weight_rejected(bad):
    with pytest.raises(ValueError, match="weight"):
        normalize_graph({"a": {"b": bad}, "b": {"a": bad}})


def test_asymmetric_weight_rejected():
    with pytest.raises(ValueError, match="symmetric|weight"):
        normalize_graph({"a": {"b": 2.0}, "b": {"a": 3.0}})


def test_asymmetric_weight_within_tolerance_is_averaged():
    _, w = normalize_graph({"a": {"b": 2.0}, "b": {"a": 2.0 + 1e-12}})
    assert w.toarray()[0, 1] == pytest.approx(2.0)


def test_missing_reverse_edge_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        normalize_graph({"a": {"b": 2.0}, "b": {}})


def test_self_loop_and_unknown_neighbour_still_rejected():
    with pytest.raises(ValueError, match="self-loop"):
        normalize_graph({"a": {"a": 1.0}})
    with pytest.raises(ValueError, match="not a node"):
        normalize_graph({"a": {"z": 1.0}, "b": {}})
```

- [ ] **Step 2: Implement** — rework `normalize_graph` to build an
  `adjacency: dict[str, dict[str, float]]` (node → neighbour → weight) instead
  of a `set`, using the per-element detection above; validate each weight
  finite and `> 0`; keep the self-loop / unknown-node checks; replace the
  set-membership symmetry check with a weight-equality check
  (`math.isclose`/`np.isclose`, `rtol=1e-9, atol=1e-12`), storing the mean;
  emit the `csr_matrix` from the weighted entries. Error messages name the
  offending node/pair and (for weights) the bad value.
- [ ] **Step 3: Run tests, confirm green, `ruff check`.**

### Task 2: Weighted `canonical_graph` round-trip

**Files:**
- Modify: `src/pylgm/effects/graph.py`
- Test: `tests/test_weighted_graph.py`

**Interfaces:**
- `canonical_graph(graph) -> tuple[(node, neighbours), ...]` — `neighbours` is a
  bare-label tuple when the graph is unweighted (unchanged) or a
  `((label, weight), …)` tuple when weighted; `dict(result)` is re-accepted by
  `normalize_graph`.

- [ ] **Step 1: Write the failing tests**

```python
def test_canonical_unweighted_is_bare_and_unchanged():
    assert canonical_graph({"a": ["b"], "b": ["a"]}) == (("a", ("b",)), ("b", ("a",)))


def test_canonical_weighted_is_pairs_and_round_trips():
    g = {"a": {"b": 2.0, "c": 1.0}, "b": {"a": 2.0}, "c": {"a": 1.0}}
    canon = canonical_graph(g)
    assert canon[0] == ("a", (("b", 2.0), ("c", 1.0)))
    # dict(canon) is re-accepted and yields the same adjacency
    n1, w1 = normalize_graph(g)
    n2, w2 = normalize_graph(dict(canon))
    assert n1 == n2 and np.array_equal(w1.toarray(), w2.toarray())


def test_canonical_is_hashable_for_frozen_spec():
    hash(canonical_graph({"a": {"b": 2.0}, "b": {"a": 2.0}}))  # must not raise
```

- [ ] **Step 2: Implement** — build the per-node neighbour tuples from `(nodes,
  w)`; if every stored weight equals `1.0`, emit bare label tuples (today's
  code path, unchanged); otherwise emit sorted `(label, weight)` pairs. Ensure
  the result is hashable (tuples of `(str, float)`).
- [ ] **Step 3: Run tests, confirm green.**

### Task 3: End-to-end CAR family on a weighted graph (no builder changes)

**Files:**
- Test: `tests/test_weighted_spatial_effects.py`

This task asserts the architecture claim: the builders already handle weights,
and unweighted behaviour is unchanged. **No `src/` change is expected here** —
if a test needs a builder edit to pass, that is a finding to record, not a
silent fix.

- [ ] **Step 1: Write the tests**

```python
# tests/test_weighted_spatial_effects.py
import numpy as np
import pandas as pd
import pytest

from pylgm import BYM2, Besag, Fixed, Gaussian, LGM, ProperCAR


def _frame():
    return pd.DataFrame({"region": ["a", "b", "c"], "y": [0.2, -0.1, 0.4]})


def test_besag_weighted_differs_from_unweighted():
    frame = _frame()
    tri = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
    weighted = {"a": {"b": 5.0, "c": 0.1}, "b": {"a": 5.0, "c": 0.1},
                "c": {"a": 0.1, "b": 0.1}}

    def fit(graph):
        model = LGM(response="y", likelihood=Gaussian(sigma=0.1),
                    predictor=Fixed("1") + Besag("region", index="region", graph=graph))
        return model.fit(frame).latent_marginals("region").mean

    assert not np.allclose(fit(tri), fit(weighted))  # weights change the smoothing


def test_besag_weight_one_matches_unweighted_exactly():
    frame = _frame()
    tri = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    tri_w = {"a": {"b": 1.0}, "b": {"a": 1.0, "c": 1.0}, "c": {"b": 1.0}}

    def fit(graph):
        model = LGM(response="y", likelihood=Gaussian(sigma=0.1),
                    predictor=Fixed("1") + Besag("region", index="region", graph=graph))
        return model.fit(frame).latent_marginals("region").mean

    assert np.allclose(fit(tri), fit(tri_w))


def test_proper_car_and_bym2_accept_weighted_graphs():
    frame = _frame()
    g = {"a": {"b": 2.0}, "b": {"a": 2.0, "c": 3.0}, "c": {"b": 3.0}}
    for effect in (ProperCAR("region", index="region", graph=g, rho=0.5),
                   BYM2("region", index="region", graph=g, precision=1.0, phi=0.5)):
        model = LGM(response="y", likelihood=Gaussian(sigma=0.1),
                    predictor=Fixed("1") + effect)
        assert model.fit(frame).latent_marginals("region").mean.shape == (3,)
```

- [ ] **Step 2: Run.** Confirm green with no `src/` edits beyond Tasks 1–2. If
  `ProperCAR`'s `rho`-validity or `BYM2`'s spectrum misbehaves on a weighted
  graph, record it (weighted graphs can shift the valid-`rho` interval and the
  scaled-structure eigenvalues — expected, not a bug) and cover it with an
  explicit test rather than changing the builder.

### Task 4: YAML weighted-graph round-trip + docs + example

**Files:**
- Verify/modify: `src/pylgm/config/model.py`
- Test: `tests/test_config_model.py` (extend)
- Docs: `docs/spatial-effects.md`, `docs/roadmap.md`
- Example: `examples/` (a small weighted-network fit)

- [ ] **Step 1:** Add a test that `pylgm.config.load_model` accepts an inline
  weighted `graph:` mapping (`{node: {nbr: weight}}`) and produces the same
  fitted means as the Python API on the same weighted graph. `.graph` files
  stay unweighted (`load_graph_file` unchanged).
- [ ] **Step 2:** Only if the test fails — the YAML loader coerces neighbours to
  a list — teach `config/model.py` to pass the neighbour mapping through
  untouched. Expect this to already work (the loader forwards the mapping into
  the spec, which calls `canonical_graph`); confirm rather than assume.
- [ ] **Step 3:** Docs — in `docs/spatial-effects.md`, document the weighted
  `graph` input on `Besag`/`ProperCAR`/`BYM2` (the mapping form, the
  nonnegative + symmetric requirement, and the economic-network reading of
  `W`); drop the "weighted graphs" item from the relevant roadmap line. Add one
  runnable `examples/` script fitting a `BYM2` over a small weighted
  ownership/exposure-style network.
- [ ] **Step 4:** Full suite green; `ruff check src tests` clean.

---

## Out of scope (later slices)

- **Time-varying `W_t`** — S7 (dynamic network).
- **Directed-influence structure** (`(I−ρW)ᵀ(I−ρW)`) — S5 (`NetworkSAR`).
- **Economic-matrix → graph constructors** and the symmetrization policy — S6.
- **Sparse assembly / large graphs** — E-sparse; weighting does not change the
  dense O(n³) ceiling, only the interpretation of `W`.

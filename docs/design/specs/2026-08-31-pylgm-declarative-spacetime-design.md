# pyLGM Declarative SpaceTime Design

**Status:** Approved 2026-08-31

## Purpose

`SpaceTime` (the Knorr-Held space-time interaction, types I-IV) ships in the
Python API but is not reachable from the YAML frontend. Unlike every effect
wired so far it is indexed by **two** columns — a `space` index and a `time`
index — not a single `index` or a list of lag `columns`. This slice exposes
`spacetime` as a YAML effect type in `pylgm.config.load_model`, reusing the
unchanged `SpaceTime` spec, adding no new numerics and no new dependency.

## Field-validation refactor (fixes a latent hole)

Adding a fourth index shape (`space`+`time`) to the per-branch rejection lists
would compound an existing bug: after the AR1/Seasonal slice, `group`/`period`
set on a `midas` effect are **silently accepted and ignored**, because
`_validate_midas` never rejected them. The accumulating per-branch "reject these
fields" lists cannot keep every combination covered.

This slice replaces them with a single **whitelist table** — `_ALLOWED_FIELDS`,
mapping each effect type to the optional fields it accepts. Any field set that is
not in a type's allowed set is rejected with `fields [...] are not valid for
effect type '...'`. Per-type *required-field* and *value* checks (e.g. `rho is
required` for `proper_car`, `order must be 1 or 2`) stay. Every existing
rejection message keeps the substring its test matches.

## Data contract

```yaml
- {name: st, type: spacetime, space: region, time: year, interaction: IV, order: 1, precision: 1.0, scale: true, graph: {a: [b], b: [a]}}
```

- `space` — **required**, the spatial index column.
- `time` — **required**, the temporal index column.
- `graph` / `graph_file` — the spatial neighbour graph, at most one. **Required
  for interaction types III and IV** (enforced by the `SpaceTime` spec);
  optional for I and II. `graph_file` accepts an R-INLA `.graph` or `.json`
  neighbour dict, exactly as for the CAR family.
- `interaction` — optional, `"I"`/`"II"`/`"III"`/`"IV"`, default `"IV"`.
- `order` — optional, `1` or `2` (temporal RW order), default `1`.
- `precision` — optional fixed float, default `1.0`.
- `scale` — optional bool, default `true`.

Estimating `precision` (a `Hyperparameter`) stays Python-API-only, as for every
other effect.

## Rejections (fail loud)

- `index`, `columns`, and any single-index/MIDAS/temporal-only field
  (`rho`, `phi`, `group`, `period`, `ridge`, `kernel`) on `spacetime` → not in
  its allowed set.
- `space` / `time` / `interaction` on any other effect type → not in their
  allowed sets (this is what the whitelist buys uniformly).
- `space` or `time` missing on `spacetime`.
- Both `graph` and `graph_file` on `spacetime`.
- Bad `interaction`, bad `order`, or a III/IV interaction with no graph — raised
  by the `SpaceTime` spec, wrapped as `ConfigurationError`.

## Out of scope

- `DynamicSpatialPanel` (per-period graphs, three coefficients) remains
  Python-only — its own slice, next.
- Estimating any hyperparameter from YAML.

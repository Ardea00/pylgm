# pyLGM Declarative DynamicSpatialPanel Design

**Status:** Approved 2026-08-31

## Purpose

`DynamicSpatialPanel` (the SDPD effect: per-period directed networks `W_t` with
contemporaneous `ρ`, temporal `γ`, and spatio-temporal-diffusion `η`
coefficients) ships in the Python API but is not reachable from the YAML
frontend. It is the **last** Python-only effect. This slice exposes
`dynamicspatialpanel` as a YAML effect type in `pylgm.config.load_model`,
reusing the unchanged `DynamicSpatialPanel` spec, adding no new numerics.

Unlike every effect wired so far it is indexed by a `unit`+`time` pair **and**
carries a per-period `graphs` mapping (`time → graph`), not a single graph. That
mapping is the one new representational problem this slice solves.

## Data contract

```yaml
# inline per-period graphs
- {name: sdpd, type: dynamicspatialpanel, unit: firm, time: year, rho: 0.3, gamma: 0.1, eta: 0.05, precision: 1.0,
   graphs: {2020: {a: [b], b: [a]}, 2021: {a: [b], b: [a]}}}

# or per-period graph files
- {name: sdpd, type: dynamicspatialpanel, unit: firm, time: year, rho: 0.3,
   graph_files: {2020: nb2020.graph, 2021: nb2021.graph}}
```

- `unit` — **required**, the panel unit index column.
- `time` — **required**, the temporal index column.
- `graphs` — a mapping `period → neighbour dict` (inline). Directed / weighted
  graphs are accepted exactly as for `SAR` — the spec canonicalizes each `W_t`.
- `graph_files` — a mapping `period → filename`; each file is an R-INLA `.graph`
  or `.json` neighbour dict, loaded via the same `load_graph_file` as the CAR /
  SAR family. **Exactly one of `graphs` or `graph_files` is required.**
- `rho` — **required** fixed float, strictly inside `(-1, 1)` (spec-enforced).
- `gamma` — optional fixed float, default `0.0`.
- `eta` — optional fixed float, default `0.0`.
- `precision` — optional fixed float, default `1.0`.

Period keys are read exactly as `DynamicSpatialPanel` reads them from a Python
`dict` (numeric-aware ordering); YAML parses `2020:` as an int key, which the
spec already handles. `T = 1` (a single-period mapping) reduces to `SAR`.

Estimating `ρ`/`γ`/`η`/`precision` (each a `Hyperparameter`) stays
Python-API-only, as for every other effect.

## Rejections (fail loud)

- `index`, `columns`, `space`, or any single-index/MIDAS/temporal/spatial-only
  field on `dynamicspatialpanel` → not in its allowed set (the `_ALLOWED_FIELDS`
  whitelist introduced in the SpaceTime slice).
- `unit`, `graphs`, or `graph_files` on any other effect type → not in their
  allowed sets.
- `unit` or `time` missing on `dynamicspatialpanel`.
- Neither or both of `graphs` / `graph_files`.
- `rho` missing.
- Bad `rho` (outside `(-1, 1)`), empty `graphs`, or a malformed per-period graph
  — raised by the `DynamicSpatialPanel` spec, wrapped as `ConfigurationError`.

## Out of scope

- Estimating any hyperparameter from YAML.
- Forecasting (`forecast_dynamic_spatial_panel`) — a runtime call, not model
  declaration.

## Completion

With `dynamicspatialpanel` wired, **every** pyLGM effect is reachable from the
YAML frontend. No Python-only effects remain.

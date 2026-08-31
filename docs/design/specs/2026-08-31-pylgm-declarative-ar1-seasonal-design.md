# pyLGM Declarative AR1 & Seasonal Design

**Status:** Approved 2026-08-31

## Purpose

`AR1` (stationary first-order autoregressive, optionally group-wise) and
`Seasonal` (drifting cyclic pattern) ship in the Python API but are not
reachable from the YAML frontend. Both are single-index temporal effects that
map onto the frontend's existing field set almost entirely, so this slice
exposes `ar1` and `seasonal` as YAML effect types in `pylgm.config.load_model`
with no new numerics and no new dependency — only schema fields, per-type
validation, and builder dispatch in `config/model.py`.

## Data contract

Both effects are indexed by a single `index` column, like the simple effects.

### `ar1` (stationary AR1, optionally group-wise)

```yaml
- {name: t, type: ar1, index: period, rho: 0.5, precision: 1.0, group: unit}
```

- `index` — required, the ordered time column.
- `rho` — optional fixed float in `(-1, 1)`, default `0.5`. Estimating `rho`
  from YAML stays Python-API-only (same convention as `proper_car` / `sar`).
- `precision` — optional fixed float, default `1.0`.
- `group` — optional column name; when given, one independent series per panel
  unit sharing `rho` and `precision` (group-wise AR1).

### `seasonal` (drifting seasonal)

```yaml
- {name: month, type: seasonal, index: month, period: 12, precision: 1.0, ridge: 1.0e-6}
```

- `index` — required, the cyclic time column.
- `period` — **required** positive int (the cycle length, e.g. `12`).
- `precision` — optional fixed float, default `1.0`.
- `ridge` — optional fixed positive float, default `1e-6` (reuses the field the
  MIDAS slice added).

## Rejections (fail loud, matching the frontend's existing style)

- `rho` on any non-`ar1`/`proper_car`/`sar` type (already enforced; `ar1`
  joins the allowed set).
- `group` on any type other than `ar1`.
- `period` missing on `seasonal`, or present on any other type.
- Spatial fields (`graph`, `graph_file`, `phi`, `scale`) on either type.

## Out of scope

- Estimating `rho` / `precision` from YAML (a `Hyperparameter`) — Python-only,
  as for every other effect.
- Irregular AR1 spacing (`ρ^Δt`) — a separate *unbuilt* Python feature, not a
  wiring gap.
- `SpaceTime` (dual space+time index) and `DynamicSpatialPanel` (per-period
  graph files, three coefficients) remain Python-only; each is its own slice.

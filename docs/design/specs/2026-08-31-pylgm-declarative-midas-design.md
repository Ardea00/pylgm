# pyLGM Declarative MIDAS Design

**Status:** Approved 2026-08-31

## Purpose

The `MIDAS` (smooth-lag) and `MIDASParametric` (exp-Almon / Beta kernel) effects
ship in the Python API and drive the hybrid-nowcast example, but neither is
reachable from the YAML frontend. The roadmap lists a "config-file `midas` type"
as *Next*; the substance already exists, so this slice closes only the thin
declarative-wiring gap: expose `midas` and `midas_parametric` as YAML effect
types in `pylgm.config.load_model`.

It reuses the existing `MIDAS` / `MIDASParametric` specs unchanged. It adds no
new numerics and no new dependency — only schema fields, per-type validation,
and builder dispatch in `config/model.py`.

## Data contract

Both MIDAS effects are indexed by a **list of HF lag columns**, not a single
`index` column like every other effect. So `_EffectModelConfig.index` becomes
optional: required for every non-MIDAS type, rejected for the MIDAS types, which
require `columns` instead.

### `midas` (smooth-lag)

```yaml
- {name: lag, type: midas, columns: [x0, x1, x2, x3], precision: 1.0, order: 2, ridge: 1.0e-6}
```

- `columns` — required, > `order` entries (enforced by the `MIDAS` spec).
- `precision` — optional fixed float, default `1.0`.
- `order` — optional, `1` or `2`, default `2`.
- `ridge` — optional fixed positive float, default `1e-6`.

### `midas_parametric` (restricted kernel)

```yaml
- {name: m, type: midas_parametric, columns: [x0, x1, x2], kernel: exp_almon}
```

- `columns` — required, ≥ 2 entries (enforced by the `MIDASParametric` spec).
- `kernel` — optional, `"beta"` or `"exp_almon"`, default `"beta"`.

The kernel shape parameters are **estimated** (EB/INLA), the effect's headline
behaviour and its default when the shapes are omitted from the Python API. YAML
keeps the existing convention — it carries no syntax for requesting estimation —
so it inherits that default. Fixed shape overrides and `prior_precision` stay
Python-API-only until a use case asks for them.

## Rejections (fail loud, matching the frontend's existing style)

- `index` on a MIDAS type → error naming `columns` as the correct field.
- `columns` on a non-MIDAS type; `kernel`/`order`/`ridge` on a type that does
  not accept them (e.g. `order`/`ridge`/`precision` on `midas_parametric`).
- Spatial fields (`graph`, `rho`, `phi`, `scale`) on a MIDAS type.

## Out of scope

`ar1`, `seasonal`, `spacetime`, and `dynamicspatialpanel` remain Python-only;
they are separate wiring slices. This slice closes only the roadmap's named
`midas` gap.

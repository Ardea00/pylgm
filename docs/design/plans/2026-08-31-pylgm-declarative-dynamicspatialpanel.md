# Declarative DynamicSpatialPanel Implementation Plan

**Goal:** Expose `dynamicspatialpanel` (SDPD) as a YAML effect type in
`pylgm.config.load_model` — the last Python-only effect. Support per-period
`graphs` (inline) or `graph_files` (per-period neighbour files).

**Architecture:** Pure frontend wiring. Add `unit`/`graphs`/`graph_files`/
`gamma`/`eta` fields and `dynamicspatialpanel` to the type Literal + the
`_ALLOWED_FIELDS` whitelist; add `_validate_dsp`; add builder dispatch that
loads per-period files and delegates to the unchanged `DynamicSpatialPanel` spec.

**Tech Stack:** Python 3.11+, pydantic, PyYAML, pytest. No new dependencies.

## Constraints

- Git identity **must** be `Ardea00` on every commit.
- Do not change effect numerics or existing YAML types.
- Preserve every existing rejection message substring the tests match.
- TDD: failing tests first, minimal implementation, then docs.

## Tasks

- [x] 1. Test: `dynamicspatialpanel` YAML loads and equals `DynamicSpatialPanel(...)` (defaults; explicit gamma/eta/precision; inline `graphs`; per-period `graph_files`).
- [x] 2. Test: rejections — `unit`/`time` missing; neither/both of `graphs`/`graph_files`; `rho` missing; `unit`/`graphs`/`graph_files` on a non-DSP type; single `index`/`space`/`columns` on DSP; bad `rho` (via spec).
- [x] 3. Implement `_ALLOWED_FIELDS["dynamicspatialpanel"]` + schema fields + `_validate_dsp` + builder dispatch.
- [x] 4. Docs: add the `dynamicspatialpanel` YAML example to `docs/spatial-effects.md`; update `docs/roadmap.md` declarative-frontend bullet (mark the frontend complete — no Python-only effects remain).

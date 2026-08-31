# Declarative AR1 & Seasonal Implementation Plan

**Goal:** Expose `ar1` and `seasonal` as YAML effect types in
`pylgm.config.load_model`.

**Architecture:** Pure frontend wiring. Add `ar1` / `seasonal` to the effect
`type` Literal; add `group` (ar1) and `period` (seasonal) fields; extend the
`rho`-allowed set to include `ar1`; add validation branches to
`_fields_match_type`; add builder dispatch delegating to the unchanged `AR1` /
`Seasonal` specs.

**Tech Stack:** Python 3.11+, pydantic, PyYAML, pytest. No new dependencies.

## Constraints

- Git identity **must** be `Ardea00` on every commit.
- Do not change effect numerics or existing YAML types. `spacetime` /
  `dynamicspatialpanel` stay rejected.
- Estimating `rho`/`precision` from YAML stays Python-only.
- TDD: failing tests first, minimal implementation, then docs.

## Tasks

- [x] 1. Test: `ar1` YAML loads and equals `AR1(...)` (defaults; explicit `rho`/`precision`; with `group`).
- [x] 2. Test: `seasonal` YAML loads and equals `Seasonal(...)` (required `period`; explicit `precision`/`ridge`).
- [x] 3. Test: rejections — `group` on a non-`ar1` type, `period` missing on `seasonal`, `period` on a non-`seasonal` type, spatial fields on either.
- [x] 4. Implement schema fields + validator branches + builder dispatch in `config/model.py`.
- [x] 5. Docs: add the `ar1` / `seasonal` YAML examples to `docs/effects.md`; update `docs/roadmap.md` declarative-frontend bullet and shrink the "Next" nowcasting note if needed.

# Declarative MIDAS Implementation Plan

**Goal:** Expose `midas` and `midas_parametric` as YAML effect types in
`pylgm.config.load_model`, closing the roadmap's named config-file MIDAS gap.

**Architecture:** Pure frontend wiring. `_EffectModelConfig.index` becomes
optional (required for non-MIDAS types, rejected for MIDAS types); add
`columns`, `order`, `ridge`, `kernel` fields; add a `_validate_midas` branch to
`_fields_match_type`; add `midas` / `midas_parametric` dispatch to
`_build_effect`, delegating to the unchanged `MIDAS` / `MIDASParametric` specs.

**Tech Stack:** Python 3.11+, pydantic, PyYAML, pytest. No new dependencies.

## Constraints

- Git identity **must** be `Ardea00` on every commit. Never iongroup/AndreaPanozzo.
- Do not change effect numerics or existing YAML types. `ar1`/`seasonal`/
  `spacetime`/`dynamicspatialpanel` stay rejected (see `test_model.py`).
- TDD: failing test first, minimal implementation, then docs.

## Tasks

- [x] 1. Test: `midas` YAML loads and equals the Python `MIDAS(...)` spec (defaults + explicit `order`/`ridge`/`precision`).
- [x] 2. Test: `midas_parametric` YAML loads and equals `MIDASParametric(...)` (default `beta` and explicit `exp_almon`).
- [x] 3. Test: rejections — `index` on a MIDAS type, `columns` on a non-MIDAS type, `kernel`/`order`/`ridge`/`precision` on the wrong MIDAS type, spatial fields on a MIDAS type, bad `kernel`.
- [x] 4. Implement schema fields + `_validate_midas` + builder dispatch in `config/model.py`.
- [x] 5. Docs: add the declarative MIDAS examples to `docs/effects.md`; correct `docs/roadmap.md` (move MIDAS/nowcast substance to Shipped, reframe the remaining gap).

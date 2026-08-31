# Declarative SpaceTime Implementation Plan

**Goal:** Expose `spacetime` as a YAML effect type in
`pylgm.config.load_model`, and replace the per-branch field-rejection lists with
a single whitelist table (fixing the latent `group`/`period`-on-`midas` hole).

**Architecture:** Pure frontend wiring. Add `space`/`time`/`interaction` fields
and `spacetime` to the type Literal; introduce `_ALLOWED_FIELDS` and validate
field applicability from it; keep per-type required/value checks; add builder
dispatch delegating to the unchanged `SpaceTime` spec.

**Tech Stack:** Python 3.11+, pydantic, PyYAML, pytest. No new dependencies.

## Constraints

- Git identity **must** be `Ardea00` on every commit.
- Do not change effect numerics or existing YAML types. `dynamicspatialpanel`
  stays rejected.
- Preserve every existing rejection message substring the tests match.
- TDD: failing tests first, minimal implementation, then docs.

## Tasks

- [x] 1. Test: `spacetime` YAML loads and equals `SpaceTime(...)` (defaults; explicit interaction/order/scale; inline graph; graph_file).
- [x] 2. Test: rejections — `space`/`time` missing; `space`/`time`/`interaction` on a non-spacetime type; single `index`/`columns`/`rho` on spacetime; both graph and graph_file; bad interaction (via spec).
- [x] 3. Test (regression): `group`/`period` on a `midas` effect are now rejected.
- [x] 4. Implement `_ALLOWED_FIELDS` whitelist + `_validate_spacetime` + builder dispatch; drop dead `_SPATIAL_EFFECTS`/`_SPATIAL_FIELDS`/`_LAG_FIELDS`.
- [x] 5. Docs: add the `spacetime` YAML example to `docs/effects.md`; update `docs/roadmap.md` declarative-frontend bullet.

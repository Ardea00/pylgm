# Development installation

```bash
python -m pip install -e ".[dev]"
```

## Running the tests

```bash
pytest -q
```

The suite is safe to run in parallel, and CI does:

```bash
pytest -q -n auto
```

That roughly halves wall time. A handful of large sparse fits (past the dense
reference guard) dominate the total, so the parallel floor is whichever of
those is slowest — running with `-n auto` locally is worth it, but `-x` and
readable tracebacks are easier serially.

The large-model tests size themselves from
`pylgm.inference.gaussian._MAX_DENSE_LATENT_DIMENSION` rather than a literal, so
they stay just past the guard if that threshold ever moves. Keep them that way:
the sparse cost grows superlinearly, and being *far* past the guard buys no
extra coverage.

See the [approved design](https://github.com/Ardea00/pylgm/blob/main/docs/design/specs/2026-07-22-pylgm-design.md).

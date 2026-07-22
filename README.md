# pyLGM

Configuration-driven latent Gaussian models for panel data, with one compiled
model representation shared by multiple inference engines.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

See the [approved design](docs/superpowers/specs/2026-07-22-pylgm-design.md).

## Gaussian foundation example

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
```

The example combines fixed effects, an IID region effect, and an intrinsic RW1
time effect. Rows with missing responses are prediction targets. Version 0.1's
exact Gaussian engine conditions on the effect precisions and observation
standard deviation declared in configuration; it does not integrate their
uncertainty.

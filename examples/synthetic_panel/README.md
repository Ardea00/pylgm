# Synthetic panel example

This small, domain-neutral panel has two regions observed over four months.
It combines a fixed intercept and covariate (`x`), an IID region effect, and
an intrinsic RW1 time effect. The blank responses in month 4 are prediction
targets: pyLGM fits the model using observed responses and returns predictions
for every input row, including rows where `y` is missing.

Run the example locally with the command-line interface:

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
```

Or use the public Python interface:

```bash
python -c 'import pandas as pd; from pylgm import Pipeline; Pipeline.from_yaml("examples/synthetic_panel/config.yaml").run(pd.read_csv("examples/synthetic_panel/data.csv"), "synthetic-run-python")'
```

Version 0.1 uses an exact Gaussian engine. It conditions on the effect
precisions and observation standard deviation declared in `config.yaml`; it
does not integrate uncertainty in those hyperparameters.

# pyLGM Gaussian Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an installable local pyLGM vertical slice that validates YAML and panel data, compiles fixed/IID/RW latent Gaussian effects, performs constrained exact Gaussian inference, and persists a reproducible result through both Python and CLI interfaces.

**Architecture:** A strict configuration and canonical Pandas panel compile into an immutable `CompiledLGM`; effect builders own design/precision construction, while the Gaussian engine consumes only the IR. This first plan uses dense reduced coordinates for constraints around sparse model matrices, producing a mathematically exact small/medium-model reference that later sparse, NUTS, Laplace, Spark, and inflation plans can test against.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Pandas, Formulaic, Pydantic 2, PyYAML, Typer, pytest, Ruff, Hatchling.

## Global Constraints

- Repository name is `Ardea00/pyLGM`; package and import name are `pylgm`; CLI command is `pylgm`.
- License is MIT.
- The runtime is local; this plan adds no HTTP service or cloud requirement.
- Inference code must not parse YAML or access Pandas/Spark objects.
- Application/data code must not implement numerical inference.
- Unknown configuration keys are errors, and all defaults must appear in persisted resolved configuration.
- Silent statistical fallback is forbidden.
- Exact Gaussian results in this plan are conditional on fixed hyperparameters.
- Every task follows red-green-refactor TDD and ends in a focused commit.

## Plan-series boundary

This plan is the first independently executable slice of the approved design. Later plans add: (1) production sparse constraints and ICAR/BYM2/space-time blocks, (2) repository-owned NUTS, (3) Spark adapters, candidate comparison and vintage-aware evaluation, (4) Italian inflation, and (5) Laplace/INLA/HMC-Laplace. Those plans consume the interfaces fixed here instead of expanding this plan beyond a reviewable scope.

---

### Task 1: Installable package and quality gate

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `README.md`
- Create: `src/pylgm/__init__.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Consumes: none.
- Produces: importable `pylgm`, `pylgm.__version__: str`, and the `pylgm` console entry point reserved for Task 8.

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_package.py
import pylgm


def test_package_exports_version() -> None:
    assert pylgm.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and verify the package is absent**

Run: `python -m pytest tests/test_package.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'pylgm'`.

- [ ] **Step 3: Add packaging and the minimal public module**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "pylgm"
version = "0.1.0"
description = "Configuration-driven latent Gaussian models for panel data"
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
authors = [{ name = "Andrea Panozzo" }]
dependencies = [
  "formulaic>=1.1",
  "numpy>=2.0",
  "pandas>=2.2",
  "pydantic>=2.8",
  "pyyaml>=6.0",
  "scipy>=1.14",
  "typer>=0.15",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "ruff>=0.9"]

[project.scripts]
pylgm = "pylgm.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/pylgm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"
```

```python
# src/pylgm/__init__.py
"""pyLGM public package."""

__version__ = "0.1.0"
```

```text
# LICENSE
MIT License

Copyright (c) 2026 Andrea Panozzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

````markdown
<!-- README.md -->
# pyLGM

Configuration-driven latent Gaussian models for panel data, with one compiled
model representation shared by multiple inference engines.

## Development installation

```bash
python -m pip install -e ".[dev]"
```

See the [approved design](docs/superpowers/specs/2026-07-22-pylgm-design.md).
````

- [ ] **Step 4: Install and run the quality gate**

Run: `python -m pip install -e ".[dev]" && python -m pytest tests/test_package.py -v && ruff check .`  
Expected: installation succeeds, one test passes, and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml LICENSE README.md src/pylgm/__init__.py tests/test_package.py
git commit -m "build: initialize pylgm package"
```

### Task 2: Strict versioned configuration

**Files:**
- Create: `src/pylgm/config/__init__.py`
- Create: `src/pylgm/config/schema.py`
- Create: `src/pylgm/config/load.py`
- Create: `src/pylgm/exceptions.py`
- Create: `tests/config/test_load.py`

**Interfaces:**
- Consumes: Pydantic and PyYAML.
- Produces: `RunConfig`, `EffectConfig`, and `load_config(path: Path) -> RunConfig`.

- [ ] **Step 1: Write strict-schema tests**

```python
# tests/config/test_load.py
from pathlib import Path

import pytest

from pylgm.config import load_config
from pylgm.exceptions import ConfigurationError


def test_load_config_resolves_defaults(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 1
data: {time: month, response: y, panel: [region]}
model:
  fixed: "1 + x"
  sigma: 0.5
  effects:
    - {name: trend, type: rw1, index: month, precision: 2.0}
"""
    )
    config = load_config(path)
    assert config.model.likelihood == "gaussian"
    assert config.model.fixed_prior_precision == 1e-6
    assert config.model.effects[0].type == "rw1"


def test_unknown_key_is_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\ndata: {time: t, response: y}\nmodel: {fixed: '1', typo: 3}\n")
    with pytest.raises(ConfigurationError, match="typo"):
        load_config(path)
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `python -m pytest tests/config/test_load.py -v`  
Expected: collection FAIL because `pylgm.config` does not exist.

- [ ] **Step 3: Implement the schema and loader**

```python
# src/pylgm/exceptions.py
class PyLGMError(Exception):
    """Base class for typed pyLGM errors."""


class ConfigurationError(PyLGMError):
    """Configuration could not be parsed or validated."""
```

```python
# src/pylgm/config/schema.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    time: str
    response: str
    panel: tuple[str, ...] = ()


class EffectConfig(StrictModel):
    name: str
    type: Literal["iid", "rw1", "rw2"]
    index: str
    precision: PositiveFloat = 1.0


class ModelConfig(StrictModel):
    likelihood: Literal["gaussian"] = "gaussian"
    fixed: str = "1"
    fixed_prior_precision: PositiveFloat = 1e-6
    sigma: PositiveFloat
    effects: tuple[EffectConfig, ...] = ()

    @model_validator(mode="after")
    def unique_effect_names(self) -> "ModelConfig":
        names = [effect.name for effect in self.effects]
        if len(names) != len(set(names)):
            raise ValueError("effect names must be unique")
        return self


class RunConfig(StrictModel):
    schema_version: Literal[1]
    data: DataConfig
    model: ModelConfig
```

```python
# src/pylgm/config/load.py
from pathlib import Path

import yaml
from pydantic import ValidationError

from pylgm.config.schema import RunConfig
from pylgm.exceptions import ConfigurationError


def load_config(path: Path) -> RunConfig:
    try:
        payload = yaml.safe_load(path.read_text())
        return RunConfig.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc
```

```python
# src/pylgm/config/__init__.py
from pylgm.config.load import load_config
from pylgm.config.schema import EffectConfig, RunConfig

__all__ = ["EffectConfig", "RunConfig", "load_config"]
```

- [ ] **Step 4: Run focused and lint tests**

Run: `python -m pytest tests/config/test_load.py -v && ruff check src/pylgm/config src/pylgm/exceptions.py tests/config`  
Expected: two tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/config src/pylgm/exceptions.py tests/config
git commit -m "feat: add strict versioned configuration"
```

### Task 3: Canonical Pandas panel contract

**Files:**
- Create: `src/pylgm/data/__init__.py`
- Create: `src/pylgm/data/panel.py`
- Modify: `src/pylgm/exceptions.py`
- Create: `tests/data/test_panel.py`

**Interfaces:**
- Consumes: `RunConfig.data` and a Pandas DataFrame.
- Produces: `CanonicalPanel.from_frame(frame, config) -> CanonicalPanel` with ordered `frame`, Boolean `observed`, and stable `key_columns`.

- [ ] **Step 1: Write panel validation tests**

```python
# tests/data/test_panel.py
import pandas as pd
import pytest

from pylgm.config.schema import DataConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import DataContractError


def test_panel_sorts_keys_and_preserves_missing_targets() -> None:
    frame = pd.DataFrame({"month": [2, 1], "region": ["A", "A"], "y": [None, 1.2]})
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y", panel=("region",)))
    assert panel.frame["month"].tolist() == [1, 2]
    assert panel.observed.tolist() == [True, False]


def test_duplicate_panel_key_fails() -> None:
    frame = pd.DataFrame({"month": [1, 1], "region": ["A", "A"], "y": [1.0, 2.0]})
    with pytest.raises(DataContractError, match="duplicate"):
        CanonicalPanel.from_frame(frame, DataConfig(time="month", response="y", panel=("region",)))
```

- [ ] **Step 2: Run tests and verify the contract is absent**

Run: `python -m pytest tests/data/test_panel.py -v`  
Expected: collection FAIL because `pylgm.data` does not exist.

- [ ] **Step 3: Implement immutable canonicalization**

Append this typed error to `src/pylgm/exceptions.py`:

```python
class DataContractError(PyLGMError):
    """Input data violates the canonical panel contract."""
```

Then create:

```python
# src/pylgm/data/panel.py
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pylgm.config.schema import DataConfig
from pylgm.exceptions import DataContractError


@dataclass(frozen=True)
class CanonicalPanel:
    frame: pd.DataFrame
    observed: np.ndarray
    key_columns: tuple[str, ...]
    response: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, config: DataConfig) -> "CanonicalPanel":
        keys = (*config.panel, config.time)
        required = {*keys, config.response}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataContractError(f"missing columns: {missing}")
        if frame.duplicated(list(keys)).any():
            raise DataContractError(f"duplicate panel keys: {keys}")
        ordered = frame.sort_values(list(keys), kind="stable").reset_index(drop=True).copy()
        observed = ordered[config.response].notna().to_numpy(dtype=bool)
        if not observed.any():
            raise DataContractError("panel contains no observed responses")
        return cls(ordered, observed, keys, config.response)
```

```python
# src/pylgm/data/__init__.py
from pylgm.data.panel import CanonicalPanel

__all__ = ["CanonicalPanel"]
```

- [ ] **Step 4: Run the contract tests**

Run: `python -m pytest tests/data/test_panel.py -v && ruff check src/pylgm/data tests/data`  
Expected: two tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/data src/pylgm/exceptions.py tests/data
git commit -m "feat: define canonical panel contract"
```

### Task 4: Immutable LGM IR and fixed effects

**Files:**
- Create: `src/pylgm/ir/__init__.py`
- Create: `src/pylgm/ir/model.py`
- Create: `src/pylgm/effects/__init__.py`
- Create: `src/pylgm/effects/fixed.py`
- Create: `tests/ir/test_fixed.py`

**Interfaces:**
- Consumes: canonical frame and fixed formula.
- Produces: `LatentBlock`, `CompiledLGM`, and `build_fixed(frame, formula, prior_precision) -> LatentBlock`.

- [ ] **Step 1: Write the fixed-block test**

```python
# tests/ir/test_fixed.py
import numpy as np
import pandas as pd

from pylgm.effects.fixed import build_fixed


def test_fixed_block_has_stable_columns_and_precision() -> None:
    frame = pd.DataFrame({"x": [2.0, 3.0]})
    block = build_fixed(frame, "1 + x", prior_precision=0.25)
    assert block.name == "fixed"
    assert block.labels == ("Intercept", "x")
    np.testing.assert_allclose(block.design.toarray(), [[1.0, 2.0], [1.0, 3.0]])
    np.testing.assert_allclose(block.precision.toarray(), np.eye(2) * 0.25)
    assert block.constraints.shape == (0, 2)
```

- [ ] **Step 2: Run the test and verify missing IR modules**

Run: `python -m pytest tests/ir/test_fixed.py -v`  
Expected: collection FAIL because `pylgm.effects.fixed` does not exist.

- [ ] **Step 3: Implement the IR and fixed builder**

```python
# src/pylgm/ir/model.py
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class LatentBlock:
    name: str
    labels: tuple[str, ...]
    design: csr_matrix
    precision: csr_matrix
    constraints: np.ndarray


@dataclass(frozen=True)
class CompiledLGM:
    y: np.ndarray
    observed: np.ndarray
    offset: np.ndarray
    design: csr_matrix
    precision: csr_matrix
    constraints: np.ndarray
    labels: tuple[str, ...]
    sigma: float
    blocks: tuple[LatentBlock, ...]
```

```python
# src/pylgm/effects/fixed.py
import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy.sparse import csr_matrix, eye

from pylgm.ir.model import LatentBlock


def build_fixed(frame: pd.DataFrame, formula: str, prior_precision: float) -> LatentBlock:
    matrix = model_matrix(formula, frame)
    design = csr_matrix(np.asarray(matrix, dtype=float))
    labels = tuple(matrix.model_spec.column_names)
    return LatentBlock(
        name="fixed",
        labels=labels,
        design=design,
        precision=eye(design.shape[1], format="csr") * prior_precision,
        constraints=np.empty((0, design.shape[1]), dtype=float),
    )
```

```python
# src/pylgm/ir/__init__.py
from pylgm.ir.model import CompiledLGM, LatentBlock

__all__ = ["CompiledLGM", "LatentBlock"]
```

```python
# src/pylgm/effects/__init__.py
from pylgm.effects.fixed import build_fixed

__all__ = ["build_fixed"]
```

- [ ] **Step 4: Run fixed-block tests**

Run: `python -m pytest tests/ir/test_fixed.py -v && ruff check src/pylgm/ir src/pylgm/effects tests/ir`  
Expected: one test passes and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/ir src/pylgm/effects tests/ir
git commit -m "feat: add immutable LGM IR and fixed effects"
```

### Task 5: IID and intrinsic random-walk effects

**Files:**
- Create: `src/pylgm/effects/iid.py`
- Create: `src/pylgm/effects/random_walk.py`
- Modify: `src/pylgm/effects/__init__.py`
- Create: `tests/effects/test_structured.py`

**Interfaces:**
- Consumes: canonical frame, index name, effect name, and fixed precision.
- Produces: `build_iid(...) -> LatentBlock` and `build_random_walk(..., order: Literal[1, 2]) -> LatentBlock`.

- [ ] **Step 1: Write structure and null-space tests**

```python
# tests/effects/test_structured.py
import numpy as np
import pandas as pd

from pylgm.effects import build_iid, build_random_walk


def test_iid_uses_first_seen_sorted_levels() -> None:
    frame = pd.DataFrame({"region": ["A", "B", "A"]})
    block = build_iid(frame, "region_effect", "region", 2.0)
    assert block.labels == ("A", "B")
    np.testing.assert_allclose(block.design.toarray(), [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(block.precision.toarray(), np.eye(2) * 2.0)


def test_rw2_has_rank_two_null_space_constraints() -> None:
    frame = pd.DataFrame({"month": [1, 2, 3, 4]})
    block = build_random_walk(frame, "trend", "month", 3.0, order=2)
    assert block.precision.shape == (4, 4)
    assert block.constraints.shape == (2, 4)
    np.testing.assert_allclose(block.constraints @ block.precision.toarray(), 0.0, atol=1e-12)
```

- [ ] **Step 2: Run tests and verify builders are absent**

Run: `python -m pytest tests/effects/test_structured.py -v`  
Expected: collection FAIL because the builders are not exported.

- [ ] **Step 3: Implement deterministic structured blocks**

```python
# src/pylgm/effects/iid.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, eye

from pylgm.ir.model import LatentBlock


def build_iid(frame: pd.DataFrame, name: str, index: str, precision: float) -> LatentBlock:
    levels = tuple(sorted(frame[index].drop_duplicates().tolist()))
    positions = {level: column for column, level in enumerate(levels)}
    rows = np.arange(len(frame))
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix((np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels)))
    return LatentBlock(name, tuple(map(str, levels)), design, eye(len(levels), format="csr") * precision, np.empty((0, len(levels))))
```

```python
# src/pylgm/effects/random_walk.py
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pylgm.ir.model import LatentBlock


def build_random_walk(
    frame: pd.DataFrame,
    name: str,
    index: str,
    precision: float,
    order: Literal[1, 2],
) -> LatentBlock:
    levels = tuple(sorted(frame[index].drop_duplicates().tolist()))
    if len(levels) <= order:
        raise ValueError(f"{name} requires more than {order} ordered levels")
    positions = {level: column for column, level in enumerate(levels)}
    rows = np.arange(len(frame))
    columns = np.array([positions[value] for value in frame[index]])
    design = csr_matrix((np.ones(len(frame)), (rows, columns)), shape=(len(frame), len(levels)))
    difference = np.diff(np.eye(len(levels)), n=order, axis=0)
    precision_matrix = csr_matrix(precision * (difference.T @ difference))
    coordinate = np.arange(len(levels), dtype=float)
    constraints = np.ones((1, len(levels)))
    if order == 2:
        constraints = np.vstack([constraints, coordinate - coordinate.mean()])
    return LatentBlock(name, tuple(map(str, levels)), design, precision_matrix, constraints)
```

Replace `src/pylgm/effects/__init__.py` with:

```python
from pylgm.effects.fixed import build_fixed
from pylgm.effects.iid import build_iid
from pylgm.effects.random_walk import build_random_walk

__all__ = ["build_fixed", "build_iid", "build_random_walk"]
```

- [ ] **Step 4: Run structured-effect tests**

Run: `python -m pytest tests/effects/test_structured.py -v && ruff check src/pylgm/effects tests/effects`  
Expected: two tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/effects tests/effects
git commit -m "feat: add IID and intrinsic random-walk effects"
```

### Task 6: Configuration-to-IR compiler

**Files:**
- Create: `src/pylgm/compiler.py`
- Create: `tests/test_compiler.py`

**Interfaces:**
- Consumes: `compile_model(config: RunConfig, panel: CanonicalPanel) -> CompiledLGM`.
- Produces: globally assembled sparse design/precision matrices, global constraints, stable labels, observed response, and fixed sigma.

- [ ] **Step 1: Write an assembly test**

```python
# tests/test_compiler.py
import pandas as pd

from pylgm.compiler import compile_model
from pylgm.config.schema import RunConfig
from pylgm.data import CanonicalPanel


def test_compiler_assembles_named_blocks() -> None:
    config = RunConfig.model_validate({
        "schema_version": 1,
        "data": {"time": "month", "response": "y", "panel": ["region"]},
        "model": {
            "fixed": "1 + x", "sigma": 1.0,
            "effects": [{"name": "trend", "type": "rw1", "index": "month", "precision": 2.0}],
        },
    })
    frame = pd.DataFrame({"month": [1, 2], "region": ["A", "A"], "x": [0.0, 1.0], "y": [1.0, None]})
    panel = CanonicalPanel.from_frame(frame, config.data)
    model = compile_model(config, panel)
    assert [block.name for block in model.blocks] == ["fixed", "trend"]
    assert model.design.shape == (2, 4)
    assert model.precision.shape == (4, 4)
    assert model.constraints.shape == (1, 4)
    assert model.observed.tolist() == [True, False]
```

- [ ] **Step 2: Run the test and verify compiler absence**

Run: `python -m pytest tests/test_compiler.py -v`  
Expected: collection FAIL because `pylgm.compiler` does not exist.

- [ ] **Step 3: Implement deterministic global assembly**

```python
# src/pylgm/compiler.py
import numpy as np
from scipy.sparse import block_diag, hstack

from pylgm.config import RunConfig
from pylgm.data import CanonicalPanel
from pylgm.effects import build_fixed, build_iid, build_random_walk
from pylgm.ir.model import CompiledLGM, LatentBlock


def _structured_blocks(config: RunConfig, panel: CanonicalPanel) -> list[LatentBlock]:
    blocks: list[LatentBlock] = []
    for effect in config.model.effects:
        if effect.type == "iid":
            blocks.append(build_iid(panel.frame, effect.name, effect.index, effect.precision))
        else:
            order = 1 if effect.type == "rw1" else 2
            blocks.append(build_random_walk(panel.frame, effect.name, effect.index, effect.precision, order))
    return blocks


def compile_model(config: RunConfig, panel: CanonicalPanel) -> CompiledLGM:
    blocks = [build_fixed(panel.frame, config.model.fixed, config.model.fixed_prior_precision)]
    blocks.extend(_structured_blocks(config, panel))
    widths = [block.design.shape[1] for block in blocks]
    total = sum(widths)
    constraint_rows: list[np.ndarray] = []
    start = 0
    for block, width in zip(blocks, widths, strict=True):
        for local in block.constraints:
            row = np.zeros(total)
            row[start : start + width] = local
            constraint_rows.append(row)
        start += width
    constraints = np.vstack(constraint_rows) if constraint_rows else np.empty((0, total))
    y = panel.frame[panel.response].fillna(0.0).to_numpy(dtype=float)
    return CompiledLGM(
        y=y,
        observed=panel.observed,
        offset=np.zeros(len(panel.frame)),
        design=hstack([block.design for block in blocks], format="csr"),
        precision=block_diag([block.precision for block in blocks], format="csr"),
        constraints=constraints,
        labels=tuple(f"{block.name}:{label}" for block in blocks for label in block.labels),
        sigma=float(config.model.sigma),
        blocks=tuple(blocks),
    )
```

- [ ] **Step 4: Run compiler and full tests**

Run: `python -m pytest tests/test_compiler.py -v && python -m pytest -q && ruff check .`  
Expected: compiler test and all prior tests pass; Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/compiler.py tests/test_compiler.py
git commit -m "feat: compile configuration into LGM IR"
```

### Task 7: Constrained exact Gaussian engine

**Files:**
- Create: `src/pylgm/inference/__init__.py`
- Create: `src/pylgm/inference/result.py`
- Create: `src/pylgm/inference/gaussian.py`
- Create: `tests/inference/test_gaussian.py`

**Interfaces:**
- Consumes: `fit_gaussian(model: CompiledLGM) -> GaussianResult`.
- Produces: latent posterior mean/covariance, observed-data conditional log marginal likelihood, labels, and prediction mean/variance.

- [ ] **Step 1: Write analytical and constraint tests**

```python
# tests/inference/test_gaussian.py
import numpy as np
from scipy.sparse import csr_matrix

from pylgm.inference import fit_gaussian
from pylgm.ir.model import CompiledLGM


def test_scalar_gaussian_matches_conjugate_solution() -> None:
    model = CompiledLGM(
        y=np.array([2.0]), observed=np.array([True]), offset=np.zeros(1),
        design=csr_matrix([[1.0]]), precision=csr_matrix([[1.0]]),
        constraints=np.empty((0, 1)), labels=("x",), sigma=1.0, blocks=(),
    )
    result = fit_gaussian(model)
    np.testing.assert_allclose(result.mean, [1.0])
    np.testing.assert_allclose(result.covariance, [[0.5]])
    np.testing.assert_allclose(result.predictive_mean, [1.0])
    np.testing.assert_allclose(result.predictive_variance, [1.5])


def test_sum_to_zero_constraint_is_satisfied() -> None:
    model = CompiledLGM(
        y=np.array([1.0, -1.0]), observed=np.array([True, True]), offset=np.zeros(2),
        design=csr_matrix(np.eye(2)), precision=csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        constraints=np.array([[1.0, 1.0]]), labels=("a", "b"), sigma=1.0, blocks=(),
    )
    result = fit_gaussian(model)
    np.testing.assert_allclose(model.constraints @ result.mean, 0.0, atol=1e-12)
```

- [ ] **Step 2: Run tests and verify engine absence**

Run: `python -m pytest tests/inference/test_gaussian.py -v`  
Expected: collection FAIL because `pylgm.inference` does not exist.

- [ ] **Step 3: Implement reduced-coordinate exact inference**

```python
# src/pylgm/inference/result.py
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianResult:
    labels: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    log_marginal_likelihood: float
    predictive_mean: np.ndarray
    predictive_variance: np.ndarray
```

```python
# src/pylgm/inference/gaussian.py
import numpy as np
from scipy.linalg import cho_factor, cho_solve, null_space

from pylgm.inference.result import GaussianResult
from pylgm.ir.model import CompiledLGM


def fit_gaussian(model: CompiledLGM) -> GaussianResult:
    latent_size = model.precision.shape[0]
    basis = null_space(model.constraints) if model.constraints.shape[0] else np.eye(latent_size)
    observed_design = model.design[model.observed]
    reduced_design = observed_design @ basis
    reduced_precision = basis.T @ model.precision.toarray() @ basis
    residual = model.y[model.observed] - model.offset[model.observed]
    variance = model.sigma**2
    posterior_precision = reduced_precision + (reduced_design.T @ reduced_design) / variance
    factor = cho_factor(posterior_precision, lower=True, check_finite=True)
    score = np.asarray(reduced_design.T @ residual / variance).reshape(-1)
    reduced_mean = cho_solve(factor, score)
    reduced_covariance = cho_solve(factor, np.eye(posterior_precision.shape[0]))
    mean = basis @ reduced_mean
    covariance = basis @ reduced_covariance @ basis.T
    sign_q, logdet_q = np.linalg.slogdet(reduced_precision)
    sign_p, logdet_p = np.linalg.slogdet(posterior_precision)
    if sign_q <= 0 or sign_p <= 0:
        raise np.linalg.LinAlgError("reduced prior and posterior precision must be positive definite")
    quadratic = residual @ residual / variance - score @ reduced_mean
    n_observed = int(model.observed.sum())
    log_marginal = -0.5 * (
        n_observed * np.log(2 * np.pi * variance) - logdet_q + logdet_p + quadratic
    )
    predictive_mean = model.offset + model.design @ mean
    latent_predictive_variance = np.einsum(
        "ij,jk,ik->i", model.design.toarray(), covariance, model.design.toarray()
    )
    return GaussianResult(
        labels=model.labels,
        mean=mean,
        covariance=covariance,
        log_marginal_likelihood=float(log_marginal),
        predictive_mean=np.asarray(predictive_mean).reshape(-1),
        predictive_variance=latent_predictive_variance + variance,
    )
```

```python
# src/pylgm/inference/__init__.py
from pylgm.inference.gaussian import fit_gaussian
from pylgm.inference.result import GaussianResult

__all__ = ["GaussianResult", "fit_gaussian"]
```

- [ ] **Step 4: Run analytical and full tests**

Run: `python -m pytest tests/inference/test_gaussian.py -v && python -m pytest -q && ruff check .`  
Expected: analytical tests and the complete suite pass; Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/inference tests/inference
git commit -m "feat: add constrained exact Gaussian inference"
```

### Task 8: Python pipeline, artifacts, and CLI

**Files:**
- Create: `src/pylgm/artifacts/__init__.py`
- Create: `src/pylgm/artifacts/run.py`
- Create: `src/pylgm/pipeline.py`
- Create: `src/pylgm/cli.py`
- Modify: `src/pylgm/__init__.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Pipeline.from_yaml(path).run(frame, output_dir) -> GaussianResult` and CSV input in the CLI.
- Produces: resolved config JSON, Gaussian arrays NPZ, summary JSON, and `pylgm fit CONFIG DATA --output DIR`.

- [ ] **Step 1: Write pipeline and CLI tests**

```python
# tests/test_pipeline.py
import json
from pathlib import Path

import pandas as pd

from pylgm import Pipeline


def test_pipeline_persists_resolved_run(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n")
    output = tmp_path / "run"
    result = Pipeline.from_yaml(config).run(pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), output)
    assert result.mean.shape == (1,)
    assert json.loads((output / "summary.json").read_text())["engine"] == "exact_gaussian"
    assert (output / "resolved_config.json").exists()
    assert (output / "posterior.npz").exists()
    assert (output / "environment.json").exists()
    assert len(json.loads((output / "summary.json").read_text())["data_fingerprint"]) == 64
```

```python
# tests/test_cli.py
from pathlib import Path

from typer.testing import CliRunner

from pylgm.cli import app


def test_cli_fits_csv(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    data = tmp_path / "data.csv"
    output = tmp_path / "run"
    config.write_text("schema_version: 1\ndata: {time: month, response: y}\nmodel: {fixed: '1', sigma: 1.0}\n")
    data.write_text("month,y\n1,1.0\n2,2.0\n")
    response = CliRunner().invoke(app, ["fit", str(config), str(data), "--output", str(output)])
    assert response.exit_code == 0
    assert "exact_gaussian" in response.stdout
```

- [ ] **Step 2: Run tests and verify pipeline absence**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -v`  
Expected: collection FAIL because `Pipeline` and `pylgm.cli` do not exist.

- [ ] **Step 3: Implement local orchestration and artifacts**

```python
# src/pylgm/artifacts/run.py
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from pylgm.config import RunConfig
from pylgm.inference import GaussianResult


def write_run(
    output: Path,
    config: RunConfig,
    result: GaussianResult,
    data_fingerprint: str,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.json").write_text(config.model_dump_json(indent=2))
    np.savez_compressed(output / "posterior.npz", mean=result.mean, covariance=result.covariance)
    summary = {
        "engine": "exact_gaussian",
        "conditional_on_fixed_hyperparameters": True,
        "data_fingerprint": data_fingerprint,
        "log_marginal_likelihood": result.log_marginal_likelihood,
        "labels": list(result.labels),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    dependencies = ["pylgm", "numpy", "scipy", "pandas", "formulaic", "pydantic"]
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in dependencies},
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True))
```

```python
# src/pylgm/pipeline.py
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pylgm.artifacts.run import write_run
from pylgm.compiler import compile_model
from pylgm.config import RunConfig, load_config
from pylgm.data import CanonicalPanel
from pylgm.inference import GaussianResult, fit_gaussian


@dataclass(frozen=True)
class Pipeline:
    config: RunConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        return cls(load_config(Path(path)))

    def run(self, frame: pd.DataFrame, output: str | Path) -> GaussianResult:
        panel = CanonicalPanel.from_frame(frame, self.config.data)
        result = fit_gaussian(compile_model(self.config, panel))
        hashed = pd.util.hash_pandas_object(panel.frame, index=True).values.tobytes()
        fingerprint = hashlib.sha256(hashed).hexdigest()
        write_run(Path(output), self.config, result, fingerprint)
        return result
```

```python
# src/pylgm/cli.py
from pathlib import Path

import pandas as pd
import typer

from pylgm.pipeline import Pipeline

app = typer.Typer(no_args_is_help=True)


@app.command()
def fit(config: Path, data: Path, output: Path = typer.Option(...)) -> None:
    result = Pipeline.from_yaml(config).run(pd.read_csv(data), output)
    typer.echo(f"engine=exact_gaussian log_marginal_likelihood={result.log_marginal_likelihood:.6f}")
```

```python
# src/pylgm/artifacts/__init__.py
from pylgm.artifacts.run import write_run

__all__ = ["write_run"]
```

Replace `src/pylgm/__init__.py` with:

```python
"""pyLGM public package."""

from pylgm.pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["Pipeline", "__version__"]
```

- [ ] **Step 4: Run pipeline, CLI, and full quality gate**

Run: `python -m pytest tests/test_pipeline.py tests/test_cli.py -v && python -m pytest -q && ruff check .`  
Expected: pipeline and CLI tests pass, the complete suite passes, and Ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add src/pylgm/artifacts src/pylgm/pipeline.py src/pylgm/cli.py src/pylgm/__init__.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: add local Gaussian pipeline and CLI"
```

### Task 9: End-to-end synthetic panel and documentation

**Files:**
- Create: `examples/synthetic_panel/data.csv`
- Create: `examples/synthetic_panel/config.yaml`
- Create: `examples/synthetic_panel/README.md`
- Modify: `README.md`
- Create: `tests/integration/test_synthetic_example.py`

**Interfaces:**
- Consumes: public CLI and Python interfaces only.
- Produces: a non-economic panel example proving that the core is domain-neutral and locally reproducible.

- [ ] **Step 1: Write an end-to-end example test**

```python
# tests/integration/test_synthetic_example.py
from pathlib import Path

import pandas as pd

from pylgm import Pipeline


def test_synthetic_example_runs(tmp_path: Path) -> None:
    root = Path("examples/synthetic_panel")
    result = Pipeline.from_yaml(root / "config.yaml").run(pd.read_csv(root / "data.csv"), tmp_path / "run")
    assert result.predictive_mean.shape == (8,)
    assert result.labels[:2] == ("fixed:Intercept", "fixed:x")
    assert all(value > 0 for value in result.predictive_variance)
```

- [ ] **Step 2: Run the test and verify fixtures are absent**

Run: `python -m pytest tests/integration/test_synthetic_example.py -v`  
Expected: FAIL with `FileNotFoundError` for `examples/synthetic_panel/config.yaml`.

- [ ] **Step 3: Add the complete example**

```yaml
# examples/synthetic_panel/config.yaml
schema_version: 1
data:
  time: month
  response: y
  panel: [region]
model:
  likelihood: gaussian
  fixed: "1 + x"
  fixed_prior_precision: 0.001
  sigma: 0.4
  effects:
    - {name: region, type: iid, index: region, precision: 2.0}
    - {name: trend, type: rw1, index: month, precision: 4.0}
```

Create `data.csv` with exactly these rows:

```csv
month,region,x,y
1,A,0.0,0.8
2,A,0.2,1.1
3,A,0.4,1.4
4,A,0.6,
1,B,0.0,1.2
2,B,0.2,1.4
3,B,0.4,1.7
4,B,0.6,
```

Document both commands in `examples/synthetic_panel/README.md`:

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
python -c 'import pandas as pd; from pylgm import Pipeline; Pipeline.from_yaml("examples/synthetic_panel/config.yaml").run(pd.read_csv("examples/synthetic_panel/data.csv"), "synthetic-run-python")'
```

Append this exact section to the root `README.md`:

````markdown
## Gaussian foundation example

```bash
pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output synthetic-run
```

The example combines fixed effects, an IID region effect, and an intrinsic RW1
time effect. Rows with missing responses are prediction targets. Version 0.1's
exact Gaussian engine conditions on the effect precisions and observation
standard deviation declared in configuration; it does not integrate their
uncertainty.
````

- [ ] **Step 4: Run final verification**

Run: `export PYLGM_VERIFY_DIR=$(mktemp -d) && pylgm fit examples/synthetic_panel/config.yaml examples/synthetic_panel/data.csv --output "$PYLGM_VERIFY_DIR/run" && python -m pytest -q && ruff check . && python -c 'import json, os, pathlib; p=pathlib.Path(os.environ["PYLGM_VERIFY_DIR"])/"run"/"summary.json"; assert json.loads(p.read_text())["conditional_on_fixed_hyperparameters"] is True'`  
Expected: CLI prints `engine=exact_gaussian` with a finite log marginal likelihood, all tests pass, Ruff reports no errors, and the summary assertion exits successfully.

- [ ] **Step 5: Commit**

```bash
git add examples/synthetic_panel README.md tests/integration/test_synthetic_example.py
git commit -m "docs: add synthetic Gaussian panel example"
```

## Completion gate

Before declaring this plan complete, run:

```bash
python -m pytest -q
ruff check .
git status --short
```

Expected: every test passes, Ruff reports `All checks passed!`, and Git status is clean. Confirm that the example result labels the engine `exact_gaussian` and its conditioning on fixed hyperparameters. Do not begin the NUTS or spatial plan until this gate passes and the Gaussian fixture outputs are retained for cross-engine tests.

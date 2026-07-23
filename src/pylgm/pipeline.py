"""Local orchestration for exact Gaussian pyLGM fits."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pylgm.artifacts.run import write_run
from pylgm.compiler import compile_model
from pylgm.config import RunConfig, load_config
from pylgm.data import CanonicalPanel
from pylgm.data.fingerprint import panel_fingerprint
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
        write_run(Path(output), self.config, result, panel_fingerprint(panel))
        return result

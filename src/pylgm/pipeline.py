"""Local orchestration for exact Gaussian pyLGM fits."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pylgm.artifacts.run import write_run
from pylgm.compiler import compile_model
from pylgm.config import RunConfig, load_config
from pylgm.data import CanonicalPanel
from pylgm.inference import GaussianResult, fit_gaussian


def _panel_fingerprint(panel: CanonicalPanel) -> str:
    """Hash canonical data values together with their schema and column order."""
    frame = panel.frame
    hashed_values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    metadata = json.dumps(
        {
            "columns": list(frame.columns),
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "key_columns": list(panel.key_columns),
            "response": panel.response,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(metadata + hashed_values).hexdigest()


@dataclass(frozen=True)
class Pipeline:
    config: RunConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        return cls(load_config(Path(path)))

    def run(self, frame: pd.DataFrame, output: str | Path) -> GaussianResult:
        panel = CanonicalPanel.from_frame(frame, self.config.data)
        result = fit_gaussian(compile_model(self.config, panel))
        write_run(Path(output), self.config, result, _panel_fingerprint(panel))
        return result

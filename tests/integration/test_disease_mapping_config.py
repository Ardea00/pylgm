from pathlib import Path

import numpy as np
import pandas as pd

from pylgm import Besag, Poisson
from pylgm.config import load_model


def test_disease_mapping_config_fits_from_files() -> None:
    """The shipped declarative disease-mapping example loads (graph_file:
    graph.json) and fits from data.csv with zero Python model code."""
    root = Path("examples/disease_mapping")
    model = load_model(root / "config.yaml")
    assert isinstance(model.likelihood, Poisson)
    assert model.offset == "log_E"
    assert any(isinstance(effect, Besag) for effect in model.predictor.effects)

    result = model.fit(pd.read_csv(root / "data.csv"), engine="laplace")
    assert np.isfinite(result.fitted_mean).all()

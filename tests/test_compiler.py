import numpy as np
import pandas as pd
import pytest

from pylgm.compiler import compile_model
from pylgm.config.schema import RunConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import ConfigurationError


def test_compiler_assembles_named_blocks() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y", "panel": ["region"]},
            "model": {
                "fixed": "1 + x",
                "fixed_prior_precision": 0.5,
                "sigma": 1.5,
                "effects": [
                    {
                        "name": "group",
                        "type": "iid",
                        "index": "group",
                        "precision": 3.0,
                    },
                    {
                        "name": "trend",
                        "type": "rw2",
                        "index": "month",
                        "precision": 2.0,
                    }
                ],
            },
        }
    )
    frame = pd.DataFrame(
        {
            "month": [1, 2, 3],
            "region": ["A", "A", "A"],
            "group": ["B", "A", "B"],
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, None, 3.0],
        }
    )
    panel = CanonicalPanel.from_frame(frame, config.data)

    model = compile_model(config, panel)

    assert [block.name for block in model.blocks] == ["fixed", "group", "trend"]
    assert model.labels == (
        "fixed:Intercept",
        "fixed:x",
        "group:A",
        "group:B",
        "trend:1",
        "trend:2",
        "trend:3",
    )
    np.testing.assert_allclose(
        model.design.toarray(),
        [[1, 0, 0, 1, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0], [1, 2, 0, 1, 0, 0, 1]],
    )
    np.testing.assert_allclose(
        model.precision.toarray(),
        [
            [0.5, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 2, -4, 2],
            [0, 0, 0, 0, -4, 8, -4],
            [0, 0, 0, 0, 2, -4, 2],
        ],
    )
    np.testing.assert_allclose(
        model.constraints,
        [[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, -1, 0, 1]],
    )
    assert model.y.tolist() == [1.0, 0.0, 3.0]
    assert model.observed.tolist() == [True, False, True]
    assert model.sigma == 1.5


def test_compiler_rejects_duplicate_qualified_labels() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {
                "sigma": 1.0,
                "effects": [
                    {
                        "name": "fixed",
                        "type": "iid",
                        "index": "group",
                        "precision": 1.0,
                    }
                ],
            },
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "group": ["Intercept"], "y": [1.0]}),
        config.data,
    )

    with pytest.raises(ConfigurationError, match="duplicate latent labels"):
        compile_model(config, panel)

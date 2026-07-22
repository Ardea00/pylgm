import pandas as pd

from pylgm.compiler import compile_model
from pylgm.config.schema import RunConfig
from pylgm.data import CanonicalPanel


def test_compiler_assembles_named_blocks() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y", "panel": ["region"]},
            "model": {
                "fixed": "1 + x",
                "sigma": 1.0,
                "effects": [
                    {
                        "name": "trend",
                        "type": "rw1",
                        "index": "month",
                        "precision": 2.0,
                    }
                ],
            },
        }
    )
    frame = pd.DataFrame(
        {
            "month": [1, 2],
            "region": ["A", "A"],
            "x": [0.0, 1.0],
            "y": [1.0, None],
        }
    )
    panel = CanonicalPanel.from_frame(frame, config.data)

    model = compile_model(config, panel)

    assert [block.name for block in model.blocks] == ["fixed", "trend"]
    assert model.design.shape == (2, 4)
    assert model.precision.shape == (4, 4)
    assert model.constraints.shape == (1, 4)
    assert model.observed.tolist() == [True, False]

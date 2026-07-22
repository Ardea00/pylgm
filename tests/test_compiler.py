import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pylgm.compiler import compile_model
from pylgm.config.schema import RunConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.ir import LatentBlock


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
    with pytest.raises(ValueError, match="reserved"):
        RunConfig.model_validate(
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


def _config_with_effect_index(index: str = "group") -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {
                "fixed": "1 + x",
                "sigma": 1.0,
                "effects": [{"name": "group", "type": "iid", "index": index}],
            },
        }
    )


def test_compiler_rejects_panel_missing_configured_data_columns() -> None:
    config = _config_with_effect_index()
    other_data = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "other_time", "response": "other_y"},
            "model": {"sigma": 1.0},
        }
    ).data
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"other_time": [1], "other_y": [1.0], "x": [0.0], "group": ["A"]}),
        other_data,
    )

    with pytest.raises(DataContractError, match="missing configured data columns"):
        compile_model(config, panel)


def test_compiler_rejects_missing_effect_index_with_typed_error() -> None:
    config = _config_with_effect_index("missing_group")
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "y": [1.0], "x": [0.0]}), config.data
    )

    with pytest.raises(DataContractError, match="effect index.*missing_group"):
        compile_model(config, panel)


@pytest.mark.parametrize("formula", ["1 + missing", "1 + ("])
def test_compiler_wraps_fixed_formula_failures(formula: str) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"fixed": formula, "sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "y": [1.0]}), config.data
    )

    with pytest.raises(CompilationError, match="fixed formula"):
        compile_model(config, panel)


def test_compiler_rejects_fixed_formula_row_dropping() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"fixed": "1 + x", "sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0], "x": [0.0, None]}),
        config.data,
    )

    with pytest.raises(CompilationError, match="missing covariates|row"):
        compile_model(config, panel)


def test_compiler_rejects_block_with_wrong_panel_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"fixed": "1", "sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), config.data
    )
    bad_block = LatentBlock(
        "bad",
        ("level",),
        csr_matrix([[1.0]]),
        csr_matrix([[1.0]]),
        np.empty((0, 1)),
    )
    monkeypatch.setattr("pylgm.compiler._structured_blocks", lambda config, panel: [bad_block])

    with pytest.raises(CompilationError, match="row count"):
        compile_model(config, panel)

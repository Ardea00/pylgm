import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from typing import cast

from pylgm import Bernoulli, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.effects import Predictor
from pylgm.compiler import compile_family, compile_gaussian_family, compile_lgm, compile_model
from pylgm.config.experiment import EvaluationConfig, ExperimentDataConfig, OriginConfig
from pylgm.config.schema import DataConfig, RunConfig
from pylgm.data import CanonicalPanel
from pylgm.evaluation import build_fold_definitions, materialize_fold
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.ir import CompiledFamily, LatentBlock
from pylgm.likelihoods import CompiledPoisson
from pylgm.parameters import Hyperparameter


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
                    },
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
    panel = CanonicalPanel(
        pd.DataFrame({"x": [0.0], "group": ["A"]}),
        np.array([True]),
        ("month",),
        "y",
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
    panel = CanonicalPanel.from_frame(pd.DataFrame({"month": [1], "y": [1.0]}), config.data)

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
    panel = CanonicalPanel.from_frame(pd.DataFrame({"month": [1, 2], "y": [1.0, 2.0]}), config.data)
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


class _MetadataOnlyPanel:
    def __init__(self, response: str, key_columns: tuple[str, ...]) -> None:
        self.response = response
        self.key_columns = key_columns

    @property
    def frame(self) -> pd.DataFrame:
        raise AssertionError("frame accessed before panel metadata validation")


def test_compiler_rejects_panel_response_metadata_mismatch_before_frame_access() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "time", "response": "y"},
            "model": {"sigma": 1.0},
        }
    )
    panel = cast(CanonicalPanel, _MetadataOnlyPanel("other_y", ("time",)))

    with pytest.raises(DataContractError, match="response metadata"):
        compile_model(config, panel)


def test_compiler_rejects_panel_key_metadata_mismatch_before_frame_access() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "time", "response": "y", "panel": ["region"]},
            "model": {"sigma": 1.0},
        }
    )
    panel = cast(CanonicalPanel, _MetadataOnlyPanel("y", ("time", "region")))

    with pytest.raises(DataContractError, match="key metadata"):
        compile_model(config, panel)


def test_qualified_label_collision_is_a_compilation_error() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "time", "response": "y"},
            "model": {
                "fixed": "0",
                "sigma": 1.0,
                "effects": [
                    {"name": "a:b", "type": "iid", "index": "first"},
                    {"name": "a", "type": "iid", "index": "second"},
                ],
            },
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"time": [1], "y": [1.0], "first": ["c"], "second": ["b:c"]}),
        config.data,
    )

    with pytest.raises(CompilationError, match="duplicate latent labels"):
        compile_model(config, panel)


@pytest.mark.parametrize("optimized", [("unknown",), ("sigma", "sigma"), (1,)])
def test_gaussian_family_compiler_rejects_invalid_optimized_names(
    optimized: tuple[object, ...],
) -> None:
    config = _config_with_effect_index()
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "y": [1.0], "x": [0.0], "group": ["A"]}),
        config.data,
    )

    with pytest.raises(CompilationError, match="optimized parameter names|unknown"):
        compile_gaussian_family(
            config.data,
            config.model,
            panel,
            optimized=optimized,  # type: ignore[arg-type]
        )


def test_nonlexical_categorical_order_aligns_folds_labels_and_rw2_precision() -> None:
    data = ExperimentDataConfig(time="phase", response="y")
    evaluation = EvaluationConfig(horizons=(2,), origins=OriginConfig(values=("middle",)))
    frame = pd.DataFrame(
        {
            "phase": pd.Categorical(
                ["early", "middle", "late"],
                categories=["middle", "late", "unused", "early"],
                ordered=True,
            ),
            "y": [3.0, 1.0, 2.0],
        }
    )
    definition = build_fold_definitions(frame, data, evaluation)[0]
    fold = materialize_fold(frame, data, evaluation, definition)
    model = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "phase", "response": "y"},
            "model": {
                "fixed": "1",
                "sigma": 1.0,
                "effects": [{"name": "trend", "type": "rw2", "index": "phase", "precision": 2.0}],
            },
        }
    )

    compiled = compile_model(model, CanonicalPanel.from_frame(fold.model_frame, model.data))
    trend = compiled.blocks[1]

    assert definition.origin == "middle"
    assert definition.target == "early"
    assert trend.labels == ("middle", "late", "early")
    np.testing.assert_allclose(
        trend.precision.toarray(),
        [[2, -4, 2], [-4, 8, -4], [2, -4, 2]],
    )


def test_compiler_translates_unordered_rw_categorical_index_to_typed_error() -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "phase", "response": "y"},
            "model": {
                "sigma": 1.0,
                "effects": [{"name": "trend", "type": "rw1", "index": "phase"}],
            },
        }
    )
    frame = pd.DataFrame(
        {
            "phase": pd.Categorical(
                ["middle", "late"], categories=["middle", "late"], ordered=False
            ),
            "y": [1.0, 2.0],
        }
    )

    with pytest.raises(CompilationError, match="categorical.*ordered"):
        compile_model(config, CanonicalPanel.from_frame(frame, config.data))


@pytest.mark.parametrize("error", [RuntimeError("fixed bug"), MemoryError("fixed oom")])
def test_unexpected_fixed_builder_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {"sigma": 1.0},
        }
    )
    panel = CanonicalPanel.from_frame(pd.DataFrame({"month": [1], "y": [1.0]}), config.data)

    def broken_fixed(*args: object, **kwargs: object) -> LatentBlock:
        raise error

    monkeypatch.setattr("pylgm.compiler.build_fixed", broken_fixed)
    with pytest.raises(type(error), match="fixed"):
        compile_model(config, panel)


@pytest.mark.parametrize("error", [RuntimeError("effect bug"), MemoryError("effect oom")])
def test_unexpected_effect_builder_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    config = RunConfig.model_validate(
        {
            "schema_version": 1,
            "data": {"time": "month", "response": "y"},
            "model": {
                "sigma": 1.0,
                "effects": [{"name": "group", "type": "iid", "index": "group"}],
            },
        }
    )
    panel = CanonicalPanel.from_frame(
        pd.DataFrame({"month": [1], "group": ["A"], "y": [1.0]}), config.data
    )

    def broken_effect(*args: object, **kwargs: object) -> LatentBlock:
        raise error

    monkeypatch.setattr("pylgm.compiler.build_iid", broken_effect)
    with pytest.raises(type(error), match="effect"):
        compile_model(config, panel)


def _panel(frame):
    return CanonicalPanel.from_frame(frame, DataConfig(time="t", response="y", panel=()))


def test_compile_poisson_uses_non_gaussian_likelihood_and_offset():
    frame = pd.DataFrame({"t": [1, 2, 3], "x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 4.0], "logexp": [0.0, 0.1, 0.2]})
    model = LGM("y", Poisson(), Fixed("1 + x"), time="t", offset="logexp")
    compiled = compile_lgm(model, _panel(frame))
    assert isinstance(compiled.likelihood, CompiledPoisson)
    np.testing.assert_allclose(compiled.offset, [0.0, 0.1, 0.2])


def test_compile_poisson_rejects_non_count_response():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 1.5]})
    model = LGM("y", Poisson(), Fixed("1"), time="t")
    with pytest.raises(DataContractError, match="non-negative integer"):
        compile_lgm(model, _panel(frame))


def test_compile_bernoulli_default_offset_is_zero():
    frame = pd.DataFrame({"t": [1, 2], "y": [0.0, 1.0]})
    model = LGM("y", Bernoulli(), Fixed("1"), time="t")
    compiled = compile_lgm(model, _panel(frame))
    np.testing.assert_allclose(compiled.offset, [0.0, 0.0])


def test_compile_rejects_missing_offset_column():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    model = LGM("y", Poisson(), Fixed("1"), time="t", offset="does_not_exist")
    with pytest.raises(DataContractError, match="offset column not found"):
        compile_lgm(model, _panel(frame))


def test_compile_rejects_empty_predictor_for_non_gaussian():
    frame = pd.DataFrame({"t": [1, 2], "y": [1.0, 2.0]})
    model = LGM("y", Poisson(), Predictor(()), time="t")
    with pytest.raises(CompilationError, match="model must contain at least one latent effect"):
        compile_lgm(model, _panel(frame))


def test_compile_family_returns_none_without_hyperparameters():
    frame = pd.DataFrame({"t": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    model = LGM("y", Gaussian(1.0), Fixed("1"), time="t")
    assert compile_family(model, _panel(frame)) is None


def test_compile_family_binds_declared_hyperparameters():
    frame = pd.DataFrame({"t": [1, 2, 3, 4], "region": ["a", "a", "b", "b"], "y": [1.0, 2.0, 3.0, 4.0]})
    model = LGM(
        "y", Gaussian(1.0),
        Fixed("1") + IID("region", index="region", precision=Hyperparameter("region_prec", initial=1.0)),
        panel=("region",), time="t",
    )
    family = compile_family(model, CanonicalPanel.from_frame(frame, DataConfig(time="t", response="y", panel=("region",))))
    assert isinstance(family, CompiledFamily)
    assert family.parameter_names == ("region_prec",)
    compiled = family.materialize({"region_prec": 2.0})
    assert compiled.y.shape[0] == 4
    region_block = next(block for block in compiled.blocks if block.name == "region")
    np.testing.assert_allclose(region_block.precision.toarray(), 2.0 * np.eye(2))

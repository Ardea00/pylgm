from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pylgm.compiler import compile_gaussian_family, compile_model
from pylgm.config.schema import DataConfig, ModelConfig, RunConfig
from pylgm.data import CanonicalPanel
from pylgm.exceptions import ModelValidationError, NumericalError
from pylgm.inference import fit_gaussian
from pylgm.ir import CompiledGaussianFamily, Hyperparameters, LatentBlock, ScalableBlock


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(time="month", response="y")


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "fixed": "1 + x",
            "fixed_prior_precision": 0.5,
            "sigma": 1.5,
            "effects": [
                {"name": "group", "type": "iid", "index": "group", "precision": 3.0},
                {"name": "trend", "type": "rw2", "index": "month", "precision": 2.0},
            ],
        }
    )


@pytest.fixture
def panel(data_config: DataConfig) -> CanonicalPanel:
    return CanonicalPanel.from_frame(
        pd.DataFrame(
            {
                "month": [1, 2, 3],
                "group": ["B", "A", "B"],
                "x": [0.0, 1.0, 2.0],
                "y": [1.0, None, 3.0],
            }
        ),
        data_config,
    )


def test_family_rescales_precision_without_rebuilding_design(
    data_config: DataConfig, model_config: ModelConfig, panel: CanonicalPanel
) -> None:
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma", "trend.precision")
    )

    low = family.materialize({"sigma": 0.5, "trend.precision": 2.0})
    high = family.materialize({"sigma": 1.0, "trend.precision": 8.0})

    np.testing.assert_allclose(low.design.toarray(), high.design.toarray())
    trend = family.block_slice("trend")
    np.testing.assert_allclose(
        high.precision[trend, trend].toarray(),
        4.0 * low.precision[trend, trend].toarray(),
    )


def test_family_materializes_sigma_and_multiple_effect_parameters(
    data_config: DataConfig, model_config: ModelConfig, panel: CanonicalPanel
) -> None:
    family = compile_gaussian_family(
        data_config,
        model_config,
        panel,
        optimized=("sigma", "group.precision", "trend.precision"),
    )

    model = family.materialize(
        {"sigma": 0.25, "group.precision": 5.0, "trend.precision": 7.0}
    )

    assert model.sigma == 0.25
    np.testing.assert_allclose(
        model.precision[family.block_slice("group"), family.block_slice("group")].toarray(),
        5.0 * np.eye(2),
    )
    assert model.precision[family.block_slice("trend"), family.block_slice("trend")].nnz


def test_family_keeps_fixed_and_unoptimized_blocks_at_configured_precisions(
    data_config: DataConfig, model_config: ModelConfig, panel: CanonicalPanel
) -> None:
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("trend.precision",)
    )

    materialized = family.materialize({"trend.precision": 8.0})

    fixed = family.block_slice("fixed")
    group = family.block_slice("group")
    np.testing.assert_allclose(
        materialized.precision[fixed, fixed].toarray(), 0.5 * np.eye(2)
    )
    np.testing.assert_allclose(
        materialized.precision[group, group].toarray(), 3.0 * np.eye(2)
    )
    assert materialized.sigma == model_config.sigma
    assert family.initial.precisions == {"group": 3.0, "trend": 2.0}


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"sigma": 1.0},
        {"sigma": 1.0, "trend.precision": 2.0, "extra": 3.0},
        {"sigma": float("nan"), "trend.precision": 2.0},
        {"sigma": float("inf"), "trend.precision": 2.0},
        {"sigma": 0.0, "trend.precision": 2.0},
        {"sigma": True, "trend.precision": 2.0},
    ],
)
def test_family_requires_exact_finite_positive_optimized_parameters(
    data_config: DataConfig,
    model_config: ModelConfig,
    panel: CanonicalPanel,
    values: Mapping[str, float],
) -> None:
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma", "trend.precision")
    )

    with pytest.raises(ValueError):
        family.materialize(values)


def test_family_and_materializations_are_value_isolated(
    data_config: DataConfig, model_config: ModelConfig, panel: CanonicalPanel
) -> None:
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma", "trend.precision")
    )
    first = family.materialize({"sigma": 0.5, "trend.precision": 2.0})
    second = family.materialize({"sigma": 1.0, "trend.precision": 8.0})

    with pytest.raises(ValueError):
        family.y[0] = 100.0
    with pytest.raises(ValueError):
        family.blocks[0].block.design.data[0] = 100.0
    with pytest.raises(ValueError):
        first.precision.data[0] = 100.0
    np.testing.assert_allclose(first.y, second.y)
    assert first.precision[family.block_slice("trend"), family.block_slice("trend")].max() < second.precision[
        family.block_slice("trend"), family.block_slice("trend")
    ].max()


def test_family_matches_compile_model_at_configured_values(
    data_config: DataConfig, model_config: ModelConfig, panel: CanonicalPanel
) -> None:
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma", "group.precision", "trend.precision")
    )
    materialized = family.materialize(
        {
            "sigma": model_config.sigma,
            "group.precision": model_config.effects[0].precision,
            "trend.precision": model_config.effects[1].precision,
        }
    )
    expected = compile_model(
        RunConfig(schema_version=1, data=data_config, model=model_config), panel
    )

    np.testing.assert_array_equal(materialized.y, expected.y)
    np.testing.assert_array_equal(materialized.observed, expected.observed)
    np.testing.assert_allclose(materialized.design.toarray(), expected.design.toarray())
    np.testing.assert_allclose(materialized.precision.toarray(), expected.precision.toarray())
    np.testing.assert_allclose(materialized.constraints, expected.constraints)
    assert materialized.labels == expected.labels
    assert materialized.sigma == expected.sigma


def test_family_hyperparameters_defensively_copy_inputs() -> None:
    precision = {"trend": 2.0}
    initial = Hyperparameters(sigma=1.0, precisions=precision)
    precision["trend"] = 100.0
    assert initial.precisions["trend"] == 2.0
    with pytest.raises(TypeError):
        initial.precisions["trend"] = 3.0  # type: ignore[index]


def _trend_block(name: str = "trend") -> LatentBlock:
    return LatentBlock(
        name,
        ("level",),
        csr_matrix([[1.0]]),
        csr_matrix([[1.0]]),
        np.empty((0, 1)),
    )


def _family_arguments() -> dict[str, object]:
    return {
        "y": np.array([2.0]),
        "observed": np.array([True]),
        "offset": np.zeros(1),
        "blocks": (ScalableBlock(_trend_block(), "trend.precision", 1.0),),
        "parameter_names": ("trend.precision",),
        "initial": Hyperparameters(sigma=1.0, precisions={"trend": 2.0}),
    }


def test_sigma_only_zero_block_family_materializes_and_exactly_fits() -> None:
    family = CompiledGaussianFamily(
        y=np.array([2.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(),
        parameter_names=("sigma",),
        initial=Hyperparameters(sigma=3.0, precisions={}),
    )

    model = family.materialize({"sigma": 0.5})
    result = fit_gaussian(model)

    assert model.design.shape == (1, 0)
    assert model.precision.shape == (0, 0)
    assert model.constraints.shape == (0, 0)
    assert model.labels == ()
    assert model.blocks == ()
    assert model.sigma == 0.5
    np.testing.assert_allclose(result.predictive_mean, [0.0])
    np.testing.assert_allclose(result.predictive_variance, [0.25])


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_names": ("trend.precision", "phantom.precision")},
        {"parameter_names": ("trend.precision", "trend.precision")},
        {"parameter_names": (1,)},
        {
            "blocks": (
                ScalableBlock(_trend_block(), "trend.precision", 1.0),
                ScalableBlock(_trend_block("other"), "trend.precision", 1.0),
            ),
            "parameter_names": ("trend.precision",),
        },
        {
            "blocks": (ScalableBlock(_trend_block(), "wrong", 1.0),),
            "parameter_names": ("wrong",),
        },
        {"parameter_names": ()},
        {"initial": Hyperparameters(sigma=1.0, precisions={})},
    ],
)
def test_family_constructor_requires_complete_parameter_bindings(
    changes: dict[str, object],
) -> None:
    arguments = _family_arguments()
    arguments.update(changes)

    with pytest.raises(ModelValidationError):
        CompiledGaussianFamily(**arguments)  # type: ignore[arg-type]


def test_family_does_not_rebuild_effects_when_materializing(
    data_config: DataConfig,
    model_config: ModelConfig,
    panel: CanonicalPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"fixed": 0, "iid": 0, "rw": 0}
    from pylgm import compiler

    original_fixed = compiler.build_fixed
    original_iid = compiler.build_iid
    original_random_walk = compiler.build_random_walk

    def count_fixed(*args: object, **kwargs: object) -> LatentBlock:
        calls["fixed"] += 1
        return original_fixed(*args, **kwargs)  # type: ignore[arg-type]

    def count_iid(*args: object, **kwargs: object) -> LatentBlock:
        calls["iid"] += 1
        return original_iid(*args, **kwargs)  # type: ignore[arg-type]

    def count_random_walk(*args: object, **kwargs: object) -> LatentBlock:
        calls["rw"] += 1
        return original_random_walk(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("pylgm.compiler.build_fixed", count_fixed)
    monkeypatch.setattr("pylgm.compiler.build_iid", count_iid)
    monkeypatch.setattr("pylgm.compiler.build_random_walk", count_random_walk)

    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma", "trend.precision")
    )
    compiled_calls = calls.copy()

    family.materialize({"sigma": 0.5, "trend.precision": 4.0})

    assert compiled_calls == {"fixed": 1, "iid": 1, "rw": 1}
    assert calls == compiled_calls


def test_family_scales_requested_effects_and_preserves_rw_structure(
    data_config: DataConfig, model_config: ModelConfig, panel: CanonicalPanel
) -> None:
    family = compile_gaussian_family(
        data_config,
        model_config,
        panel,
        optimized=("group.precision", "trend.precision"),
    )

    low = family.materialize({"group.precision": 2.0, "trend.precision": 3.0})
    high = family.materialize({"group.precision": 8.0, "trend.precision": 12.0})

    for name in ("group", "trend"):
        block = family.block_slice(name)
        np.testing.assert_allclose(
            high.precision[block, block].toarray(),
            4.0 * low.precision[block, block].toarray(),
        )
    np.testing.assert_allclose(low.design.toarray(), high.design.toarray())
    np.testing.assert_allclose(low.constraints, high.constraints)
    assert low.blocks[-1].constraints.shape == high.blocks[-1].constraints.shape


def test_family_reports_parameter_driven_precision_overflow_as_numerical() -> None:
    block = LatentBlock(
        "latent",
        ("x",),
        csr_matrix([[1.0]]),
        csr_matrix([[2.0]]),
        np.empty((0, 1)),
    )
    family = CompiledGaussianFamily(
        y=np.array([1.0]),
        observed=np.array([True]),
        offset=np.zeros(1),
        blocks=(ScalableBlock(block, "latent.precision", 1.0),),
        parameter_names=("latent.precision",),
        initial=Hyperparameters(sigma=1.0, precisions={"latent": 1.0}),
    )

    with pytest.raises(NumericalError, match="precision scaling"):
        family.materialize({"latent.precision": 1e308})


def test_family_isolates_source_values_accessors_and_materializations() -> None:
    y = np.array([1.0])
    observed = np.array([True])
    offset = np.array([0.25])
    design = csr_matrix([[1.0]])
    precision = csr_matrix([[2.0]])
    constraints = np.array([[1.0]])
    block = LatentBlock(
        "trend",
        ("level",),
        design,
        precision,
        constraints,
    )
    family = CompiledGaussianFamily(
        y=y,
        observed=observed,
        offset=offset,
        blocks=(ScalableBlock(block, "trend.precision", 1.0),),
        parameter_names=("trend.precision",),
        initial=Hyperparameters(sigma=1.0, precisions={"trend": 2.0}),
    )
    y[0] = 100.0
    observed[0] = False
    offset[0] = 100.0
    design.data[0] = 100.0
    precision.data[0] = 100.0
    constraints[0, 0] = 100.0
    first = family.materialize({"trend.precision": 2.0})
    second = family.materialize({"trend.precision": 4.0})

    for value in (family.y, family.observed, family.offset):
        with pytest.raises(ValueError):
            value.flat[0] = 100.0
    for value in (first.design, first.precision):
        with pytest.raises(ValueError):
            value.data[0] = 100.0
    with pytest.raises(ValueError):
        first.constraints[0, 0] = 100.0
    with pytest.raises(ValueError):
        family.blocks[0].block.design.data[0] = 100.0
    with pytest.raises(ValueError):
        family.blocks[0].block.precision.data[0] = 100.0
    with pytest.raises(ValueError):
        family.blocks[0].block.constraints[0, 0] = 100.0
    np.testing.assert_allclose(first.y, [1.0])
    np.testing.assert_allclose(first.observed, [True])
    np.testing.assert_allclose(first.offset, [0.25])
    np.testing.assert_allclose(first.design.toarray(), second.design.toarray())
    np.testing.assert_allclose(first.constraints, second.constraints)
    assert first.precision[0, 0] == 4.0
    assert second.precision[0, 0] == 8.0


def test_family_isolates_the_source_frame_after_panel_canonicalization(
    data_config: DataConfig, model_config: ModelConfig
) -> None:
    source = pd.DataFrame(
        {
            "month": [1, 2, 3],
            "group": ["B", "A", "B"],
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, None, 3.0],
        }
    )
    panel = CanonicalPanel.from_frame(source, data_config)
    family = compile_gaussian_family(
        data_config, model_config, panel, optimized=("sigma",)
    )
    source.loc[:, "y"] = 100.0
    source.loc[:, "x"] = 100.0

    materialized = family.materialize({"sigma": 1.0})

    np.testing.assert_allclose(materialized.y, [1.0, 0.0, 3.0])
    np.testing.assert_allclose(
        materialized.design[:, :2].toarray(), [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
    )

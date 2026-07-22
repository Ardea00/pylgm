from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from pylgm.compiler import compile_gaussian_family, compile_model
from pylgm.config.schema import DataConfig, ModelConfig, RunConfig
from pylgm.data import CanonicalPanel
from pylgm.ir import Hyperparameters


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

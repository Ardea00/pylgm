"""``RW1``/``RW2`` ``scale=True``: Sørbye-Rue scaling, as ``Besag`` and ``RW*Structure``."""

import numpy as np
import pandas as pd
import pytest

from pylgm import RW1, RW2, Gaussian, Hyperparameter, LGM
from pylgm.compiler import _build_effect_block
from pylgm.config.schema import EffectConfig, build_effect

FRAME = pd.DataFrame({"t": range(12), "y": np.sin(np.arange(12.0))})


def _generalized_variances(precision, null_dim):
    values, vectors = np.linalg.eigh(precision)
    inverse = np.zeros_like(values)
    inverse[null_dim:] = 1.0 / values[null_dim:]
    return np.einsum("ij,j,ij->i", vectors, inverse, vectors)


@pytest.mark.parametrize("effect, null_dim", [(RW1, 1), (RW2, 2)])
def test_scaled_random_walk_has_unit_geometric_mean_variance_per_precision(effect, null_dim):
    block, _ = _build_effect_block(effect("trend", "t", precision=4.0, scale=True), FRAME)
    variances = _generalized_variances(block.precision.toarray(), null_dim)
    assert np.exp(np.mean(np.log(variances))) == pytest.approx(1.0 / 4.0, rel=1e-10)


@pytest.mark.parametrize("effect", [RW1, RW2])
def test_unscaled_is_still_the_default(effect):
    default, _ = _build_effect_block(effect("trend", "t", precision=4.0), FRAME)
    explicit, _ = _build_effect_block(effect("trend", "t", precision=4.0, scale=False), FRAME)
    np.testing.assert_array_equal(default.precision.toarray(), explicit.precision.toarray())


def test_an_estimated_precision_keeps_the_scaling():
    """The hyperparameter path rebuilds the block at unit precision and rescales it."""
    precision = Hyperparameter("trend.precision", initial=4.0)
    fixed = LGM(response="y", likelihood=Gaussian(0.5),
                predictor=RW1("trend", "t", precision=4.0, scale=True)).fit(FRAME)
    family = LGM(response="y", likelihood=Gaussian(0.5),
                 predictor=RW1("trend", "t", precision=precision, scale=True))
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data import CanonicalPanel
    frame = FRAME.assign(row=range(12))
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_family(family, panel).materialize({"trend.precision": 4.0})
    reference, _ = _build_effect_block(RW1("trend", "t", precision=4.0, scale=True), FRAME)
    np.testing.assert_allclose(compiled.precision.toarray(), reference.precision.toarray())
    assert np.isfinite(fixed.log_marginal_likelihood)


def test_scale_is_a_boolean():
    with pytest.raises(ValueError, match="scale"):
        RW1("trend", "t", scale="yes")


@pytest.mark.parametrize("kind, effect", [("rw1", RW1), ("rw2", RW2)])
def test_config_accepts_scale(kind, effect):
    config = EffectConfig(name="trend", type=kind, index="t", precision=2.0, scale=True)
    assert build_effect(config, None) == effect("trend", "t", 2.0, scale=True)

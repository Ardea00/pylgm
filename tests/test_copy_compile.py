import numpy as np
import pandas as pd
import pytest

from pylgm import Copy, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.compiler import compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.exceptions import CompilationError


def _frame():
    return pd.DataFrame({
        "i": ["a", "b", "c", "a"],
        "j": ["b", "c", "a", "c"],
        "y": [1.0, 2.0, 3.0, 4.0],
        "row": range(4),
    })


def _panel(frame):
    return CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )


def _compiled(predictor, frame=None):
    frame = _frame() if frame is None else frame
    model = LGM(response="y", likelihood=Poisson(), predictor=predictor)
    return compile_lgm(model, _panel(frame))


def test_a_copy_adds_no_block_of_its_own():
    without = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    with_copy = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j")
    )
    assert len(with_copy.blocks) == len(without.blocks)
    assert with_copy.labels == without.labels


def test_a_copy_adds_its_scaled_incidence_to_the_target_design():
    frame = _frame()
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    copied = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j", scale=2.0)
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_copy = [b for b in copied.blocks if b.name == "u"][0]

    labels = list(u_base.labels)
    position = {label: k for k, label in enumerate(labels)}
    expected = u_base.design.toarray().copy()
    for row, level in enumerate(frame["j"].astype(str)):
        expected[row, position[level]] += 2.0
    assert np.allclose(u_copy.design.toarray(), expected)


def test_precision_labels_and_constraints_are_untouched_by_a_copy():
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    copied = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j")
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_copy = [b for b in copied.blocks if b.name == "u"][0]
    assert u_copy.labels == u_base.labels
    assert np.allclose(u_copy.precision.toarray(), u_base.precision.toarray())
    assert np.allclose(u_copy.constraints, u_base.constraints)


def test_a_zero_scale_copy_leaves_the_design_unchanged():
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    copied = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j", scale=0.0)
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_copy = [b for b in copied.blocks if b.name == "u"][0]
    assert np.allclose(u_copy.design.toarray(), u_base.design.toarray())


def test_copy_naming_a_missing_block_is_rejected():
    with pytest.raises(CompilationError, match="nonexistent"):
        _compiled(Fixed("1") + IID("u", index="i") + Copy("nonexistent", index="j"))


def test_two_copies_of_the_same_field_at_different_indices_both_fold_in():
    """Not a copy of a copy -- a copy has no name of its own, so that is not
    expressible. This is one field entering three times, which R-INLA allows and
    the spec does not forbid: both copies fold into the same columns."""
    frame = _frame()
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0))
    both = _compiled(
        Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=2.0) + Copy("u", index="i", scale=0.5)
    )
    u_base = [b for b in base.blocks if b.name == "u"][0]
    u_both = [b for b in both.blocks if b.name == "u"][0]
    assert len(both.blocks) == len(base.blocks)

    position = {label: k for k, label in enumerate(u_base.labels)}
    expected = u_base.design.toarray().copy()
    for row, level in enumerate(frame["j"].astype(str)):
        expected[row, position[level]] += 2.0
    for row, level in enumerate(frame["i"].astype(str)):
        expected[row, position[level]] += 0.5
    assert np.allclose(u_both.design.toarray(), expected)


def test_copy_whose_index_has_a_level_outside_the_target_is_rejected():
    # A copy reuses an existing latent field; it cannot create a new level in it.
    frame = _frame()
    frame.loc[0, "j"] = "zz"
    with pytest.raises(CompilationError, match="zz"):
        _compiled(Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j"),
                  frame=frame)


def test_a_copy_can_reference_a_weighted_block():
    """The spec allows a copy to reference a wrapped block: the copy contributes
    to that block's columns as they were built. Weighted keeps the inner name,
    so `Copy("u", ...)` finds it, and the copy's incidence is NOT re-weighted --
    the weighting belongs to the occurrence that declared it."""
    from pylgm import Weighted

    frame = _frame().assign(z=[2.0, 3.0, 4.0, 5.0])
    compiled = _compiled(
        Fixed("1") + Weighted(IID("u", index="i", precision=1.0), by="z")
        + Copy("u", index="j", scale=1.0),
        frame=frame,
    )
    u = [b for b in compiled.blocks if b.name == "u"][0]
    labels = list(u.labels)
    position = {label: k for k, label in enumerate(labels)}

    expected = np.zeros((len(frame), len(labels)))
    for row, (level_i, level_j, weight) in enumerate(
        zip(frame["i"].astype(str), frame["j"].astype(str), frame["z"])
    ):
        expected[row, position[level_i]] += weight    # weighted occurrence
        expected[row, position[level_j]] += 1.0       # unweighted copy
    assert np.allclose(u.design.toarray(), expected)


def test_copy_index_column_missing_is_rejected():
    frame = _frame().drop(columns=["j"])
    with pytest.raises(CompilationError, match="j"):
        _compiled(Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j"),
                  frame=frame)


def test_spark_required_columns_include_a_copy_index():
    pytest.importorskip("pyspark")
    from pylgm.data.spark import _required_columns

    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0) + Copy("u", index="j"),
    )
    required = _required_columns(model)
    assert "i" in required and "j" in required


def test_an_estimated_copy_scale_is_discovered_as_a_hyperparameter():
    from pylgm.compiler import _effect_hyperparameters
    from pylgm.parameters import Hyperparameter

    beta = Hyperparameter("beta", initial=1.0)
    assert [hp.name for hp in _effect_hyperparameters(Copy("u", index="j", scale=beta))] == [
        "beta"
    ]


def test_a_model_with_an_estimated_copy_scale_fits_and_reports_beta():
    """``beta`` must be more than merely finite: it must have actually moved,
    driven by data that carries real signal for it, and must not be pinned at
    its lower bound (a pinned estimate means the bound set the value, not the
    data -- indistinguishable from ``np.isfinite`` alone)."""
    from pylgm.parameters import Hyperparameter

    rng = np.random.default_rng(4)
    K = 12
    n = 300
    levels = [f"L{k}" for k in range(K)]
    u_true = rng.normal(size=K)
    beta_true = 2.0
    sigma_true = 0.05
    i_idx = rng.integers(0, K, size=n)
    j_idx = rng.integers(0, K, size=n)
    eta = 1.0 + u_true[i_idx] + beta_true * u_true[j_idx]
    frame = pd.DataFrame({
        "i": [levels[k] for k in i_idx],
        "j": [levels[k] for k in j_idx],
        "y": eta + rng.normal(scale=sigma_true, size=n),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Gaussian(sigma=sigma_true),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    )
    result = model.fit(frame, engine="exact_gaussian")
    assert np.isfinite(result.log_marginal_likelihood)
    fitted_beta = result.hyperparameters["beta"]
    assert np.isfinite(fitted_beta)
    assert abs(fitted_beta - 1.0) > 0.1
    assert abs(fitted_beta - beta_true) < 0.1
    at_bound = result.diagnostics["hyperparameters_at_bound"].split(", ")
    assert "beta" not in at_bound


def test_the_estimated_scale_is_applied_on_every_rebuild_not_only_the_template():
    """A ParametricDesignBlock rebuilds its design per draw. If the copy were
    folded only into the template, every draw after the first would silently
    lose it -- a model that fits and returns plausible numbers."""
    from pylgm.compiler import compile_family
    from pylgm.ir.family import ParametricDesignBlock
    from pylgm.parameters import Hyperparameter

    frame = _frame()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="i", precision=1.0)
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    )
    family = compile_family(model, _panel(frame))
    assert family is not None
    assert "beta" in family.parameter_names

    item = [b for b in family.blocks if b.block.name == "u"][0]
    assert isinstance(item, ParametricDesignBlock)
    at_one = item.build({"beta": 1.0}).toarray()
    at_three = item.build({"beta": 3.0}).toarray()
    # Changing beta must change the design; a template-only fold would not.
    assert not np.allclose(at_one, at_three)
    # And the difference is exactly twice the copy's incidence.
    assert np.allclose(at_three - at_one, 2.0 * (at_one - _base_design(frame)))


def _base_design(frame):
    """The u design with no copy folded in, for the difference check above."""
    base = _compiled(Fixed("1") + IID("u", index="i", precision=1.0), frame=frame)
    return [b for b in base.blocks if b.name == "u"][0].design.toarray()


def test_an_estimated_copy_scale_does_not_drop_the_targets_estimated_precision():
    """A ParametricDesignBlock folded from a Copy must still scale the
    *precision* by the target block's own Hyperparameter (here ``tau``), not
    just rebuild the design. Before the fix, converting a ScalableBlock target
    into a ParametricDesignBlock dropped ``item.parameter``/``item.scale`` on
    the floor, so the copied block's precision was byte-identical regardless
    of tau."""
    from pylgm.compiler import compile_family
    from pylgm.parameters import Hyperparameter

    frame = _frame()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=IID("u", index="i", precision=Hyperparameter("tau", initial=1.0))
        + Copy("u", index="j", scale=Hyperparameter("beta", initial=1.0)),
    )
    family = compile_family(model, _panel(frame))
    assert family is not None

    low = family.materialize({"tau": 1.0, "beta": 1.0})
    high = family.materialize({"tau": 50.0, "beta": 1.0})
    p_low = [b for b in low.blocks if b.name == "u"][0].precision.toarray()
    p_high = [b for b in high.blocks if b.name == "u"][0].precision.toarray()
    assert not np.allclose(p_low, p_high)
    nonzero = p_low != 0
    assert np.allclose(p_high[nonzero] / p_low[nonzero], 50.0)

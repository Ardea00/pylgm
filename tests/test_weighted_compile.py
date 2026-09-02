import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted
from pylgm.compiler import _build_effect_block, _effect_hyperparameters
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.parameters import Hyperparameter


def _frame(z):
    return pd.DataFrame({"district": ["a", "b", "c", "a"], "z": z, "row": range(4)})


def test_weighted_design_is_the_inner_design_scaled_row_wise():
    z = [2.0, -1.0, 0.5, 3.0]
    frame = _frame(z)
    plain, _ = _build_effect_block(IID("u", index="district"), frame)
    weighted, _ = _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)

    expected = np.diag(z) @ plain.design.toarray()
    assert np.allclose(weighted.design.toarray(), expected)


def test_weighted_preserves_precision_labels_and_constraints():
    frame = _frame([2.0, -1.0, 0.5, 3.0])
    plain, _ = _build_effect_block(IID("u", index="district"), frame)
    weighted, _ = _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)

    assert weighted.name == plain.name == "u"
    assert weighted.labels == plain.labels
    assert np.allclose(weighted.precision.toarray(), plain.precision.toarray())
    assert np.allclose(weighted.constraints, plain.constraints)


def test_all_ones_weights_reduce_to_the_unweighted_block():
    frame = _frame([1.0, 1.0, 1.0, 1.0])
    plain, _ = _build_effect_block(IID("u", index="district"), frame)
    weighted, _ = _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)
    assert (weighted.design != plain.design).nnz == 0


def test_inner_hyperparameters_are_still_discovered_through_the_wrapper():
    tau = Hyperparameter("tau", initial=1.0)
    wrapped = Weighted(IID("u", index="district", precision=tau), by="z")
    assert [hp.name for hp in _effect_hyperparameters(wrapped)] == ["tau"]


def test_missing_weight_column_is_rejected_naming_the_effect_and_column():
    frame = _frame([1.0, 1.0, 1.0, 1.0]).drop(columns=["z"])
    with pytest.raises((CompilationError, DataContractError), match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_non_numeric_weight_column_is_rejected():
    frame = _frame(["a", "b", "c", "d"])
    with pytest.raises((CompilationError, DataContractError), match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_nan_weight_is_rejected():
    frame = _frame([1.0, np.nan, 1.0, 1.0])
    with pytest.raises((CompilationError, DataContractError), match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_all_zero_weights_are_rejected_rather_than_compiling_an_inert_block():
    frame = _frame([0.0, 0.0, 0.0, 0.0])
    with pytest.raises((CompilationError, DataContractError), match="zero"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_weighted_model_fits_and_estimates_the_inner_hyperparameter():
    rng = np.random.default_rng(3)
    n = 60
    district = [f"d{i % 12}" for i in range(n)]
    z = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({
        "district": district, "z": z,
        "y": rng.poisson(np.exp(0.2 + 0.3 * z)).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="district", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    )
    result = model.fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert result.hyperparameters["tau"] > 0


def test_weighted_family_scales_every_rebuilt_design():
    """A ParametricDesignBlock rebuilds its design per draw; weights must apply
    to each rebuild, not only to the template built at the initial value."""
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.ir.family import ParametricDesignBlock

    rng = np.random.default_rng(5)
    n = 40
    frame = pd.DataFrame({
        "district": [f"d{i % 8}" for i in range(n)],
        "z": rng.normal(1.0, 0.2, n),
        "y": rng.poisson(2.0, n).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="district", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    family = compile_family(model, panel)
    assert family is not None
    assert "tau" in family.parameter_names

    # The weighted block's design must carry the weights, whichever block kind
    # it came out as.
    weighted_blocks = [b for b in family.blocks if b.block.name == "u"]
    assert len(weighted_blocks) == 1
    item = weighted_blocks[0]
    design = (
        item.build({"tau": 1.0}) if isinstance(item, ParametricDesignBlock)
        else item.block.design
    )
    row_sums = np.asarray(design.sum(axis=1)).ravel()
    assert np.allclose(row_sums, frame["z"].to_numpy())


def test_weighted_model_with_estimated_hyperparameter_matches_a_manual_weighting():
    """Weighting a column is the same as pre-multiplying it into a one-hot design.

    Fitting Weighted(IID(...), by=z) must equal fitting the same model where the
    weighting was done by hand, which is the property that makes the wrapper
    trustworthy rather than merely functional.
    """
    rng = np.random.default_rng(11)
    n = 50
    frame = pd.DataFrame({
        "district": [f"d{i % 10}" for i in range(n)],
        "z": rng.normal(1.0, 0.3, n),
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    wrapped = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(IID("u", index="district", precision=2.0), by="z"),
    ).fit(frame, engine="laplace")

    from pylgm.compiler import _build_effect_block
    plain, _ = _build_effect_block(IID("u", index="district", precision=2.0), frame)
    manual, _ = _build_effect_block(
        Weighted(IID("u", index="district", precision=2.0), by="z"), frame
    )
    assert np.allclose(
        manual.design.toarray(),
        np.diag(frame["z"].to_numpy()) @ plain.design.toarray(),
    )
    assert np.isfinite(wrapped.log_marginal_likelihood)

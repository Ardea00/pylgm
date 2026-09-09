import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted
from pylgm.compiler import _build_effect_block, _effect_hyperparameters
from pylgm.exceptions import CompilationError
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
    with pytest.raises(CompilationError, match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_missing_weight_column_raises_the_same_type_with_or_without_a_hyperparameter():
    """A bad `by` column must raise CompilationError whether the inner effect's
    precision is a plain float (routed through _build_effect_block) or a
    Hyperparameter (routed through compile_family's _append_family_blocks).
    Before, the Hyperparameter path raised a raw DataContractError instead,
    because _append_family_blocks called _weight_vector outside the wrapping
    _compiled_block gives every other builder failure."""
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame([1.0, 1.0, 1.0, 1.0]).drop(columns=["z"])
    frame = frame.assign(y=[1.0, 2.0, 3.0, 4.0])
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="district", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    )
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    with pytest.raises(CompilationError, match="z"):
        compile_family(model, panel)


def test_non_numeric_weight_column_is_rejected():
    frame = _frame(["a", "b", "c", "d"])
    with pytest.raises(CompilationError, match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_datetime_weight_column_is_rejected_not_silently_cast_to_nanoseconds():
    """A `by` column of datetime-like objects must raise DataContractError, not
    silently convert into nanosecond-since-epoch floats (~1.58e18) the way
    pd.to_numeric(..., errors="coerce") used to."""
    import datetime

    frame = _frame([1.0, 1.0, 1.0, 1.0])
    frame["z"] = [
        datetime.date(2020, 1, 1), datetime.date(2020, 1, 2),
        datetime.date(2020, 1, 3), datetime.date(2020, 1, 4),
    ]
    with pytest.raises(CompilationError, match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_datetime64_weight_column_is_rejected_at_fit_time():
    """A `by` column with datetime64[ns] dtype must raise CompilationError at fit
    time, not silently convert to nanosecond-since-epoch floats (~1.58e18)."""
    frame = _frame([1.0, 1.0, 1.0, 1.0])
    frame["z"] = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    )
    with pytest.raises(CompilationError, match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_nan_weight_is_rejected():
    frame = _frame([1.0, np.nan, 1.0, 1.0])
    with pytest.raises(CompilationError, match="z"):
        _build_effect_block(Weighted(IID("u", index="district"), by="z"), frame)


def test_all_zero_weights_are_rejected_rather_than_compiling_an_inert_block():
    frame = _frame([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(CompilationError, match="zero"):
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


def test_weighted_family_block_is_a_scaled_scalable_block():
    """compile_family's Weighted branch always yields a ScalableBlock here.

    An IID's Hyperparameter precision only ever produces a ScalableBlock (or
    ParametricBlock) template; ParametricDesignBlock is MIDASParametric's
    doing, and MIDASParametric has no `.index`, so Weighted refuses it before
    compile_family ever sees it (see _weighted_family_block). This checks the
    block kind stays a ScalableBlock and that its design still carries the
    weights, so the family path doesn't silently drop weighting the way the
    plain compile path (_build_effect_block) could.
    """
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.ir.family import ScalableBlock

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

    weighted_blocks = [b for b in family.blocks if b.block.name == "u"]
    assert len(weighted_blocks) == 1
    item = weighted_blocks[0]
    assert isinstance(item, ScalableBlock)
    row_sums = np.asarray(item.block.design.sum(axis=1)).ravel()
    assert np.allclose(row_sums, frame["z"].to_numpy())


def test_weighted_model_with_fixed_precision_fits_successfully():
    """A Weighted effect with a plain float (not Hyperparameter) precision
    still fits end to end -- the wrapper doesn't implicitly require an
    estimated precision. (The wrapped-vs-manual design identity itself is
    already covered by test_weighted_design_is_the_inner_design_scaled_row_wise.)
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
    assert np.isfinite(wrapped.log_marginal_likelihood)

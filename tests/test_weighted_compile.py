import numpy as np
import pandas as pd
import pytest

from pylgm import IID, Weighted
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

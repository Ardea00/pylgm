import numpy as np
import pandas as pd
import pytest

from pylgm import Besag, Fixed, IID, LGM, Poisson, Replicated, Weighted
from pylgm.compiler import _build_effect_block, _effect_hyperparameters
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.parameters import Hyperparameter


def _frame():
    return pd.DataFrame({
        "t": ["a", "b", "c", "a", "b", "c"],
        "firm": ["f1", "f1", "f1", "f2", "f2", "f2"],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "row": range(6),
    })


def test_precision_is_the_kronecker_product_of_identity_and_the_inner_structure():
    frame = _frame()
    inner, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=2.0), over="firm"), frame
    )
    expected = np.kron(np.eye(2), inner.precision.toarray())
    assert np.allclose(outer.precision.toarray(), expected)


def test_labels_are_replicate_major_pairs():
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.0), over="firm"), _frame()
    )
    assert outer.labels == ("f1@a", "f1@b", "f1@c", "f2@a", "f2@b", "f2@c")


def test_design_maps_each_row_to_its_own_replicate_cell():
    frame = _frame()
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.0), over="firm"), frame
    )
    dense = outer.design.toarray()
    assert dense.shape == (6, 6)
    # Row k belongs to firm frame["firm"][k] at level frame["t"][k]; with three
    # levels and replicate-major layout that is cell (firm_index * 3 + level).
    position = {label: k for k, label in enumerate(outer.labels)}
    for row, (firm, level) in enumerate(zip(frame["firm"], frame["t"])):
        assert dense[row, position[f"{firm}@{level}"]] == 1.0
        assert dense[row].sum() == 1.0


def test_a_constrained_inner_effect_gets_one_constraint_per_replicate():
    """The named risk of this slice. One shared constraint over R replicates
    leaves R-1 directions unidentified: the fit still converges and returns
    plausible numbers, and nothing else in the suite would catch it."""
    frame = _frame()
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    inner, _ = _build_effect_block(Besag("u", index="t", graph=graph, precision=1.0), frame)
    outer, _ = _build_effect_block(
        Replicated(Besag("u", index="t", graph=graph, precision=1.0), over="firm"), frame
    )

    assert inner.constraints.shape == (1, 3)
    assert outer.constraints.shape == (2, 6)
    assert np.allclose(outer.constraints, np.kron(np.eye(2), inner.constraints))
    # Each replicate's constraint touches only its own columns.
    assert np.allclose(outer.constraints[0], [1, 1, 1, 0, 0, 0])
    assert np.allclose(outer.constraints[1], [0, 0, 0, 1, 1, 1])
    assert np.linalg.matrix_rank(outer.constraints) == 2


def test_an_unconstrained_inner_effect_stays_unconstrained():
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.0), over="firm"), _frame()
    )
    assert outer.constraints.shape == (0, 6)


def test_a_single_replicate_reduces_to_the_inner_block():
    frame = _frame().assign(firm="only")
    inner, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    outer, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=2.0), over="firm"), frame
    )
    assert np.allclose(outer.precision.toarray(), inner.precision.toarray())
    assert np.allclose(outer.design.toarray(), inner.design.toarray())
    assert outer.labels == tuple(f"only@{label}" for label in inner.labels)


def test_inner_hyperparameters_are_discovered_through_the_wrapper():
    tau = Hyperparameter("tau", initial=1.0)
    wrapped = Replicated(IID("u", index="t", precision=tau), over="firm")
    assert [hp.name for hp in _effect_hyperparameters(wrapped)] == ["tau"]


def test_replicated_commutes_with_weighted():
    """Weighted touches only the design; Replicated only precision and indexing.
    The spec asserts they commute, so this pins it rather than leaving it as a
    convention to remember."""
    frame = _frame()
    a, _ = _build_effect_block(
        Replicated(Weighted(IID("u", index="t", precision=1.0), by="z"), over="firm"), frame
    )
    b, _ = _build_effect_block(
        Weighted(Replicated(IID("u", index="t", precision=1.0), over="firm"), by="z"), frame
    )
    assert a.labels == b.labels
    assert np.allclose(a.design.toarray(), b.design.toarray())
    assert np.allclose(a.precision.toarray(), b.precision.toarray())
    assert np.allclose(a.constraints, b.constraints)


def test_missing_replicate_column_is_rejected():
    frame = _frame().drop(columns=["firm"])
    with pytest.raises((CompilationError, DataContractError), match="firm"):
        _build_effect_block(Replicated(IID("u", index="t"), over="firm"), frame)


def test_a_null_in_the_replicate_column_is_rejected():
    frame = _frame()
    frame.loc[0, "firm"] = None
    with pytest.raises((CompilationError, DataContractError), match="firm"):
        _build_effect_block(Replicated(IID("u", index="t"), over="firm"), frame)


def test_a_replicated_model_fits():
    rng = np.random.default_rng(5)
    n = 60
    frame = pd.DataFrame({
        "t": [f"t{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(IID("u", index="t", precision=1.0), over="firm"),
    ).fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert len(result.labels) == 1 + 6 * 4

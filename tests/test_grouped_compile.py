import numpy as np
import pandas as pd
import pytest

from pylgm import (
    BesagStructure, Fixed, Grouped, IID, IIDStructure, LGM, Poisson,
    Replicated, RW1, RW1Structure, Weighted,
)
from pylgm.compiler import _build_effect_block
from pylgm.exceptions import CompilationError
from pylgm.parameters import Hyperparameter

GRAPH = {"r1": ["r2"], "r2": ["r1", "r3"], "r3": ["r2"]}


def _frame():
    rows = []
    for region in ("r1", "r2", "r3"):
        for t in ("a", "b"):
            rows.append({"region": region, "t": t, "z": 2.0, "y": 1.0})
    frame = pd.DataFrame(rows)
    frame["row"] = range(len(frame))
    return frame


def test_precision_is_the_kronecker_product_of_structure_and_inner():
    frame = _frame()
    inner, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t", precision=2.0), over="region",
                structure=BesagStructure(GRAPH)),
        frame,
    )
    structure = BesagStructure(GRAPH).precision(("r1", "r2", "r3")).toarray()
    # inner is built over the level set alone, so its precision is 2x2 here
    assert outer.precision.shape == (6, 6)
    assert np.allclose(
        outer.precision.toarray(), np.kron(structure, inner.precision.toarray())
    )


def test_labels_are_group_major_pairs_with_the_replicated_separator():
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t"), over="region", structure=IIDStructure()), _frame()
    )
    assert outer.labels == ("r1@a", "r1@b", "r2@a", "r2@b", "r3@a", "r3@b")


def test_an_iid_structure_reduces_grouped_to_replicated():
    frame = _frame()
    grouped, _ = _build_effect_block(
        Grouped(IID("u", index="t", precision=1.5), over="region",
                structure=IIDStructure()),
        frame,
    )
    replicated, _ = _build_effect_block(
        Replicated(IID("u", index="t", precision=1.5), over="region"), frame
    )
    assert grouped.labels == replicated.labels
    assert np.allclose(grouped.design.toarray(), replicated.design.toarray())
    assert np.allclose(grouped.precision.toarray(), replicated.precision.toarray())


def test_a_single_group_level_reduces_to_the_bare_effect():
    frame = pd.DataFrame({"region": ["r1"] * 3, "t": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
    bare, _ = _build_effect_block(IID("u", index="t", precision=2.0), frame)
    grouped, _ = _build_effect_block(
        Grouped(IID("u", index="t", precision=2.0), over="region",
                structure=IIDStructure()),
        frame,
    )
    assert np.allclose(grouped.precision.toarray(), bare.precision.toarray())
    assert np.allclose(grouped.design.toarray(), bare.design.toarray())


def test_a_correlated_structure_is_not_block_diagonal():
    """The whole point of Grouped: groups are coupled, unlike Replicated."""
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
        _frame(),
    )
    dense = outer.precision.toarray()
    # r1's cells (rows 0-1) must couple to r2's (columns 2-3)
    assert not np.allclose(dense[0:2, 2:4], 0.0)


def test_constraints_span_the_null_space_of_the_composed_precision():
    outer, _ = _build_effect_block(
        Grouped(RW1("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
        _frame(),
    )
    q = outer.precision.toarray()
    assert outer.constraints.shape[0] == q.shape[0] - np.linalg.matrix_rank(q)
    assert np.allclose(q @ outer.constraints.T, 0.0)


def test_a_group_level_outside_the_graph_is_rejected():
    frame = _frame()
    frame.loc[0, "region"] = "elsewhere"
    with pytest.raises((CompilationError, ValueError), match="elsewhere"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
            frame,
        )


def test_a_missing_group_column_is_named():
    with pytest.raises((CompilationError, ValueError), match="region"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=IIDStructure()),
            pd.DataFrame({"t": ["a", "b"], "y": [1.0, 2.0]}),
        )


def test_a_structure_whose_size_disagrees_with_the_levels_is_rejected():
    with pytest.raises((CompilationError, ValueError), match="level"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=RW1Structure()),
            pd.DataFrame({"region": ["only"], "t": ["a"], "y": [1.0]}),
        )


def test_grouped_and_weighted_commute():
    frame = _frame()
    inside, _ = _build_effect_block(
        Grouped(Weighted(IID("u", index="t"), by="z"), over="region",
                structure=BesagStructure(GRAPH)),
        frame,
    )
    outside, _ = _build_effect_block(
        Weighted(Grouped(IID("u", index="t"), over="region",
                         structure=BesagStructure(GRAPH)), by="z"),
        frame,
    )
    assert inside.labels == outside.labels
    assert np.allclose(inside.design.toarray(), outside.design.toarray())
    assert np.allclose(inside.precision.toarray(), outside.precision.toarray())
    assert np.allclose(inside.constraints, outside.constraints)


def test_an_integer_index_keeps_its_numeric_level_order():
    """The dtype guard, which slice 3 shipped commented and untested."""
    rows = [
        {"region": r, "year": y, "y": 0.0}
        for r in ("r1", "r2", "r3") for y in range(1, 13)
    ]
    frame = pd.DataFrame(rows)
    outer, _ = _build_effect_block(
        Grouped(RW1("u", index="year"), over="region", structure=BesagStructure(GRAPH)),
        frame,
    )
    expected = tuple(f"{r}@{y}" for r in ("r1", "r2", "r3") for y in range(1, 13))
    assert outer.labels == expected


def test_an_integer_index_keeps_its_numeric_level_order_in_the_family_path():
    """The same guard at the second call site, _append_family_blocks.

    ``_build_effect_block``'s Grouped branch and ``_append_family_blocks``'s
    Grouped branch each pass ``frame[index].dtype`` to their own
    ``_levels_frame`` call; a Hyperparameter on the inner effect is what routes
    a model through the family path instead of the plain compile path, so this
    is the only test in the file that exercises that second call site.
    """
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    rows = [
        {"region": r, "year": y, "y": 0.0}
        for r in ("r1", "r2", "r3") for y in range(1, 13)
    ]
    frame = pd.DataFrame(rows)
    frame["row"] = range(len(frame))
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            RW1("u", index="year", precision=Hyperparameter("tau", initial=1.0)),
            over="region", structure=BesagStructure(GRAPH),
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    compiled = compile_family(model, panel).materialize({"tau": 1.0})
    expected = tuple(
        f"u:{r}@{y}" for r in ("r1", "r2", "r3") for y in range(1, 13)
    )
    assert tuple(compiled.labels[-36:]) == expected


def test_a_grouped_model_fits_end_to_end():
    frame = _frame()
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            IID("u", index="t", precision=1.0), over="region",
            structure=BesagStructure(GRAPH),
        ),
    ).fit(frame, engine="laplace")
    assert np.isfinite(result.log_marginal_likelihood)
    assert len(result.labels) == 1 + 6

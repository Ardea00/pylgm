import numpy as np
import pandas as pd
import pytest

from pylgm import (
    AR1, BesagStructure, Fixed, Grouped, IID, IIDStructure, LGM, Poisson,
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
    """F5's constraint check needs an inner effect with a null space -- IID is
    proper, so both sides were vacuously ``(0, 12)`` regardless of ordering.
    RW1 (needs >1 level, so 4 here) makes the comparison real: ``(3, 12)``.
    """
    frame = pd.DataFrame(
        {"region": r, "t": t, "y": 1.0}
        for r in ("r1", "r2", "r3") for t in range(4)
    )
    grouped, _ = _build_effect_block(
        Grouped(RW1("u", index="t", precision=1.5), over="region",
                structure=IIDStructure()),
        frame,
    )
    replicated, _ = _build_effect_block(
        Replicated(RW1("u", index="t", precision=1.5), over="region"), frame
    )
    assert grouped.labels == replicated.labels
    assert np.allclose(grouped.design.toarray(), replicated.design.toarray())
    assert np.allclose(grouped.precision.toarray(), replicated.precision.toarray())
    # F5: constraints are exactly what the two paths build differently --
    # structure.null_basis(groups) versus np.zeros((R, 0)) -- so the
    # reduction claim is incomplete without checking them too.
    assert grouped.constraints.shape == replicated.constraints.shape == (3, 12)
    assert np.allclose(grouped.constraints, replicated.constraints)


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


def test_a_single_group_level_is_rejected_by_rw1_itself_not_the_shape_guard():
    """Despite its old name, this fires ``RW1Structure.precision``'s own "needs
    more than 1 group level(s)" message before ``grouped_block``'s shape guard
    ever runs -- see ``test_a_malformed_duck_typed_structure_is_rejected``
    below for a test that actually exercises the shape guard.
    """
    with pytest.raises((CompilationError, ValueError), match="level"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=RW1Structure()),
            pd.DataFrame({"region": ["only"], "t": ["a"], "y": [1.0]}),
        )


def test_a_malformed_duck_typed_structure_is_rejected():
    """F4: ``grouped_block``'s own shape guard, not any individual structure's.

    ``Grouped.__post_init__`` duck-types a structure by the presence of
    ``levels``/``precision``/``null_basis`` alone, so a user-supplied
    structure can reach ``grouped_block`` with all three methods present but
    a ``precision`` that returns the wrong shape -- there is no built-in
    structure that does this, so it has to be constructed by hand.
    """
    from scipy.sparse import identity as sparse_identity

    class _MismatchedStructure:
        def levels(self, observed):
            return observed

        def precision(self, levels):
            # Deliberately one row/column too many for `levels`.
            return sparse_identity(len(levels) + 1, format="csr")

        def null_basis(self, levels):
            return np.zeros((len(levels), 0))

    with pytest.raises((CompilationError, ValueError), match="shape"):
        _build_effect_block(
            Grouped(IID("u", index="t"), over="region", structure=_MismatchedStructure()),
            _frame(),
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


def test_an_integer_group_column_orders_numerically_not_lexically():
    """F1: ``group_levels`` used to sort the group column as strings, silently
    permuting an ordered outer structure's neighbourhood -- 1, 10, 11, 12, 2,
    ... instead of 1, 2, ..., 12. Needs >= 12 levels so 1/2/10/11/12
    discriminate a numeric sort from a lexical one.
    """
    rows = [
        {"yr": yr, "t": t, "y": 0.0}
        for yr in range(1, 13) for t in ("a", "b")
    ]
    frame = pd.DataFrame(rows)
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t"), over="yr", structure=RW1Structure()),
        frame,
    )
    expected = tuple(f"{yr}@{t}" for yr in range(1, 13) for t in ("a", "b"))
    assert outer.labels == expected


def test_grouped_family_rebuild_uses_the_structure_precision_not_identity():
    """F2: the ParametricBlock rebuild closure built in ``_grouped_family_block``
    must re-Kron the inner rebuild against the *structure's* precision on
    every hyperparameter draw. Swap in an identity there instead and every
    other test in this file still passes -- the family would silently ship
    uncorrelated groups while the hyperparameter stays estimated and
    reported.

    ``rho`` is the estimated hyperparameter (with ``transform="logit"``, so
    the optimizer's unconstrained draws land back inside (-1, 1)) because it
    enters the AR1 structure itself rather than multiplying it -- unlike a
    scalar precision, which would commute with the Kronecker product
    regardless of which factor the bug put it on, and so could not
    distinguish a real structure from an identity.
    """
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.effects.ar1 import ar1_structure

    periods = 4
    rows = [
        {"region": r, "t": t, "y": 0.0}
        for r in ("r1", "r2", "r3") for t in range(periods)
    ]
    frame = pd.DataFrame(rows)
    frame["row"] = range(len(frame))
    structure = BesagStructure(GRAPH)
    groups = structure.levels(("r1", "r2", "r3"))
    outer = structure.precision(groups).toarray()

    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            AR1("u", index="t", rho=Hyperparameter("rho", initial=0.1, transform="logit")),
            over="region", structure=structure,
        ),
    )
    panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="y", panel=())
    )
    family = compile_family(model, panel)
    candidates = [
        item for item in family.blocks
        if item.block.name == "u" and hasattr(item, "build")
    ]
    assert len(candidates) == 1
    build = candidates[0].build
    for rho in (0.1, 0.85):
        composed = build({"rho": rho}).toarray()
        expected = np.kron(outer, ar1_structure(periods, rho).toarray())
        assert np.allclose(composed, expected)


def test_an_unobserved_graph_node_still_gets_a_cell():
    """F3: ``group_levels`` calls ``structure.levels(observed)`` -- not
    ``observed`` unchanged -- so a ``BesagStructure`` node with no
    observations still gets a cell (see its docstring: the graph is the
    universe, not the observed levels). Mutating that call to
    ``return observed`` passes every other test in this file.
    """
    frame = pd.DataFrame({
        "region": ["r1", "r1", "r2", "r2"],
        "t": ["a", "b", "a", "b"],
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    outer, _ = _build_effect_block(
        Grouped(IID("u", index="t"), over="region", structure=BesagStructure(GRAPH)),
        frame,
    )
    assert outer.labels == ("r1@a", "r1@b", "r2@a", "r2@b", "r3@a", "r3@b")
    assert outer.precision.shape == (6, 6)
    design = outer.design.toarray()
    # r3 has no rows in the frame, so its two columns get no observations.
    assert np.allclose(design[:, 4:6], 0.0)


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


def test_an_estimated_inner_precision_scales_every_group():
    """F1: also pins the *outer* factor -- the family path's ordinary
    ScalableBlock branch (``_grouped_family_block``) composes ``item.block``
    once, up front, against ``effect.structure``'s precision; only the
    ParametricBlock rebuild closure is covered elsewhere
    (``test_grouped_family_rebuild_uses_the_structure_precision_not_identity``,
    which only exercises an AR1 structure hyperparameter). Swapping
    ``effect.structure`` for ``IIDStructure()`` in that composed = grouped_block(...)
    call materialises a literal identity between groups -- confirmed by
    mutation to fail this test (and no other test in this file's IID-precision
    family) before this strengthening, and to keep failing after.
    """
    from pylgm.compiler import compile_family
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame()
    structure = BesagStructure(GRAPH)
    groups = structure.levels(("r1", "r2", "r3"))
    outer = structure.precision(groups).toarray()
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Grouped(
            IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)),
            over="region", structure=structure,
        ),
    )
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    family = compile_family(model, panel)
    assert family is not None and "tau" in family.parameter_names
    low = [b for b in family.materialize({"tau": 1.0}).blocks if b.name == "u"][0]
    high = [b for b in family.materialize({"tau": 50.0}).blocks if b.name == "u"][0]
    nonzero = low.precision.toarray() != 0
    assert np.allclose(high.precision.toarray()[nonzero] / low.precision.toarray()[nonzero], 50.0)
    assert low.precision.shape == (6, 6)
    # The inner IID precision template is tau * I_2; the outer factor must be
    # the Besag structure's precision, not an identity -- r1 and r2 couple.
    assert np.allclose(low.precision.toarray(), np.kron(outer, 1.0 * np.eye(2)))
    assert not np.allclose(low.precision.toarray()[0:2, 2:4], 0.0)

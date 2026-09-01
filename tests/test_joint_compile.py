import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pylgm import Beta, Besag, Fixed, IID, LGM, Poisson, RW1
from pylgm.compiler import compile_joint, compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.exceptions import CompilationError, DataContractError
from pylgm.ir.model import LatentBlock
from pylgm.joint import Joint, Shared, _pad_block_rows


def _block():
    design = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    precision = csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    constraints = np.array([[1.0, 1.0]])
    return LatentBlock("u", ("a", "b"), design, precision, constraints)


def test_pad_block_rows_zero_pads_design_and_preserves_everything_else():
    block = _block()
    padded = _pad_block_rows(block, before=2, after=1)

    assert padded.design.shape == (6, 2)
    assert np.allclose(padded.design.toarray()[:2], 0.0)
    assert np.allclose(padded.design.toarray()[2:5], block.design.toarray())
    assert np.allclose(padded.design.toarray()[5:], 0.0)

    assert padded.name == block.name
    assert padded.labels == block.labels
    assert np.allclose(padded.precision.toarray(), block.precision.toarray())
    assert np.allclose(padded.constraints, block.constraints)


def test_pad_block_rows_with_no_padding_is_an_identity_on_the_design():
    # The actual contract is the early-return itself -- no padding means no new
    # sparse blocks/vstack, the same object comes back. Comparing arrays would
    # pass even if the function rebuilt an equal-but-different object, which is
    # not what this is guarding.
    block = _block()
    padded = _pad_block_rows(block, before=0, after=0)
    assert padded is block


def _frame():
    return pd.DataFrame({
        "district": ["a", "b", "c", "a", "b", "c"],
        "oral": [3.0, 5.0, 2.0, None, None, None],
        "larynx": [None, None, None, 4.0, 1.0, 6.0],
        "row": range(6),
    })


def _panel(frame, response):
    sub = frame[frame[response].notna()].reset_index(drop=True)
    return CanonicalPanel.from_frame(sub, DataConfig(time="row", response=response, panel=()))


def test_single_submodel_joint_matches_the_equivalent_lgm():
    # The strongest cheap guard: with one sub-model and no sharing, stacking is
    # the identity, so the compiled artefacts must agree exactly.
    frame = _frame()
    panel = _panel(frame, "oral")
    model = LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1") + IID("d", index="district"))

    expected = compile_lgm(model, panel)
    got = compile_joint(Joint._unchecked((model,), ()), {"oral": panel})

    assert got.design.shape == expected.design.shape
    assert (got.design != expected.design).nnz == 0
    assert (got.precision != expected.precision).nnz == 0
    assert got.labels == tuple(f"oral:{label}" for label in expected.labels)


def test_two_submodels_stack_rows_and_block_diagonalise_the_latent():
    frame = _frame()
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
        LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    compiled = compile_joint(joint, panels)

    assert compiled.design.shape[0] == 6
    assert compiled.labels == ("oral:fixed:Intercept", "larynx:fixed:Intercept")
    dense = compiled.design.toarray()
    # Each sub-model's intercept column is zero outside its own row slice.
    assert dense[3:, 0].tolist() == [0.0, 0.0, 0.0]
    assert dense[:3, 1].tolist() == [0.0, 0.0, 0.0]


def test_shared_effect_with_fixed_scales_enters_both_slices():
    frame = _frame()
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district"), scale=(1.0, 2.0))],
    )
    compiled = compile_joint(joint, panels)

    shared = [b for b in compiled.blocks if b.name == "u"][0]
    dense = shared.design.toarray()
    assert dense.shape == (6, 3)
    assert dense[:3].sum() == pytest.approx(3.0)   # scale 1.0 on three oral rows
    assert dense[3:].sum() == pytest.approx(6.0)   # scale 2.0 on three larynx rows


def test_response_validation_rejects_a_bad_response_and_names_the_outcome():
    # oral's response violates Poisson's domain (negative count); larynx is fine.
    # The compile must fail at compile time, not silently produce NaNs later,
    # and the message must name which sub-model's outcome is at fault.
    frame = _frame()
    frame.loc[0, "oral"] = -1.0
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
        LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1")),
    ])

    with pytest.raises(DataContractError, match="oral"):
        compile_joint(joint, panels)


def test_response_validation_is_scoped_to_each_submodels_observed_rows():
    # The Beta domain is the open unit interval, so the 0.0 fillna sentinel used
    # for unobserved rows is itself invalid; if validation ran over every row
    # (observed or not) this would raise. It must not: unobserved rows are
    # excluded from validation, exactly as compile_lgm does.
    frame = pd.DataFrame({
        "prop": [0.4, 0.6, 0.5, None, None, None],
        "count": [1.0, 2.0, 3.0, 4.0, 1.0, 6.0],
        "row": range(6),
    })
    prop_panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="prop", panel=())
    )
    count_panel = CanonicalPanel.from_frame(
        frame, DataConfig(time="row", response="count", panel=())
    )
    assert prop_panel.observed.tolist() == [True, True, True, False, False, False]

    joint = Joint([
        LGM(response="prop", likelihood=Beta(), predictor=Fixed("1")),
        LGM(response="count", likelihood=Poisson(), predictor=Fixed("1")),
    ])
    compiled = compile_joint(joint, {"prop": prop_panel, "count": count_panel})

    assert compiled.design.shape[0] == 12


def test_shared_block_columns_stay_aligned_when_first_seen_order_is_not_alphabetical():
    # build_iid alphabetises levels internally regardless of input row order;
    # the shared design's columns must be realigned to match, or a row lands
    # in the wrong latent column with a plausible-looking wrong answer.
    frame = pd.DataFrame({
        "district": ["c", "a", "b", "a", "b", "c"],
        "oral": [3.0, 5.0, 2.0, None, None, None],
        "larynx": [None, None, None, 4.0, 1.0, 6.0],
        "row": range(6),
    })
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(IID("u", index="district"), scale=(1.0, 2.0))],
    )
    compiled = compile_joint(joint, panels)

    shared = [b for b in compiled.blocks if b.name == "u"][0]
    assert shared.labels == ("a", "b", "c")
    dense = shared.design.toarray()
    # oral rows are district c, a, b (in that row order) with scale 1.0.
    assert dense[0].tolist() == [0.0, 0.0, 1.0]
    assert dense[1].tolist() == [1.0, 0.0, 0.0]
    assert dense[2].tolist() == [0.0, 1.0, 0.0]
    # larynx rows are district a, b, c (in that row order) with scale 2.0.
    assert dense[3].tolist() == [2.0, 0.0, 0.0]
    assert dense[4].tolist() == [0.0, 2.0, 0.0]
    assert dense[5].tolist() == [0.0, 0.0, 2.0]


def test_unshared_joint_factorises_into_the_separate_fits(shared_component_frame):
    frame, _ = shared_component_frame()
    joint = Joint([
        LGM(response="oral", likelihood=Poisson(),
            predictor=Fixed("1") + IID("d", index="district", precision=1.0)),
        LGM(response="larynx", likelihood=Poisson(),
            predictor=Fixed("1") + IID("d", index="district", precision=1.0)),
    ])
    together = joint.fit(frame, engine="laplace")

    separate = []
    for response in ("oral", "larynx"):
        sub = frame[frame[response].notna()].reset_index(drop=True)
        separate.append(
            LGM(response=response, likelihood=Poisson(),
                predictor=Fixed("1") + IID("d", index="district", precision=1.0)
                ).fit(sub, engine="laplace")
        )

    assert together.log_marginal_likelihood == pytest.approx(
        separate[0].log_marginal_likelihood + separate[1].log_marginal_likelihood, rel=1e-8
    )
    joint_oral = together.mean[[i for i, la in enumerate(together.labels) if la.startswith("oral:")]]
    assert joint_oral == pytest.approx(separate[0].mean, rel=1e-6, abs=1e-8)


def _rw1_labels_match_private(index_values):
    # Finding C1: a shared RW1's levels used to route through str(...), so an
    # int/float index sorted lexicographically ('1','10','11',...) instead of
    # numerically -- silently smoothing the wrong neighbours. A private RW1
    # over the same column is ground truth for the correct order.
    n = len(index_values)
    counts = [float(i % 4) for i in range(n)]  # Poisson-valid response, independent of the index
    frame = pd.DataFrame({
        "period": index_values * 2,
        "oral": counts + [None] * n,
        "larynx": [None] * n + counts,
        "row": range(2 * n),
    })
    oral_panel = _panel(frame, "oral")
    larynx_panel = _panel(frame, "larynx")

    private_model = LGM(
        response="oral", likelihood=Poisson(), predictor=RW1("period_effect", index="period")
    )
    private_compiled = compile_lgm(private_model, oral_panel)
    private_labels = next(
        b for b in private_compiled.blocks if b.name == "period_effect"
    ).labels

    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(RW1("period_effect", index="period"))],
    )
    joint_compiled = compile_joint(joint, {"oral": oral_panel, "larynx": larynx_panel})
    shared_labels = next(
        b for b in joint_compiled.blocks if b.name == "period_effect"
    ).labels

    return private_labels, shared_labels


def test_shared_rw1_over_integer_index_matches_the_private_labels():
    years = list(range(1, 13))  # 1..12: lexicographic order would scramble this
    private_labels, shared_labels = _rw1_labels_match_private(years)
    assert shared_labels == private_labels
    assert shared_labels == tuple(str(y) for y in years)


def test_shared_rw1_over_float_index_matches_the_private_labels():
    values = [1.5, 2.5, 3.5, 10.5, 11.5, 12.5]
    private_labels, shared_labels = _rw1_labels_match_private(values)
    assert shared_labels == private_labels
    assert shared_labels == tuple(str(v) for v in values)


def test_shared_spatial_effect_with_an_unobserved_graph_node_raises_clearly():
    # Finding I1: Besag/ProperCAR/SAR/BYM2 take their latent domain from the
    # graph, not the data. A graph node with no observed row for the shared
    # index used to blow up inside `tuple.index(x)` with no mention of the
    # shared effect or the missing node -- a normal situation whenever a graph
    # ships with more regions than the data (e.g. from a shapefile).
    frame = _frame()  # districts a, b, c
    panels = {"oral": _panel(frame, "oral"), "larynx": _panel(frame, "larynx")}
    graph = {
        "a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"],  # "d" unobserved
    }
    joint = Joint(
        [LGM(response="oral", likelihood=Poisson(), predictor=Fixed("1")),
         LGM(response="larynx", likelihood=Poisson(), predictor=Fixed("1"))],
        shared=[Shared(Besag("region", index="district", graph=graph))],
    )

    with pytest.raises(CompilationError, match="region") as excinfo:
        compile_joint(joint, panels)
    assert "district" in str(excinfo.value)
    assert "d" in str(excinfo.value)

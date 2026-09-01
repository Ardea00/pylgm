import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from pylgm import Beta, Fixed, IID, LGM, Poisson
from pylgm.compiler import compile_joint, compile_lgm
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel
from pylgm.exceptions import DataContractError
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
    block = _block()
    padded = _pad_block_rows(block, before=0, after=0)
    assert np.allclose(padded.design.toarray(), block.design.toarray())


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

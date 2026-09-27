"""The Spark adapter projects the frame onto the columns the model reads; any
column it misses is dropped before collection, so the fit fails or silently
changes. Pure function: no pyspark needed."""

from pylgm import (
    AR1, IID, IIDStructure, LGM, MIDAS, RW1, Binomial, DynamicSpatialPanel, Fixed, Gaussian, Grouped,
    MIDASParametric, Replicated, SpaceTime, Weighted, WeibullSurv,
)
from pylgm.data.spark import _required_columns

GRAPH = {"a": ["b"], "b": ["a"]}


def test_every_effect_column_is_required():
    model = LGM(
        response="y", likelihood=Gaussian(1.0), panel=("region",), time="t",
        predictor=Fixed("1 + x")
        + MIDAS("m", columns=("hf0", "hf1", "hf2"))
        + MIDASParametric("mp", columns=("hp0", "hp1"))
        + SpaceTime("st", space="area", time="q", graph=GRAPH)
        + DynamicSpatialPanel("dsp", unit="unit", time="period", graphs={0: GRAPH}, rho=0.3)
        + Replicated(IID("rep", index="r_idx"), over="r_over")
        + Grouped(RW1("grp", index="g_idx"), over="g_over", structure=IIDStructure())
        + Weighted(IID("w", index="w_idx"), by="w_by")
        + AR1("ar", index="a_idx", replicate="a_rep"),
    )
    assert {
        "hf0", "hf1", "hf2", "hp0", "hp1", "area", "q", "unit", "period",
        "r_idx", "r_over", "g_idx", "g_over", "w_idx", "w_by", "a_idx", "a_rep",
        "x", "y", "region", "t",
    } <= _required_columns(model)


def test_likelihood_columns_are_required():
    binomial = LGM(response="y", likelihood=Binomial(trials="n"), time="t", predictor=Fixed("1"))
    survival = LGM(response="y", likelihood=WeibullSurv(event="d", entry="e"), time="t",
                   predictor=Fixed("1"))
    assert "n" in _required_columns(binomial)
    assert {"d", "e"} <= _required_columns(survival)

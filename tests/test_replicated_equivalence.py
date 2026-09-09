"""`Replicated` against the shipped `AR1(group=)`, which is the same concept.

`AR1(group=)` predates this slice and is R-INLA's `replicate` under the wrong
name: `ar1_structure(..., group_count=G)` builds `I_G (x) T`, independent series
sharing rho and precision. Matching it bit for bit is what proves the general
machinery is right, using an implementation that was correct beforehand.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import AR1, Fixed, LGM, Poisson, Replicated
from pylgm.compiler import _build_effect_block


def _frame(n=72):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "year": [f"y{k % 6}" for k in range(n)],
        "firm": [f"f{k % 4}" for k in range(n)],
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })


@pytest.mark.parametrize("rho", [0.0, 0.3, -0.6, 0.9])
def test_replicated_ar1_matches_the_shipped_grouped_ar1_bit_for_bit(rho):
    frame = _frame()
    with pytest.warns(DeprecationWarning):
        legacy, _ = _build_effect_block(
            AR1("u", index="year", precision=2.0, rho=rho, group="firm"), frame
        )
    general, _ = _build_effect_block(
        Replicated(AR1("u", index="year", precision=2.0, rho=rho), over="firm"), frame
    )
    assert general.labels == legacy.labels
    assert np.allclose(general.design.toarray(), legacy.design.toarray())
    assert np.allclose(general.precision.toarray(), legacy.precision.toarray())
    assert np.allclose(general.constraints, legacy.constraints)


def test_replicated_ar1_and_grouped_ar1_fit_identically():
    frame = _frame()
    general = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Replicated(
            AR1("u", index="year", precision=2.0, rho=0.4), over="firm"),
    ).fit(frame, engine="laplace")
    with pytest.warns(DeprecationWarning):
        legacy = LGM(
            response="y", likelihood=Poisson(),
            predictor=Fixed("1") + AR1(
                "u", index="year", precision=2.0, rho=0.4, group="firm"),
        ).fit(frame, engine="laplace")
    assert general.log_marginal_likelihood == pytest.approx(
        legacy.log_marginal_likelihood, rel=1e-12
    )
    assert general.mean == pytest.approx(legacy.mean, rel=1e-10, abs=1e-12)


def test_ar1_replicate_is_the_supported_name_and_warns_for_group():
    frame = _frame()
    modern, _ = _build_effect_block(
        AR1("u", index="year", precision=2.0, rho=0.4, replicate="firm"), frame
    )
    with pytest.warns(DeprecationWarning, match="replicate"):
        legacy, _ = _build_effect_block(
            AR1("u", index="year", precision=2.0, rho=0.4, group="firm"), frame
        )
    assert modern.labels == legacy.labels
    assert np.allclose(modern.precision.toarray(), legacy.precision.toarray())


def test_ar1_rejects_both_names_at_once():
    with pytest.raises(ValueError, match="replicate"):
        AR1("u", index="year", replicate="firm", group="firm")

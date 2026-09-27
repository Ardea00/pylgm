"""Replicated(Grouped(...)): R-INLA's f(idx, model, group=g, replicate=r).

Precision I_R (x) Q_S (x) Q_E, labels replicate@group@level. With the replicated
term as the only effect and a fixed noise, replicates are independent, so the
fit must decompose exactly into one Grouped fit per replicate.
"""

import numpy as np
import pandas as pd
import pytest

from pylgm import AR1Structure, Gaussian, Grouped, Hyperparameter, LGM, Replicated, RW1


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = [(f"r{r}", f"g{g}", t) for r in range(3) for g in range(4) for t in range(6)]
    frame = pd.DataFrame(rows, columns=["rep", "grp", "t"])
    frame["y"] = rng.normal(size=len(frame)) + frame["t"] * 0.2
    frame.loc[[5, 30], "y"] = np.nan  # prediction-only rows
    return frame


def _grouped(precision=2.0):
    return Grouped(RW1("u", index="t", precision=precision), over="grp",
                   structure=AR1Structure(rho=0.6))


def _fit(effect, frame):
    frame = frame.assign(row=np.arange(len(frame)))
    return LGM("y", Gaussian(0.5), effect, panel=("row",), time="t").fit(frame)


def test_replicated_grouped_decomposes_into_one_grouped_fit_per_replicate():
    frame = _frame()
    joint = _fit(Replicated(_grouped(), over="rep"), frame)
    parts = [_fit(_grouped(), frame[frame["rep"] == r]) for r in ("r0", "r1", "r2")]
    assert joint.log_marginal_likelihood == pytest.approx(
        sum(p.log_marginal_likelihood for p in parts), abs=1e-8
    )
    assert any(label.count("@") == 2 for label in joint.labels)  # rep@grp@level
    for r, part in zip(("r0", "r1", "r2"), parts):
        rows = frame[frame["rep"] == r]
        np.testing.assert_allclose(joint.predict(rows).predictive_mean,
                                   part.predict(rows).predictive_mean, atol=1e-8)
        np.testing.assert_allclose(joint.predict(rows).predictive_variance,
                                   part.predict(rows).predictive_variance, atol=1e-8)


def test_replicated_grouped_estimates_the_shared_precision():
    """The family path: the inner precision is one hyperparameter shared by every copy."""
    effect = Replicated(_grouped(Hyperparameter("tau", initial=1.0)), over="rep")
    result = _fit(effect, _frame())
    assert np.isfinite(result.hyperparameters["tau"])
    fixed = _fit(Replicated(_grouped(result.hyperparameters["tau"]), over="rep"), _frame())
    assert fixed.log_marginal_likelihood == pytest.approx(result.log_marginal_likelihood, abs=1e-8)


def test_grouped_still_rejects_wrapping_a_replicated():
    with pytest.raises(TypeError, match="Replicated\\(Grouped"):
        Grouped(Replicated(RW1("u", index="t"), over="rep"), over="grp",
                structure=AR1Structure(rho=0.5))

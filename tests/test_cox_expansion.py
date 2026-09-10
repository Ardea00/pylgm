"""Semi-parametric survival by Poisson augmentation.

The load-bearing claim is an identity, not an approximation: with one interval
the expanded Poisson model *is* the exponential model, so the two must agree on
the coefficient exactly and on the log likelihood up to the constant that
separates them. Everything else here is downstream of that.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from pylgm import (
    ExponentialSurv, Fixed, Hyperparameter, IID, LGM, Poisson, RW1, expand_cox,
)
from pylgm.exceptions import DataContractError
from pylgm.survival import log_likelihood_offset


def _tiny():
    return pd.DataFrame({"t": [1.0, 2.5, 4.0, 0.5], "d": [1.0, 0.0, 1.0, 1.0],
                         "x": [0.0, 1.0, 0.0, 1.0]})


def _survival_frame(n=600, beta=0.8, rate=0.3, seed=0, shape=None):
    """Exponential (constant hazard) unless ``shape`` asks for a Weibull one."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    if shape is None:
        event_time = rng.exponential(1.0 / (rate * np.exp(beta * x)))
        censor = rng.exponential(1.0 / 0.15, n)
    else:
        event_time = np.exp(-beta * x / shape) * rng.weibull(shape, n)
        censor = rng.uniform(0.2, 2.0, n)
    t = np.minimum(event_time, censor)
    return pd.DataFrame({"t": t, "d": (event_time <= censor).astype(float),
                         "x": x, "row": range(n)})


def _beta(result):
    return dict(zip(result.labels, result.mean))["fixed:x"]


def _fit(expansion, frame, predictor):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LGM(response=expansion.response, likelihood=Poisson(),
                   predictor=predictor, offset=expansion.exposure).fit(frame, engine="laplace")


# --------------------------------------------------------------------------
# The expansion itself
# --------------------------------------------------------------------------
def test_expansion_splits_follow_up_at_the_break_points():
    expansion = expand_cox(_tiny(), time="t", event="d", breaks=[1.0, 3.0])
    frame = expansion.frame
    np.testing.assert_allclose(expansion.breaks, [0.0, 1.0, 3.0, np.inf])
    # subject 0 (t=1.0, event): at risk only on [0, 1), and the event lands there
    first = frame[np.isclose(frame["t"], 1.0)]
    assert list(first[expansion.interval]) == [0]
    np.testing.assert_allclose(np.exp(first[expansion.exposure]), [1.0])
    np.testing.assert_allclose(first[expansion.response], [1.0])
    # subject 2 (t=4.0, event): three rows, event only in the last
    third = frame[np.isclose(frame["t"], 4.0)]
    assert list(third[expansion.interval]) == [0, 1, 2]
    np.testing.assert_allclose(np.exp(third[expansion.exposure]), [1.0, 2.0, 1.0])
    np.testing.assert_allclose(third[expansion.response], [0.0, 0.0, 1.0])


def test_expansion_conserves_exposure_and_events():
    """Two totals that cannot change: everyone's time at risk, and how many
    events happened. A break-point or clipping error moves one of them."""
    frame = _survival_frame(n=300)
    expansion = expand_cox(frame, time="t", event="d", breaks=7)
    exposure = np.exp(expansion.frame[expansion.exposure].to_numpy(dtype=float))
    assert exposure.sum() == pytest.approx(frame["t"].sum())
    assert expansion.frame[expansion.response].sum() == pytest.approx(frame["d"].sum())
    assert (expansion.frame[expansion.response] <= 1.0).all()


def test_left_truncation_removes_time_before_entry():
    frame = pd.DataFrame({"t": [4.0], "d": [1.0], "e": [1.5], "x": [0.0]})
    expansion = expand_cox(frame, time="t", event="d", entry="e", breaks=[1.0, 3.0])
    exposure = np.exp(expansion.frame[expansion.exposure].to_numpy(dtype=float))
    # at risk on [1.5, 3) and [3, 4): nothing before entry
    assert list(expansion.frame[expansion.interval]) == [1, 2]
    np.testing.assert_allclose(exposure, [1.5, 1.0])
    assert exposure.sum() == pytest.approx(4.0 - 1.5)


def test_rows_with_no_time_at_risk_are_dropped():
    """Their offset would be log 0."""
    expansion = expand_cox(_tiny(), time="t", event="d", breaks=[1.0, 3.0])
    assert np.isfinite(expansion.frame[expansion.exposure]).all()
    assert len(expansion.frame) < len(_tiny()) * expansion.intervals


def test_break_points_default_to_event_time_quantiles():
    frame = _survival_frame(n=300)
    expansion = expand_cox(frame, time="t", event="d", breaks=5)
    assert expansion.intervals <= 5
    assert expansion.breaks[0] == 0.0 and np.isinf(expansion.breaks[-1])
    assert np.all(np.diff(expansion.breaks) > 0)


# --------------------------------------------------------------------------
# The identity
# --------------------------------------------------------------------------
def test_one_interval_is_exactly_the_exponential_model():
    """The claim the whole approach rests on.

    A single interval means a constant baseline hazard, which is what
    ``ExponentialSurv`` fits directly. Same model, two mechanisms: the
    coefficient must match to numerical precision, and the log marginal
    likelihood must match once the augmentation constant is restored.
    """
    frame = _survival_frame(n=600)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parametric = LGM(response="t", likelihood=ExponentialSurv("d"),
                         predictor=Fixed("1 + x")).fit(frame, engine="laplace")
    expansion = expand_cox(frame, time="t", event="d", breaks=1)
    augmented = _fit(expansion, expansion.frame, Fixed("1 + x"))

    assert _beta(augmented) == pytest.approx(_beta(parametric), abs=1e-6)
    assert augmented.log_marginal_likelihood + log_likelihood_offset(expansion) == pytest.approx(
        parametric.log_marginal_likelihood, abs=1e-6
    )


def test_the_offset_is_the_documented_constant():
    frame = _survival_frame(n=200)
    expansion = expand_cox(frame, time="t", event="d", breaks=4)
    events = expansion.frame[expansion.response].to_numpy(dtype=float)
    exposure = expansion.frame[expansion.exposure].to_numpy(dtype=float)
    assert log_likelihood_offset(expansion) == pytest.approx(-float(events @ exposure))


# --------------------------------------------------------------------------
# What it buys
# --------------------------------------------------------------------------
def test_recovers_the_coefficient_with_a_free_baseline():
    frame = _survival_frame(n=800, beta=0.8)
    expansion = expand_cox(frame, time="t", event="d", breaks=10)
    result = _fit(expansion, expansion.frame,
                  Fixed("1 + x") + IID("base", index=expansion.interval, precision=1.0))
    assert _beta(result) == pytest.approx(0.8, abs=0.15)


def test_beats_a_misspecified_constant_hazard_on_a_rising_one():
    """The reason to do any of this: a wrong baseline biases the coefficient, and
    an arbitrary one does not have to be guessed."""
    frame = _survival_frame(n=1200, beta=0.7, shape=2.5, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constant = LGM(response="t", likelihood=ExponentialSurv("d"),
                       predictor=Fixed("1 + x")).fit(frame, engine="laplace")
    expansion = expand_cox(frame, time="t", event="d", breaks=12)
    smooth = _fit(expansion, expansion.frame,
                  Fixed("1 + x") + RW1("base", index=expansion.interval,
                                       precision=Hyperparameter("kappa", initial=10.0)))
    assert abs(_beta(smooth) - 0.7) < abs(_beta(constant) - 0.7) / 3.0


def test_recovers_the_shape_of_a_rising_baseline_hazard():
    """log h0(t) = log(a) + (a-1) log t for a Weibull baseline; the interval
    effects should track it."""
    shape = 2.5
    frame = _survival_frame(n=1200, beta=0.7, shape=shape, seed=2)
    expansion = expand_cox(frame, time="t", event="d", breaks=12)
    result = _fit(expansion, expansion.frame,
                  Fixed("1 + x") + RW1("base", index=expansion.interval,
                                       precision=Hyperparameter("kappa", initial=10.0)))
    fitted = np.array([v for k, v in zip(result.labels, result.mean) if k.startswith("base:")])
    edges = expansion.breaks
    finite_upper = np.where(np.isfinite(edges[1:]), edges[1:], edges[-2] * 1.5)
    midpoints = 0.5 * (edges[:-1] + finite_upper)
    truth = (shape - 1.0) * np.log(midpoints)
    assert np.corrcoef(fitted, truth)[0, 1] > 0.95


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize("column, value, match", [
    ("t", 0.0, "positive"),
    ("t", float("inf"), "finite"),
    ("d", 2.0, "must be 0 or 1"),
])
def test_rejects_an_invalid_response_contract(column, value, match):
    frame = _tiny()
    frame.loc[0, column] = value
    with pytest.raises(DataContractError, match=match):
        expand_cox(frame, time="t", event="d", breaks=2)


def test_rejects_entry_at_or_after_exit():
    frame = _tiny().assign(e=[0.0, 0.0, 5.0, 0.0])   # 5.0 >= t of 4.0
    with pytest.raises(DataContractError, match="entry"):
        expand_cox(frame, time="t", event="d", entry="e", breaks=2)


def test_rejects_a_missing_column():
    with pytest.raises(DataContractError, match="column not found"):
        expand_cox(_tiny(), time="t", event="nope", breaks=2)


def test_rejects_colliding_output_names():
    frame = _tiny().assign(interval=1)
    with pytest.raises(ValueError, match="already exists"):
        expand_cox(frame, time="t", event="d", breaks=2)


def test_rejects_data_with_no_events():
    frame = _tiny().assign(d=0.0)
    with pytest.raises(DataContractError, match="no events"):
        expand_cox(frame, time="t", event="d", breaks=3)


@pytest.mark.parametrize("breaks, match", [
    (0, "at least 1"),
    ([2.0, 1.0], "strictly increasing"),
    ([-1.0], "positive"),
    ([], "1-D array"),
])
def test_rejects_impossible_break_points(breaks, match):
    with pytest.raises(ValueError, match=match):
        expand_cox(_tiny(), time="t", event="d", breaks=breaks)

"""Group-wise AR1: one independent series per level of a grouping column."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from pylgm import AR1, Fixed, Gaussian, Hyperparameter, LGM
from pylgm.effects.ar1 import ar1_structure, build_ar1
from pylgm.effects.spec import Predictor


def _panel(groups, periods, rho, sd, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        x = np.empty(periods)
        x[0] = rng.standard_normal()
        for t in range(1, periods):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * rng.standard_normal()
        for t in range(periods):
            rows.append({"firm": f"f{g:02d}", "t": t, "y": x[t] + sd * rng.standard_normal()})
    return pd.DataFrame(rows)


def test_group_structure_is_block_diagonal():
    single = ar1_structure(5, 0.6).toarray()
    grouped = ar1_structure(5, 0.6, group_count=3).toarray()
    assert grouped.shape == (15, 15)
    assert np.allclose(grouped, block_diag(single, single, single))


def test_group_count_one_matches_ungrouped():
    assert np.allclose(
        ar1_structure(6, -0.4, group_count=1).toarray(), ar1_structure(6, -0.4).toarray()
    )


def test_group_count_must_be_positive():
    with pytest.raises(ValueError, match="group_count"):
        ar1_structure(5, 0.5, group_count=0)


def test_grouped_fit_equals_separate_per_group_fits():
    """The defining property: G groups are exactly G independent AR1 fits.

    Fixed hyperparameters and no fixed effect, so nothing couples the groups --
    any cross-group leakage in the precision or the design shows up here.
    """
    periods = 15

    def series(seed):
        r = np.random.default_rng(seed)
        x = np.empty(periods)
        x[0] = r.standard_normal()
        for t in range(1, periods):
            x[t] = 0.6 * x[t - 1] + 0.8 * r.standard_normal()
        return x + 0.2 * r.standard_normal(periods)

    ya, yb = series(1), series(2)
    both = pd.DataFrame(
        {"g": ["a"] * periods + ["b"] * periods,
         "t": list(range(periods)) * 2,
         "y": np.concatenate([ya, yb])}
    )
    joint = LGM(
        response="y",
        predictor=Predictor((AR1("d", "t", group="g", precision=1.5, rho=0.6),)),
        likelihood=Gaussian(sigma=0.2),
    ).fit(both).mean

    separate = np.concatenate([
        LGM(
            response="y",
            predictor=Predictor((AR1("d", "t", precision=1.5, rho=0.6),)),
            likelihood=Gaussian(sigma=0.2),
        ).fit(pd.DataFrame({"t": range(periods), "y": y})).mean
        for y in (ya, yb)
    ])
    assert np.allclose(joint, separate, atol=1e-9)


def test_grouped_ar1_recovers_rho_from_a_panel():
    # 8 x 30 recovers rho to 0.714, the same estimate a 12 x 40 panel gives,
    # in about a sixth of the time.
    frame = _panel(groups=8, periods=30, rho=0.75, sd=0.3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + AR1(
            "dyn", index="t", group="firm",
            rho=Hyperparameter("dyn.rho", initial=0.0, transform="logit"),
            precision=Hyperparameter("dyn.precision", initial=1.0),
        ),
        likelihood=Gaussian(sigma=0.3),
    )
    result = model.fit(frame)
    assert result.hyperparameters["dyn.rho"] == pytest.approx(0.75, abs=0.12)
    # one latent cell per (firm, period), plus the intercept
    assert result.mean.shape[0] == 8 * 30 + 1


def test_grouped_labels_and_design_are_group_major():
    frame = pd.DataFrame({"g": ["b", "a", "b", "a"], "t": [0, 0, 1, 1], "y": [0.0] * 4})
    block = build_ar1(frame, "d", "t", precision=1.0, rho=0.5, group="g")
    assert block.labels == ("a@0", "a@1", "b@0", "b@1")
    # row 0 is (b, 0) -> group 1, level 0 -> cell 1 * 2 + 0 = 2
    assert block.design.toarray()[0].argmax() == 2


def test_predict_scores_grouped_rows():
    frame = _panel(groups=4, periods=12, rho=0.6, sd=0.3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + AR1("dyn", index="t", group="firm", precision=1.0, rho=0.6),
        likelihood=Gaussian(sigma=0.3),
    )
    result = model.fit(frame)
    predicted = result.predict(frame).predictive_mean
    assert np.isfinite(predicted).all()
    assert np.corrcoef(predicted, frame["y"])[0, 1] > 0.5
    # a subset of groups still scores, on the fitted posterior
    subset = frame[frame["firm"].isin(["f00", "f02"])]
    assert np.isfinite(result.predict(subset).predictive_mean).all()


def test_predict_rejects_an_unseen_group():
    frame = _panel(groups=3, periods=10, rho=0.6, sd=0.3)
    model = LGM(
        response="y",
        predictor=Fixed("1") + AR1("dyn", index="t", group="firm", precision=1.0, rho=0.6),
        likelihood=Gaussian(sigma=0.3),
    )
    result = model.fit(frame)
    unseen = frame.head(3).copy()
    unseen["firm"] = "brand_new"
    with pytest.raises(ValueError, match="group/level"):
        result.predict(unseen)


def test_missing_group_column_is_named():
    frame = pd.DataFrame({"t": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="group column 'firm' not found"):
        build_ar1(frame, "d", "t", precision=1.0, rho=0.5, group="firm")


def test_ungrouped_ar1_is_unchanged():
    frame = pd.DataFrame({"t": [0, 1, 2, 3], "y": [1.0, 2.0, 3.0, 4.0]})
    block = build_ar1(frame, "d", "t", precision=1.0, rho=0.5)
    assert block.labels == ("0", "1", "2", "3")
    assert block.precision.shape == (4, 4)

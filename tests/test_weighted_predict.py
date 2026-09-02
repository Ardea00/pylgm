import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted


def _fitted():
    rng = np.random.default_rng(17)
    n = 60
    frame = pd.DataFrame({
        "district": [f"d{i % 12}" for i in range(n)],
        "z": rng.normal(1.0, 0.4, n),
        "y": rng.poisson(3.0, n).astype(float),
        "row": range(n),
    })
    model = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(IID("u", index="district", precision=2.0), by="z"),
    )
    return frame, model.fit(frame, engine="laplace")


def test_predicting_on_the_fit_rows_reproduces_the_fitted_means():
    frame, result = _fitted()
    prediction = result.predict(frame)
    assert prediction.predictive_mean == pytest.approx(
        result.predictive_mean, rel=1e-12, abs=1e-12
    )


def test_prediction_uses_new_data_weights_not_the_fitted_ones():
    """The weights are data, not fitted state: doubling z on new rows must
    change the linear predictor. If predict rebuilt an unweighted design, or
    reused the fit-time weights, this would not move."""
    frame, result = _fitted()
    doubled = frame.assign(z=frame["z"] * 2.0)
    base = result.predict(frame).predictive_mean
    scaled = result.predict(doubled).predictive_mean
    assert not np.allclose(base, scaled)


def test_predict_rejects_new_data_missing_the_weight_column():
    frame, result = _fitted()
    with pytest.raises(ValueError, match="z"):
        result.predict(frame.drop(columns=["z"]))


def test_predict_rejects_a_datetime_weight_column():
    """Mirrors the compile-time check: a datetime `by` column at predict time
    must raise, not silently convert to nanosecond-since-epoch floats."""
    import datetime

    frame, result = _fitted()
    bad = frame.copy()
    bad["z"] = [datetime.date(2020, 1, 1 + i % 27) for i in range(len(bad))]
    with pytest.raises(ValueError, match="z"):
        result.predict(bad)

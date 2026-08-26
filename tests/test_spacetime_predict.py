import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, LGM, SpaceTime

GRAPH = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}


def _frame():
    rows = [(s, t) for s in ["a", "b", "c"] for t in range(5)]
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"area": [r[0] for r in rows], "t": [r[1] for r in rows], "y": rng.normal(size=len(rows))}
    )


def test_predict_on_seen_cells_roundtrips():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=1.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(_frame())
    new = pd.DataFrame({"area": ["a", "c"], "t": [0, 4], "y": [0.0, 0.0]})
    pred = result.predict(new)
    assert pred.predictive_mean.shape == (2,)
    assert np.isfinite(pred.predictive_mean).all()


def test_predict_rejects_unseen_area():
    model = LGM(
        response="y",
        predictor=Fixed("1") + SpaceTime("st", space="area", time="t", interaction="I", precision=1.0),
        likelihood=Gaussian(sigma=0.5),
    )
    result = model.fit(_frame())
    new = pd.DataFrame({"area": ["z"], "t": [0], "y": [0.0]})
    with pytest.raises(ValueError, match="not in the fitted"):
        result.predict(new)

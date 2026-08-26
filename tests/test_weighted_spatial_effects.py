import numpy as np
import pandas as pd

from pylgm import BYM2, Besag, Fixed, Gaussian, LGM, ProperCAR


def _frame():
    return pd.DataFrame({"region": ["a", "b", "c"], "y": [0.2, -0.1, 0.4]})


def test_besag_weighted_differs_from_unweighted():
    frame = _frame()
    tri = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
    weighted = {"a": {"b": 5.0, "c": 0.1}, "b": {"a": 5.0, "c": 0.1},
                "c": {"a": 0.1, "b": 0.1}}

    def fit(graph):
        model = LGM(response="y", likelihood=Gaussian(sigma=0.1),
                    predictor=Fixed("1") + Besag("region", index="region", graph=graph))
        return model.fit(frame).latent_marginals("region").mean

    assert not np.allclose(fit(tri), fit(weighted))  # weights change the smoothing


def test_besag_weight_one_matches_unweighted_exactly():
    frame = _frame()
    tri = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    tri_w = {"a": {"b": 1.0}, "b": {"a": 1.0, "c": 1.0}, "c": {"b": 1.0}}

    def fit(graph):
        model = LGM(response="y", likelihood=Gaussian(sigma=0.1),
                    predictor=Fixed("1") + Besag("region", index="region", graph=graph))
        return model.fit(frame).latent_marginals("region").mean

    assert np.allclose(fit(tri), fit(tri_w))


def test_proper_car_and_bym2_accept_weighted_graphs():
    frame = _frame()
    g = {"a": {"b": 2.0}, "b": {"a": 2.0, "c": 3.0}, "c": {"b": 3.0}}
    for effect in (ProperCAR("region", index="region", graph=g, rho=0.5),
                   BYM2("region", index="region", graph=g, precision=1.0, phi=0.5)):
        model = LGM(response="y", likelihood=Gaussian(sigma=0.1),
                    predictor=Fixed("1") + effect)
        assert model.fit(frame).latent_marginals("region").mean.shape == (3,)

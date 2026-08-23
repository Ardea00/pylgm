import numpy as np
import pandas as pd

from pylgm import Fixed, Gaussian, Hyperparameter, LGM, ProperCAR
from pylgm.priors import PCPrecision


def _chain(n):
    return {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}


def _spatial_frame(n=12, rho_true=0.95, seed=0):
    # simulate smooth spatial signal on a chain, strong dependence
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho_true * x[i - 1] + rng.normal(scale=0.3)
    x -= x.mean()
    y = x + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"region": [str(i) for i in range(n)], "y": y})


def _estimate_rho(rho_true, seed):
    frame = _spatial_frame(rho_true=rho_true, seed=seed)
    rho = Hyperparameter("region.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(response="y",
                predictor=Fixed("1")
                + ProperCAR("region", index="region", graph=_chain(12), rho=rho, precision=tau),
                likelihood=Gaussian(sigma=0.1))
    return model.fit(frame).hyperparameters["region.rho"]


def test_rho_estimated_by_empirical_bayes():
    # A single realization at n=12 is weakly identified (a lucky seed can give a
    # high estimate even for an independent truth), so average a few seeds and
    # assert the estimate TRACKS the simulated dependence: strong-vs-weak must
    # separate. This is what makes the test discriminating rather than a
    # coin-flip threshold on one draw.
    seeds = (0, 1, 2, 3, 4)
    strong = [_estimate_rho(0.95, s) for s in seeds]
    weak = [_estimate_rho(0.0, s) for s in seeds]
    assert all(-1.0 < e < 1.0 for e in strong + weak)
    assert np.mean(strong) > np.mean(weak) + 0.3


def test_rho_integrated_by_inla():
    frame = _spatial_frame()
    graph = _chain(12)
    rho = Hyperparameter("region.rho", initial=0.0, transform="logit")
    tau = Hyperparameter("region.precision", initial=1.0, prior=PCPrecision(upper_sd=1.0, alpha=0.01))
    model = LGM(response="y",
                predictor=Fixed("1") + ProperCAR("region", index="region", graph=graph, rho=rho, precision=tau),
                likelihood=Gaussian(sigma=0.1))
    result = model.fit(frame, hyperparameters="integrate")
    marg = result.hyperparameter_marginals()
    assert "region.rho" in marg and "region.precision" in marg
    m = marg["region.rho"]
    assert -1.0 < float(m.mean[0]) < 1.0
    assert float(m.variance[0]) > 0.0


def test_plugin_rho_still_works():
    frame = _spatial_frame()
    model = LGM(response="y",
                predictor=Fixed("1") + ProperCAR("region", index="region", graph=_chain(12), rho=0.9),
                likelihood=Gaussian(sigma=0.1))
    assert model.fit(frame) is not None

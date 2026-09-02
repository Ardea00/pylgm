"""Spatially-varying coefficient: the model this slice exists to enable.

    log mu_i = alpha + z_i * u_{s(i)},    u ~ IID(tau)

The effect of covariate z varies by region. Recovering the simulated u from
data is what proves the weighting enters the predictor the way it is meant to,
rather than merely compiling.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, IID, LGM, Poisson, Weighted
from pylgm.parameters import Hyperparameter

N_REGIONS, PER_REGION = 15, 40
SIGMA_U = 0.5


def _simulate(seed=23):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, SIGMA_U, N_REGIONS)
    region = np.repeat(np.arange(N_REGIONS), PER_REGION)
    z = rng.normal(0.0, 1.0, N_REGIONS * PER_REGION)
    eta = 0.5 + z * u[region]
    y = rng.poisson(np.exp(eta)).astype(float)
    return u, pd.DataFrame({
        "region": region, "z": z, "y": y, "row": range(len(y)),
    })


def test_spatially_varying_coefficient_is_recovered():
    u_true, frame = _simulate()
    result = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(
            IID("u", index="region", precision=Hyperparameter("tau", initial=1.0)), by="z"
        ),
    ).fit(frame, engine="laplace")

    fitted = np.array([
        dict(zip(result.labels, result.mean))[f"u:{i}"] for i in range(N_REGIONS)
    ])
    # Correlation is the honest summary: shrinkage biases magnitudes toward zero,
    # so requiring the values themselves to match would be requiring the prior
    # not to work.
    assert np.corrcoef(fitted, u_true)[0, 1] > 0.8
    assert result.hyperparameters["tau"] > 0


def test_a_constant_weight_column_collapses_to_an_ordinary_iid():
    """With z constant, z*u_s is just a rescaled ordinary IID, so the weighted
    fit must match the unweighted one on the same data up to that scale."""
    _, frame = _simulate()
    frame = frame.assign(z=1.0)
    weighted = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + Weighted(IID("u", index="region", precision=2.0), by="z"),
    ).fit(frame, engine="laplace")
    plain = LGM(
        response="y", likelihood=Poisson(),
        predictor=Fixed("1") + IID("u", index="region", precision=2.0),
    ).fit(frame, engine="laplace")

    assert weighted.log_marginal_likelihood == pytest.approx(
        plain.log_marginal_likelihood, rel=1e-9
    )
    assert weighted.mean == pytest.approx(plain.mean, rel=1e-7, abs=1e-9)

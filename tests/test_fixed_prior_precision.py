import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
from scipy.optimize import minimize_scalar

from pylgm import Fixed, Gaussian, LGM, Poisson
from pylgm.joint import Joint
from pylgm.parameters import Hyperparameter


def _ridge_data(intercept=2.0, n=80, p=15, seed=0):
    rng = default_rng(seed)
    names = [f"x{i}" for i in range(p)]
    X = rng.normal(size=(n, p))
    beta_true = rng.normal(scale=0.5, size=p)
    y = intercept + X @ beta_true + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame(X, columns=names)
    frame["y"] = y
    formula = "1 + " + " + ".join(names)
    return frame, formula, names, beta_true


def test_hyperparameter_prior_precision_is_accepted_and_float_still_validated():
    hp = Hyperparameter("tau_beta", initial=1.0)
    effect = Fixed("1 + x", prior_precision=hp)
    assert effect.prior_precision is hp

    for bad in (-1.0, 0.0, True, "a"):
        with pytest.raises(ValueError):
            Fixed("1 + x", prior_precision=bad)


def test_learned_ridge_matches_the_exact_gaussian_oracle():
    frame, formula, names, _ = _ridge_data()
    sigma = 0.5
    model = LGM(
        response="y",
        likelihood=Gaussian(sigma),
        predictor=Fixed(formula, prior_precision=Hyperparameter("tau_beta", initial=1.0)),
    )
    result = model.fit(frame)
    tau_hat = result.hyperparameters["tau_beta"]

    p = len(names)
    X = frame[names].to_numpy(dtype=float)
    X1 = np.column_stack([np.ones(len(frame)), X])
    y = frame["y"].to_numpy(dtype=float)

    def lml_of(tau):
        Qprior = np.diag([1e-6] + [tau] * p)
        Qpost = Qprior + X1.T @ X1 / sigma**2
        m = np.linalg.solve(Qpost, X1.T @ y / sigma**2)
        sign_prior, logdet_prior = np.linalg.slogdet(Qprior)
        sign_post, logdet_post = np.linalg.slogdet(Qpost)
        n = len(y)
        return (
            0.5 * logdet_prior
            - 0.5 * logdet_post
            - n / 2 * np.log(2 * np.pi * sigma**2)
            - 0.5 * (y @ y / sigma**2 - m @ Qpost @ m)
        ), m

    lml_hat, m_hat = lml_of(tau_hat)

    assert result.log_marginal_likelihood == pytest.approx(lml_hat, abs=1e-6)

    opt = minimize_scalar(
        lambda log_tau: -lml_of(np.exp(log_tau))[0],
        bounds=(-10, 10),
        method="bounded",
        options={"xatol": 1e-8},
    )
    tau_oracle = np.exp(opt.x)
    assert tau_hat == pytest.approx(tau_oracle, rel=2e-2)

    beta_mean = result.latent_marginals("fixed").mean
    np.testing.assert_allclose(beta_mean, m_hat, atol=1e-6)


def test_intercept_is_not_shrunk():
    frame, formula, names, _ = _ridge_data(intercept=50.0)
    model = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed(formula, prior_precision=Hyperparameter("tau_beta", initial=100.0)),
    )
    result = model.fit(frame)

    X1 = np.column_stack([np.ones(len(frame)), frame[names].to_numpy(dtype=float)])
    y = frame["y"].to_numpy(dtype=float)
    ols, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ols_intercept = ols[0]

    beta_mean = result.latent_marginals("fixed").mean
    assert abs(beta_mean[0] - ols_intercept) < 0.5


def test_learned_ridge_on_a_non_gaussian_likelihood():
    rng = default_rng(1)
    n = 200
    p = 5
    names = [f"x{i}" for i in range(p)]
    X = rng.normal(size=(n, p))
    b = rng.normal(scale=0.2, size=p)
    eta = 0.5 + X @ b
    y = rng.poisson(np.exp(eta))
    frame = pd.DataFrame(X, columns=names)
    frame["y"] = y
    formula = "1 + " + " + ".join(names)

    model = LGM(
        response="y",
        likelihood=Poisson(),
        predictor=Fixed(formula, prior_precision=Hyperparameter("tau_beta", initial=1.0)),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(frame, engine="laplace")

    tau_hat = result.hyperparameters["tau_beta"]
    assert np.isfinite(tau_hat)
    assert tau_hat > 0
    assert not any("edge" in str(w.message) for w in caught)


def test_learned_ridge_integrates():
    frame, formula, names, _ = _ridge_data()
    model = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed(formula, prior_precision=Hyperparameter("tau_beta", initial=1.0)),
    )
    result = model.fit(frame, hyperparameters="integrate")
    assert np.isfinite(result.log_marginal_likelihood)
    assert "tau_beta" in result.hyperparameter_marginals()


def test_learned_ridge_inside_a_joint():
    frame, formula, names, _ = _ridge_data()

    standalone = LGM(
        response="y",
        likelihood=Gaussian(0.5),
        predictor=Fixed(formula, prior_precision=Hyperparameter("tau_beta", initial=1.0)),
    ).fit(frame)
    standalone_tau = standalone.hyperparameters["tau_beta"]

    other = pd.DataFrame({"z": [1.0, 2.0, 3.0, 4.0, 5.0]})

    joint_frame = pd.concat(
        [
            frame.assign(z=np.nan),
            other.assign(**{"y": np.nan, **{name: np.nan for name in names}}),
        ],
        ignore_index=True,
    )

    joint = Joint(
        [
            LGM(
                response="y",
                likelihood=Gaussian(0.5),
                predictor=Fixed(formula, prior_precision=Hyperparameter("tau_beta", initial=1.0)),
            ),
            LGM(response="z", likelihood=Gaussian(1.0), predictor=Fixed("1")),
        ]
    )
    result = joint.fit(joint_frame, engine="laplace")

    assert "tau_beta" in result.hyperparameters
    assert result.hyperparameters["tau_beta"] == pytest.approx(standalone_tau, rel=5e-2)

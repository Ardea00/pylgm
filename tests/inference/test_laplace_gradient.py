"""Analytic Laplace LML gradient (dense engine) against refit central differences."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pylgm.inference import fit_laplace
from pylgm.ir.model import CompiledLGM
from pylgm.likelihoods import (
    CompiledBernoulli, CompiledCensoredHurdle, CompiledGamma, CompiledGaussian, CompiledMixture,
    CompiledNegativeBinomial, CompiledPoisson, CompiledWeibullSurv,
)

N = 8
TIGHT = 1e-12  # Newton tolerance: the envelope theorem needs a stationary mode


def _prior(kind):
    w = np.eye(N, k=1) + np.eye(N, k=-1)
    d = np.diag(w.sum(1))
    if kind == "iid":
        return (lambda t: (2.0 + t) * np.eye(N)), csr_matrix(np.eye(N)), False
    return (lambda t: (1.5 + t) * (d - w)), csr_matrix(d - w), True  # rw1 + sum-to-zero


def _hurdle(t):
    m = 3
    mask = np.zeros(2 * m + N, dtype=bool)
    mask[: 2 * m] = True
    hurdle = CompiledCensoredHurdle(np.arange(m), np.arange(m, 2 * m), np.full(m, -0.3), 0.8 + t)
    return CompiledMixture(((mask, hurdle), (~mask, CompiledGaussian(0.9 + t))), 2 * m + N)


_Y = np.array([0.5, 2, 1.2, 4, 3, 0.3, 5, 1.0])
_COUNTS = np.array([0, 2, 1, 4, 3, 0, 5, 1.0])
# name -> (likelihood(t), y, has an estimable scalar)
_LIKELIHOODS = {
    "poisson": (lambda t: CompiledPoisson(), _COUNTS, False),
    "bernoulli": (lambda t: CompiledBernoulli(), np.array([0, 1, 1, 0, 1, 0, 0, 1.0]), False),
    "negbin": (lambda t: CompiledNegativeBinomial(2.0 + t), _COUNTS, True),
    "gamma": (lambda t: CompiledGamma(3.0 + t), _Y, True),
    "weibull": (lambda t: CompiledWeibullSurv(1.3 + t, np.array([1, 0, 1, 1, 0, 1, 0.0])), _Y, True),
    "gaussian": (lambda t: CompiledGaussian(float(np.sqrt(0.7 + t))), np.linspace(-1, 1, N), True),
    "hurdle+gaussian": (_hurdle, np.r_[np.zeros(6), np.linspace(-1, 1, N)], True),
}


def _model(constrained, q, likelihood, y, extra):
    rows = np.arange(y.size)
    design = np.zeros((y.size, N))
    design[rows, rows % N] = 1.0
    design[rows, (rows + 1) % N] += 0.2
    block = np.ones((1, N)) if constrained else np.empty((0, N))
    kwargs = {}
    if extra:
        row = np.zeros((1, N))
        row[0, :3] = 1.0
        block = np.vstack([block, row])
        kwargs = {"extra_constraints": row, "extra_constraint_rhs": np.array([0.4])}
    return CompiledLGM(
        y=y, observed=rows != rows[-1] - 1, offset=0.1 * np.cos(rows),
        design=csr_matrix(design), precision=csr_matrix(q), constraints=block,
        labels=tuple(map(str, range(N))), likelihood=likelihood, blocks=(), **kwargs,
    )


@pytest.mark.parametrize("prior", ["iid", "rw1"])
@pytest.mark.parametrize("name", list(_LIKELIHOODS))
@pytest.mark.parametrize("direction", ["q", "lik", "both"])
@pytest.mark.parametrize("extra", [False, True])
def test_laplace_lml_gradient_matches_refit_central_difference(prior, name, direction, extra):
    build, d_q, constrained = _prior(prior)
    likelihood, y, estimable = _LIKELIHOODS[name]
    use_q, use_lik = direction in {"q", "both"}, direction in {"lik", "both"}
    if use_lik and not estimable:
        pytest.skip("likelihood has no estimable scalar")

    def fit(t, **kw):
        model = _model(constrained, build(t * use_q), likelihood(t * use_lik), y, extra)
        return fit_laplace(model, tolerance=TIGHT, **kw)

    h = 1e-5
    d_lik = (likelihood(-h), likelihood(h), 2 * h) if use_lik else None
    result = fit(0.0, lml_directions=((d_q if use_q else None, d_lik),))
    (grad,) = result.diagnostics["lml_gradient"]
    numeric = (fit(h).log_marginal_likelihood - fit(-h).log_marginal_likelihood) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)


def test_laplace_float_direction_is_gaussian_variance_and_key_omitted_otherwise():
    build, d_q, _ = _prior("iid")
    likelihood, y, _ = _LIKELIHOODS["gaussian"]
    model = _model(False, build(0.0), likelihood(0.0), y, False)
    h = 1e-5
    (grad,) = fit_laplace(model, tolerance=TIGHT, lml_directions=((None, 1.0),)).diagnostics[
        "lml_gradient"
    ]

    def lml(t):
        refit = _model(False, build(0.0), likelihood(t), y, False)
        return fit_laplace(refit, tolerance=TIGHT).log_marginal_likelihood

    np.testing.assert_allclose(grad, (lml(h) - lml(-h)) / (2 * h), rtol=1e-6)
    poisson = _model(False, build(0.0), CompiledPoisson(), _COUNTS, False)
    assert "lml_gradient" not in fit_laplace(poisson, lml_directions=((d_q, 1.0),)).diagnostics
    assert "lml_gradient" not in fit_laplace(poisson).diagnostics


def test_laplace_empirical_bayes_analytic_matches_finite_differences_with_fewer_fits(monkeypatch):
    import pandas as pd

    from pylgm import IID, LGM, Fixed, Hyperparameter, NegativeBinomial
    from pylgm.inference import laplace
    from pylgm.optimization import empirical_bayes

    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(30), 4)
    mu = np.exp(0.3 + 0.6 * rng.normal(size=30)[g])
    frame = pd.DataFrame({"t": np.arange(g.size), "g": g, "y": rng.negative_binomial(2, 2 / (2 + mu))})
    model = LGM(
        "y", NegativeBinomial(Hyperparameter("phi", initial=1.0, lower=0.1, upper=50.0)),
        Fixed("1") + IID("g", index="g", precision=Hyperparameter("tau", initial=1.0)), time="t",
    )
    counts = {"fits": 0, "directions": 0}
    original = laplace._fit_laplace_dense

    def counting(model, *args, **kwargs):
        counts["fits"] += 1
        counts["directions"] += kwargs.get("lml_directions") is not None
        return original(model, *args, **kwargs)

    monkeypatch.setattr(laplace, "_fit_laplace_dense", counting)
    analytic = model.fit(frame, engine="laplace")
    fits_analytic, counts["fits"] = counts["fits"], 0
    assert counts["directions"] > 0
    monkeypatch.setattr(empirical_bayes, "_ANALYTIC_GRADIENT", False)
    reference = model.fit(frame, engine="laplace")
    assert fits_analytic < counts["fits"] / 3
    for name in ("phi", "tau"):
        assert analytic.hyperparameters[name] == pytest.approx(reference.hyperparameters[name], rel=1e-3)

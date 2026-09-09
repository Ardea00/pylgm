"""Calibration harness tests.

The load-bearing test here is ``test_detects_*``: a calibration check that cannot
fail is worse than no check at all, and this repo has found two near-vacuous
tests by mutating the implementation and confirming the failure. So the harness
is pointed at a deliberately corrupted posterior and required to reject it.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import norm

from pylgm import BYM2, RW1, RW2, Besag, Binomial, Fixed, Gaussian, IID, LGM, Poisson
from pylgm.inference.result import (
    GaussianMarginals,
    SkewNormalMarginals,
    TabulatedMarginals,
)
from pylgm.validation import calibrate, pit, simulate_latent, simulate_response


def _frame(n=60, groups=10):
    return pd.DataFrame({
        "g": [f"a{i % groups}" for i in range(n)],
        "y": np.zeros(n),
        "row": range(n),
    })


def _gaussian_model(sigma=0.7):
    return LGM(
        response="y",
        likelihood=Gaussian(sigma=sigma),
        predictor=Fixed("1", prior_precision=1.0) + IID("u", index="g", precision=2.0),
    )


# --------------------------------------------------------------------------
# The self-test: a Gaussian likelihood with a Gaussian latent strategy has an
# EXACT posterior, so its PIT is uniform by construction. This case cannot fail
# for statistical reasons -- only if the simulator, the indexing or the CDF is
# wrong. It is the harness checking itself.
# --------------------------------------------------------------------------
def test_exact_gaussian_posterior_is_calibrated():
    report = calibrate(_gaussian_model(), _frame(), replicates=256, seed=11)
    assert report.ok, f"exact posterior reported as miscalibrated:\n{report}"
    assert len(report.entries) == 5
    assert all(0.0 <= value <= 1.0 for e in report.entries for value in e.values)


def test_report_is_reproducible_from_its_seed():
    kwargs = dict(replicates=64, seed=5)
    first = calibrate(_gaussian_model(), _frame(), **kwargs)
    second = calibrate(_gaussian_model(), _frame(), **kwargs)
    np.testing.assert_array_equal(first.entries[0].values, second.entries[0].values)


# --------------------------------------------------------------------------
# Mutation tests -- the harness must reject a posterior that is wrong.
# --------------------------------------------------------------------------
def _corrupt_cdf(monkeypatch, *, sd_factor=1.0, mean_shift=0.0):
    def cdf(self, x):
        sd = np.sqrt(self.variance)
        centre = self.mean + mean_shift * sd
        return norm.cdf((np.asarray(x, dtype=float) - centre) / (sd * sd_factor))

    monkeypatch.setattr(GaussianMarginals, "cdf", cdf)


# Magnitudes here are chosen for ~1.0 detection power at 256 replicates, measured
# rather than guessed: a 0.5-sigma shift and a 0.6x/1.6x SD error. Smaller, more
# "impressive" corruptions were tried first and rejected -- a 0.1-sigma shift has
# only 0.02 power at this size, so a test asserting it would be passing on the
# luck of one seed while implying a sensitivity the harness does not have. The
# real sensitivity curve is documented in docs/calibration.md.
def test_detects_a_shifted_posterior_mean(monkeypatch):
    """A location error, which the KS statistic on the raw PIT catches."""
    _corrupt_cdf(monkeypatch, mean_shift=0.5)
    report = calibrate(_gaussian_model(), _frame(), replicates=256, seed=11)
    assert not report.ok
    assert min(e.pvalue for e in report.entries) < report.threshold


def test_detects_an_overconfident_posterior(monkeypatch):
    """A dispersion error, which plain KS is nearly blind to.

    The PIT keeps its median at 0.5 and pushes mass into both tails, so it is the
    folded (dispersion) statistic that must catch this -- asserted specifically,
    so that losing the folded test would fail here rather than pass by accident.
    """
    _corrupt_cdf(monkeypatch, sd_factor=0.6)
    report = calibrate(_gaussian_model(), _frame(), replicates=256, seed=11)
    assert not report.ok
    assert min(e.dispersion_pvalue for e in report.entries) < report.threshold


def test_detects_an_underconfident_posterior(monkeypatch):
    _corrupt_cdf(monkeypatch, sd_factor=1.6)
    report = calibrate(_gaussian_model(), _frame(), replicates=256, seed=11)
    assert not report.ok
    assert min(e.dispersion_pvalue for e in report.entries) < report.threshold


# --------------------------------------------------------------------------
# Guards. Each rejects a case that would otherwise fail silently or absurdly.
# --------------------------------------------------------------------------
def test_near_improper_prior_is_rejected():
    """``Fixed`` defaults to prior_precision=1e-6, i.e. a prior SD of 1000.

    Sensible for fitting, useless for calibration -- simulating an intercept from
    N(0, 1000^2) yields data no likelihood survives. Must be a loud error.
    """
    model = LGM(response="y", likelihood=Gaussian(sigma=0.7),
                predictor=Fixed("1") + IID("u", index="g", precision=2.0))
    with pytest.raises(ValueError, match="near-improper"):
        calibrate(model, _frame(), replicates=4)


def test_improper_prior_direction_is_rejected():
    """A null-space direction the constraints do not pin down cannot be sampled.

    Constructed directly, because the effect library always pairs a rank-deficient
    precision with a constraint basis for its null space -- which is exactly why
    the supported intrinsic effects work.
    """
    singular = csr_matrix(np.diag([1.0, 0.0]))
    compiled = SimpleNamespace(
        precision=singular,
        constraints=np.empty((0, 2)),
        constraint_rhs=np.empty(0),
    )
    with pytest.raises(NotImplementedError, match="not positive definite"):
        simulate_latent(compiled, np.random.default_rng(0))


def test_unsupported_likelihood_is_rejected():
    compiled = type("C", (), {
        "design": np.eye(2), "offset": np.zeros(2),
        "likelihood": type("L", (), {"response_mean": staticmethod(lambda eta: eta)})(),
    })()
    with pytest.raises(NotImplementedError, match="no simulator"):
        simulate_response(compiled, np.zeros(2), np.random.default_rng(0))


def test_replicates_and_alpha_are_validated():
    with pytest.raises(ValueError, match="replicates"):
        calibrate(_gaussian_model(), _frame(), replicates=1)
    with pytest.raises(ValueError, match="alpha"):
        calibrate(_gaussian_model(), _frame(), replicates=4, alpha=1.5)


def test_unknown_tracked_label_is_rejected():
    with pytest.raises(KeyError):
        calibrate(_gaussian_model(), _frame(), replicates=4, indices=["nope"])


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------
def test_simulate_latent_recovers_the_prior_covariance():
    """The sampler must produce N(0, Q^-1) -- checked against the dense inverse."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _frame()
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_lgm(_gaussian_model(), panel)
    rng = np.random.default_rng(0)
    draws = np.array([simulate_latent(compiled, rng) for _ in range(20000)])
    expected = np.linalg.inv(compiled.precision.toarray())
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(np.cov(draws.T), expected, atol=0.05)


def test_pit_is_elementwise_for_every_marginal_representation():
    """Guards a real inconsistency: SkewNormalMarginals.cdf returns (p,) while
    TabulatedMarginals.cdf returns the (p, len(x)) cross product."""
    x = np.array([-1.0, 0.0, 1.0])
    expected = norm.cdf(x)
    p, grid = 3, 5

    gaussian = GaussianMarginals(np.zeros(p), np.ones(p))
    skew = SkewNormalMarginals(
        np.full((p, grid), 1.0 / grid), np.zeros((p, grid)),
        np.ones((p, grid)), np.zeros((p, grid)),
    )
    support = np.linspace(-9, 9, 3001)
    tabulated = TabulatedMarginals(
        np.tile(support, (p, 1)), np.tile(norm.pdf(support), (p, 1))
    )
    for marginals in (gaussian, skew, tabulated):
        result = pit(marginals, x)
        assert result.shape == (p,), type(marginals).__name__
        np.testing.assert_allclose(result, expected, atol=1e-4)


def test_gaussian_marginals_cdf_inverts_quantile():
    marginals = GaussianMarginals(np.array([0.0, 1.0]), np.array([1.0, 4.0]))
    np.testing.assert_allclose(marginals.cdf(marginals.quantile(0.3)), 0.3, atol=1e-12)
    with pytest.raises(ValueError, match="shape"):
        marginals.cdf(np.zeros(3))


# --------------------------------------------------------------------------
# Non-Gaussian: this is a measurement, not a pass/fail. The Gaussian latent
# strategy is a known approximation for count data; asserting it is calibrated
# would be asserting a falsehood. Assert only that the harness drives the path.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("likelihood, kwargs", [
    (Poisson(), {}),
    (Binomial(trials="n"), {"trials": True}),
])
def test_runs_against_non_gaussian_likelihoods(likelihood, kwargs):
    frame = _frame(n=40, groups=8)
    if kwargs.get("trials"):
        frame["n"] = 12
    model = LGM(
        response="y", likelihood=likelihood,
        predictor=Fixed("1", prior_precision=4.0) + IID("u", index="g", precision=9.0),
    )
    report = calibrate(model, frame, replicates=32, seed=3, engine="laplace")
    assert len(report.entries) == 5
    assert np.isfinite([e.pvalue for e in report.entries]).all()
    assert "verdict" in str(report)


def test_gaussian_marginals_cdf_handles_a_degenerate_component():
    """Zero variance is a point mass, not a division by zero.

    ``GaussianMarginals`` validates ``variance >= 0``, and ``quantile`` already
    degrades gracefully at zero; ``cdf`` must not be the one place that warns.
    """
    marginals = GaussianMarginals(np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    with np.errstate(all="raise"):
        values = marginals.cdf(np.array([0.5, 2.0, 0.0]))
    np.testing.assert_allclose(values, [norm.cdf(0.5), 1.0, 0.0])


# --------------------------------------------------------------------------
# Intrinsic (rank-deficient) effects. Their precision is singular, and the
# block's constraints are a basis of its null space, so the field is sampled on
# the subspace where the improper prior becomes proper. The Gaussian likelihood
# keeps the posterior exact, so these are self-tests too: they cannot fail for
# statistical reasons.
# --------------------------------------------------------------------------
def _ring_graph(size=10):   # must cover every level _frame() produces
    return {f"a{i}": [f"a{(i + 1) % size}", f"a{(i - 1) % size}"] for i in range(size)}


def _intrinsic_frame(n=60):
    frame = _frame(n=n)
    frame["t"] = [i % 10 for i in range(n)]
    return frame


# Genuinely rank-deficient: precision singular, constraints spanning its null space.
INTRINSIC = {
    "rw1": lambda: RW1("w", index="t", precision=2.0),
    "rw2": lambda: RW2("w", index="t", precision=2.0),
    "besag": lambda: Besag("s", index="g", graph=_ring_graph(), precision=2.0),
}

# BYM2 is NOT one of them: its phi-mixture of a scaled Besag and an IID part is
# proper by construction, so it compiles to a full-rank precision with zero
# constraints and was already sampleable before constrained sampling existed.
STRUCTURED_PROPER = {
    "bym2": lambda: BYM2("s", index="g", graph=_ring_graph(), precision=2.0, phi=0.6),
}


@pytest.mark.parametrize("name", sorted({**INTRINSIC, **STRUCTURED_PROPER}))
def test_structured_effects_are_calibrated(name):
    build = {**INTRINSIC, **STRUCTURED_PROPER}[name]
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.7),
        predictor=Fixed("1", prior_precision=1.0) + build(),
    )
    report = calibrate(model, _intrinsic_frame(), replicates=256, seed=17)
    assert report.ok, f"exact posterior reported as miscalibrated:\n{report}"


def test_bym2_is_full_rank_and_needs_no_constraint():
    """Pins the classification above: if BYM2 ever became intrinsic, the
    constraint-satisfaction test below would need to cover it."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    model = LGM(response="y", likelihood=Gaussian(sigma=0.7),
                predictor=Fixed("1", prior_precision=1.0) + STRUCTURED_PROPER["bym2"]())
    frame = _intrinsic_frame()
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_lgm(model, panel)
    precision = compiled.precision.toarray()
    assert compiled.constraints.shape[0] == 0
    assert np.linalg.matrix_rank(precision) == precision.shape[0]


@pytest.mark.parametrize("name", sorted(INTRINSIC))
def test_intrinsic_sampler_satisfies_its_constraints_exactly(name):
    """The constraint is satisfied by construction, not by a correction step, so
    it should hold to machine precision rather than to a tolerance."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    frame = _intrinsic_frame()
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.7),
        predictor=Fixed("1", prior_precision=1.0) + INTRINSIC[name](),
    )
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_lgm(model, panel)
    assert compiled.constraints.shape[0] > 0, "expected an intrinsic constraint"
    rng = np.random.default_rng(0)
    draws = np.array([simulate_latent(compiled, rng) for _ in range(200)])
    residual = np.abs(compiled.constraints @ draws.T).max()
    assert residual < 1e-10, f"constraint violated by {residual}"


def test_intrinsic_sampler_matches_the_constrained_prior_covariance():
    """The engines reparametrise as x = B z with z ~ N(0, (B^T Q B)^-1), so the
    prior covariance of x is B (B^T Q B)^-1 B^T. Pin that, not a hand-derived
    pseudo-inverse -- agreeing with the engine is the whole point."""
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel
    from pylgm.inference.gaussian import _constraint_null_space

    frame = _intrinsic_frame()
    model = LGM(response="y", likelihood=Gaussian(sigma=0.7),
                predictor=Fixed("1", prior_precision=1.0) + RW1("w", index="t", precision=2.0))
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_lgm(model, panel)
    rng = np.random.default_rng(0)
    draws = np.array([simulate_latent(compiled, rng) for _ in range(40000)])

    precision = compiled.precision.toarray()
    basis = _constraint_null_space(compiled.constraints, precision.shape[0])
    expected = basis @ np.linalg.inv(basis.T @ precision @ basis) @ basis.T
    np.testing.assert_allclose(np.cov(draws.T), expected, atol=0.05)
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=0.05)


@pytest.mark.parametrize("constraints, description", [
    ([{"u:a0": 1.0, "u:a1": -1.0}], "homogeneous"),
    ([({"u:a0": 1.0}, 2.5)], "nonzero rhs"),
    ([{"u:a0": 1.0, "u:a1": -1.0}, ({"u:a2": 1.0, "u:a3": 1.0}, 1.5)], "mixed"),
])
def test_extra_constraints_are_calibrated(constraints, description):
    """User-supplied ``A x = e`` rides the same reparametrisation; a nonzero rhs
    additionally shifts the prior mean of z, which is conditioning by kriging."""
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.7),
        predictor=Fixed("1", prior_precision=1.0) + IID("u", index="g", precision=2.0),
        constraints=constraints,
    )
    report = calibrate(model, _frame(), replicates=256, seed=17)
    assert report.ok, f"{description}:\n{report}"


def test_nonzero_rhs_constraint_holds_in_the_sample():
    from pylgm.compiler import compile_lgm
    from pylgm.config.schema import DataConfig
    from pylgm.data.panel import CanonicalPanel

    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.7),
        predictor=Fixed("1", prior_precision=1.0) + IID("u", index="g", precision=2.0),
        constraints=[({"u:a0": 1.0, "u:a1": 1.0}, 1.5)],
    )
    frame = _frame()
    panel = CanonicalPanel.from_frame(frame, DataConfig(time="row", response="y", panel=()))
    compiled = compile_lgm(model, panel)
    rng = np.random.default_rng(0)
    draws = np.array([simulate_latent(compiled, rng) for _ in range(200)])
    achieved = (compiled.constraints @ draws.T).T
    assert np.abs(achieved - compiled.constraint_rhs).max() < 1e-10


# ==========================================================================
# Phase 2: hyperparameters drawn from their priors and integrated over.
#
# The check is now of the marginal p(x | y) rather than p(x | y, theta), which
# is the quantity a user of hyperparameters="integrate" actually reads.
# ==========================================================================
from pylgm import Hyperparameter                                    # noqa: E402
from pylgm.optimization.empirical_bayes import OptimizationBounds   # noqa: E402
from pylgm.optimization.transforms import IdentityTransform, LogTransform  # noqa: E402
from pylgm.priors import GaussianPrior, PCPrecision                 # noqa: E402
from pylgm.validation import _PriorSampler, sample_prior           # noqa: E402


def _hyper_model(prior=None, lower=None, upper=None):
    prior = PCPrecision(upper_sd=1.0, alpha=0.01) if prior is None else prior
    return LGM(
        response="y", likelihood=Gaussian(sigma=0.7),
        predictor=Fixed("1", prior_precision=1.0)
        + IID("u", index="g", precision=Hyperparameter(
            "tau", initial=2.0, prior=prior, lower=lower, upper=upper)),
    )


def test_sample_prior_reproduces_the_pc_precision_construction():
    """PCPrecision is Exp(lambda) on the KLD distance tau^-1/2, lambda =
    -log(alpha)/upper_sd. Pinning the construction, not just "it runs"."""
    from scipy.stats import kstest

    upper_sd, alpha = 1.0, 0.01
    prior = PCPrecision(upper_sd=upper_sd, alpha=alpha)
    bounds = OptimizationBounds(1.0, 1e-6, 1e6, transform=LogTransform())
    rng = np.random.default_rng(0)
    draw = _PriorSampler(prior, bounds)          # what calibrate builds, once
    draws = np.array([draw(rng) for _ in range(4000)])
    rate = -np.log(alpha) / upper_sd
    assert kstest(draws ** -0.5, "expon", args=(0.0, 1.0 / rate)).pvalue > 0.01
    # and the one-shot public helper agrees with it
    assert sample_prior(prior, bounds, np.random.default_rng(7)) == pytest.approx(
        _PriorSampler(prior, bounds)(np.random.default_rng(7))
    )


def test_sample_prior_reproduces_a_gaussian_prior():
    from scipy.stats import kstest

    bounds = OptimizationBounds(0.0, -20.0, 20.0, transform=IdentityTransform())
    rng = np.random.default_rng(0)
    draw = _PriorSampler(GaussianPrior(mean=0.5, precision=4.0), bounds)
    draws = np.array([draw(rng) for _ in range(4000)])
    assert kstest(draws, "norm", args=(0.5, 0.5)).pvalue > 0.01


def test_sample_prior_respects_the_declared_bounds():
    """The engine only ever searches within the bounds, so the prior it uses is
    the truncated one; drawing outside would be miscalibration nobody caused."""
    bounds = OptimizationBounds(1.0, 0.5, 2.0, transform=LogTransform())
    rng = np.random.default_rng(0)
    draw = _PriorSampler(PCPrecision(upper_sd=1.0, alpha=0.01), bounds)
    draws = np.array([draw(rng) for _ in range(500)])
    assert draws.min() >= 0.5 and draws.max() <= 2.0


def test_hyperparameter_without_a_prior_is_rejected():
    model = LGM(
        response="y", likelihood=Gaussian(sigma=0.7),
        predictor=Fixed("1", prior_precision=1.0)
        + IID("u", index="g", precision=Hyperparameter("tau", initial=2.0)),
    )
    with pytest.raises(NotImplementedError, match="declare none"):
        calibrate(model, _frame(), replicates=4)


def test_integrated_fit_is_calibrated_when_the_prior_is_tight():
    """The phase-2 self-test.

    With theta pinned to a narrow interval the theta-mixture collapses to
    essentially one component, so the reported marginal is very nearly exact and
    the PIT must be uniform. If the prior draw, the per-replicate rebuild of
    Q(theta), or the integrate path were wrong, this is where it shows -- and it
    cannot fail for the mixture-shape reason the next test characterises.
    """
    report = calibrate(_hyper_model(lower=1.9, upper=2.1), _frame(),
                       replicates=128, seed=23)
    assert report.ok, f"near-fixed theta reported as miscalibrated:\n{report}"


def test_skew_normal_marginals_are_better_calibrated_than_the_gaussian_collapse():
    """The measurement F4 exists to make.

    Under ``hyperparameters="integrate"`` the true latent marginal is a mixture
    over the theta grid. ``latent_strategy="gaussian"`` reports only that
    mixture's first two moments, and a moment-matched Gaussian has the right
    variance but the wrong shape -- which the dispersion statistic sees and the
    location statistic does not. ``simplified_laplace`` tracks the shape.

    Asserted as an ordering rather than an absolute threshold: the ordering is
    stable across seeds, while whether either crosses a fixed p-value at this
    many replicates is not.
    """
    model, frame = _hyper_model(), _frame()
    collapsed = calibrate(model, frame, replicates=128, seed=31)
    skewed = calibrate(model, frame, replicates=128, seed=31,
                       latent_strategy="simplified_laplace")
    worst = lambda report: min(e.dispersion_pvalue for e in report.entries)  # noqa: E731
    assert worst(skewed) > worst(collapsed), (
        f"expected simplified_laplace to be better calibrated\n"
        f"gaussian: {worst(collapsed):.3g}\nsimplified_laplace: {worst(skewed):.3g}"
    )

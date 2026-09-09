"""Calibration validation for pyLGM posteriors (SBC via the probability
integral transform).

Simulation-based calibration asks a question no goodness-of-fit statistic can:
*if the data really came from this model, does the reported posterior cover the
truth at the advertised rate?* Draw a latent field from the prior, simulate a
response from it, refit, and ask where the truth falls in the reported marginal:

    u = F(x_true),   F = the posterior marginal CDF

Under exact inference ``u`` is Uniform(0, 1) exactly. Talts et al.
(arXiv:1804.06788) phrase this as a rank among ``L`` posterior draws; ``u`` is
the ``L -> inf`` limit, and it is the right form here because pyLGM reports
densities rather than draws -- no sampler, no Monte-Carlo noise in the
statistic, and an exact KS test instead of binned ranks.

WHAT IS AND IS NOT ASSERTED
---------------------------
The latent field is drawn from the *compiled* precision -- the same ``Q`` the
engine then conditions on. That is deliberate: a separately hand-written
generator that disagreed with the IR would show up as miscalibration with no way
to tell which side was wrong. The cost is that an error in IR assembly is
invisible here, because both sides would be wrong identically. That case is
covered by the effect oracle tests (``tests/test_grouped_spacetime_oracle.py``,
``tests/effects/test_sorbye_scaling.py``), which pin ``Q`` against closed-form
values. **Oracles pin the model; this pins the inference given the model.**
Neither substitutes for the other.

Uniformity of a scalar marginal PIT is necessary, not sufficient, for joint
correctness: a posterior can be marginally calibrated and jointly wrong. This
checks each reported marginal, which is what pyLGM reports and what users read.

Intrinsic effects are supported: the latent field is drawn in the same
constrained parametrisation the engines use, so a rank-deficient ``Q`` whose null
space its constraints span is sampled on the subspace where its prior is proper.

Hyperparameters are held fixed. Calibration *over* a hyperparameter prior needs
``hyperparameters="integrate"`` and a prior sampler; calibration *of* a
hyperparameter marginal additionally needs the natural-scale Gaussian collapse
in ``optimization/inla.py`` replaced. Both are out of scope here.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import kstest

from .compiler import _model_hyperparameters, compile_lgm
from .config.schema import DataConfig
from .data.panel import CanonicalPanel
from .inference.gaussian import (
    _constraint_null_space,
    _constraint_particular_solution,
)
from .likelihoods import (
    CompiledBernoulli,
    CompiledBinomial,
    CompiledGaussian,
    CompiledPoisson,
)

__all__ = [
    "CalibrationReport",
    "IndexCalibration",
    "calibrate",
    "pit",
    "simulate_latent",
    "simulate_response",
]


def simulate_latent(compiled, rng) -> np.ndarray:
    """Draw one latent field from a compiled model's prior.

    Sampled in **the engine's own parametrisation**, not a re-derivation of it.
    Both engines handle a constraint ``A x = e`` by reparametrising onto the null
    space of ``A``: with ``B`` an orthonormal basis of that null space and ``x_p``
    a particular solution, they set ``x = x_p + B z`` and give ``z`` the prior
    precision ``B^T Q B`` and prior mean ``-(B^T Q B)^-1 B^T Q x_p``. Drawing
    ``z`` from exactly that and mapping it back makes the simulator agree with
    the engine by construction rather than by argument -- which matters, because
    any disagreement would surface as miscalibration with no way to tell whose
    fault it was.

    This subsumes both cases in one path. Unconstrained, ``B`` is the identity and
    ``x_p`` is ``None``, leaving the ordinary ``N(0, Q^-1)``. For an intrinsic
    effect (``RW1``, ``RW2``, ``Besag``, ``BYM2``) the block's constraints are a
    basis of ``Q``'s null space, so ``B^T Q B`` is positive definite even though
    ``Q`` is singular, and ``B z`` lands in the subspace where the improper prior
    is proper -- the constraint is satisfied by construction, with no correction
    step. User-supplied ``extraconstr`` rows, including a nonzero right-hand side,
    ride the same path through ``x_p``.
    """
    # ponytail: dense throughout. Calibration models are small by construction
    # (tens of latent nodes); swap in a sparse factor if that stops being true.
    precision = compiled.precision.toarray()
    latent_size = precision.shape[0]
    constraints = compiled.constraints
    basis = _constraint_null_space(constraints, latent_size)
    reduced = basis.T @ precision @ basis
    try:
        lower = np.linalg.cholesky(reduced)
    except np.linalg.LinAlgError as error:
        raise NotImplementedError(
            "the constrained prior precision B^T Q B is not positive definite, so "
            "this prior is improper and cannot be sampled. That means Q is "
            "rank-deficient in a direction the constraints do not pin down -- an "
            "intrinsic effect whose null space is only partly constrained."
        ) from error
    # B^T Q B = L L^T, so solving L^T z = w with w ~ N(0, I) gives Cov(z) = (B^T Q B)^-1.
    z = solve_triangular(lower.T, rng.standard_normal(reduced.shape[0]), lower=False)
    x_p = _constraint_particular_solution(constraints, compiled.constraint_rhs, latent_size)
    if x_p is None:
        return basis @ z
    # Nonzero rhs shifts the prior mean of z by -(B^T Q B)^-1 B^T Q x_p, which is
    # what conditioning by kriging on A x = e amounts to in this parametrisation.
    linear = basis.T @ (precision @ x_p)
    mean = solve_triangular(
        lower.T, solve_triangular(lower, -linear, lower=True), lower=False
    )
    return x_p + basis @ (z + mean)


def simulate_response(compiled, latent: np.ndarray, rng) -> np.ndarray:
    """Draw a response vector from the likelihood at ``eta = A x + offset``.

    The mean comes from the likelihood's own ``response_mean``, so the link and
    any per-row auxiliary data (binomial trials) are applied by the shipped code
    rather than restated here.
    """
    eta = compiled.design @ np.asarray(latent, dtype=float) + compiled.offset
    likelihood = compiled.likelihood
    mean = np.asarray(likelihood.response_mean(eta), dtype=float)
    if isinstance(likelihood, CompiledGaussian):
        return rng.normal(mean, likelihood.sigma)
    if isinstance(likelihood, CompiledPoisson):
        return rng.poisson(mean).astype(float)
    if isinstance(likelihood, CompiledBernoulli):
        return rng.binomial(1, mean).astype(float)
    if isinstance(likelihood, CompiledBinomial):
        trials = np.asarray(likelihood.trials, dtype=float)
        return rng.binomial(trials.astype(int), mean / trials).astype(float)
    raise NotImplementedError(
        f"no simulator for {type(likelihood).__name__}; calibration supports "
        "Gaussian, Poisson, Bernoulli and Binomial responses"
    )


def pit(marginals, truth: np.ndarray) -> np.ndarray:
    """``F_i(truth_i)`` for every component of a latent-marginal object."""
    values = np.asarray(marginals.cdf(np.asarray(truth, dtype=float)), dtype=float)
    # GaussianMarginals.cdf and SkewNormalMarginals.cdf are elementwise and
    # return (p,); TabulatedMarginals.cdf returns the (p, len(x)) cross product.
    # The library is inconsistent here, so normalise rather than assume.
    if values.ndim == 2:
        values = np.diagonal(values)
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class IndexCalibration:
    """One latent component's PIT values, tested two ways.

    A single uniformity test is not enough, and this was measured rather than
    assumed. The two ways a marginal goes wrong leave different fingerprints:

    * a **location** error (posterior mean off) shifts the PIT distribution
      sideways -- the KS statistic is powerful against this, catching a 0.1-sigma
      shift at 128 replicates;
    * a **dispersion** error (posterior variance off) leaves the PIT median at
      0.5 and pushes mass symmetrically toward both tails, producing a U- or
      dome-shaped histogram that KS barely sees. Against a 10% inflated SD, plain
      KS has ~0.01 power even at 1024 replicates.

    Folding the PIT about its centre -- testing ``|u - 0.5| * 2`` for uniformity,
    which it is under exact inference -- turns that symmetric deviation back into
    a one-sided one the same KS machinery detects (~0.5 power at 1024).

    So ``pvalue`` is the location test and ``dispersion_pvalue`` the scale test,
    and a component passes only if both do.
    """

    label: str
    index: int
    statistic: float
    pvalue: float
    dispersion_statistic: float
    dispersion_pvalue: float
    values: np.ndarray

    @property
    def mean_pit(self) -> float:
        return float(np.mean(self.values))


@dataclass(frozen=True)
class CalibrationReport:
    """Per-component calibration of a model's reported latent marginals.

    ``ok`` applies a Bonferroni correction across the tested components, because
    the components share replicates and an uncorrected sweep over many of them
    manufactures failures.
    """

    replicates: int
    entries: tuple[IndexCalibration, ...]
    alpha: float
    seed: int

    @property
    def threshold(self) -> float:
        # Two tests per component, so the correction counts both.
        return self.alpha / max(2 * len(self.entries), 1)

    def _passed(self, entry: IndexCalibration) -> bool:
        return (
            entry.pvalue >= self.threshold
            and entry.dispersion_pvalue >= self.threshold
        )

    @property
    def ok(self) -> bool:
        return all(self._passed(entry) for entry in self.entries)

    @property
    def failures(self) -> tuple[IndexCalibration, ...]:
        return tuple(e for e in self.entries if not self._passed(e))

    def __str__(self) -> str:
        width = max((len(e.label) for e in self.entries), default=9)
        head = (
            f"calibration over {self.replicates} replicates "
            f"(seed {self.seed}, Bonferroni threshold p >= {self.threshold:.4g})\n"
            f"  {'component'.ljust(width)}  {'p(location)':>12}  {'p(dispersion)':>14}"
            f"  {'mean PIT':>9}\n"
        )
        rows = "\n".join(
            f"  {e.label.ljust(width)}  {e.pvalue:12.4g}  {e.dispersion_pvalue:14.4g}  "
            f"{e.mean_pit:9.4f}{'' if self._passed(e) else '   <-- MISCALIBRATED'}"
            for e in self.entries
        )
        verdict = "PASS" if self.ok else f"FAIL ({len(self.failures)} component(s))"
        return f"{head}{rows}\n  verdict: {verdict}"


def _tracked_indices(labels, requested, limit: int = 5) -> tuple[int, ...]:
    """Default to a spread across the latent vector rather than all of it.

    PIT values are independent across replicates at a fixed component, but
    *dependent* across components within one replicate -- they share a dataset.
    Testing a handful keeps the multiplicity correction mild and keeps the
    per-component sample size, not the component count, as the power knob.
    """
    width = len(labels)
    if requested is None:
        if width <= limit:
            return tuple(range(width))
        return tuple(sorted({0, *(round(k * (width - 1) / (limit - 1)) for k in range(limit))}))
    resolved = []
    for item in requested:
        if isinstance(item, str):
            try:
                resolved.append(labels.index(item))
            except ValueError:
                raise KeyError(f"unknown latent label {item!r}") from None
        else:
            index = int(item)
            if not 0 <= index < width:
                raise IndexError(f"latent index {index} out of range for width {width}")
            resolved.append(index)
    if not resolved:
        raise ValueError("indices must not be empty")
    return tuple(resolved)


def _check_fixed_hyperparameters(model) -> None:
    """Reject a declared ``Hyperparameter``, which would test the wrong thing.

    The latent field is simulated from ``Q`` at the hyperparameter's ``initial``
    value, but ``fit`` re-estimates it from each simulated dataset by empirical
    Bayes. The reported posterior is then conditional on ``theta_hat(y)`` rather
    than on the theta that generated the data, so the PIT is not uniform even
    under exact inference -- and it comes out *looking* roughly calibrated, so
    the check would pass while measuring something other than what it claims.

    Calibrating over a hyperparameter prior needs draws from that prior and
    ``hyperparameters="integrate"``; neither is implemented here.
    """
    declared = _model_hyperparameters(model)
    if declared:
        names = ", ".join(sorted({repr(hp.name) for _, hp in declared}))
        raise NotImplementedError(
            f"calibration holds hyperparameters fixed, but this model declares {names}. "
            "Empirical Bayes would re-estimate them from each simulated dataset, so the "
            "check would silently test a different quantity. Replace them with the fixed "
            "values you want to calibrate at, e.g. IID(..., precision=9.0)."
        )


def _check_proper_prior(compiled, max_prior_sd: float) -> None:
    """Reject near-improper priors, which make simulated data meaningless.

    ``Fixed`` defaults to ``prior_precision=1e-6`` -- a prior SD of 1000. That is
    a sensible default for *fitting* (it is deliberately uninformative) and
    useless for calibration: drawing an intercept from ``N(0, 1000^2)`` produces
    a linear predictor no likelihood can survive. SBC needs a proper prior, so
    this is a hard error naming the offending component.
    """
    diagonal = compiled.precision.diagonal()
    loose = np.nonzero(diagonal < 1.0 / max_prior_sd**2)[0]
    if loose.size:
        names = ", ".join(repr(compiled.labels[i]) for i in loose[:4])
        raise ValueError(
            f"prior on {names} has SD above max_prior_sd={max_prior_sd:g} "
            "(near-improper), so simulating from it would not produce usable data. "
            "Calibration needs proper priors: tighten it, e.g. "
            "Fixed('1', prior_precision=1.0), or raise max_prior_sd deliberately."
        )


def calibrate(
    model,
    frame,
    *,
    replicates: int = 128,
    indices=None,
    seed: int = 0,
    alpha: float = 0.01,
    max_prior_sd: float = 100.0,
    **fit_kwargs,
) -> CalibrationReport:
    """Check that ``model``'s reported latent marginals are calibrated.

    ``frame`` supplies the design -- its index columns, sizes and any covariates
    are used as-is, and its response column is overwritten by simulated data, so
    the values already in it do not matter.

    Extra keyword arguments are passed to ``model.fit`` (``engine="laplace"`` for
    a non-Gaussian likelihood, ``latent_strategy=...`` to compare approximations).

    >>> report = calibrate(model, frame, engine="laplace")   # doctest: +SKIP
    >>> print(report)                                        # doctest: +SKIP
    """
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    if not 0 < alpha < 1:
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    response = model.response
    working = frame.copy()
    panel = CanonicalPanel.from_frame(
        working, DataConfig(time=_time_column(working), response=response, panel=())
    )
    compiled = compile_lgm(model, panel)
    _check_fixed_hyperparameters(model)
    # The caller should not have to know that a non-Gaussian likelihood needs the
    # Laplace engine -- the compiled likelihood already says so.
    fit_kwargs.setdefault(
        "engine",
        "exact_gaussian" if isinstance(compiled.likelihood, CompiledGaussian) else "laplace",
    )
    _check_proper_prior(compiled, max_prior_sd)
    tracked = _tracked_indices(list(compiled.labels), indices)

    rng = np.random.default_rng(seed)
    collected = np.empty((replicates, len(tracked)))
    for replicate in range(replicates):
        truth = simulate_latent(compiled, rng)
        working[response] = simulate_response(compiled, truth, rng)
        result = model.fit(working, **fit_kwargs)
        # The handoff records four separate column-order bugs in this codebase.
        # A silent misalignment here would compare component i's posterior with
        # component j's truth and read as miscalibration, so pin it.
        if tuple(result.labels) != tuple(compiled.labels):
            raise RuntimeError(
                "fitted labels do not match the compiled labels; the PIT would "
                "compare mismatched components"
            )
        collected[replicate] = pit(result.latent_marginals(), truth)[list(tracked)]

    entries = tuple(
        _calibration_entry(compiled.labels[index], index, collected[:, position])
        for position, index in enumerate(tracked)
    )
    return CalibrationReport(
        replicates=replicates, entries=entries, alpha=alpha, seed=seed
    )


def _calibration_entry(label: str, index: int, values: np.ndarray) -> IndexCalibration:
    location = kstest(values, "uniform")
    # |u - 0.5| * 2 is Uniform(0, 1) too when u is, and a variance error that
    # leaves the median at 0.5 shows up here as a one-sided shift.
    dispersion = kstest(np.abs(values - 0.5) * 2.0, "uniform")
    return IndexCalibration(
        label=label,
        index=index,
        statistic=float(location.statistic),
        pvalue=float(location.pvalue),
        dispersion_statistic=float(dispersion.statistic),
        dispersion_pvalue=float(dispersion.pvalue),
        values=values,
    )


def _time_column(working) -> str:
    """The column a CanonicalPanel orders rows by, added to the copy if absent.

    Row order is irrelevant to calibration -- the response is overwritten anyway --
    so requiring the caller to supply an ordering column would be a papercut with
    nothing behind it.
    """
    for candidate in ("row", "time", "t"):
        if candidate in working.columns:
            return candidate
    working["__calibration_row"] = range(len(working))
    return "__calibration_row"

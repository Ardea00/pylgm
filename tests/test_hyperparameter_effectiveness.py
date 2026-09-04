"""Structural guard: every declared Hyperparameter must actually affect the fit.

This project has shipped five separate instances of one failure mode -- a
user-declared ``Hyperparameter`` that is registered, optimised and reported in
``result.hyperparameters`` while having *zero effect* on the compiled model,
with no error and plausible-looking numbers:

1. A ``Shared`` effect's precision.
2. ``Weighted``'s inner effect.
3. ``Copy``'s own scale.
4. A copied target's precision (``_copied_family_block`` dropping the target's
   own ``parameter``/``scale`` when folding a copy in).
5. The prediction context under ``hyperparameters="integrate"`` (Finding C1:
   ``build_prediction_context`` freezing a copy scale's *initial* value
   instead of its fitted marginal mean).

Nothing in the compiler structurally cross-checks declared -> registered ->
effective: ``_family_optimization_inputs`` builds ``initial`` from declared
Hyperparameters and ``bounds`` from ``family.parameter_bounds`` and never
compares either against what the compiled design/precision actually do. This
module is that missing cross-check, applied directly to ``compile_family``'s
output: for every model below, and for every one of its non-likelihood-scalar
hyperparameters, materializing the family at two different values for that
one parameter (holding every other parameter at its initial) must change the
compiled design or precision. If a future change introduces a sixth dead
parameter, one of the cases below should start failing.

The model list is the reviewer's own sweep table from the final-slice review
that found instance 5 above.
"""
import numpy as np
import pandas as pd
import pytest

from pylgm import (
    AR1,
    Besag,
    BYM2,
    Copy,
    Fixed,
    Hyperparameter,
    IID,
    LGM,
    Poisson,
    ProperCAR,
    Replicated,
    RW1,
    SAR,
    Seasonal,
    Weighted,
)
from pylgm.compiler import _model_hyperparameters, compile_family
from pylgm.config.schema import DataConfig
from pylgm.data.panel import CanonicalPanel

N = 12
LEVELS = [str(k) for k in range(N)]


def _chain_graph(n=N):
    return {str(i): [str(j) for j in (i - 1, i + 1) if 0 <= j <= n - 1] for i in range(n)}


def _ring_graph(n=N):
    return {str(i): [str((i + 1) % n)] for i in range(n)}


CHAIN_GRAPH = _chain_graph()
RING_GRAPH = _ring_graph()


def _frame(seed=0, n_rows=90):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "i": rng.choice(LEVELS, n_rows),
        "j": rng.choice(LEVELS, n_rows),
        "k": rng.choice(LEVELS, n_rows),
        "t": rng.choice(LEVELS, n_rows),
        "w": rng.normal(1.0, 0.1, n_rows),
        "firm": rng.choice(["f0", "f1", "f2"], n_rows),
        "z": rng.normal(1.0, 0.1, n_rows),
        "y": rng.poisson(3.0, n_rows).astype(float),
        "row": range(n_rows),
    })


FRAME = _frame()
PANEL = CanonicalPanel.from_frame(FRAME, DataConfig(time="row", response="y", panel=()))


def _tau(name="u.tau"):
    return Hyperparameter(name, initial=1.0)


def _rho(name="u.rho"):
    return Hyperparameter(name, initial=0.0, transform="logit")


def _phi(name="u.phi"):
    return Hyperparameter(name, initial=0.5, transform="logit")


def _beta(name="beta"):
    return Hyperparameter(name, initial=1.0)


def _model(predictor):
    return LGM(response="y", likelihood=Poisson(), predictor=predictor)


# The reviewer's sweep table: every combination of a target effect's own
# estimated structural parameter(s) with a fixed and/or estimated Copy scale.
MODEL_TABLE = [
    (
        "IID(tau)+Copy(fixed)",
        _model(
            Fixed("1") + IID("u", index="i", precision=_tau())
            + Copy("u", index="j", scale=2.0)
        ),
    ),
    (
        "AR1(rho)+Copy(fixed)",
        _model(
            Fixed("1") + AR1("u", index="i", precision=1.0, rho=_rho())
            + Copy("u", index="j", scale=2.0)
        ),
    ),
    (
        "IID(1)+Copy(beta)",
        _model(
            Fixed("1") + IID("u", index="i", precision=1.0)
            + Copy("u", index="j", scale=_beta())
        ),
    ),
    (
        "AR1(tau,rho-fixed)+Copy(beta)",
        _model(
            Fixed("1") + AR1("u", index="i", precision=_tau(), rho=0.3)
            + Copy("u", index="j", scale=_beta())
        ),
    ),
    (
        "IID(tau)+Copy(beta)",
        _model(
            Fixed("1") + IID("u", index="i", precision=_tau())
            + Copy("u", index="j", scale=_beta())
        ),
    ),
    (
        "Weighted(IID(tau))+Copy(beta)",
        _model(
            Fixed("1") + Weighted(IID("u", index="i", precision=_tau()), by="w")
            + Copy("u", index="j", scale=_beta())
        ),
    ),
    (
        "Seasonal(tau)+Copy(fixed)",
        _model(
            Fixed("1") + Seasonal("u", index="i", period=4, precision=_tau())
            + Copy("u", index="j", scale=2.0)
        ),
    ),
    (
        "Besag(tau)+Copy(beta)",
        _model(
            Fixed("1") + Besag("u", index="i", graph=CHAIN_GRAPH, precision=_tau())
            + Copy("u", index="j", scale=_beta())
        ),
    ),
    (
        "RW1(tau)+Copy(fixed)+Copy(beta)",
        _model(
            Fixed("1") + RW1("u", index="i", precision=_tau())
            + Copy("u", index="j", scale=2.0)
            + Copy("u", index="k", scale=_beta())
        ),
    ),
    (
        "BYM2(phi)+Copy(fixed)",
        _model(
            Fixed("1") + BYM2("u", index="i", graph=CHAIN_GRAPH, precision=1.0, phi=_phi())
            + Copy("u", index="j", scale=2.0)
        ),
    ),
    (
        "ProperCAR(rho,tau)+Copy(fixed)",
        _model(
            Fixed("1")
            + ProperCAR("u", index="i", graph=CHAIN_GRAPH, rho=_rho(), precision=_tau())
            + Copy("u", index="j", scale=2.0)
        ),
    ),
    (
        "SAR(rho,tau)+Copy(fixed)",
        _model(
            Fixed("1") + SAR("u", index="i", graph=RING_GRAPH, rho=_rho(), precision=_tau())
            + Copy("u", index="j", scale=2.0)
        ),
    ),
    (
        "Replicated(IID(tau))",
        _model(Fixed("1") + Replicated(
            IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)), over="firm")),
    ),
    (
        "Replicated(AR1(rho))",
        # initial=0.2, not 0.5: AR1 rho's default (-1, 1) bounds make
        # _alternate_value's primary candidate land at ~0.4999995 regardless of
        # `initial` (0.75 of the span from -1), so initial=0.5 would coincide
        # with the alternate almost exactly and this row would not exercise a
        # real change -- see _alternate_value's fallback, which only guards
        # exact/near-exact collisions, not this one 5e-7 short of its epsilon.
        _model(Fixed("1") + Replicated(
            AR1(
                "u", index="t", precision=1.0,
                rho=Hyperparameter("rho", initial=0.2, transform="logit"),
            ),
            over="firm")),
    ),
    (
        "Replicated(Weighted(IID(tau)))",
        _model(Fixed("1") + Replicated(
            Weighted(IID("u", index="t", precision=Hyperparameter("tau", initial=1.0)), by="z"),
            over="firm")),
    ),
]


def _alternate_value(bounds) -> float:
    """A second value for ``bounds`` guaranteed to lie in its declared domain.

    Three-quarters of the way across ``[lower, upper]``, falling back to a
    quarter of the way across when that lands on (or near) ``initial`` itself.
    The tolerance is a fraction of the span, not of ``initial``: an absolute
    guard let AR1's ``rho`` land 5e-7 from a declared ``initial=0.5``, inside
    ``np.allclose``'s tolerance, so the perturbation registered as no change
    and the row failed spuriously -- either way strictly inside the interval
    ``OptimizationBounds`` already validated, so no per-parameter/per-
    transform special-casing is needed here.
    """
    span = bounds.upper - bounds.lower
    candidate = bounds.lower + 0.75 * span
    if abs(candidate - bounds.initial) < 1e-6 * span:
        candidate = bounds.lower + 0.25 * span
    return candidate


def _likelihood_scalar_names(model) -> set:
    return {
        hp.name for target, hp in _model_hyperparameters(model)
        if target in ("sigma", "phi", "shape")
    }


def _cases():
    cases = []
    for label, model in MODEL_TABLE:
        family = compile_family(model, PANEL)
        scalar_names = _likelihood_scalar_names(model)
        testable = [name for name in family.parameter_names if name not in scalar_names]
        assert testable, f"{label} declares no testable (non-likelihood-scalar) hyperparameter"
        for name in testable:
            cases.append((label, model, name))
    return cases


CASES = _cases()


@pytest.mark.parametrize(
    "label, model, name", CASES, ids=[f"{label}::{name}" for label, _, name in CASES]
)
def test_hyperparameter_changes_the_compiled_model(label, model, name):
    family = compile_family(model, PANEL)
    baseline = {n: family.parameter_bounds[n].initial for n in family.parameter_names}
    variant = dict(baseline)
    variant[name] = _alternate_value(family.parameter_bounds[name])

    at_baseline = family.materialize(baseline)
    at_variant = family.materialize(variant)

    design_changed = not np.allclose(
        at_baseline.design.toarray(), at_variant.design.toarray()
    )
    precision_changed = not np.allclose(
        at_baseline.precision.toarray(), at_variant.precision.toarray(), atol=1e-10
    )
    assert design_changed or precision_changed, (
        f"{label}: hyperparameter {name!r} is declared, registered in "
        "family.parameter_names, and optimizable, but two different values for "
        "it produced an identical compiled design AND precision -- it has zero "
        "effect on the fit."
    )
